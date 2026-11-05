# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset RTC evaluation preserves temporal overlap and relative EE anchors."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets")

from examples.rtc import eval_dataset  # noqa: E402
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.rtc import RTCConfig  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AbsoluteEEActionsStep,
    EEStateStep,
    NormalizerProcessorStep,
    RelativeEEActionsStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.factory import make_policy_processor_pipelines  # noqa: E402
from lerobot.utils.constants import ACTION, LANGUAGE_PERSISTENT, OBS_STATE  # noqa: E402
from lerobot.utils.rotation_representations import RotationRepresentation  # noqa: E402


class _IndexDataset:
    def __init__(self, ranges: list[tuple[int, int]]) -> None:
        self.meta = SimpleNamespace(
            episodes=[{"dataset_from_index": start, "dataset_to_index": end} for start, end in ranges]
        )
        self.reads: list[int] = []

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self.reads.append(index)
        episode = next(
            i
            for i, row in enumerate(self.meta.episodes)
            if row["dataset_from_index"] <= index < row["dataset_to_index"]
        )
        return {"index": torch.tensor(index), "episode_index": torch.tensor(episode)}


def _evaluator(tmp_path, *, chunk: int = 8, delay: int = 2, horizon: int = 3):
    evaluator = object.__new__(eval_dataset.RTCEvaluator)
    evaluator.cfg = SimpleNamespace(
        policy=SimpleNamespace(chunk_size=chunk, rtc_training_max_delay=2),
        rtc=RTCConfig(execution_horizon=horizon),
        inference_delay=delay,
        device="cpu",
        output_dir=str(tmp_path / "rtc"),
        seed=7,
    )
    evaluator.device = "cpu"
    return evaluator


@pytest.mark.parametrize("chunk", [6, 7, 8])
@pytest.mark.parametrize("endpoint", ["first", "last"])
def test_sampling_keeps_both_full_chunks_in_the_same_episode(tmp_path, monkeypatch, chunk, endpoint):
    evaluator = _evaluator(tmp_path, chunk=chunk)
    evaluator.dataset = _IndexDataset([(0, chunk), (chunk, chunk + 30)])
    monkeypatch.setattr(eval_dataset.random, "choice", lambda ranges: ranges[-1])
    monkeypatch.setattr(
        eval_dataset.random, "randint", lambda first, last: first if endpoint == "first" else last
    )

    first, second, offset = evaluator._sample_overlapping_pair()

    assert offset == chunk // 2
    assert second["index"].item() == first["index"].item() + offset
    assert first["episode_index"].item() == second["episode_index"].item() == 1
    assert chunk <= first["index"].item()
    assert second["index"].item() + chunk <= chunk + 30
    assert first["index"].shape == second["index"].shape == (1,)


@pytest.mark.parametrize(
    "delay,horizon,error",
    [
        (-1, 3, "inference_delay"),
        (5, 3, "inference_delay"),
        (2, 0, "execution_horizon"),
        (2, 5, "execution_horizon"),
    ],
)
def test_evaluator_rejects_prefix_bounds_outside_real_overlap_before_loading_data(
    tmp_path, delay, horizon, error
):
    cfg = _evaluator(tmp_path, delay=delay, horizon=horizon).cfg

    with pytest.raises(ValueError, match=error):
        eval_dataset.RTCEvaluator(cfg)


def test_sampling_does_not_join_short_adjacent_episodes(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.dataset = _IndexDataset([(0, 8), (8, 16)])

    with pytest.raises(ValueError, match="episode with at least 12 frames"):
        evaluator._sample_overlapping_pair()

    assert not evaluator.dataset.reads


def test_sampling_collates_language_columns(tmp_path):
    """Language rows hold ``None`` fields, which PyTorch's default collate rejects."""
    row = {"role": "assistant", "content": "reach", "style": "subtask", "camera": None, "tool_calls": None}

    class _LanguageDataset(_IndexDataset):
        def __getitem__(self, index: int) -> dict:
            return {**super().__getitem__(index), LANGUAGE_PERSISTENT: [row]}

    evaluator = _evaluator(tmp_path)
    evaluator.dataset = _LanguageDataset([(0, 30)])

    first, second, _ = evaluator._sample_overlapping_pair()

    assert first[LANGUAGE_PERSISTENT] == second[LANGUAGE_PERSISTENT] == [[row]]
    assert first["index"].shape == (1,)


def test_trained_evaluator_rejects_delay_above_checkpoint_maximum_before_loading_data(tmp_path):
    cfg = _evaluator(tmp_path, delay=3, horizon=3).cfg  # Fits the 4-step overlap, exceeds the trained 2.
    cfg.rtc.mode = "trained"

    with pytest.raises(ValueError, match="inference_delay"):
        eval_dataset.RTCEvaluator(cfg)


@pytest.mark.parametrize("max_delay,horizon", [(None, 3), (0, 3), (3, 2), (3, 6)])
def test_trained_evaluator_validates_checkpoint_bounds_before_loading_data(tmp_path, max_delay, horizon):
    cfg = _evaluator(tmp_path, horizon=horizon).cfg
    cfg.rtc.mode = "trained"
    if max_delay is None:
        del cfg.policy.rtc_training_max_delay
        cfg.policy.type = "smolvla"
    else:
        cfg.policy.rtc_training_max_delay = max_delay

    with pytest.raises(ValueError, match="rtc_training_max_delay|execution_horizon"):
        eval_dataset.RTCEvaluator(cfg)


def test_reanchoring_applies_the_deployment_continuation_checks(tmp_path):
    evaluator = _evaluator(tmp_path)
    model, canonical = torch.zeros(4, 10), torch.full((4, 8), float("nan"))

    evaluator.preprocessor = SimpleNamespace(steps=[])
    assert torch.equal(evaluator._reanchor_prefix(model, canonical), model)  # No anchor to change.

    evaluator.preprocessor = SimpleNamespace(
        steps=[RelativeEEActionsStep(enabled=True, rot_repr=RotationRepresentation.rot6d)]
    )
    with pytest.raises(ValueError, match="canonical continuation"):
        evaluator._reanchor_prefix(model, canonical)


@pytest.mark.parametrize("mode", ["guided", "trained"])
def test_evaluation_reanchors_the_previous_tail_after_preserving_its_original_anchor(
    tmp_path, monkeypatch, mode
):
    evaluator = _evaluator(tmp_path)
    evaluator.cfg.rtc.mode = mode
    width, chunk, offset = 10, 8, 4
    relative = RelativeEEActionsStep(enabled=True, rot_repr=RotationRepresentation.rot6d)
    stats = {ACTION: {"mean": torch.linspace(-0.3, 0.3, width), "std": torch.linspace(0.5, 1.5, width)}}
    normalization = {
        "features": {ACTION: PolicyFeature(FeatureType.ACTION, (width,))},
        "norm_map": {FeatureType.ACTION: NormalizationMode.MEAN_STD},
        "stats": stats,
    }
    evaluator.preprocessor, evaluator.postprocessor = make_policy_processor_pipelines(
        input_steps=[
            relative,
            EEStateStep(enabled=True, rot_repr=RotationRepresentation.rot6d),
            NormalizerProcessorStep(**normalization),
        ],
        output_steps=[
            UnnormalizerProcessorStep(**normalization),
            AbsoluteEEActionsStep(enabled=True, relative_step=relative),
        ],
    )
    positions = torch.arange(chunk + offset, dtype=torch.float32)
    targets = torch.zeros(chunk + offset, 8)
    targets[:, 0] = positions * 0.05
    targets[:, 1] = positions * 0.03
    targets[:, 6] = 1.0
    targets[:, 7] = positions * 0.01
    first_state = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.5]])
    second_state = torch.tensor([[2.0, 3.0, 4.0, 0.0, 0.0, 2**-0.5, 2**-0.5, 0.5]])
    first = {OBS_STATE: first_state, ACTION: targets[:chunk].unsqueeze(0)}
    second = {OBS_STATE: second_state, ACTION: targets[offset:].unsqueeze(0)}
    monkeypatch.setattr(evaluator, "_sample_overlapping_pair", lambda: (first, second, offset))
    calls = {}

    def init_policy(name: str, rtc_enabled: bool, rtc_debug: bool):
        def predict(batch: dict, **kwargs) -> torch.Tensor:
            calls[name] = {"actions": batch[ACTION].clone(), "kwargs": kwargs, "enabled": rtc_enabled}
            actions = batch[ACTION].clone()
            if rtc_enabled and mode == "trained":
                delay = kwargs["inference_delay"]
                actions[:, :delay] = kwargs["prev_chunk_left_over"][:delay]
            return actions

        return SimpleNamespace(
            predict_action_chunk=predict,
            config=SimpleNamespace(chunk_size=chunk, max_action_dim=width, num_steps=2),
            model=SimpleNamespace(
                sample_noise=lambda shape, device: (
                    torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
                )
            ),
            rtc_processor=SimpleNamespace(
                reset_tracker=lambda: None,
                tracker=SimpleNamespace(get_all_steps=lambda: []),
                get_all_debug_steps=lambda: [],
            ),
        )

    monkeypatch.setattr(evaluator, "_init_policy", init_policy)
    monkeypatch.setattr(evaluator, "_destroy_policy", lambda policy, name: None)
    monkeypatch.setattr(evaluator, "plot_tracked_data", lambda *args: None)
    monkeypatch.setattr(evaluator, "plot_final_actions_comparison", lambda *args: None)

    evaluator.run_evaluation()

    assert list(calls) == ["policy_prev_chunk", "policy_no_rtc", "policy_rtc"]
    prefix = calls["policy_rtc"]["kwargs"]["prev_chunk_left_over"]
    # Both samples describe the same trajectory. Correct slicing and anchoring
    # therefore make the prefix equal the overlapping clean training targets.
    torch.testing.assert_close(
        prefix, calls["policy_rtc"]["actions"][0, : chunk - offset], atol=1e-5, rtol=1e-5
    )
    assert prefix.shape == (chunk - offset, width)
    assert calls["policy_rtc"]["kwargs"]["inference_delay"] == 2
    torch.testing.assert_close(
        calls["policy_no_rtc"]["kwargs"]["noise"], calls["policy_rtc"]["kwargs"]["noise"]
    )
    assert not calls["policy_no_rtc"]["enabled"]
    assert calls["policy_rtc"]["enabled"]
