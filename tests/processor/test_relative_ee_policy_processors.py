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

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AbsoluteActionsProcessorStep,
    AbsoluteEEActionsStep,
    EEStateStep,
    NormalizerProcessorStep,
    RelativeActionsProcessorStep,
    RelativeEEActionsStep,
    TokenizerProcessorStep,
    TransitionKey,
    batch_to_transition,
    tokenizer_processor,
)
from lerobot.processor.relative_ee_action_processor import (  # noqa: E402
    ee_to_rot_repr,
    quat_to_rotmat,
    to_relative_ee_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.rotation_representations import ROT_REPR_DIM, RotationRepresentation  # noqa: E402

SUPPORTED_REPRESENTATIONS = [r for r in RotationRepresentation if ROT_REPR_DIM[r] is not None]


def _model_space_stats(rot_repr: RotationRepresentation) -> dict[str, dict[str, torch.Tensor]]:
    dim = 4 + ROT_REPR_DIM[rot_repr]
    stats = {
        "mean": torch.zeros(dim),
        "std": torch.ones(dim),
        "min": -torch.ones(dim),
        "max": torch.ones(dim),
    }
    return {OBS_STATE: dict(stats), ACTION: dict(stats)}


def _relative_ee_smolvla_config(rot_repr: RotationRepresentation) -> SmolVLAConfig:
    config = SmolVLAConfig(use_relative_ee=True, device="cpu", rotation_representation=rot_repr)
    dim = 4 + ROT_REPR_DIM[rot_repr]
    config.input_features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (dim,))}
    config.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (dim,))}
    return config


def _step(pipeline, step_type):
    return next(step for step in pipeline.steps if isinstance(step, step_type))


class _FakeTokenizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, text: list[str], max_length: int, **_kwargs: Any) -> dict[str, torch.Tensor]:
        self.prompts = list(text)
        shape = (len(text), max_length)
        return {"input_ids": torch.zeros(shape, dtype=torch.long), "attention_mask": torch.ones(shape)}

    def save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / "tokenizer_config.json").write_text("{}")


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: _FakeTokenizer()
    )


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_smolvla_relative_ee_pipeline_composition(rot_repr):
    config = _relative_ee_smolvla_config(rot_repr)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, _model_space_stats(rot_repr))

    pre_types = [type(step) for step in preprocessor.steps]
    relative_step = _step(preprocessor, RelativeEEActionsStep)
    absolute_step = _step(postprocessor, AbsoluteEEActionsStep)

    # The relative step needs the raw 8D state, so it must run before the state conversion and normalization.
    assert (
        pre_types.index(RelativeEEActionsStep)
        < pre_types.index(EEStateStep)
        < pre_types.index(NormalizerProcessorStep)
    )
    assert relative_step.state_frame == config.observation_delta_indices.index(0)
    assert absolute_step.relative_step is relative_step
    assert relative_step.rot_repr == rot_repr
    assert _step(preprocessor, EEStateStep).rot_repr == rot_repr
    assert config.action_delta_indices == list(range(config.chunk_size))


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_smolvla_relative_ee_processors_save_load_and_reconnect(rot_repr):
    config = _relative_ee_smolvla_config(rot_repr)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, _model_space_stats(rot_repr))

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor.save_pretrained(tmpdir)
        postprocessor.save_pretrained(tmpdir)
        loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(config, pretrained_path=tmpdir)

    relative_step = _step(loaded_preprocessor, RelativeEEActionsStep)
    absolute_step = _step(loaded_postprocessor, AbsoluteEEActionsStep)
    assert absolute_step.relative_step is relative_step
    assert relative_step.state_frame == 0
    assert relative_step.rot_repr == rot_repr
    assert _step(loaded_preprocessor, EEStateStep).rot_repr == rot_repr

    # Check behavior after deserialization as well as restored configuration.
    state = torch.tensor([[0.2, -0.1, 0.3, 0, 0, 0, 1, 0.5]])
    target = torch.tensor([[[0.4, 0.1, 0.2, 0, 0, 0.6, 0.8, 0.7]]])
    encoded = relative_step(batch_to_transition({OBS_STATE: state, ACTION: target}))[TransitionKey.ACTION]
    assert encoded.shape == (1, 1, 4 + ROT_REPR_DIM[rot_repr])
    decoded = absolute_step(batch_to_transition({ACTION: encoded}))[TransitionKey.ACTION]
    torch.testing.assert_close(decoded[..., :3], target[..., :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(decoded[..., -1:], target[..., -1:])
    torch.testing.assert_close(
        quat_to_rotmat(decoded[..., 3:7]), quat_to_rotmat(target[..., 3:7]), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("identity_norm", [False, True])
def test_legacy_relative_ee_checkpoint_loads_without_rotation_fields(tmp_path, identity_norm):
    config = _relative_ee_smolvla_config(RotationRepresentation.rot6d)
    config.rot6d_identity_norm = identity_norm
    config.save_pretrained(tmp_path)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        config, _model_space_stats(RotationRepresentation.rot6d)
    )
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)

    # Reproduce the schema before rotation representations became configurable.
    for path in tmp_path.glob("*.json"):
        saved = json.loads(path.read_text())
        saved.pop("rotation_representation", None)
        for step in saved.get("steps", []):
            step.get("config", {}).pop("rot_repr", None)
        path.write_text(json.dumps(saved))

    loaded_config = SmolVLAConfig.from_pretrained(tmp_path)
    assert loaded_config.rotation_representation == RotationRepresentation.rot6d
    loaded_pre, loaded_post = make_pre_post_processors(loaded_config, pretrained_path=tmp_path)
    relative = _step(loaded_pre, RelativeEEActionsStep)
    absolute = _step(loaded_post, AbsoluteEEActionsStep)
    assert absolute.relative_step is relative
    assert _step(loaded_pre, EEStateStep).rot_repr == RotationRepresentation.rot6d

    state = torch.tensor([[0.2, -0.1, 0.3, 0, 0, 0, 1, 0.5]])
    target = torch.tensor([[[0.4, 0.1, 0.2, 0, 0, 0.6, 0.8, 0.7]]])
    encoded = relative(batch_to_transition({OBS_STATE: state, ACTION: target}))[TransitionKey.ACTION]
    assert encoded.shape[-1] == 10
    decoded = absolute(batch_to_transition({ACTION: encoded}))[TransitionKey.ACTION]
    torch.testing.assert_close(decoded[..., :3], target[..., :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(decoded[..., -1:], target[..., -1:])
    torch.testing.assert_close(
        quat_to_rotmat(decoded[..., 3:7]), quat_to_rotmat(target[..., 3:7]), atol=1e-6, rtol=0
    )


def test_smolvla_default_rotation_is_rot6d():
    assert SmolVLAConfig(device="cpu").rotation_representation == RotationRepresentation.rot6d


def _stats_with_identity_rot6d() -> dict[str, torch.Tensor]:
    """Real-looking stats on position and gripper, identity stats on the rot6d dims [3:9]."""
    stats = {
        "mean": torch.full((10,), 0.2),
        "std": torch.full((10,), 0.5),
        "min": torch.full((10,), -0.5),
        "max": torch.full((10,), 0.5),
        "q01": torch.full((10,), -0.5),
        "q10": torch.full((10,), -0.4),
        "q50": torch.zeros(10),
        "q90": torch.full((10,), 0.4),
        "q99": torch.full((10,), 0.5),
    }
    identity = {
        "mean": 0.0,
        "std": 1.0,
        "min": -1.0,
        "max": 1.0,
        "q01": -1.0,
        "q10": -1.0,
        "q90": 1.0,
        "q99": 1.0,
    }
    for name, value in identity.items():
        stats[name][3:9] = value
    return stats


@pytest.mark.parametrize(
    "mode", [NormalizationMode.MEAN_STD, NormalizationMode.MIN_MAX, NormalizationMode.QUANTILES]
)
def test_identity_rot6d_stats_are_a_noop_under_the_normalizer(mode):
    action = torch.tensor([[0.1, -0.1, 0.05, 0.98, 0.0, -0.05, 0.01, 0.995, 0.03, 0.5]])
    step = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (10,))},
        norm_map={FeatureType.ACTION: mode},
        stats={ACTION: _stats_with_identity_rot6d()},
    )
    normalized = step({ACTION: action.clone()})[ACTION]

    # The rot6d block passes through unchanged, while position and gripper are rescaled.
    torch.testing.assert_close(normalized[0, 3:9], action[0, 3:9], atol=1e-6, rtol=0)
    assert not torch.allclose(normalized[0, :3], action[0, :3])
    assert not torch.allclose(normalized[0, 9], action[0, 9])


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_smolvla_identity_rotation_normalization_requires_rot6d(rot_repr):
    if rot_repr == RotationRepresentation.rot6d:
        config = SmolVLAConfig(
            use_relative_ee=True, rotation_representation=rot_repr, rot6d_identity_norm=True
        )
        assert config.rot6d_identity_norm
    else:
        with pytest.raises(ValueError, match="rot6d_identity_norm"):
            SmolVLAConfig(use_relative_ee=True, rotation_representation=rot_repr, rot6d_identity_norm=True)


def _pi05_config(rot_repr: RotationRepresentation, *, mode: str = "ee", memory: bool = False) -> PI05Config:
    config = PI05Config(
        use_relative_ee=mode == "ee",
        use_relative_actions=mode == "joint",
        use_proprioceptive_memory=memory,
        memory_frames=3,
        device="cpu",
        rotation_representation=rot_repr,
    )
    dim = 4 + ROT_REPR_DIM[rot_repr] if mode == "ee" else 8
    config.input_features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (dim,))}
    config.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (dim,))}
    return config


def _pi05_quantile_stats(dim: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {"q01": -2 * torch.ones(dim), "q99": 2 * torch.ones(dim)},
        ACTION: {"q01": -4 * torch.ones(dim), "q99": 4 * torch.ones(dim)},
    }


def _pi05_ee_batch(memory: bool) -> dict[str, Any]:
    state = torch.tensor([[0.25, 0.5, -0.25, 0, 0, 0.6, 0.8, 0.5]])
    if memory:
        state = state[:, None].expand(-1, 3, -1).clone()
        state[:, :-1, :3] += 2  # A first-frame anchor would produce different actions.
    return {
        OBS_STATE: state,
        ACTION: torch.tensor([[[0.4, 0.1, 0.2, 0, 0.6, 0, 0.8, 0.7], [0.3, 0.2, 0.1, 0, 0, 0, 1, 0.2]]]),
        "task": ["pick_up"],
    }


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("reload", [False, True], ids=["fresh", "reloaded"])
def test_pi05_relative_ee_pipeline_roundtrip(
    tmp_path: Path, rot_repr: RotationRepresentation, memory: bool, reload: bool
) -> None:
    config = _pi05_config(rot_repr, memory=memory)
    dim = config.output_features[ACTION].shape[0]
    pre, post = make_pi05_pre_post_processors(config, _pi05_quantile_stats(dim))
    if reload:
        config.save_pretrained(tmp_path)
        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
        config = PI05Config.from_pretrained(tmp_path)
        pre, post = make_pre_post_processors(config, pretrained_path=tmp_path)

    generic_relative = next(s for s in pre.steps if type(s) is RelativeActionsProcessorStep)
    generic_absolute = next(s for s in post.steps if type(s) is AbsoluteActionsProcessorStep)
    ee_relative = _step(pre, RelativeEEActionsStep)
    ee_absolute = _step(post, AbsoluteEEActionsStep)
    assert not generic_relative.enabled and not generic_absolute.enabled
    assert generic_absolute.relative_step is generic_relative
    assert ee_absolute.relative_step is ee_relative
    assert ee_relative.enabled and ee_absolute.enabled and _step(pre, EEStateStep).enabled
    assert ee_relative.rot_repr == rot_repr
    assert config.max_state_dim == config.max_action_dim == 32

    batch = _pi05_ee_batch(memory)
    state, target = batch[OBS_STATE], batch[ACTION]
    current_state = state[:, -1] if memory else state
    processed = pre(batch)
    torch.testing.assert_close(processed[OBS_STATE], ee_to_rot_repr(state, rot_repr) / 2)
    torch.testing.assert_close(processed[ACTION], to_relative_ee_actions(target, current_state, rot_repr) / 4)
    assert processed[ACTION].shape == (1, 2, dim)
    assert processed[OBS_STATE].shape == ((1, 3, dim) if memory else (1, dim))

    # Inspect the actual text sent to the tokenizer, without downloading PaliGemma.
    prompt = _step(pre, TokenizerProcessorStep).input_tokenizer.prompts[0]
    if memory:
        assert prompt == "Task: pick up;\nAction: "
    else:
        state_tokens = prompt.split("State: ")[1].split(";\nAction: ")[0].split()
        assert prompt.startswith("Task: pick up, State: ")
        assert len(state_tokens) == dim
        # Raw xyz [0.25, 0.5, -0.25] must first normalize to [0.125, 0.25, -0.125].
        assert state_tokens[:3] == ["144", "160", "112"]
        assert state_tokens[-1] == "160"  # normalized gripper 0.25

    decoded = post(processed[ACTION])
    torch.testing.assert_close(decoded[..., :3], target[..., :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(decoded[..., -1:], target[..., -1:], atol=1e-6, rtol=0)
    torch.testing.assert_close(
        quat_to_rotmat(decoded[..., 3:7]), quat_to_rotmat(target[..., 3:7]), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("mode", ["joint", "absolute"])
def test_pi05_non_ee_modes_survive_pipeline_reload(tmp_path: Path, mode: str) -> None:
    config = _pi05_config(RotationRepresentation.rot6d, mode=mode)
    pre, post = make_pi05_pre_post_processors(config, _pi05_quantile_stats(8))
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre, post = make_pre_post_processors(config, pretrained_path=tmp_path)

    generic_relative = next(s for s in pre.steps if type(s) is RelativeActionsProcessorStep)
    generic_absolute = next(s for s in post.steps if type(s) is AbsoluteActionsProcessorStep)
    assert generic_absolute.relative_step is generic_relative
    assert generic_relative.enabled == generic_absolute.enabled == (mode == "joint")
    assert _step(post, AbsoluteEEActionsStep).relative_step is _step(pre, RelativeEEActionsStep)
    assert not _step(pre, RelativeEEActionsStep).enabled
    assert not _step(pre, EEStateStep).enabled
    assert not _step(post, AbsoluteEEActionsStep).enabled

    batch = _pi05_ee_batch(memory=False)
    state, target = batch[OBS_STATE], batch[ACTION]
    processed = pre(batch)
    expected_actions = target - state[:, None] if mode == "joint" else target
    torch.testing.assert_close(processed[ACTION], expected_actions / 4)
    torch.testing.assert_close(processed[OBS_STATE], state / 2)
    torch.testing.assert_close(post(processed[ACTION]), target, atol=1e-6, rtol=0)
