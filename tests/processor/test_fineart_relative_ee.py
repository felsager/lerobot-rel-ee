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

"""Relative-EE and prompt parity between FineART and the working PI0.5 pipeline."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("datasets")

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig  # noqa: E402
from lerobot.policies.fineart_vla.processor_fineart_vla import (  # noqa: E402
    FineARTVLATextTokenizerStep,
    make_fineart_vla_pre_post_processors,
)
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AbsoluteEEActionsStep,
    RelativeEEActionsStep,
    TokenizerProcessorStep,
    tokenizer_processor,
)
from lerobot.processor.relative_ee_action_processor import quat_to_rotmat  # noqa: E402
from lerobot.utils.constants import ACTION, ACTION_TOKENS, OBS_STATE  # noqa: E402
from lerobot.utils.rotation_representations import ROT_REPR_DIM, RotationRepresentation  # noqa: E402


class _TextTokenizer:
    vocab_size = 4096
    bos_token_id = 2
    eos_token = "</s>"
    added_tokens_encoder = {"<loc0000>": 256000}

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, text: str | list[str], max_length: int, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.prompts = [text] if isinstance(text, str) else list(text)
        shape = (len(self.prompts), max_length)
        result = {"input_ids": torch.zeros(shape, dtype=torch.long), "attention_mask": torch.ones(shape)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = torch.zeros(*shape, 2, dtype=torch.long)
        return result

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [10, 11] if text == "Action: " else [12]

    def save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / "tokenizer_config.json").write_text("{}")


class _ActionTokenizer:
    def __init__(self) -> None:
        self.actions: np.ndarray | None = None

    def __call__(self, actions: np.ndarray) -> list[list[int]]:
        self.actions = np.asarray(actions).copy()
        return [[1, 2, 3] for _ in actions]

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "processor_config.json").write_text("{}")


@pytest.fixture(autouse=True)
def _no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *a, **kw: _TextTokenizer()
    )
    monkeypatch.setattr(
        tokenizer_processor.AutoProcessor, "from_pretrained", lambda *a, **kw: _ActionTokenizer()
    )


def _config(rot_repr: RotationRepresentation, **kwargs: Any) -> FineARTVLAConfig:
    dim = 4 + ROT_REPR_DIM[rot_repr]
    return FineARTVLAConfig(
        device="cpu",
        recipe=None,
        use_pi05_action_prompt=True,
        text_loss_weight=0,
        use_relative_ee=True,
        rotation_representation=rot_repr,
        auto_fit_fast_tokenizer=False,
        chunk_size=2,
        n_action_steps=2,
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (dim,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (dim,))},
        **kwargs,
    )


def _stats(dim: int, identity_rotation: bool) -> dict[str, dict[str, torch.Tensor]]:
    scales = torch.full((dim,), 4.0)
    if identity_rotation:
        scales[3:9] = 1
    return {key: {"q01": -scales.clone(), "q99": scales.clone()} for key in (ACTION, OBS_STATE)}


def _batch() -> dict[str, Any]:
    return {
        OBS_STATE: torch.tensor([[0.25, 0.5, -0.25, 0, 0, 0.6, 0.8, 0.5]]),
        ACTION: torch.tensor([[[0.4, 0.1, 0.2, 0, 0.6, 0, 0.8, 0.7], [0.3, 0.2, 0.1, 0, 0, 0, 1, 0.2]]]),
        "task": ["pick_up"],
    }


@pytest.mark.parametrize("rot_repr", [r for r in RotationRepresentation if ROT_REPR_DIM[r] is not None])
@pytest.mark.parametrize("use_proprio", [False, True])
@pytest.mark.parametrize("reload", [False, True])
def test_action_prompt_and_ee_roundtrip_match_pi05(tmp_path, rot_repr, use_proprio, reload):
    identity = rot_repr == RotationRepresentation.rot6d
    config = _config(
        rot_repr, use_proprio=use_proprio, rot6d_identity_norm=identity, prompt_control_prefix=True
    )
    stats = _stats(config.output_features[ACTION].shape[0], identity)
    pre, post = make_fineart_vla_pre_post_processors(config, stats)
    baseline_pre, _ = make_pi05_pre_post_processors(config, stats)
    if reload:
        config.save_pretrained(tmp_path)
        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
        config = PreTrainedConfig.from_pretrained(tmp_path)
        pre, post = make_pre_post_processors(config, pretrained_path=tmp_path)
    batch = _batch()
    expected = baseline_pre(deepcopy(batch))
    processed = pre(deepcopy(batch))
    torch.testing.assert_close(processed[ACTION], expected[ACTION])
    torch.testing.assert_close(processed[OBS_STATE], expected[OBS_STATE])
    tokenizer = next(step for step in pre.steps if isinstance(step, TokenizerProcessorStep)).input_tokenizer
    baseline_tokenizer = next(
        step for step in baseline_pre.steps if isinstance(step, TokenizerProcessorStep)
    ).input_tokenizer
    assert tokenizer.prompts == baseline_tokenizer.prompts
    assert ("State:" in tokenizer.prompts[0]) == use_proprio
    assert "<control mode> end effector cam frame </control mode>" in tokenizer.prompts[0]
    assert ACTION_TOKENS in processed
    relative = next(step for step in pre.steps if isinstance(step, RelativeEEActionsStep))
    absolute = next(step for step in post.steps if isinstance(step, AbsoluteEEActionsStep))
    assert absolute.relative_step is relative
    decoded = post(processed[ACTION])
    torch.testing.assert_close(decoded[..., :3], batch[ACTION][..., :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(decoded[..., -1], batch[ACTION][..., -1])
    torch.testing.assert_close(
        quat_to_rotmat(decoded[..., 3:7]), quat_to_rotmat(batch[ACTION][..., 3:7]), atol=1e-6, rtol=0
    )

    # Real inference has no action target; it must still capture a fresh raw EE anchor.
    runtime = {OBS_STATE: batch[OBS_STATE].clone(), "task": ["pick_up"]}
    runtime[OBS_STATE][:, :3] += 1
    pre(runtime)
    assert (tokenizer.prompts == baseline_tokenizer.prompts) == (not use_proprio)
    shifted = post(processed[ACTION])
    torch.testing.assert_close(shifted[..., :3], decoded[..., :3] + 1, atol=1e-6, rtol=0)


@pytest.mark.parametrize("use_proprio", [False, True])
def test_recipe_prompt_honors_state_and_control_flags(tmp_path, use_proprio):
    config = FineARTVLAConfig(
        device="cpu",
        use_relative_ee=True,
        use_proprio=use_proprio,
        prompt_control_prefix=True,
        enable_fast_action_loss=False,
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (10,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (10,))},
    )
    pre, post = make_fineart_vla_pre_post_processors(config, _stats(10, True))
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre, _ = make_pre_post_processors(config, pretrained_path=tmp_path)
    pre(_batch())
    step = next(step for step in pre.steps if isinstance(step, FineARTVLATextTokenizerStep))
    prompt = step._tokenizer.prompts[0]
    assert ("State:" in prompt) == use_proprio
    assert "<control mode> end effector cam frame </control mode>" in prompt


def test_action_only_prompt_rejects_language_loss():
    with pytest.raises(ValueError, match="text_loss_weight=0"):
        FineARTVLAConfig(device="cpu", use_pi05_action_prompt=True)
