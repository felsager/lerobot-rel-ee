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

import tempfile

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AbsoluteEEActionsStep,
    EEStateStep,
    NormalizerProcessorStep,
    RelativeEEActionsStep,
    tokenizer_processor,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


def _model_space_stats() -> dict[str, dict[str, torch.Tensor]]:
    stats = {"mean": torch.zeros(10), "std": torch.ones(10), "min": -torch.ones(10), "max": torch.ones(10)}
    return {OBS_STATE: dict(stats), ACTION: dict(stats)}


def _relative_ee_smolvla_config() -> SmolVLAConfig:
    config = SmolVLAConfig(use_relative_ee=True, device="cpu")
    config.input_features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (10,))}
    config.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (10,))}
    return config


def _step(pipeline, step_type):
    return next(step for step in pipeline.steps if isinstance(step, step_type))


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: object()
    )


def test_smolvla_relative_ee_pipeline_composition():
    config = _relative_ee_smolvla_config()
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, _model_space_stats())

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
    assert config.action_delta_indices == list(range(config.chunk_size))


def test_smolvla_relative_ee_processors_save_load_and_reconnect():
    config = _relative_ee_smolvla_config()
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, _model_space_stats())

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor.save_pretrained(tmpdir)
        postprocessor.save_pretrained(tmpdir)
        loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(config, pretrained_path=tmpdir)

    relative_step = _step(loaded_preprocessor, RelativeEEActionsStep)
    absolute_step = _step(loaded_postprocessor, AbsoluteEEActionsStep)
    assert absolute_step.relative_step is relative_step
    assert relative_step.state_frame == 0


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
