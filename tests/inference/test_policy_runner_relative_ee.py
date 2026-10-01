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

"""Canonical EE input validation and conversion at the remote runner boundary."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.inference import FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.processor import AbsoluteEEActionsStep, EEStateStep, RelativeEEActionsStep
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from tests.inference.policy_helpers import ConformingPolicy, tiny_config

_CAMERA = "observation.images.camera"
_STATE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")


class _EEPolicy(ConformingPolicy):
    name = "relative_ee_input_conformance"

    def predict_action_chunk(self, batch, **kwargs):
        self.prepared_state = batch[OBS_STATE].clone()
        action = torch.tensor([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5])
        return action.expand(1, self.config.chunk_size, 10).clone()


@pytest.fixture
def ee_deployment():
    config = tiny_config()
    config.input_features[OBS_STATE] = PolicyFeature(FeatureType.STATE, (10,))
    config.input_features[_CAMERA] = PolicyFeature(FeatureType.VISUAL, (3, 16, 16))
    config.output_features[ACTION] = PolicyFeature(FeatureType.ACTION, (10,))
    policy = _EEPolicy(config)
    relative = RelativeEEActionsStep()
    pre, post = make_policy_processor_pipelines(
        input_steps=[relative, EEStateStep()],
        output_steps=[AbsoluteEEActionsStep(relative_step=relative)],
    )
    features = (
        FeatureSpec(
            OBS_STATE, (8,), "float32", names=_STATE_NAMES, semantics="xyz, quaternion xyzw, gripper"
        ),
        FeatureSpec(OBS_ENV_STATE, (3,), "float32", semantics="environment state"),
        FeatureSpec(_CAMERA, (16, 16, 3), "uint8", kind="rgb", semantics="RGB pixels"),
    )
    return (
        policy,
        pre,
        post,
        {
            "action_interval": 1 / 30,
            "features": features,
            "action_feature": FeatureSpec(
                ACTION, (8,), "float32", names=_STATE_NAMES, semantics="absolute EE"
            ),
            "language_enabled": False,
        },
    )


@pytest.mark.parametrize("validator", ["default", "smolvla"])
def test_runner_accepts_raw_ee_state_and_preserves_wire_schema(ee_deployment, monkeypatch, validator):
    policy, pre, post, kwargs = ee_deployment
    if validator == "smolvla":
        pytest.importorskip("transformers")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        monkeypatch.setattr(
            _EEPolicy, "validate_chunk_input_features", SmolVLAPolicy.validate_chunk_input_features
        )
        monkeypatch.setattr(policy.config, "resize_imgs_with_padding", None, raising=False)
    runner = PolicyRunner(policy, pre, post, **kwargs)
    state = np.array([1, 2, 3, 0, 0, 0, 1, 0.5], dtype=np.float32)
    observation = ObservationSnapshot(
        {
            OBS_STATE: state,
            OBS_ENV_STATE: np.zeros(3, dtype=np.float32),
            _CAMERA: np.zeros((16, 16, 3), np.uint8),
        },
        capture_time=1.0,
        task="pick up the cube",
    )
    result = runner.predict(observation)
    torch.testing.assert_close(
        result.canonical_actions[0], torch.from_numpy(state) + torch.tensor([0.1, 0, 0, 0, 0, 0, 0, 0])
    )
    assert policy.prepared_state.shape == (1, 10)
    assert runner.capabilities.features == kwargs["features"]
    assert runner.capabilities.features[0].names == _STATE_NAMES
    assert policy.config.input_features[OBS_STATE].shape == (10,)


@pytest.mark.parametrize("shape", [(7,), (10,)])
def test_runner_rejects_noncanonical_ee_state_shape(ee_deployment, shape):
    policy, pre, post, kwargs = ee_deployment
    kwargs["features"] = (replace(kwargs["features"][0], shape=shape, names=()), *kwargs["features"][1:])
    with pytest.raises(ValueError, match="8D canonical state"):
        PolicyRunner(policy, pre, post, **kwargs)


@pytest.mark.parametrize("missing", [False, True])
def test_runner_requires_enabled_ee_state_conversion(ee_deployment, missing):
    policy, pre, post, kwargs = ee_deployment
    if missing:
        pre.steps.pop()
    else:
        pre.steps[-1].enabled = False
    with pytest.raises(ValueError, match="enabled EEStateStep"):
        PolicyRunner(policy, pre, post, **kwargs)


def test_runner_still_rejects_unrelated_input_mismatch(ee_deployment):
    policy, pre, post, kwargs = ee_deployment
    kwargs["features"] = tuple(
        replace(feature, shape=(2,)) if feature.name == OBS_ENV_STATE else feature
        for feature in kwargs["features"]
    )
    with pytest.raises(ValueError, match="Tensor feature shape differs.*observation.environment_state"):
        PolicyRunner(policy, pre, post, **kwargs)
