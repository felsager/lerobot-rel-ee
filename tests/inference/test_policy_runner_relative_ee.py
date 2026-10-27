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
from lerobot.inference import ExecutionMode, FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import AbsoluteEEActionsStep, EEStateStep, RelativeEEActionsStep
from lerobot.processor.factory import make_policy_processor_pipelines
from lerobot.processor.relative_ee_action_processor import ee_to_rot_repr, to_absolute_ee_actions
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.rotation_representations import RotationRepresentation, get_rotation_dim
from tests.inference.policy_helpers import ConformingPolicy, tiny_config

_CAMERA = "observation.images.camera"
_STATE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")


class _EEPolicy(ConformingPolicy):
    name = "relative_ee_input_conformance"

    def predict_action_chunk(self, batch, **kwargs):
        self.last_kwargs = kwargs
        self.prepared_state = batch[OBS_STATE].clone()
        action = ee_to_rot_repr(
            torch.tensor([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5]),
            self.config.rotation_representation,
        )
        return action.expand(1, self.config.chunk_size, action.shape[-1]).clone()


@pytest.fixture(
    params=[
        representation
        for representation in RotationRepresentation
        if representation != RotationRepresentation.riemannian_fancy_stuff
    ]
)
def ee_deployment(request):
    config = tiny_config()
    config.rotation_representation = request.param
    model_dim = 4 + get_rotation_dim(request.param)
    config.input_features[OBS_STATE] = PolicyFeature(FeatureType.STATE, (model_dim,))
    config.input_features[_CAMERA] = PolicyFeature(FeatureType.VISUAL, (3, 16, 16))
    config.output_features[ACTION] = PolicyFeature(FeatureType.ACTION, (model_dim,))
    policy = _EEPolicy(config)
    relative = RelativeEEActionsStep(enabled=True, rot_repr=request.param)
    pre, post = make_policy_processor_pipelines(
        input_steps=[relative, EEStateStep(enabled=True, rot_repr=request.param)],
        output_steps=[AbsoluteEEActionsStep(enabled=True, relative_step=relative)],
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
    model_dim = 4 + get_rotation_dim(policy.config.rotation_representation)
    assert policy.prepared_state.shape == (1, model_dim)
    assert runner.capabilities.features == kwargs["features"]
    assert runner.capabilities.features[0].names == _STATE_NAMES
    assert policy.config.input_features[OBS_STATE].shape == (model_dim,)


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


def test_runner_rejects_mismatched_state_and_action_rotation_representations(ee_deployment):
    policy, pre, post, kwargs = ee_deployment
    pre.steps[-1].rot_repr = (
        RotationRepresentation.quaternion
        if pre.steps[0].rot_repr != RotationRepresentation.quaternion
        else RotationRepresentation.rot6d
    )
    with pytest.raises(ValueError, match="rotation representations must match"):
        PolicyRunner(policy, pre, post, **kwargs)


def test_guided_rtc_reanchors_each_rotation_representation(ee_deployment):
    policy, pre, post, kwargs = ee_deployment
    policy.config.rtc_config = RTCConfig(execution_horizon=4)
    runner = PolicyRunner(policy, pre, post, modes=(ExecutionMode.RTC_GUIDED,), **kwargs)
    features = {feature.name: np.zeros(feature.shape, dtype=feature.dtype) for feature in kwargs["features"]}
    features[OBS_STATE][6] = 1.0
    first = runner.predict(ObservationSnapshot(features, 1.0, "pick"), mode=ExecutionMode.RTC_GUIDED)
    features[OBS_STATE] = np.array([1, 2, 3, 0, 0, 2**-0.5, 2**-0.5, 0.5], dtype=np.float32)
    second = runner.predict(
        ObservationSnapshot(features, 2.0, "pick"),
        mode=ExecutionMode.RTC_GUIDED,
        model_continuation=first.model_actions[3:],
        canonical_continuation=first.canonical_actions[3:],
    )
    prefix = policy.last_kwargs["prev_chunk_left_over"]
    decoded = to_absolute_ee_actions(
        prefix, torch.from_numpy(features[OBS_STATE]), policy.config.rotation_representation
    )
    torch.testing.assert_close(decoded, first.canonical_actions[3:7], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(second.canonical_actions).all()
    assert second.canonical_actions.shape == (8, 8)


def test_runner_rejects_model_action_width_inconsistent_with_rotation_representation(ee_deployment):
    policy, pre, post, kwargs = ee_deployment
    expected_dim = 4 + get_rotation_dim(policy.config.rotation_representation)
    policy.config.output_features[ACTION] = PolicyFeature(FeatureType.ACTION, (expected_dim + 1,))
    with pytest.raises(ValueError, match=f"{expected_dim}D model actions"):
        PolicyRunner(policy, pre, post, **kwargs)
