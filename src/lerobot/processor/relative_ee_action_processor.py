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

"""Processor steps for fixed-reference relative end-effector action chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from roma import (
    euler_to_rotmat,
    quat_normalize,
    rotmat_to_euler,
    rotmat_to_rotvec,
    rotmat_to_unitquat,
    rotvec_to_rotmat,
    special_gramschmidt,
    unitquat_to_rotmat,
)
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation_representations import ROT_REPR_DIM, RotationRepresentation

from .pipeline import ProcessorStep, ProcessorStepRegistry


def rotmat_to_rot6d(rotmat: Tensor) -> Tensor:
    """First two columns of R, flattened row by row: [r00, r01, r10, r11, r20, r21]."""
    if rotmat.shape[-2:] != (3, 3):
        raise ValueError(f"expected [..., 3, 3], got {tuple(rotmat.shape)}")
    return rotmat[..., :2].flatten(-2)


def rotmat_to_rotvec_4d(rotmat: Tensor) -> Tensor:
    rotvec = rotmat_to_rotvec(rotmat)
    angle = torch.norm(rotvec, dim=-1).unsqueeze(-1)
    axis = rotvec / torch.where(angle != 0, angle, 1)
    return torch.cat([axis, angle], dim=-1)


def rotvec_4d_to_rotmat(rotvec_4d: Tensor) -> Tensor:
    rotvec = rotvec_4d[..., -1].unsqueeze(-1) * rotvec_4d[..., :3]
    return rotvec_to_rotmat(rotvec)


def rot6d_to_rotmat(rot6d: Tensor) -> Tensor:
    if rot6d.shape[-1] != 6:
        raise ValueError(f"expected [..., 6], got {tuple(rot6d.shape)}")
    return special_gramschmidt(rot6d.unflatten(-1, (3, 2)), epsilon=1e-8)


def quat_to_rotmat(quat: Tensor) -> Tensor:
    if quat.shape[-1] != 4:
        raise ValueError(f"expected [..., 4], got {tuple(quat.shape)}")
    return unitquat_to_rotmat(quat_normalize(quat))


def rotmat_to_riemannian_fancy_stuff(rotmat: Tensor) -> Tensor:
    raise NotImplementedError


def riemannian_fancy_stuff_to_rotmat(riemannian_fancy_stuff: Tensor) -> Tensor:
    raise NotImplementedError


rotmat_to_rot_repr = {
    RotationRepresentation.quaternion: rotmat_to_unitquat,
    RotationRepresentation.euler_angles: partial(rotmat_to_euler, "xyz"),
    RotationRepresentation.rot6d: rotmat_to_rot6d,
    RotationRepresentation.axis_angle_3d: rotmat_to_rotvec,
    RotationRepresentation.axis_angle_4d: rotmat_to_rotvec_4d,
    RotationRepresentation.riemannian_fancy_stuff: rotmat_to_riemannian_fancy_stuff,
}

rot_repr_to_rotmat = {
    RotationRepresentation.quaternion: quat_to_rotmat,
    RotationRepresentation.euler_angles: partial(euler_to_rotmat, "xyz"),
    RotationRepresentation.rot6d: rot6d_to_rotmat,
    RotationRepresentation.axis_angle_3d: rotvec_to_rotmat,
    RotationRepresentation.axis_angle_4d: rotvec_4d_to_rotmat,
    RotationRepresentation.riemannian_fancy_stuff: riemannian_fancy_stuff_to_rotmat,
}


def ee_to_rot_repr(state: Tensor, rot_repr: RotationRepresentation) -> Tensor:
    """Convert 8D EE vectors (7D pose + 1D gripper) to (4+m)D ((3+m)D pose + 1D gripper) rotation representation."""
    translation, quat, gripper = torch.split(state, [3, 4, 1], dim=-1)
    return torch.cat([translation, rotmat_to_rot_repr[rot_repr](quat_to_rotmat(quat)), gripper], dim=-1)


def absolute_ee_to_relative(reference: Tensor, target: Tensor, rot_repr: RotationRepresentation) -> Tensor:
    """Express absolute 8D EE targets (7D pose + 1D gripper) in the EE frame of an 8D reference as (4+m)D ((3+m)D pose + 1D gripper)."""

    reference_translation, reference_quat, _ = torch.split(reference, [3, 4, 1], dim=-1)
    target_translation, target_quat, target_gripper = torch.split(target, [3, 4, 1], dim=-1)
    reference_rotation = quat_to_rotmat(reference_quat)
    target_rotation = quat_to_rotmat(target_quat)
    reference_rotation_inv = reference_rotation.transpose(-2, -1)
    relative_rotation = reference_rotation_inv @ target_rotation
    relative_translation = (
        reference_rotation_inv @ (target_translation - reference_translation).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [relative_translation, rotmat_to_rot_repr[rot_repr](relative_rotation), target_gripper], dim=-1
    )


def relative_ee_to_absolute(relative: Tensor, reference: Tensor, rot_repr: RotationRepresentation) -> Tensor:
    """Convert (4+m)D relative EE targets ((3+m)D pose + 1D gripper) in the EE frame of an 8D reference to absolute 8D (7D pose + 1D gripper)."""
    relative_translation, relative_rot, relative_gripper = torch.split(
        relative, [3, ROT_REPR_DIM[rot_repr], 1], dim=-1
    )
    reference_translation, reference_quat, _ = torch.split(reference, [3, 4, 1], dim=-1)

    reference_rotation = quat_to_rotmat(reference_quat)
    absolute_rotation = reference_rotation @ rot_repr_to_rotmat[rot_repr](relative_rot)
    absolute_translation = reference_translation + (
        reference_rotation @ relative_translation.unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat([absolute_translation, rotmat_to_unitquat(absolute_rotation), relative_gripper], dim=-1)


def to_relative_ee_actions(actions: Tensor, state: Tensor, rot_repr: RotationRepresentation) -> Tensor:
    if actions.shape[-1] != 8 or state.shape[-1] != 8:
        raise ValueError(
            "Relative EE requires 8D [xyz, qx, qy, qz, qw, gripper] input; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    actions = actions.float()
    state = state.to(device=actions.device, dtype=torch.float32)
    if actions.ndim == state.ndim + 1:  # chunk [B, H, D] with one reference [B, 8]
        state = state.unsqueeze(-2)
    return absolute_ee_to_relative(state, actions, rot_repr)


def to_absolute_ee_actions(actions: Tensor, state: Tensor, rot_repr: RotationRepresentation) -> Tensor:
    if actions.shape[-1] != 4 + ROT_REPR_DIM[rot_repr] or state.shape[-1] != 8:
        raise ValueError(
            f"Relative EE requires {4 + ROT_REPR_DIM[rot_repr]}D actions and an 8D reference state; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    actions = actions.float()
    state = state.to(device=actions.device, dtype=torch.float32)
    if actions.ndim == state.ndim + 1:  # chunk [B, H, D] with one reference [B, 8]
        state = state.unsqueeze(-2)
    return relative_ee_to_absolute(actions, state, rot_repr)


@ProcessorStepRegistry.register("relative_ee_actions")
@dataclass
class RelativeEEActionsStep(RelativeActionsProcessorStep):
    """Convert absolute 8D EE action chunks to (4+m)D, relative to the EE frame of the 8D reference state."""

    rot_repr: RotationRepresentation = field(default=RotationRepresentation.rot6d, kw_only=True)
    enabled: bool = False
    state_frame: int | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            state = self._current_state(state)
            if not self._chunk_in_flight():
                self._last_state = state.detach().clone()
        action = transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return transition
        result = transition.copy()
        result[TransitionKey.ACTION] = to_relative_ee_actions(action, state, self.rot_repr)
        return result

    def _current_state(self, state: Tensor) -> Tensor:
        if state.ndim == 2:
            return state
        if self.state_frame is None:
            raise ValueError(
                f"Got a stacked state {tuple(state.shape)} but state_frame is unset; "
                "set it from observation_delta_indices.index(0)"
            )
        return state[:, self.state_frame]

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "state_frame": self.state_frame, "rot_repr": self.rot_repr}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.enabled:
            return features
        result = {feature_type: dict(values) for feature_type, values in features.items()}
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            feature = action_features[ACTION]
            action_features[ACTION] = PolicyFeature(
                type=feature.type, shape=(4 + ROT_REPR_DIM[self.rot_repr],)
            )
        return result


@ProcessorStepRegistry.register("absolute_ee_actions")
@dataclass
class AbsoluteEEActionsStep(AbsoluteActionsProcessorStep):
    """Convert (4+m)D EE action chunks, relative to the EE frame of the 8D reference state, back to absolute 8D."""

    enabled: bool = False
    relative_step: RelativeEEActionsStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError("AbsoluteEEActionsStep requires a paired RelativeEEActionsStep")
        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError("Relative EE postprocessing requires the preprocessor to run first")
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        result = transition.copy()
        result[TransitionKey.ACTION] = to_absolute_ee_actions(action, state, self.relative_step.rot_repr)
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.enabled:
            return features
        result = {feature_type: dict(values) for feature_type, values in features.items()}
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            feature = action_features[ACTION]
            action_features[ACTION] = PolicyFeature(type=feature.type, shape=(8,))
        return result


@ProcessorStepRegistry.register("ee_state")
@dataclass
class EEStateStep(ProcessorStep):
    """Convert the absolute 8D EE state (7D pose + 1D gripper) to absolute (4+m)D ((3+m)D pose + 1D gripper) for the vlm input."""

    rot_repr: RotationRepresentation = RotationRepresentation.rot6d
    enabled: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            return transition

        result = transition.copy()
        result_observation = dict(observation)
        result_observation[OBS_STATE] = ee_to_rot_repr(state, self.rot_repr)
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "rot_repr": self.rot_repr}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.enabled:
            return features
        result = {feature_type: dict(values) for feature_type, values in features.items()}
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            feature = observation_features[OBS_STATE]
            observation_features[OBS_STATE] = PolicyFeature(
                type=feature.type, shape=(4 + ROT_REPR_DIM[self.rot_repr],)
            )
        return result
