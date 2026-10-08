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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import roma
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import TransitionKey, batch_to_transition
from lerobot.processor.relative_action_processor import bind_relative_anchor
from lerobot.processor.relative_ee_action_processor import (
    AbsoluteEEActionsStep,
    EEStateStep,
    RelativeEEActionsStep,
    ee_to_rot_repr,
    quat_to_rotmat,
    rot6d_to_rotmat,
    rot_repr_to_rotmat,
    rotmat_to_rotvec_4d,
    rotvec_4d_to_rotmat,
    to_absolute_ee_actions,
    to_relative_ee_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.rotation_representations import ROT_REPR_DIM, RotationRepresentation

ROT6D = RotationRepresentation.rot6d
SUPPORTED_REPRESENTATIONS = [
    representation for representation in RotationRepresentation if ROT_REPR_DIM[representation] is not None
]


def _random_ee(*shape: int) -> torch.Tensor:
    """Random 8D EE vectors [x, y, z, qx, qy, qz, qw, gripper]."""
    return torch.cat(
        [torch.randn(*shape, 3) * 0.3, roma.random_unitquat(shape), torch.rand(*shape, 1)], dim=-1
    )


def _gripper_down_ee(*shape: int) -> torch.Tensor:
    """8D EE vectors with the tool z-axis pointing straight down (a 180 deg rotation), random q/-q signs."""
    psi = torch.rand(shape) * 2 * torch.pi
    zeros = torch.zeros(shape)
    quat = torch.stack([torch.cos(psi / 2), torch.sin(psi / 2), zeros, zeros], dim=-1)
    sign = torch.where(torch.rand(*shape, 1) < 0.5, -1.0, 1.0)
    return torch.cat([torch.randn(*shape, 3) * 0.3, sign * quat, torch.rand(*shape, 1)], dim=-1)


def _assert_same_ee(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5) -> None:
    """Positions and grippers compared directly, rotations by angle (q and -q are the same rotation)."""
    torch.testing.assert_close(actual[..., :3], expected[..., :3], atol=atol, rtol=0)
    torch.testing.assert_close(actual[..., 7:], expected[..., 7:], atol=atol, rtol=0)
    angle = roma.rotmat_geodesic_distance(
        quat_to_rotmat(actual[..., 3:7]), quat_to_rotmat(expected[..., 3:7])
    )
    assert angle.max() < atol, f"max rotation error {angle.max():.2e} rad"


def _homogeneous(ee: torch.Tensor) -> torch.Tensor:
    """8D EE vectors -> 4x4 homogeneous transforms in float64 (gripper dropped)."""
    ee = ee.double()
    transform = torch.eye(4, dtype=torch.float64).expand(*ee.shape[:-1], 4, 4).clone()
    transform[..., :3, :3] = quat_to_rotmat(ee[..., 3:7])
    transform[..., :3, 3] = ee[..., :3]
    return transform


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_roundtrip_chunk(rot_repr):
    torch.manual_seed(0)
    state, actions = _random_ee(4), _random_ee(4, 50)
    relative = to_relative_ee_actions(actions, state, rot_repr=rot_repr)
    assert relative.shape == (4, 50, 4 + ROT_REPR_DIM[rot_repr])
    _assert_same_ee(to_absolute_ee_actions(relative, state, rot_repr=rot_repr), actions)


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_pairs_each_sample_with_its_own_state_when_batch_equals_horizon(rot_repr):
    # A round trip alone can't catch wrong pairing: decoding undoes the same mistake.
    torch.manual_seed(1)
    state, actions = _random_ee(8), _random_ee(8, 8)
    batched = to_relative_ee_actions(actions, state, rot_repr=rot_repr)
    per_sample = torch.cat(
        [to_relative_ee_actions(actions[i : i + 1], state[i : i + 1], rot_repr=rot_repr) for i in range(8)]
    )
    torch.testing.assert_close(batched, per_sample, atol=1e-6, rtol=0)
    _assert_same_ee(to_absolute_ee_actions(batched, state, rot_repr=rot_repr), actions)


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_single_action_decode_matches_chunk_decode(rot_repr):
    torch.manual_seed(2)
    state, actions = _random_ee(4), _random_ee(4, 50)
    relative = to_relative_ee_actions(actions, state, rot_repr=rot_repr)
    chunk = to_absolute_ee_actions(relative, state, rot_repr=rot_repr)
    for t in (0, 17, 49):
        _assert_same_ee(to_absolute_ee_actions(relative[:, t], state, rot_repr=rot_repr), chunk[:, t])


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_gripper_down_roundtrip_with_mixed_quaternion_signs(rot_repr):
    torch.manual_seed(3)
    state, actions = _gripper_down_ee(4), _gripper_down_ee(4, 50)
    relative = to_relative_ee_actions(actions, state, rot_repr=rot_repr)
    _assert_same_ee(to_absolute_ee_actions(relative, state, rot_repr=rot_repr), actions)
    # q and -q are the same rotation, so flipping every sign must not change the encoding.
    flipped_state, flipped_actions = state.clone(), actions.clone()
    flipped_state[..., 3:7] *= -1
    flipped_actions[..., 3:7] *= -1
    torch.testing.assert_close(
        to_relative_ee_actions(flipped_actions, flipped_state, rot_repr=rot_repr), relative, atol=1e-6, rtol=0
    )


def test_relative_ee_pose_relative_to_itself_is_identity():
    # Known values: zero translation, identity rotation (columns [1,0,0],[0,1,0] -> [1,0,0,1,0,0]).
    torch.manual_seed(4)
    state = _random_ee(4)
    expected = torch.cat(
        [torch.zeros(4, 3), torch.tensor([1.0, 0, 0, 1, 0, 0]).expand(4, 6), state[:, 7:]], dim=-1
    )
    torch.testing.assert_close(
        to_relative_ee_actions(state, state, rot_repr=ROT6D), expected, atol=1e-5, rtol=0
    )


def test_relative_ee_known_values_pin_frame_and_6d_layout():
    s = 0.5**0.5  # quaternion [0, 0, s, s] = +90 deg about z
    # Rotation layout: target rotated +90 deg about z from an identity reference.
    # R_z(90) = [[0,-1,0],[1,0,0],[0,0,1]]; first two columns, row by row -> [0,-1, 1,0, 0,0].
    reference = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1, 0.3]])
    target = torch.tensor([[0.0, 0, 0, 0, 0, s, s, 0.7]])
    expected = torch.tensor([[0.0, 0, 0, 0, -1, 1, 0, 0, 0, 0.7]])
    torch.testing.assert_close(
        to_relative_ee_actions(target, reference, rot_repr=ROT6D), expected, atol=1e-6, rtol=0
    )

    # EE frame: the reference gripper is turned +90 deg about z, so its x-axis points along base +y.
    # A target 1 m along base +y is therefore 1 m along the gripper's own x-axis.
    reference = torch.tensor([[1.0, 0, 0, 0, 0, s, s, 0.3]])
    target = torch.tensor([[1.0, 1, 0, 0, 0, s, s, 0.7]])
    expected = torch.tensor([[1.0, 0, 0, 1, 0, 0, 1, 0, 0, 0.7]])
    torch.testing.assert_close(
        to_relative_ee_actions(target, reference, rot_repr=ROT6D), expected, atol=1e-6, rtol=0
    )


def test_relative_ee_matches_inverse_reference_transform():
    # Pins the convention itself: EE frame, T_rel = inv(T_ref) @ T. A round trip can't.
    torch.manual_seed(5)
    state, actions = _random_ee(4), _random_ee(4, 50)
    relative = to_relative_ee_actions(actions, state, rot_repr=ROT6D)
    expected = torch.linalg.inv(_homogeneous(state))[:, None] @ _homogeneous(actions)
    torch.testing.assert_close(relative[..., :3].double(), expected[..., :3, 3], atol=1e-5, rtol=0)
    torch.testing.assert_close(
        rot6d_to_rotmat(relative[..., 3:9]).double(), expected[..., :3, :3], atol=1e-5, rtol=0
    )


# ---------------------------------------------------------------------------
# Processor steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_training_transition_through_steps(rot_repr):
    # SmolVLA loads observation frames [0], so the training state arrives as [B, 1, 8].
    torch.manual_seed(6)
    state, actions = _random_ee(2, 1), _random_ee(2, 50)
    actions[:, 0] = state[:, 0]  # first target equals the reference

    transition = batch_to_transition({OBS_STATE: state, ACTION: actions})
    transition = RelativeEEActionsStep(enabled=True, state_frame=0, rot_repr=rot_repr)(transition)
    transition = EEStateStep(enabled=True, rot_repr=rot_repr)(transition)

    relative = transition[TransitionKey.ACTION]
    assert relative.shape == (2, 50, 4 + ROT_REPR_DIM[rot_repr])
    assert transition[TransitionKey.OBSERVATION][OBS_STATE].shape == (2, 1, 4 + ROT_REPR_DIM[rot_repr])
    torch.testing.assert_close(
        relative, to_relative_ee_actions(actions, state[:, 0], rot_repr=rot_repr), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(relative[:, 0, :3], torch.zeros(2, 3), atol=1e-5, rtol=0)


def test_relative_ee_state_frame_selects_the_current_frame():
    # Diffusion-style observation frames [-1, 0]: the current frame is index 1.
    torch.manual_seed(7)
    state, actions = _random_ee(2, 2), _random_ee(2, 10)
    transition = RelativeEEActionsStep(enabled=True, state_frame=1, rot_repr=ROT6D)(
        batch_to_transition({OBS_STATE: state, ACTION: actions})
    )
    torch.testing.assert_close(
        transition[TransitionKey.ACTION],
        to_relative_ee_actions(actions, state[:, 1], rot_repr=ROT6D),
        atol=1e-6,
        rtol=0,
    )


def test_relative_ee_stacked_state_without_state_frame_raises():
    transition = batch_to_transition({OBS_STATE: _random_ee(2, 1), ACTION: _random_ee(2, 10)})
    with pytest.raises(ValueError, match="state_frame is unset"):
        RelativeEEActionsStep(enabled=True, rot_repr=ROT6D)(transition)


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_ee_state_step_matches_ee_to_rot_repr(rot_repr):
    # The stats are computed with ee_to_rot_repr, so the step must produce exactly the same values.
    state = _random_ee(3)
    stepped = EEStateStep(enabled=True, rot_repr=rot_repr)(batch_to_transition({OBS_STATE: state}))[
        TransitionKey.OBSERVATION
    ][OBS_STATE]
    torch.testing.assert_close(stepped, ee_to_rot_repr(state, rot_repr=rot_repr), atol=0, rtol=0)


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_absolute_ee_step_decodes_against_cached_reference(rot_repr):
    torch.manual_seed(8)
    relative_step = RelativeEEActionsStep(enabled=True, rot_repr=rot_repr)
    absolute_step = AbsoluteEEActionsStep(enabled=True, relative_step=relative_step)
    state = _random_ee(1)
    relative_action = to_relative_ee_actions(_random_ee(1), state, rot_repr=rot_repr)

    relative_step(
        batch_to_transition({OBS_STATE: state})
    )  # inference preprocess: no action, caches the state
    decoded = absolute_step(batch_to_transition({ACTION: relative_action}))[TransitionKey.ACTION]

    _assert_same_ee(decoded, to_absolute_ee_actions(relative_action, state, rot_repr=rot_repr))


def test_relative_ee_reference_is_held_while_chunk_is_in_flight():
    relative_step = RelativeEEActionsStep(enabled=True, rot_repr=ROT6D)
    queue = {"size": 0}
    policy = SimpleNamespace(count_queued_actions=lambda: queue["size"])
    assert bind_relative_anchor(policy, SimpleNamespace(steps=[relative_step])) is relative_step

    first, second, third = _random_ee(1), _random_ee(1), _random_ee(1)
    relative_step(batch_to_transition({OBS_STATE: first}))  # queue empty: new chunk, cache first
    queue["size"] = 49
    relative_step(batch_to_transition({OBS_STATE: second}))  # chunk in flight: keep first
    torch.testing.assert_close(relative_step.get_cached_state(), first, atol=0, rtol=0)
    queue["size"] = 0
    relative_step(batch_to_transition({OBS_STATE: third}))  # queue drained: re-anchor
    torch.testing.assert_close(relative_step.get_cached_state(), third, atol=0, rtol=0)


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
@pytest.mark.parametrize("enabled", [False, True])
def test_relative_ee_steps_get_config_roundtrip(rot_repr, enabled):
    # Loading a saved pipeline rebuilds each step from its get_config().
    for step in (
        RelativeEEActionsStep(enabled=enabled, state_frame=0, rot_repr=rot_repr),
        EEStateStep(enabled=enabled, rot_repr=rot_repr),
        AbsoluteEEActionsStep(enabled=enabled),
    ):
        rebuilt = type(step)(**step.get_config())
        assert rebuilt.get_config() == step.get_config()
    assert (
        RelativeEEActionsStep(
            **RelativeEEActionsStep(state_frame=1, rot_repr=rot_repr).get_config()
        ).state_frame
        == 1
    )


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_steps_transform_features(rot_repr):
    features = {
        PipelineFeatureType.OBSERVATION: {OBS_STATE: PolicyFeature(FeatureType.STATE, (8,))},
        PipelineFeatureType.ACTION: {ACTION: PolicyFeature(FeatureType.ACTION, (8,))},
    }
    model_side = EEStateStep(enabled=True, rot_repr=rot_repr).transform_features(
        RelativeEEActionsStep(enabled=True, rot_repr=rot_repr).transform_features(features)
    )
    assert model_side[PipelineFeatureType.OBSERVATION][OBS_STATE].shape == (4 + ROT_REPR_DIM[rot_repr],)
    assert model_side[PipelineFeatureType.ACTION][ACTION].shape == (4 + ROT_REPR_DIM[rot_repr],)

    robot_side = AbsoluteEEActionsStep(enabled=True).transform_features(model_side)
    assert robot_side[PipelineFeatureType.ACTION][ACTION].shape == (8,)

    # Disabled steps leave the declared shapes alone, and the input is never mutated.
    assert RelativeEEActionsStep(enabled=False, rot_repr=rot_repr).transform_features(features) == features
    assert EEStateStep(enabled=False, rot_repr=rot_repr).transform_features(features) == features
    assert features[PipelineFeatureType.ACTION][ACTION].shape == (8,)


def test_relative_ee_steps_are_disabled_by_default():
    transition = batch_to_transition({OBS_STATE: _random_ee(2, 1), ACTION: _random_ee(2, 10)})
    for step in (RelativeEEActionsStep(), EEStateStep(), AbsoluteEEActionsStep()):
        assert not step.enabled
        assert step(transition) is transition


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _ee_x(x: float, gripper: float) -> list[float]:
    """8D EE vector at position (x, 0, 0) with identity rotation."""
    return [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, gripper]


def test_relative_ee_stats_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.datasets import compute_rel_ee_stats as stats_module

    actions = np.array([_ee_x(0.0, 0.0), _ee_x(1.0, 0.5)], dtype=np.float32)
    states = actions.copy()
    episodes = np.array([0, 0])
    key = stats_module.relative_ee_stats_cache_key
    original = key(actions, states, episodes, 50, rotation_representation=ROT6D)
    assert original == key(actions.copy(), states.copy(), episodes.copy(), 50, rotation_representation=ROT6D)
    assert original == key(
        actions.astype(np.float64), np.asfortranarray(states), episodes, 50, rotation_representation=ROT6D
    )
    assert original != key(actions, states, episodes, 25, rotation_representation=ROT6D)
    assert len(
        {key(actions, states, episodes, 50, rotation_representation=r) for r in SUPPORTED_REPRESENTATIONS}
    ) == len(SUPPORTED_REPRESENTATIONS)
    changed = actions.copy()
    changed[0, 0] += 0.1
    assert original != key(changed, states, episodes, 50, rotation_representation=ROT6D)
    assert original != key(actions, changed, episodes, 50, rotation_representation=ROT6D)
    assert original != key(actions, states, np.array([0, 1]), 50, rotation_representation=ROT6D)
    assert original != key(actions[:1], states[:1], episodes[:1], 50, rotation_representation=ROT6D)
    assert original != key(actions[::-1], states[::-1], episodes[::-1], 50, rotation_representation=ROT6D)
    monkeypatch.setattr(
        stats_module, "_RELATIVE_EE_STATS_VERSION", stats_module._RELATIVE_EE_STATS_VERSION + 1
    )
    assert original != key(actions, states, episodes, 50, rotation_representation=ROT6D)


def test_relative_ee_stats_cache_reuse(tmp_path: Path) -> None:
    from lerobot.datasets.compute_rel_ee_stats import (
        compute_relative_ee_stats,
        load_or_compute_relative_ee_stats,
    )

    poses = np.array([_ee_x(0, 0), _ee_x(1, 1)], dtype=np.float32)
    data = {ACTION: poses, OBS_STATE: poses.copy(), "episode_index": np.array([0, 0])}
    with patch(
        "lerobot.datasets.compute_rel_ee_stats.compute_relative_ee_stats", wraps=compute_relative_ee_stats
    ) as compute:
        raw = load_or_compute_relative_ee_stats(data, 2, tmp_path, rot_repr=ROT6D)
        identity = load_or_compute_relative_ee_stats(data, 2, tmp_path, identity_rot6d=True, rot_repr=ROT6D)
        loaded = load_or_compute_relative_ee_stats(data, 2, tmp_path, rot_repr=ROT6D)
        assert compute.call_count == 1
        for key in (ACTION, OBS_STATE):
            for name in raw[key]:
                np.testing.assert_array_equal(loaded[key][name], raw[key][name])
            np.testing.assert_array_equal(identity[key]["mean"][3:9], 0)
            np.testing.assert_array_equal(identity[key]["std"][3:9], 1)
        load_or_compute_relative_ee_stats(data, 1, tmp_path, rot_repr=ROT6D)
        assert compute.call_count == 2
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_stats_anchor_on_state_and_stay_within_episodes(rot_repr):
    # Imported here so the tests above still run without the dataset extra installed.
    from lerobot.datasets.compute_rel_ee_stats import compute_relative_ee_stats

    actions = np.array(
        [_ee_x(0.0, 0.0), _ee_x(1.0, 0.2), _ee_x(10.0, 0.4), _ee_x(11.0, 0.6)], dtype=np.float32
    )
    states = np.array(
        [_ee_x(0.0, 0.0), _ee_x(0.5, 0.2), _ee_x(10.0, 0.4), _ee_x(10.5, 0.6)], dtype=np.float32
    )
    stats = compute_relative_ee_stats(
        {ACTION: actions, OBS_STATE: states, "episode_index": np.array([0, 0, 1, 1])},
        chunk_size=2,
        rot_repr=rot_repr,
    )
    # Per episode: (0, 1) from t=0 and (0.5) from t=1; the target past the episode end is excluded.
    # Anchoring on the action instead of the state would give (0, 1) and (0), so a mean of 1/3.
    np.testing.assert_allclose(stats[ACTION]["mean"][0], 0.5, atol=1e-6)
    np.testing.assert_allclose(stats[OBS_STATE]["mean"][0], (0 + 0.5 + 10 + 10.5) / 4, atol=1e-6)
    assert stats[ACTION]["mean"].shape == (4 + ROT_REPR_DIM[rot_repr],)
    assert stats[OBS_STATE]["mean"].shape == (4 + ROT_REPR_DIM[rot_repr],)


def test_relative_ee_stats_identity_rot6d():
    from lerobot.datasets.compute_rel_ee_stats import compute_relative_ee_stats

    actions = np.array(
        [_ee_x(0.0, 0.0), _ee_x(1.0, 0.2), _ee_x(10.0, 0.4), _ee_x(11.0, 0.6)], dtype=np.float32
    )
    stats = compute_relative_ee_stats(
        {ACTION: actions, OBS_STATE: actions.copy(), "episode_index": np.array([0, 0, 1, 1])},
        chunk_size=2,
        identity_rot6d=True,
        rot_repr=ROT6D,
    )
    for key in (ACTION, OBS_STATE):
        np.testing.assert_allclose(stats[key]["min"][3:9], -1.0)
        np.testing.assert_allclose(stats[key]["max"][3:9], 1.0)
        np.testing.assert_allclose(stats[key]["mean"][3:9], 0.0)
        np.testing.assert_allclose(stats[key]["std"][3:9], 1.0)


@pytest.mark.parametrize("selection", [None, [1, 2], [3, 0, 3, 1]])
@pytest.mark.parametrize("formatter", [None, "torch"])
@pytest.mark.parametrize("fixed", [False, True])
def test_arrow_relative_columns_preserve_current_hash(selection, formatter, fixed):
    import hashlib

    import pyarrow as pa
    from datasets import Dataset, concatenate_datasets

    from lerobot.datasets.compute_rel_ee_stats import _load_relative_ee_columns, relative_ee_stats_cache_key

    poses = [_ee_x(i * 0.13, i / 4) for i in range(4)]
    feature_type = pa.list_(pa.float32(), 8) if fixed else pa.list_(pa.float32())
    table = pa.table(
        {
            ACTION: pa.array(poses, type=feature_type),
            OBS_STATE: pa.array(poses[::-1], type=feature_type),
            "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
        }
    )
    ds = concatenate_datasets([Dataset(table.slice(0, 2)), Dataset(table.slice(2, 2))])
    if selection is not None:
        ds = ds.select(selection)
    ds = ds.with_format(formatter)
    expected = {
        k: np.asarray(ds[k], dtype=dtype)
        for k, dtype in [(ACTION, np.float32), (OBS_STATE, np.float32), ("episode_index", np.int64)]
    }
    original_format = ds.format.copy()
    actual = _load_relative_ee_columns(ds)
    assert ds.format == original_format
    legacy = hashlib.sha256(f"relative-ee:2:50:{ROT6D}".encode())
    for key, dtype in [(ACTION, "<f4"), (OBS_STATE, "<f4"), ("episode_index", "<i8")]:
        np.testing.assert_array_equal(actual[key], expected[key])
        values = np.ascontiguousarray(expected[key], dtype=dtype)
        legacy.update(f"{key}:{values.shape}:{dtype}:".encode())
        legacy.update(values.tobytes())
    assert (
        relative_ee_stats_cache_key(
            actual[ACTION], actual[OBS_STATE], actual["episode_index"], 50, rotation_representation=ROT6D
        )
        == legacy.hexdigest()
    )


def test_arrow_relative_cache_hits_existing_cache(tmp_path):
    from datasets import Dataset

    from lerobot.datasets.compute_rel_ee_stats import load_or_compute_relative_ee_stats

    poses = np.array([_ee_x(0, 0), _ee_x(1, 1)], dtype=np.float32)
    columns = {ACTION: poses, OBS_STATE: poses.copy(), "episode_index": np.array([0, 0])}
    expected = load_or_compute_relative_ee_stats(columns, 2, tmp_path, rot_repr=ROT6D)
    dataset = Dataset.from_dict(columns).with_format("torch")
    with patch(
        "lerobot.datasets.compute_rel_ee_stats.compute_relative_ee_stats",
        side_effect=AssertionError("cache miss"),
    ):
        actual = load_or_compute_relative_ee_stats(dataset, 2, tmp_path, rot_repr=ROT6D)
    for key in expected:
        for stat in expected[key]:
            np.testing.assert_array_equal(expected[key][stat], actual[key][stat])


def test_empty_relative_columns_preserve_current_hash() -> None:
    import hashlib

    from lerobot.datasets.compute_rel_ee_stats import relative_ee_stats_cache_key

    poses = np.empty((0, 8), dtype=np.float32)
    episodes = np.empty(0, dtype=np.int64)
    legacy = hashlib.sha256(f"relative-ee:2:50:{ROT6D}".encode())
    for name, array, dtype in [
        (ACTION, poses, "<f4"),
        (OBS_STATE, poses, "<f4"),
        ("episode_index", episodes, "<i8"),
    ]:
        legacy.update(f"{name}:{array.shape}:{dtype}:".encode())
        legacy.update(array.tobytes())
    assert (
        relative_ee_stats_cache_key(poses, poses, episodes, 50, rotation_representation=ROT6D)
        == legacy.hexdigest()
    )


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_axis_angle_4d_identity_is_finite_and_zero(shape):
    matrices = torch.eye(3).expand(*shape, 3, 3)
    encoded = rotmat_to_rotvec_4d(matrices)
    assert encoded.shape == (*shape, 4)
    torch.testing.assert_close(encoded, torch.zeros_like(encoded), atol=0, rtol=0)
    torch.testing.assert_close(rotvec_4d_to_rotmat(encoded), matrices)


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_geometry_matches_transform_for_each_representation(rot_repr):
    torch.manual_seed(12)
    state, actions = _random_ee(3), _random_ee(3, 5)
    relative = to_relative_ee_actions(actions, state, rot_repr)
    expected = torch.linalg.inv(_homogeneous(state))[:, None] @ _homogeneous(actions)
    torch.testing.assert_close(relative[..., :3].double(), expected[..., :3, 3], atol=1e-5, rtol=0)
    torch.testing.assert_close(
        rot_repr_to_rotmat[rot_repr](relative[..., 3:-1]).double(),
        expected[..., :3, :3],
        atol=1e-5,
        rtol=0,
    )
    converted_state = ee_to_rot_repr(state, rot_repr)
    torch.testing.assert_close(
        rot_repr_to_rotmat[rot_repr](converted_state[..., 3:-1]),
        quat_to_rotmat(state[..., 3:7]),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_relative_ee_statistics_match_model_samples(rot_repr):
    from lerobot.datasets.compute_rel_ee_stats import compute_relative_ee_stats

    torch.manual_seed(13)
    states, actions = _random_ee(4), _random_ee(4)
    data = {ACTION: actions.numpy(), OBS_STATE: states.numpy(), "episode_index": np.zeros(4, dtype=np.int64)}
    stats = compute_relative_ee_stats(data, 2, rot_repr)
    action_samples = torch.cat(
        [to_relative_ee_actions(actions[t : min(t + 2, 4)], states[t], rot_repr) for t in range(4)]
    ).numpy()
    state_samples = ee_to_rot_repr(states, rot_repr).numpy()
    for key, samples in [(ACTION, action_samples), (OBS_STATE, state_samples)]:
        for name, expected in [
            ("mean", samples.mean(axis=0)),
            ("min", samples.min(axis=0)),
            ("max", samples.max(axis=0)),
        ]:
            np.testing.assert_allclose(stats[key][name], expected, atol=1e-6)


@pytest.mark.parametrize("step_type", [RelativeEEActionsStep, EEStateStep])
def test_missing_ee_representation_uses_legacy_rot6d(step_type):
    assert step_type().rot_repr == ROT6D


@pytest.mark.parametrize("rot_repr", SUPPORTED_REPRESENTATIONS)
def test_rotation_decoder_identity(rot_repr):
    encodings = {
        RotationRepresentation.quaternion: [0.0, 0, 0, 1],
        RotationRepresentation.euler_angles: [0.0, 0, 0],
        RotationRepresentation.rot6d: [1.0, 0, 0, 1, 0, 0],
        RotationRepresentation.axis_angle_3d: [0.0, 0, 0],
        RotationRepresentation.axis_angle_4d: [0.0, 0, 0, 0],
    }
    decoded = rot_repr_to_rotmat[rot_repr](torch.tensor(encodings[rot_repr]).expand(2, -1))
    torch.testing.assert_close(decoded, torch.eye(3).expand(2, 3, 3))
