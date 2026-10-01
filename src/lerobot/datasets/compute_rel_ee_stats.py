"""Statistics for the model-facing relative end-effector action and state."""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.processor.relative_ee_action_processor import absolute_ee_to_relative, ee_to_rot6d
from lerobot.utils.constants import ACTION, OBS_STATE

from .compute_stats import RunningQuantileStats

logger = logging.getLogger(__name__)

# rot6d sits at dims [3:9] of both 10D model vectors: [pos(3), rot6d(6), gripper(1)].
_ROT6D_SLICE = slice(3, 9)

# Statistics that make every NormalizationMode a no-op on a dimension: MIN_MAX needs
# min=-1/max=1; QUANTILES need q01=q10=-1, q90=q99=1; MEAN_STD needs mean=0/std=1.
# (q50 and count are unused by the normalizer.)
_IDENTITY_ROT6D_STATS: dict[str, float] = {
    "mean": 0.0,
    "std": 1.0,
    "min": -1.0,
    "max": 1.0,
    "q01": -1.0,
    "q10": -1.0,
    "q90": 1.0,
    "q99": 1.0,
}


def _force_identity_rot6d_stats(stats: dict[str, dict[str, np.ndarray]]) -> None:
    """Overwrite the rot6d statistics of the action and state in place so normalization leaves rotation unchanged.

    rot6d entries are rotation-matrix entries already bounded in [-1, 1]. Scaling them with
    per-dimension statistics distorts their coupled geometry and, for near-identity relative
    rotations, over-weights the near-constant diagonal entries.
    """
    for key in (ACTION, OBS_STATE):
        feature_stats = stats.get(key)
        if not feature_stats:
            continue
        for stat_name, value in _IDENTITY_ROT6D_STATS.items():
            array = feature_stats.get(stat_name)
            if array is not None:
                array[_ROT6D_SLICE] = value


def compute_relative_ee_stats(
    hf_dataset,
    chunk_size: int,
    identity_rot6d: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute statistics for the model-facing 10D EE action and state.

    Action: every chunk ``action[t : t + chunk_size]`` relative to the EE frame of ``state[t]``,
    within one episode and without targets past the episode end (the padded ones).
    State: the absolute 10D state (9D pose + 1D gripper).
    ``identity_rot6d`` forces identity normalization stats on the six rotation dims.
    """
    actions = np.asarray(hf_dataset[ACTION], dtype=np.float32)
    states = np.asarray(hf_dataset[OBS_STATE], dtype=np.float32)
    for name, values in ((ACTION, actions), (OBS_STATE, states)):
        if values.ndim != 2 or values.shape[1] != 8:
            raise ValueError(
                f"Relative EE requires {name} shape [frames, 8] (7D pose + 1D gripper), got {values.shape}"
            )
    if chunk_size < 1:
        raise ValueError(f"Relative EE requires chunk_size >= 1, got {chunk_size}")

    episode_indices = np.asarray(hf_dataset["episode_index"])
    action_stats = RunningQuantileStats()
    state_stats = RunningQuantileStats()
    num_action_targets = 0

    for episode_index in np.unique(episode_indices):
        frame_indices = np.flatnonzero(episode_indices == episode_index)
        episode_actions = torch.from_numpy(actions[frame_indices])
        episode_states = torch.from_numpy(states[frame_indices])

        state_stats.update(ee_to_rot6d(episode_states).numpy())

        for batch_start in range(0, len(frame_indices), 20_000):
            base_indices = torch.arange(batch_start, min(batch_start + 20_000, len(frame_indices)))
            target_indices = base_indices[:, None] + torch.arange(chunk_size)[None, :]
            valid = target_indices < len(frame_indices)
            targets = episode_actions[target_indices[valid]]
            references = episode_states[base_indices[:, None].expand_as(target_indices)[valid]]
            relative_actions = absolute_ee_to_relative(references, targets)
            action_stats.update(relative_actions.numpy())
            num_action_targets += len(relative_actions)

    if len(states) < 2 or num_action_targets < 2:
        raise ValueError("Relative EE statistics require at least two selected dataset frames")

    logger.info(
        "Computed relative EE statistics from %d states and %d unpadded action targets",
        len(states),
        num_action_targets,
    )

    stats = {
        ACTION: action_stats.get_statistics(),
        OBS_STATE: state_stats.get_statistics(),
    }
    if identity_rot6d:
        _force_identity_rot6d_stats(stats)
    return stats
