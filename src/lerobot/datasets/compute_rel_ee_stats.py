"""Statistics for the model-facing relative end-effector action and state."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset
from huggingface_hub.utils import WeakFileLock

from lerobot.processor.relative_ee_action_processor import absolute_ee_to_relative, ee_to_rot6d
from lerobot.utils.constants import ACTION, OBS_STATE

from .compute_stats import RunningQuantileStats
from .io_utils import load_stats, write_stats

logger = logging.getLogger(__name__)

# Bump when EE conversion, target selection, or statistics computation changes.
_RELATIVE_EE_STATS_VERSION = 1


def relative_ee_stats_cache_key(
    actions: np.ndarray,
    states: np.ndarray,
    episode_indices: np.ndarray,
    chunk_size: int,
) -> str:
    """Hash selected rows in order, excluding normalization choices and video data."""
    digest = hashlib.sha256(f"relative-ee:{_RELATIVE_EE_STATS_VERSION}:{chunk_size}".encode())
    for name, values, dtype in (
        (ACTION, actions, "<f4"),
        (OBS_STATE, states, "<f4"),
        ("episode_index", episode_indices, "<i8"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(f"{name}:{array.shape}:{dtype}:".encode())
        if array.size:
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


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


def _load_relative_ee_columns(hf_dataset: Dataset | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Read only needed Arrow columns, preserving selected row order and duplicates.

    Going through Dataset's Arrow formatter honors its indices mapping; reading
    hf_dataset.data directly would silently hash excluded or reordered rows.
    """
    dtypes = {ACTION: np.float32, OBS_STATE: np.float32, "episode_index": np.int64}
    if not isinstance(hf_dataset, Dataset):
        return {key: np.asarray(hf_dataset[key], dtype=dtype) for key, dtype in dtypes.items()}
    table = hf_dataset.select_columns(list(dtypes)).with_format("arrow")[:]
    columns = {}
    for key, dtype in dtypes.items():
        array = table.column(key).combine_chunks()
        if key in (ACTION, OBS_STATE):
            if not (pa.types.is_list(array.type) or pa.types.is_fixed_size_list(array.type)):
                raise ValueError(f"Relative EE requires {key} to contain 8D vectors")
            if not np.all(array.value_lengths().to_numpy(zero_copy_only=False) == 8):
                raise ValueError(f"Relative EE requires {key} shape [frames, 8]")
            values = array.flatten().to_numpy(zero_copy_only=False).reshape(len(array), 8)
        else:
            values = array.to_numpy(zero_copy_only=False)
        columns[key] = np.asarray(values, dtype=dtype)
    return columns


def load_or_compute_relative_ee_stats(
    hf_dataset,
    chunk_size: int,
    cache_dir: Path,
    identity_rot6d: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Reuse raw derived stats; apply rotation normalization overrides only in memory."""
    started = time.perf_counter()
    logger.info("Loading relative EE statistics columns (Arrow for Hugging Face datasets)")
    columns = _load_relative_ee_columns(hf_dataset)
    logger.info(
        "Loaded %d relative EE rows in %.2fs; hashing cache key",
        len(columns[ACTION]),
        time.perf_counter() - started,
    )
    hash_started = time.perf_counter()
    key = relative_ee_stats_cache_key(
        columns[ACTION], columns[OBS_STATE], columns["episode_index"], chunk_size
    )
    logger.info("Relative EE cache key %s computed in %.2fs", key, time.perf_counter() - hash_started)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / key
    with WeakFileLock(cache_path.with_suffix(".lock")):
        stats = load_stats(cache_path)
        if stats is None:
            logger.info("Computing relative EE statistics for %s", cache_path)
            stats = compute_relative_ee_stats(columns, chunk_size)
            with TemporaryDirectory(dir=cache_dir) as temporary_dir:
                write_stats(stats, Path(temporary_dir))
                Path(temporary_dir).replace(cache_path)
        else:
            logger.info("Loaded relative EE statistics from %s", cache_path)
    if identity_rot6d:
        _force_identity_rot6d_stats(stats)
    return stats


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
