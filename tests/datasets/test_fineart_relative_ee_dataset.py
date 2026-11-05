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

import numpy as np
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize("eval_split", [0.0, 0.5])
def test_fineart_relative_ee_training_dataset(tmp_path: Path, eval_split: float) -> None:
    root = tmp_path / "dataset"
    recorded = LeRobotDataset.create(
        repo_id="test/fineart_ee",
        fps=10,
        root=root,
        features={key: {"dtype": "float32", "shape": (8,), "names": None} for key in (ACTION, OBS_STATE)},
        use_videos=False,
    )
    for episode in range(2):
        for frame in range(4):
            state = np.array([episode + frame * 0.1, 0, 0, 0, 0, 0, 1, 0.5], dtype=np.float32)
            action = state.copy()
            action[0] += 0.05
            recorded.add_frame({OBS_STATE: state, ACTION: action, "task": "move the block"})
        recorded.save_episode()
    recorded.finalize()

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="test/fineart_ee",
            root=str(root),
            eval_split=eval_split,
            use_imagenet_stats=False,
            video_backend="pyav",
        ),
        policy=FineARTVLAConfig(
            device="cpu",
            use_relative_ee=True,
            rot6d_identity_norm=True,
            use_proprio=False,
            use_pi05_action_prompt=True,
            recipe=None,
            text_loss_weight=0,
            chunk_size=2,
            n_action_steps=2,
        ),
    )
    training, evaluation = make_train_eval_datasets(cfg)

    # Storage and samples stay in raw quaternion space; normalization stats use rot6d.
    assert training[0][ACTION].shape == (2, 8)
    assert training[0][OBS_STATE].shape == (8,)
    for key in (ACTION, OBS_STATE):
        assert training.meta.stats[key]["q01"].shape == (10,)
        np.testing.assert_array_equal(training.meta.stats[key]["q01"][3:9], -np.ones(6))
        np.testing.assert_array_equal(training.meta.stats[key]["q99"][3:9], np.ones(6))

    if eval_split:
        assert training.episodes == [0]
        assert evaluation is not None
        assert evaluation.episodes == [1]
        for key in (ACTION, OBS_STATE):
            np.testing.assert_array_equal(evaluation.meta.stats[key]["q01"], training.meta.stats[key]["q01"])
    else:
        assert evaluation is None
