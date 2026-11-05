# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import json
import pickle
import random
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.datasets import factory
from lerobot.datasets.factory import make_dataset, make_train_eval_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.task_paraphrases import TaskParaphrases

SOURCE = "move the {object} to the marker."
REPLACEMENTS = [SOURCE, "put the {object} on the marker.", "carry the {object} to the marker."]
TASK = SOURCE.format(object="red block")


def write_bank(tmp_path: Path, bank: object) -> Path:
    path = tmp_path / "paraphrases.json"
    path.write_text(json.dumps(bank))
    return path


@pytest.mark.parametrize("object_name", ["red block", "blue block", "purple block", "light blue block"])
def test_resamples_without_changing_captured_values(tmp_path, monkeypatch, object_name):
    augment = TaskParaphrases(write_bank(tmp_path, {SOURCE: REPLACEMENTS}))
    choices = iter(range(6))
    monkeypatch.setattr(random, "choice", lambda values: values[next(choices) % len(values)])
    task = SOURCE.format(object=object_name)
    expected = [template.format(object=object_name) for template in REPLACEMENTS]
    assert [augment(task) for _ in range(6)] == expected * 2
    assert augment("a different instruction") == "a different instruction"
    assert augment(task.upper()) == task.upper()
    # Compiled templates and cached candidates also survive DataLoader spawn pickling.
    restored = pickle.loads(pickle.dumps(augment))
    assert restored._cache == augment._cache


def test_template_matching_escapes_literals_and_repeated_fields(tmp_path):
    augment = TaskParaphrases(
        write_bank(tmp_path, {"move {object} (then hold {object}).": ["hold {object}."]})
    )
    assert augment("move red (then hold red).") == "hold red."
    mismatched = "move red (then hold blue)."
    assert augment(mismatched) == mismatched
    missing_punctuation = "move red Xthen hold redXY"
    assert augment(missing_punctuation) == missing_punctuation


@pytest.mark.parametrize(
    "bank",
    [
        [],
        {},
        {SOURCE: []},
        {SOURCE: "not a list"},
        {SOURCE: [None]},
        {SOURCE: [""]},
        {SOURCE: ["drop the object"]},
        {SOURCE: ["move {other}"]},
        {"move {object.name}": ["move {object.name}"]},
        {"move {object!r}": ["move {object!r}"]},
        {"move {object:>10}": ["move {object:>10}"]},
        {"move {}": ["move {}"]},
        {"move {object": ["move {object"]},
    ],
)
def test_rejects_invalid_banks(tmp_path, bank):
    with pytest.raises(ValueError):
        TaskParaphrases(write_bank(tmp_path, bank))


@pytest.fixture
def training_config(tmp_path):
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="test/paraphrases",
        fps=10,
        root=root,
        features={"action": {"dtype": "float32", "shape": (2,), "names": None}},
        use_videos=False,
    )
    for _ in range(4):
        for frame in range(6):
            dataset.add_frame({"action": np.array([frame, -frame], dtype=np.float32), "task": TASK})
        dataset.save_episode()
    dataset.finalize()
    write_bank(root / "meta", {SOURCE: REPLACEMENTS})
    return SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="test/paraphrases",
            root=str(root),
            paraphrases=True,
            use_imagenet_stats=False,
            video_backend="pyav",
        ),
        trainable_config=SimpleNamespace(
            observation_delta_indices=None, action_delta_indices=[0, 1], reward_delta_indices=None
        ),
        seed=42,
        num_workers=0,
        tolerance_s=1e-4,
        rename_map={},
        resume=False,
    )


def test_map_access_paths_preserve_chunks_and_dataset_files(training_config, monkeypatch):
    root = Path(training_config.dataset.root)
    original = {
        path: hashlib.sha256(path.read_bytes()).digest() for path in root.rglob("*") if path.is_file()
    }
    training_config.dataset.paraphrases = False
    canonical = make_dataset(training_config)
    training_config.dataset.paraphrases = True
    augmented = make_dataset(training_config)
    monkeypatch.setattr(random, "choice", lambda values: values[-1])
    expected_task = REPLACEMENTS[-1].format(object="red block")
    for items in ([augmented[0]], augmented[:2], augmented.__getitems__([0, 1])):
        for item in items:
            assert item["task"] == expected_task
            reference = canonical[int(item["index"])]
            assert reference["task"] == TASK
            for key in item.keys() - {"task"}:
                torch.testing.assert_close(item[key], reference[key])
    assert canonical.task_paraphrases is None
    assert augmented.get_raw_item(0).get("task", TASK) == TASK
    assert all(hashlib.sha256(path.read_bytes()).digest() == digest for path, digest in original.items())


def test_eval_split_keeps_canonical_instructions(training_config, monkeypatch):
    training_config.dataset.eval_split = 0.5
    train, evaluation = make_train_eval_datasets(training_config)
    monkeypatch.setattr(random, "choice", lambda values: values[-1])
    assert train[0]["task"] == REPLACEMENTS[-1].format(object="red block")
    assert evaluation[0]["task"] == TASK
    assert evaluation.task_paraphrases is None
    assert set(train.episodes).isdisjoint(evaluation.episodes)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_resamples_reproducibly(training_config, streaming, workers):
    training_config.dataset.streaming = streaming
    training_config.num_workers = workers

    def collect():
        random.seed(47)
        dataset = make_dataset(training_config)
        loader = DataLoader(
            dataset, batch_size=4, num_workers=workers, generator=torch.Generator().manual_seed(47)
        )
        iterator = iter(loader)
        try:
            return [task for batch in islice(iterator, 6) for task in batch["task"]]
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    first = collect()
    assert len(set(first)) > 1
    assert set(first) <= {template.format(object="red block") for template in REPLACEMENTS}
    assert collect() == first


@pytest.mark.parametrize("streaming", [False, True])
def test_missing_metadata_bank_requires_explicit_opt_in(training_config, streaming):
    training_config.dataset.streaming = streaming
    (Path(training_config.dataset.root) / "meta" / "paraphrases.json").unlink()
    with pytest.raises(FileNotFoundError, match="meta/paraphrases.json"):
        make_dataset(training_config)
    training_config.dataset.paraphrases = False
    assert make_dataset(training_config).task_paraphrases is None


def test_paraphrases_boolean_cli_flag():
    cfg = draccus.parse(DatasetConfig, args=["--repo_id=test/paraphrases", "--paraphrases=true"])
    assert cfg.paraphrases is True
    assert DatasetConfig(repo_id="test/paraphrases").paraphrases is False


def test_uses_resolved_dataset_metadata_root(training_config, tmp_path, monkeypatch):
    dataset = make_dataset(training_config)
    snapshot = tmp_path / "resolved_snapshot"
    (snapshot / "meta").mkdir(parents=True)
    write_bank(snapshot / "meta", {SOURCE: ["carry {object}."]})
    dataset.meta.root = snapshot
    monkeypatch.setattr(factory, "LeRobotDataset", lambda *args, **kwargs: dataset)
    loaded = make_dataset(training_config)
    assert loaded.task_paraphrases(TASK) == "carry red block."
