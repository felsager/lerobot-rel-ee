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

"""FAST fitting uses the same relative-EE action space as the policy processors."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
fft = pytest.importorskip("scipy.fft")

from tokenizers import ByteLevelBPETokenizer  # noqa: E402
from tokenizers.trainers import BpeTrainer  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.fineart_vla import fit_fast_tokenizer as fitting  # noqa: E402
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig  # noqa: E402
from lerobot.processor import (  # noqa: E402
    NormalizerProcessorStep,
    RelativeEEActionsStep,
    batch_to_transition,
)
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.rotation_representations import ROT_REPR_DIM, RotationRepresentation  # noqa: E402


class _Fitter:
    def __init__(self):
        self.actions = None

    def fit(self, actions):
        self.actions = actions.copy()
        return self

    def __call__(self, actions):
        return actions

    def decode(self, actions):
        return actions

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "processor_config.json").write_text("{}")


@pytest.mark.parametrize("rot_repr", [r for r in RotationRepresentation if ROT_REPR_DIM[r] is not None])
def test_fit_uses_transformed_normalized_actions_and_selected_episodes(tmp_path, monkeypatch, rot_repr):
    monkeypatch.setenv("RANK", "0")
    fitter = _Fitter()
    monkeypatch.setattr(fitting, "_load_fast_fitter", lambda name: fitter)
    # Each row of the selected episode is the same target. A rotated reference makes
    # subtraction of pose components observably different from the required SE(3) transform.
    state = [0.25, 0.5, -0.25, 0, 0, 0.6, 0.8, 0.5]
    target = [0.4, 0.1, 0.2, 0, 0.6, 0, 0.8, 0.7]
    root = tmp_path / "dataset"
    shard = root / "data/chunk-000/file-000.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * 6 + [1] * 6,
                "observation.state": [state] * 12,
                "action": [target] * 6 + [[100, 100, 100, 0, 0, 0, 1, 0]] * 6,
            }
        ),
        shard,
    )
    dim = 4 + ROT_REPR_DIM[rot_repr]
    scale = torch.full((dim,), 4.0)
    if rot_repr == RotationRepresentation.rot6d:
        scale[3:9] = 1  # Match rot6d_identity_norm=True, including near-constant dimensions.
    stats = {"q01": -scale, "q99": scale}
    kwargs = {
        "dataset_repo_id": "test/ee",
        "dataset_root": root,
        "cache_dir": tmp_path / "cache",
        "n_samples": 4,
        "validation_samples": 2,
        "chunk_size": 2,
        "episodes": [0, 1],
        "exclude_episodes": [1],
        "use_relative_ee": True,
        "rotation_representation": rot_repr,
        "action_stats": stats,
    }
    path = fitting.fit_fast_tokenizer(**kwargs)
    transition = batch_to_transition(
        {
            "observation.state": torch.tensor([state]),
            ACTION: torch.tensor([[target, target]]),
        }
    )
    relative = RelativeEEActionsStep(enabled=True, state_frame=-1, rot_repr=rot_repr)
    normalize = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (dim,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={ACTION: stats},
    )
    expected = normalize(relative(transition))["action"].numpy()
    np.testing.assert_allclose(fitter.actions, np.repeat(expected, 4, axis=0), atol=1e-6)
    assert (Path(path) / "reconstruction_validation.json").is_file()
    assert (Path(path) / "reconstruction_examples.npz").is_file()
    # The published tokenizer cache can be reused without loading a fitting backend.
    monkeypatch.setattr(fitting, "_load_fast_fitter", lambda name: pytest.fail("unexpected refit"))
    assert fitting.fit_fast_tokenizer(**kwargs) == path


def test_cache_separates_ee_rotation_and_normalization():
    kwargs = {
        "dataset_repo_id": "test/ee",
        "base_tokenizer_name": "test/fast",
        "n_samples": 4,
        "chunk_size": 2,
        "normalization_mode": "QUANTILES",
    }
    absolute = fitting._dataset_signature(**kwargs)
    relative = fitting._dataset_signature(**kwargs, use_relative_ee=True)
    euler = fitting._dataset_signature(
        **kwargs, use_relative_ee=True, rotation_representation=RotationRepresentation.euler_angles
    )
    identity = fitting._dataset_signature(**kwargs, use_relative_ee=True, action_stats={"q01": [-1] * 10})
    assert len({absolute, relative, euler, identity}) == 4


def test_factory_forwards_training_episodes_and_ee_fit_options(monkeypatch, tmp_path):
    captured = {}

    def fit(**kwargs):
        captured.update(kwargs)
        return "fitted"

    monkeypatch.setattr(fitting, "fit_fast_tokenizer", fit)
    config = FineARTVLAConfig(
        device="cpu", use_relative_ee=True, text_loss_weight=0, use_pi05_action_prompt=True, recipe=None
    )
    config._runtime_dataset_meta = SimpleNamespace(repo_id="test/ee", root=tmp_path, revision="selected")
    # Stop before tokenizer creation; the fit resolver is what this test exercises.
    import lerobot.policies.fineart_vla.processor_fineart_vla as processor

    class ReachedFitError(Exception):
        pass

    def resolve(*args, **kwargs):
        result = fitting.resolve_fast_tokenizer(*args, **kwargs)
        assert result == "fitted"
        raise ReachedFitError

    monkeypatch.setattr(processor, "resolve_fast_tokenizer", resolve)
    # Avoid constructing a language tokenizer before reaching the fitting boundary.
    monkeypatch.setattr(processor, "TokenizerProcessorStep", lambda **kwargs: None)
    config.input_features = {}
    config.output_features = {}
    with pytest.raises(ReachedFitError):
        make_pre_post_processors(
            config, dataset_episodes=[2, 5], dataset_stats={"action": {"q01": [-1] * 10}}
        )
    assert captured["episodes"] == [2, 5]
    assert captured["use_relative_ee"] is True
    assert captured["rotation_representation"] == RotationRepresentation.rot6d
    assert captured["action_stats"]["q01"] == [-1] * 10


def test_fit_rejects_combined_relative_modes(tmp_path):
    with pytest.raises(ValueError, match="joint-relative"):
        fitting.fit_fast_tokenizer(
            dataset_repo_id="test/ee", cache_dir=tmp_path, use_relative_ee=True, use_relative_actions=True
        )


def test_fast_byte_alphabet_preserves_unseen_coefficients(tmp_path, monkeypatch):
    # Smooth chunks leave many coefficient values unseen during fitting. Reproduce
    # the upstream initial alphabet with real BPE, not an identity tokenizer mock.
    rng = np.random.default_rng(42)
    rng.uniform(-1, 1, (32, 50, 10))
    actions = np.repeat(rng.uniform(-1, 1, (32, 1, 10)), 50, axis=1).astype(np.float32)
    coefficients = np.around(fft.dct(actions, axis=1, norm="ortho") * 10).astype(int)
    minimum, maximum = int(coefficients[:24].min()), int(coefficients[:24].max())
    strings = ["".join(map(chr, np.maximum(chunk.flatten() - minimum, 0))) for chunk in coefficients]
    bpe = ByteLevelBPETokenizer()
    bpe._tokenizer.train_from_iterator(
        strings[:24],
        trainer=BpeTrainer(
            vocab_size=1024,
            min_frequency=2,
            show_progress=False,
            special_tokens=[],
            initial_alphabet=[chr(i) for i in range(maximum - minimum + 1)],
            max_token_length=10000,
        ),
    )
    fitted = SimpleNamespace(
        bpe_tokenizer=PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=False),
        scale=10,
        min_token=minimum,
        vocab_size=1024,
    )
    assert any(
        fitted.bpe_tokenizer.decode(fitted.bpe_tokenizer(text)["input_ids"]) != text for text in strings[24:]
    )

    fitting._ensure_byte_level_alphabet(fitted, actions[:24])
    # Encoding is lossless even for coefficient characters absent from training.
    for text in [*strings[24:], "".join(map(chr, range(1024)))]:
        assert fitted.bpe_tokenizer.decode(fitted.bpe_tokenizer(text)["input_ids"]) == text

    fitted.bpe_tokenizer.save_pretrained(tmp_path)
    reloaded = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    reconstructed = []
    for text in strings[24:]:
        decoded = reloaded.decode(reloaded(text)["input_ids"])
        decoded_coefficients = np.array(list(map(ord, decoded))).reshape(50, 10) + minimum
        reconstructed.append(fft.idct(decoded_coefficients / 10, axis=0, norm="ortho"))
    assert np.sqrt(np.mean((np.stack(reconstructed) - actions[24:]) ** 2)) < 0.01

    # A backend that already preserves all bytes must not be retrained.
    monkeypatch.setattr(fitting, "ByteLevelBPETokenizer", lambda: pytest.fail("unnecessary BPE refit"))
    fitting._ensure_byte_level_alphabet(fitted, actions[:24])
