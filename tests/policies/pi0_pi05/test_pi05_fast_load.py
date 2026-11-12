"""PI05 OpenPI-format loading skips random init only when the checkpoint covers every weight."""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

import safetensors.torch  # noqa: E402
import transformers.utils  # noqa: E402
from transformers import PreTrainedModel  # noqa: E402

from lerobot.policies.pi05 import modeling_pi05  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, _skip_weight_init  # noqa: E402

_ORIGINAL_KAIMING = nn.init.kaiming_uniform_


class _TinyPolicy(PI05Policy):
    """Two linear layers in place of the full model; records whether init was skipped."""

    skipped_init: list[bool] = []

    def __init__(self, config, **kwargs):
        nn.Module.__init__(self)
        type(self).skipped_init.append(nn.init.kaiming_uniform_ is not _ORIGINAL_KAIMING)
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def _convert_openpi_state_dict(self, state_dict):
        return state_dict


@pytest.fixture
def checkpoint(monkeypatch):
    weights = {}
    monkeypatch.setattr(transformers.utils, "cached_file", lambda *a, **k: "model.safetensors")
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path: dict(weights))
    _TinyPolicy.skipped_init = []
    return weights


def _weights(*names):
    return {
        f"{name}.{param}": torch.full((4, 4) if param == "weight" else (4,), 7.0)
        for name in names
        for param in ("weight", "bias")
    }


def test_complete_checkpoint_builds_once_without_random_init(checkpoint):
    checkpoint.update(_weights("a", "b"))

    policy = _TinyPolicy.from_pretrained("repo", config=object())

    assert _TinyPolicy.skipped_init == [True]
    for name, tensor in policy.state_dict().items():
        assert torch.equal(tensor, checkpoint[name]), name


def test_missing_weights_rebuild_with_random_init(checkpoint):
    checkpoint.update(_weights("a"))

    policy = _TinyPolicy.from_pretrained("repo", config=object(), strict=False)

    assert _TinyPolicy.skipped_init == [True, False]
    assert torch.equal(policy.a.weight, checkpoint["a.weight"])
    # Kaiming-uniform bound for fan_in=4 is 0.5; uninitialised memory would not respect it.
    assert policy.b.weight.abs().max() <= 0.5
    assert not torch.equal(policy.b.weight, torch.zeros(4, 4))


def test_unreadable_checkpoint_returns_initialised_model(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(transformers.utils, "cached_file", fail)
    _TinyPolicy.skipped_init = []

    _TinyPolicy.from_pretrained("repo", config=object())

    assert _TinyPolicy.skipped_init == [False]


def test_skip_weight_init_restores_functions_after_error():
    original_initialize = PreTrainedModel.initialize_weights
    tensor = torch.zeros(3)

    with pytest.raises(RuntimeError), _skip_weight_init():
        assert nn.init.normal_(tensor) is tensor
        assert torch.equal(tensor, torch.zeros(3))
        raise RuntimeError

    assert nn.init.kaiming_uniform_ is _ORIGINAL_KAIMING
    assert all(
        getattr(nn.init, name).__module__ == "torch.nn.init" for name in modeling_pi05._TORCH_INIT_FUNCTIONS
    )
    assert PreTrainedModel.initialize_weights is original_initialize
