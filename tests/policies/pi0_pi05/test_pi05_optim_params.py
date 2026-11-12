from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


class _FakePolicy(nn.Module):
    """Stand-in with PI05's parameter naming, so the test needs no weights."""

    def __init__(self, scale: float):
        super().__init__()
        self.config = SimpleNamespace(optimizer_lr=1e-4, optimizer_vlm_lr_scale=scale)
        self.model = nn.Module()
        self.model.paligemma_with_expert = nn.Module()
        self.model.paligemma_with_expert.paligemma = nn.Linear(2, 2)
        self.model.paligemma_with_expert.gemma_expert = nn.Linear(2, 2)
        self.model.action_in_proj = nn.Linear(2, 2)

    get_optim_params = PI05Policy.get_optim_params


def test_default_scale_is_one():
    assert PI05Config.__dataclass_fields__["optimizer_vlm_lr_scale"].default == 1.0


def test_scale_one_returns_all_parameters():
    policy = _FakePolicy(1.0)
    assert len(list(policy.get_optim_params())) == len(list(policy.parameters()))


def test_scale_splits_vlm_group():
    policy = _FakePolicy(0.1)
    rest, vlm = policy.get_optim_params()
    assert "lr" not in rest
    assert vlm["lr"] == 1e-4 * 0.1
    vlm_ids = {id(p) for p in policy.model.paligemma_with_expert.paligemma.parameters()}
    assert {id(p) for p in vlm["params"]} == vlm_ids
    assert len(rest["params"]) == 4
    assert not vlm_ids & {id(p) for p in rest["params"]}

    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, 1e-4 * 0.1]


def test_frozen_vlm_has_no_vlm_group():
    policy = _FakePolicy(0.1)
    policy.model.paligemma_with_expert.paligemma.requires_grad_(False)
    assert len(policy.get_optim_params()) == 1
