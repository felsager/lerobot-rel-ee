#!/usr/bin/env python

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
"""The --eval_hook schedule, loading, and call on the live policy."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_train import (  # noqa: E402
    _is_eval_hook_step,
    _load_eval_hook,
    _run_eval_hook,
)


def test_hook_runs_in_the_second_half_and_after_the_last_step():
    cfg = SimpleNamespace(eval_hook="hook.py:f", eval_hook_freq=2000, eval_hook_start=0.5, steps=10_000)
    assert [s for s in range(1, 10_001) if _is_eval_hook_step(s, cfg)] == [6000, 8000, 10_000]
    cfg.steps = 10_500
    assert [s for s in range(1, 10_501) if _is_eval_hook_step(s, cfg)] == [6000, 8000, 10_000, 10_500]
    cfg.eval_hook = None
    assert not any(_is_eval_hook_step(s, cfg) for s in range(1, 10_501))


def test_hook_loads_from_a_file_or_a_module(tmp_path):
    (tmp_path / "my_hook.py").write_text("def score(policy, pre, post):\n    return {'x': 1.0}\n")
    assert _load_eval_hook(f"{tmp_path}/my_hook.py:score")(None, None, None) == {"x": 1.0}
    assert _load_eval_hook("json:dumps") is __import__("json").dumps


def test_hook_sees_eval_mode_without_gradients_and_leaves_training_rng_alone():
    policy = torch.nn.Linear(2, 2)
    seen = []

    def hook(model, pre, post):
        seen.append((model.training, torch.is_grad_enabled()))
        torch.rand(100)  # consumes the global RNG inside the hook
        return {"real_frames/angle": 3.0}

    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    assert _run_eval_hook(hook, policy, None, None, None) == {"real_frames/angle": 3.0}
    assert torch.equal(torch.rand(3), expected)
    assert seen == [(False, False)] and policy.training


class ShadowEMA:
    """The EMAModel surface used by _ema_weights, with a shadow of all-ones weights."""

    def store(self, params):
        self.saved = [p.detach().clone() for p in params]

    def copy_to(self, params):
        for p in params:
            p.data.fill_(1.0)

    def restore(self, params):
        for p, saved in zip(params, self.saved, strict=True):
            p.data.copy_(saved)


def test_hook_also_reports_the_ema_weights():
    policy = torch.nn.Linear(2, 1, bias=False)
    policy.weight.data.zero_()
    metrics = _run_eval_hook(
        lambda m, pre, post: {"w": m.weight.sum().item()}, policy, ShadowEMA(), None, None
    )
    assert metrics == {"w": 0.0, "ema/w": 2.0}
    assert policy.weight.sum().item() == 0.0
