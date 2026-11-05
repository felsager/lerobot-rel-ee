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

"""Exercise FineART's real fused RTC paths with a small, download-free action model."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.fineart_vla import modeling_fineart_vla as modeling  # noqa: E402
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    ACTION_CODE_TOKEN_MASK,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

_BATCH, _CHUNK, _ACTION_DIM, _WIDTH = 2, 4, 10, 12


class _TinyBackbone(nn.Module):
    """Capture attention/conditioning while exposing independent action predictions."""

    def __init__(self, repeats: int) -> None:
        super().__init__()
        self.prediction = nn.Parameter(torch.full((_BATCH, repeats * _CHUNK, _WIDTH), 0.1))
        self.tokens = nn.Embedding(16, _WIDTH)
        self.lm_head = nn.Linear(_WIDTH, 16, bias=False)
        layer = SimpleNamespace(self_attn=SimpleNamespace(q_proj=SimpleNamespace(weight=torch.ones(1))))
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[layer])), lm_head=self.lm_head
        )
        self.call: dict = {}

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        return self.tokens(tokens)

    def forward(self, **kwargs):
        self.call = kwargs
        return (kwargs["inputs_embeds"][0], self.prediction), None


class _TinyCore(nn.Module):
    """Replace expensive embeddings/backbone, retaining the policy's training orchestration."""

    def __init__(self, config: FineARTVLAConfig) -> None:
        super().__init__()
        self.config = config
        self.paligemma_with_expert = _TinyBackbone(config.flow_num_repeats)
        self.action_out_proj = nn.Identity()
        self.noisy_actions: Tensor | None = None
        self.model_time: Tensor | None = None

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return (2 + torch.arange(shape[0], device=device))[:, None, None].expand(shape)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.linspace(0.2, 0.8, batch_size, device=device)

    def embed_prefix(self, images, image_masks, tokens: Tensor, masks: Tensor):
        return self.paligemma_with_expert.embed_language_tokens(tokens), masks, torch.zeros_like(masks)

    def embed_suffix(self, actions: Tensor, time: Tensor):
        self.noisy_actions, self.model_time = actions, time
        pad = torch.ones(actions.shape[:2], dtype=torch.bool)
        attention = torch.zeros_like(pad)
        attention[:, 0] = True
        condition = time[..., None].expand(*time.shape, _WIDTH)
        return actions, pad, attention, condition

    def _prepare_attention_masks_4d(self, mask: Tensor, dtype=None) -> Tensor:
        return mask[:, None]

    def _apply_checkpoint(self, fn, *args):
        return fn(*args)


def _policy(repeats: int, max_delay: int, *, fast: bool = False, ki: bool = False):
    config = FineARTVLAConfig(
        device="cpu",
        use_relative_ee=True,
        max_action_dim=_WIDTH,
        chunk_size=_CHUNK,
        n_action_steps=_CHUNK,
        rtc_training_max_delay=max_delay,
        flow_num_repeats=repeats,
        enable_fast_action_loss=fast,
        knowledge_insulation=ki,
        text_loss_weight=0.5 if fast else 0.0,
        flow_loss_weight=2.0,
        fast_action_loss_weight=0.25,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,))},
    )
    policy = modeling.FineARTVLAPolicy.__new__(modeling.FineARTVLAPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.model = _TinyCore(config)
    policy._preprocess_images = lambda batch: ([], [])
    if repeats > 1:
        policy.register_buffer(
            "_flow_block_diag", torch.block_diag(*[torch.ones(_CHUNK, _CHUNK).bool()] * repeats)
        )
    return policy


def _batch(*, fast: bool = False) -> dict[str, Tensor]:
    # Relative-EE processors have already transformed/normalized these ten action coordinates.
    batch = {
        ACTION: torch.arange(_BATCH * _CHUNK * _ACTION_DIM).reshape(_BATCH, _CHUNK, _ACTION_DIM) / 100,
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 3], [1, 2, 3]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(_BATCH, 3, dtype=torch.bool),
        "predict_actions": torch.tensor([True, False]),
    }
    if fast:
        batch.update(
            {
                "text_labels": torch.tensor([[-100, 2, 3], [-100, 2, 3]]),
                ACTION_TOKENS: torch.tensor([[4, 5, 6], [4, 5, 6]]),
                ACTION_TOKEN_MASK: torch.ones(_BATCH, 3, dtype=torch.bool),
                ACTION_CODE_TOKEN_MASK: torch.ones(_BATCH, 3, dtype=torch.bool),
            }
        )
    return batch


def _fixed_prefixes(monkeypatch, repeats: int, enabled: bool) -> Tensor | None:
    delays = torch.tensor([2, 0, 1, 2, 0, 1])[: repeats * _BATCH]
    prefix_mask = torch.arange(_CHUNK)[None] < delays[:, None] if enabled else None

    def sample(batch_size, horizon, max_delay, device):
        assert (batch_size, horizon) == (repeats * _BATCH, _CHUNK)
        assert max_delay == (2 if enabled else 0)
        return prefix_mask

    monkeypatch.setattr(modeling, "sample_training_rtc_prefix_mask", sample)
    return prefix_mask


@pytest.mark.parametrize("repeats", [1, 3])
@pytest.mark.parametrize("max_delay", [0, 2])
@pytest.mark.parametrize("reduction", ["mean", "none"])
@pytest.mark.parametrize("ki", [False, True])
def test_fused_rtc_inputs_postfix_loss_and_gradients(monkeypatch, repeats, max_delay, reduction, ki):
    """Unequal delays preserve per-example routing, repeated draws and padded dimensions."""
    policy = _policy(repeats, max_delay, ki=ki)
    batch = _batch()
    prefix_mask = _fixed_prefixes(monkeypatch, repeats, max_delay > 0)
    loss, _ = policy.forward(batch, reduction=reduction)
    core = policy.model
    actions = policy.prepare_action(batch).repeat(repeats, 1, 1)
    noise = core.sample_noise(actions.shape, actions.device)
    time = core.sample_time(actions.shape[0], actions.device)
    if prefix_mask is None:
        expected_time = time
        predicted = torch.ones_like(actions[..., :_ACTION_DIM], dtype=torch.bool)
    else:
        expected_time = torch.where(prefix_mask, 0.0, time[:, None])
        predicted = (~prefix_mask)[..., None].expand(-1, -1, _ACTION_DIM)
        torch.testing.assert_close(core.noisy_actions[prefix_mask], actions[prefix_mask])
    torch.testing.assert_close(core.model_time, expected_time)
    expanded_time = expected_time[:, None, None] if expected_time.ndim == 1 else expected_time[..., None]
    torch.testing.assert_close(core.noisy_actions, expanded_time * noise + (1 - expanded_time) * actions)

    error = (noise - actions - 0.1).square()[..., :_ACTION_DIM]
    per_draw = (error * predicted).sum((1, 2)) / predicted.sum((1, 2))
    per_example = per_draw.reshape(repeats, _BATCH).mean(0)
    expected_loss = per_example[0] if reduction == "mean" else per_example * batch["predict_actions"]
    torch.testing.assert_close(loss, policy.config.flow_loss_weight * expected_loss)

    call = core.paligemma_with_expert.call
    assert call.get("suppress_prefix_grads", False) == ki
    condition = call["adarms_cond"][1]
    if repeats == 1:
        torch.testing.assert_close(condition[..., 0], expected_time)
    else:
        expected_condition = time[:, None].expand(-1, _CHUNK) if max_delay == 0 else expected_time
        expected_condition = expected_condition.reshape(repeats, _BATCH, _CHUNK).transpose(0, 1)
        torch.testing.assert_close(condition[..., 0], expected_condition.reshape(_BATCH, repeats * _CHUNK))
        attention = call["attention_mask"][:, 0]
        prefix_len = batch[OBS_LANGUAGE_TOKENS].shape[1]
        assert not attention[:, :prefix_len, prefix_len:].any()
        for draw in range(repeats):
            rows = attention[:, prefix_len + draw * _CHUNK : prefix_len + (draw + 1) * _CHUNK]
            assert rows[:, :, prefix_len + draw * _CHUNK : prefix_len + (draw + 1) * _CHUNK].all()
            assert rows[:, :, prefix_len:].sum() == _BATCH * _CHUNK * _CHUNK

    loss.sum().backward()
    gradient = core.paligemma_with_expert.prediction.grad
    assert not gradient[1].any()  # Recipe routing excludes this example from action supervision.
    assert not gradient[..., _ACTION_DIM:].any()  # Model-only padding has no loss.
    draw_gradient = gradient.reshape(_BATCH, repeats, _CHUNK, _WIDTH).transpose(0, 1)
    draw_gradient = draw_gradient.reshape(repeats * _BATCH, _CHUNK, _WIDTH)
    if prefix_mask is not None:
        assert not draw_gradient[prefix_mask].any()
    assert draw_gradient[..., :_ACTION_DIM].abs().sum() > 0


@pytest.mark.parametrize("repeats", [1, 3])
@pytest.mark.parametrize("ki", [False, True])
def test_rtc_keeps_text_fast_objectives_and_prevents_action_leakage(monkeypatch, repeats, ki):
    """FAST/text remain supervised while flow cannot attend to target action tokens."""
    policy = _policy(repeats, 2, fast=True, ki=ki)
    batch = _batch(fast=True)
    _fixed_prefixes(monkeypatch, repeats, True)
    loss, metrics = policy.forward(batch)
    assert metrics["text_loss"] > 0
    assert metrics["fast_action_loss"] > 0
    assert metrics["loss"] == pytest.approx(
        2 * metrics["flow_loss"] + 0.5 * metrics["text_loss"] + 0.25 * metrics["fast_action_loss"]
    )
    call = policy.model.paligemma_with_expert.call
    prefix_len, fast_len = batch[OBS_LANGUAGE_TOKENS].shape[1], batch[ACTION_TOKENS].shape[1]
    action_start = prefix_len + fast_len
    assert not call["attention_mask"][:, :, action_start:, prefix_len:action_start].any()
    assert call["attention_mask"][:, :, action_start:, :prefix_len].all()
    positions = call["position_ids"][:, action_start:].reshape(_BATCH, repeats, _CHUNK)
    torch.testing.assert_close(
        positions, (prefix_len + torch.arange(_CHUNK))[None, None].expand_as(positions)
    )

    loss.backward()
    assert policy.model.paligemma_with_expert.lm_head.weight.grad.abs().sum() > 0
    assert policy.model.paligemma_with_expert.prediction.grad.abs().sum() > 0

    policy.config.rtc_training_max_delay = 0
    _fixed_prefixes(monkeypatch, repeats, False)
    _, ordinary_metrics = policy.forward(batch)
    assert ordinary_metrics["text_loss"] == metrics["text_loss"]
    assert ordinary_metrics["fast_action_loss"] == metrics["fast_action_loss"]


def test_pi05_prompt_rtc_keeps_fineart_knowledge_insulation():
    """A clean prefix must retain FineART's KI-aware flow-only execution path."""
    policy = _policy(1, 2, ki=True)
    core = policy.model
    batch = _batch()
    actions = policy.prepare_action(batch)
    prefix_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    losses = modeling.FineARTVLAPytorch.forward(
        core,
        [],
        [],
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        actions,
        core.sample_noise(actions.shape, actions.device),
        core.sample_time(_BATCH, actions.device),
        prefix_mask=prefix_mask,
    )
    assert losses.shape == actions.shape
    assert core.paligemma_with_expert.call["suppress_prefix_grads"]
    assert not core.paligemma_with_expert.call["inputs_embeds"][0].requires_grad
    assert not core.model_time[prefix_mask].any()
    torch.testing.assert_close(core.noisy_actions[prefix_mask], actions[prefix_mask])
