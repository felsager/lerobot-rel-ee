#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.gemma.configuration_gemma import GemmaConfig  # noqa: E402
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding  # noqa: E402

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs  # noqa: E402
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig  # noqa: E402
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPytorch  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    PI05Policy,
    PI05Pytorch,
    compute_layer_complete,
)
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from lerobot.policies.pi_gemma import _get_pi_gemma_decoder_layer_base  # noqa: E402
from lerobot.policies.rtc.training import reduce_training_rtc_loss  # noqa: E402
from lerobot.processor import tokenizer_processor  # noqa: E402
from lerobot.processor.relative_ee_action_processor import to_relative_ee_actions  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from lerobot.utils.rotation_representations import ROT_REPR_DIM, RotationRepresentation  # noqa: E402


def test_pi05_training_rtc_uses_clean_prefix_and_per_token_time():
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    noise = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    time = torch.tensor([0.25])
    prefix_mask = torch.tensor([[True, True, False, False]])

    x_t, _, model_time = make_flow_matching_inputs(
        actions, noise, time, convention=FlowConvention.NOISE_AT_ONE, prefix_mask=prefix_mask
    )

    assert model_time.tolist() == [[0.0, 0.0, 0.25, 0.25]]
    assert torch.equal(x_t[:, :2], actions[:, :2])
    assert torch.equal(x_t[:, 2:], 0.25 * noise[:, 2:] + 0.75 * actions[:, 2:])


def test_pi05_training_rtc_loss_excludes_clean_prefix():
    losses = torch.tensor([[[100.0], [100.0], [2.0], [4.0]]])
    prefix_mask = torch.tensor([[True, True, False, False]])

    loss = reduce_training_rtc_loss(losses, prefix_mask, reduction="mean")

    assert loss.item() == pytest.approx(3.0)


@pytest.mark.parametrize("config_cls", [PI05Config, FineARTVLAConfig])
@pytest.mark.parametrize("max_delay", [-1, 5])
def test_pi05_config_rejects_invalid_training_rtc_delay(config_cls: type[PI05Config], max_delay: int) -> None:
    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        config_cls(chunk_size=5, n_action_steps=5, rtc_training_max_delay=max_delay)


@pytest.mark.parametrize("config_cls", [PI05Config, FineARTVLAConfig])
@pytest.mark.parametrize("max_delay", [0, 4])
def test_relative_ee_training_rtc_config_roundtrip(
    config_cls: type[PI05Config], max_delay: int, tmp_path: Path
) -> None:
    config = config_cls(
        device="cpu", use_relative_ee=True, chunk_size=5, n_action_steps=5, rtc_training_max_delay=max_delay
    )
    config.save_pretrained(tmp_path)

    restored = config_cls.from_pretrained(tmp_path)

    assert restored.rtc_training_max_delay == max_delay
    assert restored.use_relative_ee
    assert restored.rotation_representation == config.rotation_representation


class _Tokenizer:
    def __call__(self, text: list[str], max_length: int, **_kwargs: Any) -> dict[str, torch.Tensor]:
        shape = (len(text), max_length)
        return {"input_ids": torch.zeros(shape, dtype=torch.long), "attention_mask": torch.ones(shape)}


class _TinyTrainingExpert(torch.nn.Module):
    """Replace transformer weights while retaining the policy and core training forwards."""

    def __init__(self, batch_size: int, horizon: int, width: int) -> None:
        super().__init__()
        self.prediction = torch.nn.Parameter(torch.ones(batch_size, horizon, width))
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            self_attn=SimpleNamespace(q_proj=SimpleNamespace(weight=torch.zeros(1)))
                        )
                    ]
                )
            )
        )

    def forward(
        self, *, inputs_embeds: list[torch.Tensor], **_kwargs: Any
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], None]:
        return (inputs_embeds[0], self.prediction), None


def _tiny_training_policy(
    config: PI05Config, model_cls: type[PI05Pytorch] = PI05Pytorch
) -> tuple[PI05Policy, dict[str, torch.Tensor]]:
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.config = config
    model.gradient_checkpointing_enabled = False
    model.action_in_proj = torch.nn.Linear(config.max_action_dim, config.max_action_dim)
    model.time_mlp_in = torch.nn.Linear(config.max_action_dim, config.max_action_dim)
    model.time_mlp_out = torch.nn.Linear(config.max_action_dim, config.max_action_dim)
    model.action_out_proj = torch.nn.Identity()
    model.paligemma_with_expert = _TinyTrainingExpert(2, config.chunk_size, config.max_action_dim)
    model.embed_prefix = lambda _images, _img_masks, tokens, *_args: (
        torch.zeros(tokens.shape[0], 1, config.max_action_dim),
        torch.ones(tokens.shape[0], 1, dtype=torch.bool),
        torch.zeros(tokens.shape[0], 1, dtype=torch.bool),
    )
    model.sample_noise = lambda shape, device: torch.full(shape, 0.75, device=device)
    model.sample_time = lambda _batch_size, device: torch.tensor([0.25, 0.75], device=device)
    recorded: dict[str, torch.Tensor] = {}
    embed_suffix = model.embed_suffix

    def record_suffix(
        noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recorded["noisy_actions"] = noisy_actions.detach().clone()
        recorded["time"] = timestep.detach().clone()
        return embed_suffix(noisy_actions, timestep)

    model.embed_suffix = record_suffix
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.model = model
    policy._preprocess_images = lambda _batch: ([], [])
    return policy, recorded


@pytest.mark.parametrize("rot_repr", [r for r in RotationRepresentation if ROT_REPR_DIM[r] is not None])
@pytest.mark.parametrize("max_delay", [0, 2])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_pi05_relative_ee_training_rtc_pipeline(
    monkeypatch: pytest.MonkeyPatch, rot_repr: RotationRepresentation, max_delay: int, reduction: str
) -> None:
    """Train on normalized local poses; clean prefixes and padded features contribute no gradient."""
    monkeypatch.setattr(tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *_a, **_kw: _Tokenizer())
    action_dim = 4 + ROT_REPR_DIM[rot_repr]
    config = PI05Config(
        device="cpu",
        use_relative_ee=True,
        rotation_representation=rot_repr,
        rtc_training_max_delay=max_delay,
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=12,
        tokenizer_max_length=8,
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (action_dim,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (action_dim,))},
    )
    stats = {
        OBS_STATE: {"q01": -2 * torch.ones(action_dim), "q99": 2 * torch.ones(action_dim)},
        ACTION: {"q01": -4 * torch.ones(action_dim), "q99": 4 * torch.ones(action_dim)},
    }
    preprocessor, _postprocessor = make_pi05_pre_post_processors(config, stats)
    state = torch.tensor([[0.1, 0.2, 0.3, 0, 0, 0, 1, 0.5], [1, 2, 3, 0, 0, 2**-0.5, 2**-0.5, 0.8]])
    targets = (
        torch.tensor(
            [
                [0.2, 0.4, 0.6, 0, 0, 0, 1, 0.1],
                [0.3, 0.5, 0.7, 0.6, 0, 0, 0.8, 0.2],
                [0.4, 0.6, 0.8, 0, 0.6, 0, 0.8, 0.3],
                [0.5, 0.7, 0.9, 0, 0, 0.6, 0.8, 0.4],
            ]
        )
        .expand(2, -1, -1)
        .clone()
    )
    batch = preprocessor({OBS_STATE: state, ACTION: targets, "task": ["pick", "place"]})
    expected_actions = to_relative_ee_actions(targets, state, rot_repr) / 4
    torch.testing.assert_close(batch[ACTION], expected_actions)

    draws = []

    def fixed_delays(low: int, high: int, size: tuple[int], *, device: torch.device) -> torch.Tensor:
        draws.append((low, high, size))
        return torch.tensor([0, max_delay], device=device)

    monkeypatch.setattr(torch, "randint", fixed_delays)
    policy, recorded = _tiny_training_policy(config)
    expert = policy.model.paligemma_with_expert
    prefix_mask = torch.arange(config.chunk_size)[None] < torch.tensor([0, max_delay])[:, None]
    with torch.no_grad():
        expert.prediction[prefix_mask] = 100
        expert.prediction[..., action_dim:] = 100

    loss, info = policy.forward(batch, reduction=reduction)

    assert draws == ([(0, max_delay + 1, (2,))] if max_delay else [])
    time = torch.tensor([0.25, 0.75])[:, None].expand(2, config.chunk_size).clone()
    time[prefix_mask] = 0
    torch.testing.assert_close(recorded["time"], time if max_delay else time[:, 0])
    padded_actions = torch.nn.functional.pad(expected_actions, (0, config.max_action_dim - action_dim))
    expected_noisy_actions = time[..., None] * 0.75 + (1 - time[..., None]) * padded_actions
    torch.testing.assert_close(recorded["noisy_actions"], expected_noisy_actions)

    target_velocity = 0.75 - expected_actions
    losses = (expert.prediction[..., :action_dim] - target_velocity).square()
    postfix = ~prefix_mask
    expected_per_sample = torch.stack([losses[i, postfix[i]].mean() for i in range(2)])
    expected_loss = losses[postfix].mean() if reduction == "mean" else expected_per_sample
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(torch.tensor(info["loss_per_dim"]), losses[postfix].mean(dim=0))

    loss.sum().backward()
    gradient = expert.prediction.grad
    assert torch.count_nonzero(gradient[prefix_mask]) == 0
    assert torch.count_nonzero(gradient[..., action_dim:]) == 0
    assert torch.count_nonzero(gradient[..., :action_dim][postfix]) > 0


# ``compute_layer_complete`` hardcodes 8 attention heads when reshaping the joint
# attention output, so the miniature stack below has to match that.
_HEADS, _HEAD_DIM = 8, 4
_WIDTH = _HEADS * _HEAD_DIM


def _mini_gemma_config(use_adarms: bool) -> GemmaConfig:
    config = GemmaConfig(
        hidden_size=_WIDTH,
        intermediate_size=2 * _WIDTH,
        num_hidden_layers=1,
        num_attention_heads=_HEADS,
        num_key_value_heads=_HEADS,
        head_dim=_HEAD_DIM,
        vocab_size=32,
    )
    config.use_adarms = use_adarms
    config.adarms_cond_dim = _WIDTH if use_adarms else None
    config._attn_implementation = "eager"  # noqa: SLF001
    return config


def _mini_two_stream_layers() -> tuple:
    """VLM stream without AdaRMS, action-expert stream with it — the pi05 layout."""
    layer_cls = _get_pi_gemma_decoder_layer_base()
    vlm_config = _mini_gemma_config(use_adarms=False)
    expert_config = _mini_gemma_config(use_adarms=True)
    layers = (layer_cls(vlm_config, 0), layer_cls(expert_config, 0))
    return layers, GemmaRotaryEmbedding(vlm_config)


class _MiniJointExpert(torch.nn.Module):
    """A real two-stream attention layer for training graph-capture regressions."""

    def __init__(self, *, knowledge_insulation: bool) -> None:
        super().__init__()
        layers, self.rotary_emb = _mini_two_stream_layers()
        self.layers = torch.nn.ModuleList(layers)
        self.knowledge_insulation = knowledge_insulation
        self.paligemma = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[layers[0]]))
        )
        # Make the action expert depend on its per-action time conditions.
        torch.nn.init.normal_(layers[1].input_layernorm.dense.weight, std=0.02)
        torch.nn.init.normal_(layers[1].post_attention_layernorm.dense.weight, std=0.02)

    def forward(
        self,
        *,
        inputs_embeds: list[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: list[torch.Tensor | None],
        suppress_prefix_grads: bool = False,
        **_kwargs: Any,
    ) -> tuple[list[torch.Tensor], None]:
        result = compute_layer_complete(
            inputs_embeds,
            attention_mask,
            position_ids,
            adarms_cond,
            layers=self.layers,
            rotary_emb=self.rotary_emb,
            knowledge_insulation=self.knowledge_insulation,
            suppress_prefix_grads=suppress_prefix_grads,
        )
        return result, None


@pytest.mark.parametrize(
    "config_cls,model_cls", [(PI05Config, PI05Pytorch), (FineARTVLAConfig, FineARTVLAPytorch)]
)
def test_training_rtc_core_fullgraph_matches_eager_loss_and_gradients(
    config_cls: type[PI05Config], model_cls: type[PI05Pytorch]
) -> None:
    """Capture real per-token time, AdaRMS attention and postfix loss on CPU without pretrained weights."""
    torch.manual_seed(42)
    config = config_cls(
        device="cpu",
        use_relative_ee=True,
        rtc_training_max_delay=2,
        chunk_size=4,
        n_action_steps=4,
        max_action_dim=12,
    )
    policy, _recorded = _tiny_training_policy(config, model_cls)
    model = policy.model
    # Keep the actual core forward and suffix embedding, replacing only model scale.
    del model.embed_suffix
    model.action_in_proj = torch.nn.Linear(config.max_action_dim, _WIDTH)
    model.time_mlp_in = torch.nn.Linear(_WIDTH, _WIDTH)
    model.time_mlp_out = torch.nn.Linear(_WIDTH, _WIDTH)
    model.action_out_proj = torch.nn.Linear(_WIDTH, config.max_action_dim)
    model.paligemma_with_expert = _MiniJointExpert(
        knowledge_insulation=bool(getattr(config, "knowledge_insulation", False))
    )
    model.embed_prefix = lambda _images, _img_masks, tokens, *_args: (
        torch.zeros(tokens.shape[0], 1, _WIDTH),
        torch.ones(tokens.shape[0], 1, dtype=torch.bool),
        torch.zeros(tokens.shape[0], 1, dtype=torch.bool),
    )
    compiled_model = deepcopy(model)
    tokens = torch.zeros(2, 1, dtype=torch.long)
    masks = torch.ones(2, 1, dtype=torch.bool)

    def loss_fn(core, actions, noise, time, prefix_mask):
        losses = core([], [], tokens, masks, actions, noise, time, prefix_mask=prefix_mask)
        return reduce_training_rtc_loss(losses[..., :10], prefix_mask, reduction="mean")

    compiled_loss = torch.compile(
        lambda *args: loss_fn(compiled_model, *args), backend="eager", fullgraph=True
    )
    actions = torch.randn(2, 4, config.max_action_dim)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.25, 0.75])
    for delays in ([0, 2], [2, 1]):
        prefix_mask = torch.arange(4)[None] < torch.tensor(delays)[:, None]
        expected = loss_fn(model, actions, noise, time, prefix_mask)
        actual = compiled_loss(actions, noise, time, prefix_mask)
        torch.testing.assert_close(actual, expected)
        expected_grads = torch.autograd.grad(expected, tuple(model.parameters()), allow_unused=True)
        actual_grads = torch.autograd.grad(actual, tuple(compiled_model.parameters()), allow_unused=True)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            if expected_grad is None:
                assert actual_grad is None
            else:
                torch.testing.assert_close(actual_grad, expected_grad)
        # This guards against a trivial graph whose output never uses per-action conditioning.
        time_grad_index = list(dict(model.named_parameters())).index("time_mlp_in.weight")
        assert expected_grads[time_grad_index].abs().sum() > 0


def test_pi05_expert_layer_accepts_per_action_time_conditioning():
    """The pi05 training path feeds a (B, chunk, width) AdaRMS cond to the action expert.

    pi05 injects the flow timestep only through AdaRMS, so training-time RTC's per-action
    timesteps have to survive this two-stream layer unchanged in shape.
    """
    torch.manual_seed(0)
    batch, prefix_tokens, chunk = 2, 3, 4
    layers, rotary_emb = _mini_two_stream_layers()

    inputs_embeds = [torch.randn(batch, prefix_tokens, _WIDTH), torch.randn(batch, chunk, _WIDTH)]
    total = prefix_tokens + chunk
    position_ids = torch.arange(total)[None].expand(batch, total)
    attention_mask = torch.zeros(batch, 1, total, total)
    # Expert stream only: prefix (VLM) tokens keep the plain norm, as in PI05FlowMatching.
    adarms_cond = [None, torch.randn(batch, chunk, _WIDTH)]

    out_prefix, out_suffix = compute_layer_complete(
        inputs_embeds,
        attention_mask,
        position_ids,
        adarms_cond,
        layers=layers,
        rotary_emb=rotary_emb,
    )

    assert out_prefix.shape == (batch, prefix_tokens, _WIDTH)
    assert out_suffix.shape == (batch, chunk, _WIDTH)


def test_pi05_expert_layer_per_action_time_matches_scalar_when_uniform():
    """A per-action cond repeated across the chunk must reproduce the scalar-cond output."""
    torch.manual_seed(0)
    batch, prefix_tokens, chunk = 2, 3, 4
    layers, rotary_emb = _mini_two_stream_layers()
    # dense is zero-initialised by design; give it signal so a broken modulation shows up.
    torch.nn.init.normal_(layers[1].input_layernorm.dense.weight, std=0.02)
    torch.nn.init.normal_(layers[1].post_attention_layernorm.dense.weight, std=0.02)

    inputs_embeds = [torch.randn(batch, prefix_tokens, _WIDTH), torch.randn(batch, chunk, _WIDTH)]
    total = prefix_tokens + chunk
    position_ids = torch.arange(total)[None].expand(batch, total)
    attention_mask = torch.zeros(batch, 1, total, total)
    cond = torch.randn(batch, _WIDTH)

    def run(expert_cond):
        return compute_layer_complete(
            [t.clone() for t in inputs_embeds],
            attention_mask,
            position_ids,
            [None, expert_cond],
            layers=layers,
            rotary_emb=rotary_emb,
        )[1]

    torch.testing.assert_close(run(cond[:, None, :].expand(batch, chunk, _WIDTH)), run(cond))
