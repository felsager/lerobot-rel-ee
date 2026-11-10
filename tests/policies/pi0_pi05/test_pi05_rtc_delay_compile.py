"""CPU regressions for changing live RTC delays across PI05's compile boundary."""

from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature, RTCAttentionSchedule  # noqa: E402
from lerobot.policies.common.flow_matching import euler_integrate  # noqa: E402
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402
from lerobot.policies.rtc.modeling_rtc import RTCProcessor  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


def make_policy(*, compile_model, schedule=RTCAttentionSchedule.EXP, debug=False, real_sampler=False):
    """Exercise real predict_action_chunk without allocating VLM/expert weights."""
    config = PI05Config(
        device="cpu",
        compile_model=compile_model,
        chunk_size=50,
        n_action_steps=50,
        rtc_config=RTCConfig(execution_horizon=10, prefix_attention_schedule=schedule, debug=debug),
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    )
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.rtc_processor = RTCProcessor(config.rtc_config)
    policy._preprocess_images = lambda _batch: ([], [])
    policy._prepare_memory_states = lambda _batch: (None, None)

    @torch.no_grad()
    def sample_actions(
        images,
        img_masks,
        tokens,
        masks,
        states=None,
        state_masks=None,
        noise=None,
        rtc_prefix_weights=None,
        **kwargs,
    ):
        def velocity(x, timestep):
            return 0.2 * x.sin() + 0.1 * x.roll(1, dims=1) + timestep[:, None, None] * 0.01

        return euler_integrate(
            velocity,
            noise,
            3,
            rtc_enabled=True,
            rtc_processor=policy.rtc_processor,
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
            rtc_prefix_weights=rtc_prefix_weights,
        )

    policy.model = SimpleNamespace(sample_actions=sample_actions)
    if real_sampler:
        config.num_inference_steps = 3
        language_config = SimpleNamespace(_attn_implementation="eager")
        paligemma = SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(config=language_config))
        )

        def embed_prefix(images, img_masks, tokens, masks, states, state_masks):
            return (
                torch.zeros(1, 2, 8),
                torch.ones(1, 2, dtype=torch.bool),
                torch.zeros(1, 2, dtype=torch.bool),
            )

        def denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
            return 0.2 * x_t.sin() + 0.1 * x_t.roll(1, dims=1) + timestep[:, None, None] * 0.01

        model = SimpleNamespace(
            config=config,
            rtc_processor=policy.rtc_processor,
            _rtc_enabled=policy._rtc_enabled,
            embed_prefix=embed_prefix,
            use_typed_attention_masks=False,
            _prepare_attention_masks_4d=lambda value, **kwargs: value,
            paligemma_with_expert=SimpleNamespace(
                paligemma=paligemma, forward=lambda **kwargs: ((None, None), None)
            ),
            denoise_step=denoise_step,
            precompute_denoise_times=False,
        )
        model.sample_actions = MethodType(PI05Pytorch.sample_actions, model)
        policy.model = model
    return policy


def batch():
    return {
        OBS_LANGUAGE_TOKENS: torch.zeros((1, 4), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((1, 4), dtype=torch.bool),
    }


@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("prefix_length", [6, 10])
def test_precomputed_weights_preserve_full_guided_output(schedule, prefix_length):
    eager = make_policy(compile_model=False, schedule=schedule)
    prepared = make_policy(compile_model=True, schedule=schedule)
    generator = torch.Generator().manual_seed(42)
    noise = torch.randn((1, 50, 8), generator=generator)
    previous = torch.randn((1, prefix_length, 8), generator=generator)
    for delay in (0, 1, 4, 10, 123):
        kwargs = {"noise": noise, "prev_chunk_left_over": previous, "inference_delay": delay}
        actual = prepared.predict_action_chunk(batch(), **kwargs)
        expected = eager.predict_action_chunk(batch(), **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert kwargs["inference_delay"] == delay


@pytest.mark.parametrize("path", ["guided", "debug", "trained", "disabled"])
def test_sampler_keeps_delay_metadata_only_when_needed(path):
    policy = make_policy(compile_model=True, debug=path == "debug")
    if path == "trained":
        policy.rtc_processor.rtc_config.mode = "trained"
        policy.config.rtc_training_max_delay = 10
    if path == "disabled":
        policy.rtc_processor.rtc_config.enabled = False
    calls = []

    def sample(*_args, **kwargs):
        calls.append(kwargs)
        return kwargs["noise"]

    policy.model.sample_actions = sample
    policy.predict_action_chunk(
        batch(),
        noise=torch.zeros((1, 50, 8)),
        prev_chunk_left_over=torch.zeros((1, 10, 8)),
        inference_delay=4,
    )
    call = calls[0]
    assert (call["trained_prefix"] is not None) == (path == "trained")
    if path != "guided":
        assert call["inference_delay"] == 4
        assert call["rtc_prefix_weights"] is None
    else:
        assert "inference_delay" not in call
        torch.testing.assert_close(
            call["rtc_prefix_weights"],
            policy.rtc_processor.get_prefix_weights(4, 10, 50),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("real_sampler", [False, True])
def test_changing_live_delay_reuses_compiled_graphs(real_sampler):
    policy = make_policy(compile_model=True, real_sampler=real_sampler)
    noise = torch.randn((1, 50, 8), generator=torch.Generator().manual_seed(7))
    previous = noise[:, :10].clone()
    eager = make_policy(compile_model=False, real_sampler=real_sampler)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    try:
        policy.model.sample_actions = torch.compile(policy.model.sample_actions, backend=backend)
        # Match startup: no-prefix call, then guidance at delays 0 and 1.
        policy.predict_action_chunk(batch(), noise=noise, inference_delay=0)
        for delay in (0, 1):
            policy.predict_action_chunk(
                batch(), noise=noise, prev_chunk_left_over=previous, inference_delay=delay
            )
        warmed_graphs = len(graphs)
        assert warmed_graphs > 0
        for delay in (4, 10, 123, 2, 0, 9):
            kwargs = {"noise": noise, "prev_chunk_left_over": previous, "inference_delay": delay}
            actual = policy.predict_action_chunk(batch(), **kwargs)
            expected = eager.predict_action_chunk(batch(), **kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert len(graphs) == warmed_graphs, f"New graph at live delay {delay}"
    finally:
        torch._dynamo.reset()


def test_precomputed_weights_require_chunk_length():
    processor = RTCProcessor(RTCConfig())
    with pytest.raises(ValueError, match="chunk length"):
        processor.denoise_step(
            torch.zeros(1, 50, 8),
            torch.zeros(1, 10, 8),
            4,
            torch.tensor(0.5),
            lambda x: x.sin(),
            prefix_weights=torch.ones(10),
        )


def test_compiled_sampler_follows_live_rtc_tuning():
    """A stale constant in a compiled graph would silently ignore a reload, so compare with eager."""
    noise = torch.randn((1, 50, 8), generator=torch.Generator().manual_seed(23))
    previous = noise[:, :20].clone() * 0.5
    settings = [
        {
            "max_guidance_weight": 10.0,
            "execution_horizon": 10,
            "prefix_attention_schedule": RTCAttentionSchedule.EXP,
        },
        {
            "max_guidance_weight": 2.0,
            "execution_horizon": 10,
            "prefix_attention_schedule": RTCAttentionSchedule.EXP,
        },
        {
            "max_guidance_weight": 2.0,
            "execution_horizon": 18,
            "prefix_attention_schedule": RTCAttentionSchedule.EXP,
        },
        {
            "max_guidance_weight": 2.0,
            "execution_horizon": 18,
            "prefix_attention_schedule": RTCAttentionSchedule.LINEAR,
        },
    ]

    def run(policy, values):
        for name, value in values.items():
            setattr(policy.rtc_processor.rtc_config, name, value)
        prefix = previous[:, : values["execution_horizon"]]
        return policy.predict_action_chunk(
            batch(), noise=noise, prev_chunk_left_over=prefix, inference_delay=4
        )

    eager = make_policy(compile_model=False, real_sampler=True)
    expected = [run(eager, values) for values in settings]
    assert all(not torch.equal(a, b) for a, b in zip(expected, expected[1:], strict=False))
    policy = make_policy(compile_model=True, real_sampler=True)
    torch._dynamo.reset()
    try:
        policy.model.sample_actions = torch.compile(
            policy.model.sample_actions, backend=lambda graph, _inputs: graph.forward
        )
        for values, want in zip(settings + settings, expected + expected, strict=True):
            torch.testing.assert_close(run(policy, values), want, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


def test_reloading_execution_horizon_reuses_compiled_graphs():
    """Horizon reloads must not recompile, although the engine trims the prefix to the horizon."""
    noise = torch.randn((1, 50, 8), generator=torch.Generator().manual_seed(11))
    previous = noise[:, :30].clone() * 0.5
    eager = make_policy(compile_model=False, real_sampler=True)
    policy = make_policy(compile_model=True, real_sampler=True)
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def run(model, horizon, weight, delay=4):
        model.rtc_processor.rtc_config.execution_horizon = horizon
        model.rtc_processor.rtc_config.max_guidance_weight = weight
        prefix = previous[:, :horizon]  # As prepare_continuation fits it to the horizon.
        return model.predict_action_chunk(
            batch(), noise=noise, prev_chunk_left_over=prefix, inference_delay=delay
        )

    torch._dynamo.reset()
    try:
        policy.model.sample_actions = torch.compile(policy.model.sample_actions, backend=backend)
        policy.predict_action_chunk(batch(), noise=noise, inference_delay=0)
        for delay in (0, 1):
            run(policy, 10, 10.0, delay)
        warmed_graphs = len(graphs)
        for horizon, weight in ((18, 10.0), (6, 10.0), (25, 10.0), (10, 10.0)):
            torch.testing.assert_close(
                run(policy, horizon, weight), run(eager, horizon, weight), rtol=0, atol=0
            )
            assert len(graphs) == warmed_graphs, f"New graph at horizon {horizon}, weight {weight}"
    finally:
        torch._dynamo.reset()
