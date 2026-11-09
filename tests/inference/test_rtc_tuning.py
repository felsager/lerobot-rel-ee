"""Retuning a paused RTC engine applies on the worker and invalidates compiled readiness."""

import pytest

from tests.inference.rtc_helpers import make_engine, stop
from tests.utils import wait_for


def test_rtc_tuning_applies_on_worker_and_clears_compiled_readiness():
    engine = make_engine(use_torch_compile=True)
    engine._policy.config.rtc_config = engine._rtc_config  # Real policies share one config.
    engine._compile_warmup_done.set()  # As after a finished warmup.
    engine.start()
    try:
        engine.update_rtc_tuning(queue_threshold=3, execution_horizon=6, max_guidance_weight=2.0)
        assert engine._rtc_queue_threshold == 3 and not engine.ready
        assert engine.rtc_tuning["execution_horizon"] == 6, "visible before the worker applies it"
        assert wait_for(lambda: engine._rtc_config.execution_horizon == 6)
        assert engine._rtc_config.max_guidance_weight == 2.0
        engine._compile_warmup_done.set()
        engine.update_rtc_tuning(queue_threshold=4, execution_horizon=6)
        assert engine.ready, "an unchanged RTC field needs no new graphs"
        with pytest.raises(ValueError, match="without a restart"):
            engine.update_rtc_tuning(queue_threshold=3, mode="trained")
        with pytest.raises(ValueError, match="execution_horizon must be"):
            engine.update_rtc_tuning(queue_threshold=3, execution_horizon=99)
        with pytest.raises(ValueError, match="finite and positive"):
            engine.update_rtc_tuning(queue_threshold=3, max_guidance_weight=float("nan"))
        assert engine._rtc_queue_threshold == 4, "rejected updates change nothing"
    finally:
        stop(engine)
