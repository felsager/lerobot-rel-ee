"""Paused compilation must not generate executable actions or consume live freshness."""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tests.inference.rtc_helpers import make_engine, observe, stop
from tests.utils import wait_for


def test_preparation_finishes_while_paused_without_publication_or_extra_reads():
    engine = make_engine(use_torch_compile=True, max_observation_age_s=0.01)
    engine.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(engine.prepare_compilation, {"j1.pos": 2.0, "j2.pos": 2.0}, timeout_s=3)
            assert wait_for(lambda: len(engine._policy.calls) == 1)
            assert not engine._policy_active.is_set()
            assert not engine.dispatch_allowed()
            assert engine.action_queue.empty()
            time.sleep(0.03)  # Compilation outlives the normal observation freshness bound.
            engine._policy.release.release(3)
            future.result(timeout=3)
        assert engine.ready and not engine.failed
        assert [delay for _, delay in engine._policy.calls] == [0, 0, 1]
        assert engine._policy.calls[0][0] is None
        assert all(prefix.shape == (4, 2) for prefix, _ in engine._policy.calls[1:])
        assert engine.action_queue.empty() and engine._runtime.pending is None
        assert not engine._runtime.active and not engine._policy_active.is_set()
        time.sleep(0.03)
        assert len(engine._policy.calls) == 3, "no extra live call after preparation"
        engine._runtime.max_age = 0.5  # Allow the worker scheduling interval for the live test.
        engine.resume()
        observe(engine, 7.0)
        assert wait_for(lambda: len(engine._policy.action_observations) == 4)
        torch.testing.assert_close(engine._policy.action_observations[-1], torch.full((1, 2), 7.0))
        assert engine._policy.calls[-1][0] is None
    finally:
        stop(engine)


@pytest.mark.parametrize("kind", ["cancel", "timeout"])
def test_preparation_cancellation_never_marks_ready_or_publishes(kind):
    from threading import Event

    engine = make_engine(use_torch_compile=True)
    cancel = Event()
    engine.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                engine.prepare_compilation,
                {"j1.pos": 0.0, "j2.pos": 0.0},
                timeout_s=0.15 if kind == "timeout" else 3,
                cancelled=cancel.is_set,
            )
            assert wait_for(lambda: len(engine._policy.calls) == 1)
            if kind == "cancel":
                cancel.set()
            with pytest.raises(InterruptedError if kind == "cancel" else TimeoutError):
                future.result(timeout=2)
            assert not engine.ready
            engine._policy.release.release()
            assert wait_for(lambda: engine._compile_preparation is None)
        assert not engine.ready and engine.action_queue.empty()
        assert not engine._runtime.active and len(engine._policy.calls) == 1
    finally:
        stop(engine)


def test_preparation_errors_surface_without_false_readiness(monkeypatch):
    engine = make_engine(use_torch_compile=True)

    def fail(*args, **kwargs):
        raise ValueError("compiler error")

    monkeypatch.setattr(engine._policy, "predict_action_chunk", fail)
    engine.start()
    try:
        with pytest.raises(RuntimeError, match="Compilation preparation failed") as error:
            engine.prepare_compilation({"j1.pos": 0.0, "j2.pos": 0.0}, timeout_s=2)
        assert isinstance(error.value.__cause__, ValueError)
        assert not engine.ready and engine.action_queue.empty()
    finally:
        stop(engine)


def test_preparation_rejects_active_inference():
    engine = make_engine(use_torch_compile=True)
    engine.start()
    engine.resume()
    try:
        with pytest.raises(RuntimeError, match="Pause inference"):
            engine.prepare_compilation({"j1.pos": 0.0, "j2.pos": 0.0}, timeout_s=2)
    finally:
        stop(engine)
