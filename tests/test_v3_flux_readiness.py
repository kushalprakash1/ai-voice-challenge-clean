import asyncio
import inspect
import threading
import time

import pytest

from voiceprobe.v3.asterisk_live import (
    DEFAULT_FLUX_CONNECT_TIMEOUT_SECONDS,
    _AsyncV3Runtime,
    _FluxReadinessGate,
    _observe_flux_connected_state,
)


class FakeRuntime(_AsyncV3Runtime):
    def __init__(self, run) -> None:
        super().__init__(api_key="test", speech_task=None, recorder=None)  # type: ignore[arg-type]
        self._run = run

    def _thread_main(self) -> None:
        try:
            self._run(self)
        finally:
            self.stopped.set()


class FakeFlux:
    def __init__(self, *, connected: bool = False) -> None:
        self._connection_established_event = asyncio.Event()
        if connected:
            self._connection_established_event.set()

    def connect(self) -> None:
        self._connection_established_event.set()


def test_flux_connect_after_waiter_registration() -> None:
    async def scenario() -> None:
        flux = FakeFlux()
        gate = _FluxReadinessGate()
        gate.mark_pipeline_started()
        waiter = asyncio.create_task(_observe_flux_connected_state(flux, gate))
        await asyncio.sleep(0)
        flux.connect()
        assert await waiter is True
        assert gate.ready.wait(0)

    asyncio.run(scenario())


def test_flux_connect_immediately_after_observer_installation() -> None:
    async def scenario() -> None:
        flux = FakeFlux()
        gate = _FluxReadinessGate()
        waiter = asyncio.create_task(_observe_flux_connected_state(flux, gate))
        flux.connect()
        gate.mark_pipeline_started()
        assert await waiter is True
        assert gate.ready.wait(0)

    asyncio.run(scenario())


def test_flux_already_connected_before_wait_begins() -> None:
    async def scenario() -> None:
        flux = FakeFlux(connected=True)
        gate = _FluxReadinessGate()
        gate.mark_pipeline_started()
        assert await _observe_flux_connected_state(flux, gate) is True
        assert gate.ready.wait(0)

    asyncio.run(scenario())


def test_flux_no_connection_still_times_out() -> None:
    async def scenario() -> None:
        flux = FakeFlux()
        gate = _FluxReadinessGate()
        gate.mark_pipeline_started()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                _observe_flux_connected_state(flux, gate), timeout=0.01
            )
        assert not gate.ready.is_set()

    asyncio.run(scenario())


def test_early_metrics_cannot_satisfy_or_poison_readiness() -> None:
    gate = _FluxReadinessGate()
    early_metrics = object()
    del early_metrics
    assert not gate.ready.is_set()
    gate.mark_pipeline_started()
    assert not gate.ready.is_set()
    assert gate.mark_flux_connected() is True
    assert gate.ready.wait(0)


def test_duplicate_connected_notification_completes_once() -> None:
    gate = _FluxReadinessGate()
    gate.mark_pipeline_started()
    assert gate.mark_flux_connected() is True
    assert gate.mark_flux_connected() is False
    assert gate.ready_observations == 1


def test_ordinary_runtime_default_remains_exactly_ten_seconds() -> None:
    default = inspect.signature(_AsyncV3Runtime.start).parameters["timeout"].default
    assert default == DEFAULT_FLUX_CONNECT_TIMEOUT_SECONDS == 10.0


def test_flux_connection_before_scalable_timeout_succeeds() -> None:
    def connect(runtime: FakeRuntime) -> None:
        time.sleep(0.02)
        runtime._readiness.mark_pipeline_started()
        runtime._readiness.mark_flux_connected()
        runtime._stop_requested.wait()

    runtime = FakeRuntime(connect)
    runtime.start(timeout=0.2)
    assert runtime.connected.is_set()
    runtime.stop()
    assert runtime.stopped.is_set()


def test_true_readiness_timeout_cancels_and_joins_background_runner() -> None:
    cleanup_observed = threading.Event()

    def await_cancellation(runtime: FakeRuntime) -> None:
        runtime._stop_requested.wait()
        cleanup_observed.set()

    runtime = FakeRuntime(await_cancellation)
    with pytest.raises(Exception, match="timed out waiting for Deepgram Flux"):
        runtime.start(timeout=0.01)

    assert cleanup_observed.is_set()
    assert runtime.stopped.is_set()
    assert runtime._thread is not None
    assert not runtime._thread.is_alive()


def test_connection_after_timeout_cannot_reverse_terminal_failure() -> None:
    gate = _FluxReadinessGate()
    gate.mark_pipeline_started()
    gate.mark_failed()
    assert gate.mark_flux_connected() is False
    assert not gate.ready.is_set()
    assert gate.ready_observations == 0
