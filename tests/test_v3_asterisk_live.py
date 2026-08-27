from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import UUID

import pytest

from voiceprobe.media.live_asr import TYPE_HANGUP, TYPE_PCM_8KHZ, TYPE_UUID
from voiceprobe.runner import AssessmentCallRequest
from voiceprobe.telephony.ami import OriginateResult
from voiceprobe.v3.asterisk_live import (
    _recording_context,
    execute_v3_asterisk_media,
    project_v3_flow_snapshot,
    scenario_termination_failure_reason,
)
from voiceprobe.v3.flow_state import FlowSnapshot, FlowStage


@dataclass(frozen=True)
class _ProjectionCase:
    complete: bool
    slot: str | None
    confirmation: str | None


def _snapshot(case: _ProjectionCase) -> FlowSnapshot:
    confirmed = (
        frozenset({FlowStage.CONFIRMATION})
        if case.complete
        else frozenset()
    )
    return FlowSnapshot(
        communicated=frozenset(),
        confirmed=confirmed,
        current_stage=(
            FlowStage.COMPLETE if case.complete else FlowStage.CONFIRMATION
        ),
        complete=case.complete,
        accepted_slot_text=case.slot,
        booking_confirmation_text=case.confirmation,
    )


def test_projection_does_not_invent_legacy_day_or_time() -> None:
    result = project_v3_flow_snapshot(
        _snapshot(
            _ProjectionCase(
                complete=False,
                slot="2:30 PM",
                confirmation=None,
            )
        )
    )

    assert result.objective_complete is False
    assert result.booking_confirmed is False
    assert result.offer_accepted is True
    assert result.offered_day is None
    assert result.offered_time is None
    assert result.accepted_slot_text == "2:30 PM"


def test_accepted_socket_closes_when_recorder_setup_fails(monkeypatch) -> None:
    class FakeConnection:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = FakeConnection()

    class FailingRecorder:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise RuntimeError("synthetic recorder setup failure")

    monkeypatch.setattr(
        "voiceprobe.v3.asterisk_live.RunArtifactRecorder",
        FailingRecorder,
    )

    with (
        pytest.raises(RuntimeError, match="synthetic recorder setup failure"),
        _recording_context(connection, root=None, scenario=None),
    ):
        pass

    assert connection.closed is True


def test_uuid_starts_runtime_and_processes_pcm_before_originate_response(
    monkeypatch,
) -> None:
    call_id = UUID("11111111-2222-4333-8444-555555555555")
    release_originate = threading.Event()
    pcm_submitted = threading.Event()
    events: list[str] = []

    class Pending:
        def done(self) -> bool:
            return release_originate.is_set()

        def result(self) -> OriginateResult:
            assert release_originate.wait(timeout=2.0)
            return OriginateResult(
                action_id="action-1",
                audiosocket_call_id=call_id,
                asterisk_unique_id="asterisk-123.456",
                channel="Local/+12025550100@voiceprobe-test",
                response="Success",
                reason="4",
            )

        def cancel(self) -> None:
            release_originate.set()

    pending = Pending()

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def settimeout(self, timeout: float) -> None:
            del timeout

        def close(self) -> None:
            events.append("connection_closed")

    connection = Connection()

    class Server:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def setsockopt(self, *args) -> None:
            del args

        def bind(self, address) -> None:
            del address

        def listen(self, backlog: int) -> None:
            del backlog

        def settimeout(self, timeout: float) -> None:
            del timeout

        def accept(self):
            return connection, ("127.0.0.1", 12345)

    class Recorder:
        run_id = "run-test"
        elapsed_seconds = 0.1

        def record_event(self, event: str, **details) -> None:
            del details
            events.append(event)

        def finalize(self, **kwargs) -> None:
            del kwargs

    recorder = Recorder()

    @contextmanager
    def recording_context(*args, **kwargs):
        del args, kwargs
        yield recorder

    class Monitor:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def observe_inbound(self, pcm: bytes) -> None:
            del pcm

        def observe_outbound(self, pcm: bytes) -> None:
            del pcm

    class Boundary:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def start_idle_silence(self, *, stop) -> None:
            del stop
            events.append("idle_started")

        def forward_inbound_pcm(self, pcm: bytes, *, submit_pcm) -> None:
            submit_pcm(pcm)

        def join_idle_silence(self, *, timeout: float) -> None:
            del timeout

    class Runtime:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.objective_complete = threading.Event()
            self.scenario_terminal = threading.Event()
            self.error_event = threading.Event()
            self.error = None

        def start(self) -> None:
            events.append("runtime_started")

        def submit_pcm(self, pcm: bytes) -> None:
            assert pcm == b"pcm"
            events.append("pcm_submitted")
            pcm_submitted.set()

        def flush_and_snapshot(self):
            return self.snapshot()

        def snapshot(self):
            return _snapshot(_ProjectionCase(False, None, None))

        def scenario_metadata(self):
            return {}

        def record_persona_final_evidence(self) -> None:
            pass

        def stop(self) -> None:
            events.append("runtime_stopped")

    messages = iter(
        [(TYPE_UUID, call_id.bytes), (TYPE_PCM_8KHZ, b"pcm"), (TYPE_HANGUP, b"")]
    )
    monkeypatch.setattr("voiceprobe.v3.asterisk_live.socket.socket", lambda *a, **k: Server())
    monkeypatch.setattr("voiceprobe.v3.asterisk_live.LiveAudioMonitor.from_environment", lambda: Monitor())
    monkeypatch.setattr("voiceprobe.v3.asterisk_live._recording_context", recording_context)
    monkeypatch.setattr("voiceprobe.v3.asterisk_live._recv_message_polling", lambda *a, **k: next(messages))
    monkeypatch.setattr("voiceprobe.v3.asterisk_live.KokoroTelephonyRenderer", lambda **kwargs: object())
    monkeypatch.setattr("voiceprobe.v3.asterisk_live.AudioSocketKokoroSpeechTask", lambda **kwargs: object())
    monkeypatch.setattr("voiceprobe.v3.asterisk_live.AudioSocketV3MediaBoundary", Boundary)
    monkeypatch.setattr("voiceprobe.v3.asterisk_live._AsyncV3Runtime", Runtime)

    outcome: list[object] = []

    def execute() -> None:
        outcome.append(
            execute_v3_asterisk_media(
                request=AssessmentCallRequest(
                    execution_id="execution-test",
                    position=1,
                    scenario_id="autonomous-phone-diagnostic",
                    originating_number="+12025550101",
                    destination="+12025550100",
                    max_duration_seconds=30,
                ),
                call_id=call_id,
                start_originate=lambda: pending,
                pipeline=object(),
                voice="af_heart",
                tts_pcm_cache=None,
                deepgram_api_key="synthetic-key",
                artifact_root="unused",
                host="127.0.0.1",
                port=0,
                accept_timeout_seconds=1.0,
                hangup_observer=None,
                ami_error_type=RuntimeError,
                classify_termination=lambda **kwargs: SimpleNamespace(value="remote_hangup"),
                termination_failure_reason=lambda **kwargs: None,
            )
        )

    worker = threading.Thread(target=execute)
    worker.start()
    assert pcm_submitted.wait(timeout=1.0)
    assert release_originate.is_set() is False
    assert events.index("idle_started") < events.index("runtime_started")
    assert events.index("runtime_started") < events.index("pcm_submitted")

    release_originate.set()
    worker.join(timeout=2.0)
    assert worker.is_alive() is False
    assert len(outcome) == 1
    assert outcome[0].originate.asterisk_unique_id == "asterisk-123.456"


def test_projection_treats_v3_complete_as_authoritative_booking_confirmation() -> None:
    result = project_v3_flow_snapshot(
        _snapshot(
            _ProjectionCase(
                complete=True,
                slot="2:30 PM",
                confirmation="You're booked Friday at 2:30 PM.",
            )
        )
    )

    assert result.objective_complete is True
    assert result.booking_confirmed is True
    assert result.offer_accepted is True
    assert result.booking_confirmation_text == "You're booked Friday at 2:30 PM."


def test_medication_projection_and_failure_are_scenario_specific() -> None:
    snapshot = _snapshot(_ProjectionCase(False, None, None))
    projection = project_v3_flow_snapshot(
        snapshot,
        scenario_id="medication-refill-correction",
        scenario_metadata={
            "scenario_stage": "correct_dose",
            "dose_correction_spoken": True,
            "dose_acknowledged": False,
            "objective_complete": False,
        },
    )

    reason = scenario_termination_failure_reason(
        status="max_duration_termination", projection=projection
    )
    assert projection.objective_complete is False
    assert projection.booking_confirmed is False
    assert "medication-refill-correction objective" in reason
    assert "dose_correction_spoken=True" in reason
    assert "scheduling objective" not in reason
    assert "booking_confirmed" not in reason
    assert "offered_day" not in reason


def test_self_pay_projection_uses_its_own_objective_state() -> None:
    snapshot = _snapshot(_ProjectionCase(False, None, None))
    projection = project_v3_flow_snapshot(
        snapshot,
        scenario_id="self-pay-location-switch",
        scenario_metadata={"objective_complete": True},
    )
    assert projection.objective_complete is True
    assert projection.booking_confirmed is False


def test_adapter_v3_environment_switch_is_explicit(monkeypatch) -> None:
    from voiceprobe.telephony.asterisk_adapter import (
        v3_live_enabled_from_environment,
    )

    monkeypatch.delenv("VOICEPROBE_V3_LIVE", raising=False)
    assert v3_live_enabled_from_environment() is False

    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    assert v3_live_enabled_from_environment() is True

    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "off")
    assert v3_live_enabled_from_environment() is False

    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "definitely")
    with pytest.raises(ValueError):
        v3_live_enabled_from_environment()


def test_recording_bridge_persists_remote_and_patient_turns() -> None:
    import asyncio

    from voiceprobe.v3.asterisk_live import _RecordingPipecatRuntimeBridge

    class Recorder:
        def __init__(self) -> None:
            self.transcript = []
            self.events = []
            self.metrics = []

        def record_transcript_turn(self, *, speaker, text, **metadata) -> None:
            self.transcript.append((speaker, text, metadata))

        def record_event(self, event, **details) -> None:
            self.events.append((event, details))

        def record_turn_metrics(self, metrics) -> None:
            self.metrics.append(dict(metrics))

    class Sink:
        def __init__(self) -> None:
            self.frames = []

        async def queue_frames(self, frames) -> None:
            self.frames.extend(frames)

    recorder = Recorder()
    sink = Sink()
    bridge = _RecordingPipecatRuntimeBridge(
        recorder=recorder,
        tts_frame_factory=lambda text: ("tts", text),
    )
    bridge.bind_frame_sink(sink)

    result = asyncio.run(
        bridge.runtime.process_turns(
            ["What insurance do you have?"],
            ingress_reason="test_flux_eot",
        )
    )

    assert result.response_ready
    assert recorder.transcript[0][0] == "agent"
    assert recorder.transcript[0][1] == "What insurance do you have?"
    assert recorder.transcript[0][2]["source"] == "deepgram_flux_eot"

    assert recorder.transcript[1][0] == "patient"
    assert "Blue Cross" in recorder.transcript[1][1]
    assert recorder.transcript[1][2]["delivery_status"] == "queued_for_tts"

    assert sink.frames == [("tts", result.decision.text)]
    assert recorder.events[-1][0] == "v3_runtime_decision"
    assert recorder.events[-1][1]["decision_kind"] == result.decision.kind.value
    assert recorder.metrics[-1]["decision_kind"] == result.decision.kind.value
    assert recorder.metrics[-1]["response_ready"] is True


def test_recording_bridge_persists_wait_turn_without_patient_speech() -> None:
    import asyncio

    from voiceprobe.v3.asterisk_live import _RecordingPipecatRuntimeBridge

    class Recorder:
        def __init__(self) -> None:
            self.transcript = []
            self.events = []
            self.metrics = []

        def record_transcript_turn(self, *, speaker, text, **metadata) -> None:
            self.transcript.append((speaker, text, metadata))

        def record_event(self, event, **details) -> None:
            self.events.append((event, details))

        def record_turn_metrics(self, metrics) -> None:
            self.metrics.append(dict(metrics))

    recorder = Recorder()
    bridge = _RecordingPipecatRuntimeBridge(
        recorder=recorder,
        tts_frame_factory=lambda text: ("tts", text),
    )

    result = asyncio.run(
        bridge.runtime.process_turns(
            ["Thanks."],
            ingress_reason="test_flux_eot",
        )
    )

    assert not result.response_ready
    assert [(speaker, text) for speaker, text, _ in recorder.transcript] == [
        ("agent", "Thanks.")
    ]
    assert recorder.events[-1][0] == "v3_runtime_decision"
    assert recorder.metrics[-1]["decision_kind"] == "wait"
    assert recorder.metrics[-1]["response_ready"] is False
