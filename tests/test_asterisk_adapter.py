from __future__ import annotations

import threading
from collections.abc import Callable
from uuid import UUID

import pytest

from voiceprobe.runner import (
    AssessmentCallRequest,
    CallExecutionError,
)
from voiceprobe.safety import (
    ALLOWED_TEST_NUMBER,
    UnsafeDestinationError,
)
from voiceprobe.telephony.ami import (
    AsteriskAMIConfig,
    AsteriskHangupResult,
    OriginateResult,
)
from voiceprobe.telephony.asterisk_adapter import (
    AsteriskAssessmentCallAdapter,
    AsteriskMediaOutcome,
    _MonitoredOriginate,
)

CALLER = "+12025550101"
CALL_ID = UUID("11111111-2222-4333-8444-555555555555")


def make_request(
    *,
    destination: str = ALLOWED_TEST_NUMBER,
    originating_number: str = CALLER,
    max_duration_seconds: int = 180,
) -> AssessmentCallRequest:
    return AssessmentCallRequest(
        execution_id="execution-test",
        position=1,
        scenario_id="autonomous-phone-diagnostic",
        originating_number=originating_number,
        destination=destination,
        max_duration_seconds=max_duration_seconds,
    )


class FakeAMIClient:
    def __init__(
        self,
        events: list[str],
    ) -> None:
        self.events = events

    def connect(self) -> str:
        self.events.append("ami_connect")
        return "Asterisk Call Manager/5.0"

    def login(
        self,
        *,
        events: str = "off",
    ) -> None:
        self.events.append(f"ami_login:{events}")

    def originate_audiosocket(
        self,
        destination: str,
        *,
        call_id: UUID | None = None,
        timeout_ms: int = 30_000,
    ) -> OriginateResult:
        self.events.append("ami_originate")

        assert destination == ALLOWED_TEST_NUMBER
        assert call_id == CALL_ID
        assert timeout_ms == 30_000

        return OriginateResult(
            action_id="action-1",
            audiosocket_call_id=CALL_ID,
            asterisk_unique_id="asterisk-123.456",
            channel=(f"Local/{destination}@voiceprobe-test"),
            response="Success",
            reason="4",
        )

    def close(self) -> None:
        self.events.append("ami_close")

    def hangup(self, *, unique_id: str, channel: str) -> None:
        assert unique_id == "asterisk-123.456"
        assert channel.startswith("Local/")
        self.events.append("ami_hangup")


def test_pending_originate_allows_media_readiness_before_response() -> None:
    events: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class DelayedAMIClient(FakeAMIClient):
        def originate_audiosocket(self, *args, **kwargs) -> OriginateResult:
            events.append("ami_originate_entered")
            entered.set()
            assert release.wait(timeout=2.0)
            return super().originate_audiosocket(*args, **kwargs)

    originate = _MonitoredOriginate(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        ami_client_factory=lambda config: DelayedAMIClient(events),
        destination=ALLOWED_TEST_NUMBER,
        call_id=CALL_ID,
    )

    pending = originate.start()
    assert entered.wait(timeout=1.0)
    events.append("audiosocket_accepted")
    events.append("uuid_validated")
    events.append("idle_silence_started")
    assert pending.done() is False

    release.set()
    result = pending.result()
    originate.close()

    assert result.audiosocket_call_id == CALL_ID
    assert events.index("idle_silence_started") < events.index("ami_originate")
    assert events.count("ami_originate") == 1


def test_pending_originate_propagates_failure_after_media_appears() -> None:
    events: list[str] = []
    release = threading.Event()

    class FailingAMIClient(FakeAMIClient):
        def originate_audiosocket(self, *args, **kwargs) -> OriginateResult:
            del args, kwargs
            events.append("ami_originate")
            assert release.wait(timeout=2.0)
            raise RuntimeError("synthetic OriginateResponse failure")

    originate = _MonitoredOriginate(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        ami_client_factory=lambda config: FailingAMIClient(events),
        destination=ALLOWED_TEST_NUMBER,
        call_id=CALL_ID,
    )
    pending = originate.start()
    events.append("audiosocket_accepted")
    release.set()

    with pytest.raises(RuntimeError, match="OriginateResponse failure"):
        pending.result()

    originate.close()
    assert events.count("ami_originate") == 1


def test_pending_originate_cancel_closes_ami_and_joins_worker() -> None:
    events: list[str] = []
    entered = threading.Event()
    closed = threading.Event()

    class CancellableAMIClient(FakeAMIClient):
        def originate_audiosocket(self, *args, **kwargs) -> OriginateResult:
            del args, kwargs
            events.append("ami_originate")
            entered.set()
            assert closed.wait(timeout=2.0)
            raise RuntimeError("AMI transport closed")

        def close(self) -> None:
            super().close()
            closed.set()

    originate = _MonitoredOriginate(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        ami_client_factory=lambda config: CancellableAMIClient(events),
        destination=ALLOWED_TEST_NUMBER,
        call_id=CALL_ID,
    )
    pending = originate.start()
    assert entered.wait(timeout=1.0)
    pending.cancel()

    assert pending.done() is True
    assert events.count("ami_originate") == 1
    assert events.count("ami_close") == 1


def test_cancel_race_retains_successful_correlation_for_hangup() -> None:
    events: list[str] = []
    result_ready = threading.Event()
    publish_result = threading.Event()
    original_closed = threading.Event()
    clients: list[FakeAMIClient] = []

    class RacingAMIClient(FakeAMIClient):
        def originate_audiosocket(self, *args, **kwargs) -> OriginateResult:
            result = super().originate_audiosocket(*args, **kwargs)
            result_ready.set()
            assert publish_result.wait(timeout=2.0)
            return result

        def close(self) -> None:
            super().close()
            original_closed.set()

    def client_factory(config: AsteriskAMIConfig) -> FakeAMIClient:
        del config
        client = RacingAMIClient(events)
        clients.append(client)
        return client

    originate = _MonitoredOriginate(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        ami_client_factory=client_factory,
        destination=ALLOWED_TEST_NUMBER,
        call_id=CALL_ID,
    )
    pending = originate.start()
    assert result_ready.wait(timeout=1.0)

    cleanup = threading.Thread(target=pending.cancel)
    cleanup.start()
    assert original_closed.wait(timeout=1.0)
    publish_result.set()
    cleanup.join(timeout=2.0)
    assert cleanup.is_alive() is False

    result = pending.result()
    originate.hangup_best_effort()
    originate.close()

    assert result.asterisk_unique_id == "asterisk-123.456"
    assert result.channel.startswith("Local/")
    assert events.count("ami_originate") == 1
    assert events.count("ami_hangup") == 1
    assert len(clients) == 2
    assert events.count("ami_close") == 2


def test_hangup_read_starts_only_after_pending_originate_reader_finishes() -> None:
    events: list[str] = []
    originate_reading = threading.Event()
    release = threading.Event()

    class SerializedAMIClient(FakeAMIClient):
        def originate_audiosocket(self, *args, **kwargs) -> OriginateResult:
            originate_reading.set()
            assert release.wait(timeout=2.0)
            result = super().originate_audiosocket(*args, **kwargs)
            originate_reading.clear()
            return result

        def wait_for_hangup(
            self, *, unique_id: str, channel: str, max_events: int = 2000
        ) -> AsteriskHangupResult:
            del max_events
            assert originate_reading.is_set() is False
            events.append("ami_wait_for_hangup")
            return AsteriskHangupResult(
                unique_id=unique_id,
                channel=channel,
                linked_id=unique_id,
                cause="16",
                cause_text="Normal Clearing",
                tech_cause=None,
            )

    originate = _MonitoredOriginate(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        ami_client_factory=lambda config: SerializedAMIClient(events),
        destination=ALLOWED_TEST_NUMBER,
        call_id=CALL_ID,
    )
    pending = originate.start()
    assert originate_reading.wait(timeout=1.0)

    with pytest.raises(CallExecutionError, match="only once"):
        originate.start()

    release.set()
    result = pending.result()
    hangup = originate.wait_for_hangup()
    originate.close()

    assert result.action_id == "action-1"
    assert result.asterisk_unique_id == "asterisk-123.456"
    assert hangup.unique_id == result.asterisk_unique_id
    assert events.count("ami_originate") == 1


def build_adapter(
    events: list[str],
    media_executor: Callable[
        [
            AssessmentCallRequest,
            UUID,
            Callable[[], OriginateResult],
        ],
        AsteriskMediaOutcome,
    ],
) -> AsteriskAssessmentCallAdapter:
    fake_client = FakeAMIClient(events)

    return AsteriskAssessmentCallAdapter(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        expected_originating_number=CALLER,
        ami_client_factory=(lambda config: fake_client),
        call_id_factory=(lambda: CALL_ID),
        media_executor=media_executor,
    )


def successful_media_executor(
    events: list[str],
) -> Callable[
    [
        AssessmentCallRequest,
        UUID,
        Callable[[], OriginateResult],
    ],
    AsteriskMediaOutcome,
]:
    def execute(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[
            [],
            OriginateResult,
        ],
    ) -> AsteriskMediaOutcome:
        assert request.scenario_id == "autonomous-phone-diagnostic"
        assert call_id == CALL_ID

        # The production executor must have its listener ready before
        # invoking the one allowed originate callback.
        events.append("media_listening")

        originate_result = originate()

        events.append("media_complete")

        return AsteriskMediaOutcome(
            call_id=call_id,
            artifact_run_id="run-test-1",
            duration_seconds=84.25,
            originate=originate_result,
        )

    return execute


def test_successful_call_maps_asterisk_and_artifact_evidence() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    result = adapter.execute_call(make_request())

    assert result.provider_call_id == ("asterisk-123.456")
    assert result.artifact_run_id == ("run-test-1")
    assert result.duration_seconds == (84.25)
    assert result.provider_cost_usd is None

    assert events == [
        "media_listening",
        "ami_connect",
        "ami_login:call",
        "ami_originate",
        "media_complete",
        "ami_close",
    ]


def test_unsafe_destination_is_rejected_before_any_side_effect() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    with pytest.raises(UnsafeDestinationError):
        adapter.execute_call(make_request(destination="+12025550101"))

    assert events == []


def test_originating_number_mismatch_is_rejected_before_any_side_effect() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    with pytest.raises(
        CallExecutionError,
        match="originating number",
    ):
        adapter.execute_call(make_request(originating_number=("+12025550102")))

    assert events == []


def test_invalid_max_duration_is_rejected_before_any_side_effect() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    with pytest.raises(
        CallExecutionError,
        match="duration",
    ):
        adapter.execute_call(make_request(max_duration_seconds=601))

    assert events == []



def test_five_minute_duration_is_accepted_by_adapter() -> None:
    events: list[str] = []

    adapter = build_adapter(
        events,
        successful_media_executor(events),
    )

    result = adapter.execute_call(
        make_request(
            max_duration_seconds=300,
        )
    )

    assert result.duration_seconds == 84.25
    assert events.count("ami_originate") == 1

def test_media_call_id_mismatch_is_rejected() -> None:
    events: list[str] = []

    different_call_id = UUID("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee")

    def mismatched_media(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[
            [],
            OriginateResult,
        ],
    ) -> AsteriskMediaOutcome:
        del request

        events.append("media_listening")

        originate_result = originate()

        return AsteriskMediaOutcome(
            call_id=different_call_id,
            artifact_run_id="run-test-1",
            duration_seconds=2.0,
            originate=originate_result,
        )

    adapter = build_adapter(
        events,
        mismatched_media,
    )

    with pytest.raises(
        CallExecutionError,
        match="call ID",
    ):
        adapter.execute_call(make_request())

    assert events.count("ami_originate") == 1
    assert "ami_hangup" in events


def test_media_failure_requests_hangup_before_closing_ami() -> None:
    events: list[str] = []

    def failing_media(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[[], OriginateResult],
    ) -> AsteriskMediaOutcome:
        del request, call_id
        originate()
        raise RuntimeError("synthetic media setup failure")

    adapter = build_adapter(events, failing_media)

    with pytest.raises(RuntimeError, match="synthetic media setup failure"):
        adapter.execute_call(make_request())

    assert events[-2:] == ["ami_hangup", "ami_close"]


def test_hangup_failure_does_not_mask_original_media_failure() -> None:
    events: list[str] = []

    class HangupFailingAMIClient(FakeAMIClient):
        def hangup(self, *, unique_id: str, channel: str) -> None:
            super().hangup(unique_id=unique_id, channel=channel)
            raise RuntimeError("synthetic_hangup_cleanup_failure")

    fake_client = HangupFailingAMIClient(events)

    def failing_media(
        request: AssessmentCallRequest,
        call_id: UUID,
        originate: Callable[[], OriginateResult],
    ) -> AsteriskMediaOutcome:
        del request, call_id
        originate()
        raise RuntimeError("synthetic_original_media_failure")

    adapter = AsteriskAssessmentCallAdapter(
        ami_config=AsteriskAMIConfig(
            username="voiceprobe",
            secret="synthetic-test-secret",
        ),
        expected_originating_number=CALLER,
        ami_client_factory=lambda config: fake_client,
        call_id_factory=lambda: CALL_ID,
        media_executor=failing_media,
    )

    with pytest.raises(RuntimeError, match="synthetic_original_media_failure"):
        adapter.execute_call(make_request())

    assert events.count("ami_hangup") == 1
    assert events[-1] == "ami_close"
