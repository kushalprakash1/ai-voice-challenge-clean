from __future__ import annotations

import json
import resource
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from voiceprobe import run_campaign_case
from voiceprobe.campaign import (
    CAMPAIGN_CONFIRMATION_TOKEN,
    CampaignCaseRequest,
    CampaignCaseSpec,
    CampaignCaseStatus,
    authorize_live_campaign,
    build_campaign_plan,
)
from voiceprobe.campaign_evidence import (
    CampaignEvidenceError,
    initialize_lifecycle,
    lifecycle_path,
    update_lifecycle,
)
from voiceprobe.campaign_subprocess import SubprocessCampaignCaseExecutor
from voiceprobe.policy import CallPolicy
from voiceprobe.telephony.ami import AsteriskAMIConfig
from voiceprobe.telephony.asterisk_adapter import AsteriskAssessmentCallAdapter
from voiceprobe.telephony.audiosocket_dispatcher import AudioSocketDispatcher

ORIGINATING_NUMBER = "+12025550101"


class FakeProcess:
    next_pid = 4000

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.returncode = returncode
        self.killed = False

    def communicate(self, timeout=None):
        del timeout
        return self.stdout_text, self.stderr_text

    def kill(self):
        self.killed = True
        self.returncode = -9


def authorization(*, repetitions: int = 1):
    policy = CallPolicy(
        originating_number=ORIGINATING_NUMBER,
        dry_run=False,
    )
    plan = build_campaign_plan(
        policy,
        cases=(
            CampaignCaseSpec(
                "autonomous-phone-diagnostic",
                repetitions=repetitions,
                evaluation_focus="booking state retention",
            ),
        ),
        max_parallel_calls=min(2, repetitions),
        campaign_id="campaign-subprocess-test",
    )
    return authorize_live_campaign(
        plan,
        live_requested=True,
        confirmation_token=CAMPAIGN_CONFIRMATION_TOKEN,
    )


def request_for(auth, position: int = 1) -> CampaignCaseRequest:
    case = auth.plan.cases[position - 1]
    return CampaignCaseRequest(
        campaign_id=auth.plan.campaign_id,
        position=case.position,
        case_id=case.case_id,
        scenario_id=case.scenario_id,
        originating_number=auth.plan.originating_number,
        destination=auth.plan.destination,
        max_duration_seconds=auth.plan.max_call_duration_seconds,
        evaluation_focus=case.evaluation_focus,
    )


@pytest.mark.parametrize("ambient", (None, "0"))
def test_campaign_worker_environment_explicitly_selects_v3(
    monkeypatch, ambient
) -> None:
    if ambient is None:
        monkeypatch.delenv("VOICEPROBE_V3_LIVE", raising=False)
    else:
        monkeypatch.setenv("VOICEPROBE_V3_LIVE", ambient)

    environment = SubprocessCampaignCaseExecutor._worker_environment()

    assert environment["VOICEPROBE_V3_LIVE"] == "1"


def test_incident_regression_absent_ambient_mode_cannot_select_legacy(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VOICEPROBE_V3_LIVE", raising=False)
    worker_environment = SubprocessCampaignCaseExecutor._worker_environment()
    monkeypatch.setenv(
        "VOICEPROBE_V3_LIVE", worker_environment["VOICEPROBE_V3_LIVE"]
    )

    adapter = AsteriskAssessmentCallAdapter(
        ami_config=AsteriskAMIConfig(username="test", secret="synthetic"),
        expected_originating_number=ORIGINATING_NUMBER,
    )

    assert adapter._media_executor == adapter._execute_v3_media_call
    assert adapter._media_executor != adapter._execute_media_call


def test_live_worker_contract_fails_closed_before_runtime_setup(monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "0")

    with pytest.raises(RuntimeError, match="not active"):
        run_campaign_case._validate_live_media_contract(selected_media_mode="v3")

    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    with pytest.raises(RuntimeError, match="media mode v3"):
        run_campaign_case._validate_live_media_contract(selected_media_mode="legacy")


def test_live_worker_contract_requires_deepgram_before_runtime_setup(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VOICEPROBE_V3_LIVE", "1")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        run_campaign_case._validate_live_media_contract(selected_media_mode="v3")


def test_subprocess_worker_command_preserves_original_live_boundary(
    tmp_path, monkeypatch
) -> None:
    auth = authorization()
    observed: dict[str, object] = {}
    secret = "synthetic-deepgram-secret-that-must-not-be-artifacted"
    monkeypatch.setenv("DEEPGRAM_API_KEY", secret)

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        call_id = command[command.index("--call-id") + 1]
        worker_port = int(command[command.index("--worker-port") + 1])
        payload = "VOICEPROBE_CAMPAIGN_CASE_RESULT=" + json.dumps(
            {
                "status": "completed",
                "execution_id": "campaign-subprocess-test-c001",
                "artifact_run_id": "artifact-1",
                "duration_seconds": 12.5,
                "call_id": call_id,
                "worker_port": worker_port,
                "selected_media_mode": "v3",
            }
        )
        payload += "\n"
        process = FakeProcess(stdout=payload)
        lifecycle_file = tmp_path / auth.plan.campaign_id / "cases" / (
            "001-autonomous-phone-diagnostic-r01.lifecycle.json"
        )
        lifecycle_file.parent.mkdir(parents=True, exist_ok=True)
        lifecycle_file.write_text(
            json.dumps(
                {
                    "worker_pid": process.pid,
                    "execution_id": "campaign-subprocess-test-c001",
                    "call_uuid": call_id,
                    "worker_port": worker_port,
                    "selected_media_mode": "v3",
                }
            )
        )
        return process

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=fake_run,
        )
        result = executor.execute_case(request_for(auth))

        assert dispatcher.registered_count == 0

    command = observed["command"]
    kwargs = observed["kwargs"]

    assert isinstance(command, list)
    assert "-m" in command
    assert "voiceprobe.run_campaign_case" in command
    assert "--live" in command
    assert "--confirm" in command
    assert command[command.index("--media-mode") + 1] == "v3"
    assert "AUTHORIZE_ASSESSMENT_CALLS" in command
    assert "--lifecycle-path" not in command
    assert "shell" not in kwargs
    assert kwargs["text"] is True
    assert kwargs["env"]["VOICEPROBE_V3_LIVE"] == "1"
    assert secret not in " ".join(command)
    assert result.status is CampaignCaseStatus.COMPLETED
    assert result.artifact_run_id == "artifact-1"
    assert result.call_duration_seconds == 12.5
    assert result.worker_port == 9200
    assert result.call_uuid
    assert result.route_registered is True
    assert result.route_consumed is False
    assert result.route_released is True
    assert result.subprocess_exit_code == 0
    assert result.timed_out is False
    assert secret not in "".join(
        path.read_text() for path in tmp_path.rglob("*.log")
    )


def test_subprocess_worker_rejects_request_not_in_authorized_campaign(tmp_path) -> None:
    auth = authorization()
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        del command, kwargs
        calls += 1
        raise AssertionError("process runner must not be reached")

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=fake_run,
        )
        request = replace(
            request_for(auth),
            campaign_id="different-campaign",
        )

        try:
            executor.execute_case(request)
        except RuntimeError as error:
            assert "authorization" in str(error)
        else:
            raise AssertionError("Unauthorized campaign request was accepted.")

    assert calls == 0
    assert executor.maximum_simultaneous_active_workers == 0


def test_subprocess_timeout_is_failed_evidence_and_route_is_released(tmp_path) -> None:
    auth = authorization()

    class TimedOutProcess(FakeProcess):
        def communicate(self, timeout=None):
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("worker", timeout)
            return "partial worker stdout\n", "partial worker stderr\n"

    def timed_out(command, **kwargs):
        del command, kwargs
        return TimedOutProcess()

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=timed_out,
        )
        result = executor.execute_case(request_for(auth))

        assert dispatcher.registered_count == 0

    assert result.status is CampaignCaseStatus.FAILED
    assert "hard process timeout" in (result.error or "")
    assert result.timed_out is True
    assert result.subprocess_exit_code == -9
    assert executor.active_worker_count == 0
    assert result.route_released is True


def test_subprocess_launch_failure_is_isolated(tmp_path) -> None:
    auth = authorization()

    def launch_failure(command, **kwargs):
        del command, kwargs
        raise OSError("synthetic process launch failure")

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=launch_failure,
        )
        result = executor.execute_case(request_for(auth))

    assert result.status is CampaignCaseStatus.FAILED
    assert "Unable to launch campaign worker" in (result.error or "")
    assert executor.maximum_simultaneous_active_workers == 0


def test_subprocess_worker_collects_lightweight_process_evidence(tmp_path) -> None:
    auth = authorization()

    def fake_run(command, **kwargs):
        del kwargs
        lifecycle_path = (
            tmp_path
            / auth.plan.campaign_id
            / "cases"
            / ("001-autonomous-phone-diagnostic-r01.lifecycle.json")
        )
        lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
        lifecycle_path.write_text(
            json.dumps(
                {
                    "worker_pid": FakeProcess.next_pid + 1,
                    "execution_id": "campaign-subprocess-test-c001",
                    "call_uuid": str(
                        __import__("uuid").uuid5(
                            __import__("uuid").NAMESPACE_URL,
                            "voiceprobe:campaign-subprocess-test:autonomous-phone-diagnostic-r01:1",
                        )
                    ),
                    "worker_port": 9200,
                    "selected_media_mode": "v3",
                    "cpu_user_seconds": 1.25,
                    "cpu_system_seconds": 0.5,
                    "max_rss": 204800,
                }
            )
        )
        payload = (
            'VOICEPROBE_CAMPAIGN_CASE_RESULT={"status":"completed",'
            '"execution_id":"campaign-subprocess-test-c001",'
            '"artifact_run_id":"artifact-1","call_id":"'
            + json.loads(lifecycle_path.read_text())["call_uuid"]
            + '","worker_port":9200,"selected_media_mode":"v3"}\n'
        )
        return FakeProcess(stdout=payload)

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=fake_run,
        )
        result = executor.execute_case(request_for(auth))

    assert result.worker_pid is not None
    assert result.cpu_user_seconds == 1.25
    assert result.cpu_system_seconds == 0.5
    assert result.max_rss == 204800


@pytest.mark.parametrize(
    ("campaign_id", "case_id"),
    (
        ("../escape", "safe-case"),
        ("safe-campaign", "../escape"),
        ("/tmp/escape", "safe-case"),
    ),
)
def test_lifecycle_path_rejects_traversal_and_absolute_segments(
    campaign_id, case_id
) -> None:
    with pytest.raises(CampaignEvidenceError):
        lifecycle_path(campaign_id=campaign_id, position=1, case_id=case_id)


def test_lifecycle_path_is_expected_and_cannot_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = lifecycle_path(campaign_id="safe-campaign", position=2, case_id="safe-case")
    assert path == Path(
        "artifacts/campaigns/safe-campaign/cases/002-safe-case.lifecycle.json"
    )
    initialize_lifecycle(path, {"terminal": False})
    update_lifecycle(path, {"terminal": True})
    assert json.loads(path.read_text()) == {"terminal": True}
    with pytest.raises(CampaignEvidenceError, match="already exists"):
        initialize_lifecycle(path, {"forged": True})


def test_lifecycle_path_rejects_symlink_escape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    campaign_root = Path("artifacts/campaigns")
    campaign_root.mkdir(parents=True)
    (campaign_root / "safe-campaign").symlink_to(outside, target_is_directory=True)
    path = lifecycle_path(
        campaign_id="safe-campaign", position=1, case_id="safe-case"
    )
    with pytest.raises(CampaignEvidenceError, match="symbolic link"):
        initialize_lifecycle(path, {"terminal": False})
    assert not (outside / "cases").exists()


def test_overlapping_subprocess_lifetimes_produce_peak_two(tmp_path) -> None:
    auth = authorization(repetitions=2)
    barrier = threading.Barrier(2)

    class OverlappingProcess(FakeProcess):
        def communicate(self, timeout=None):
            del timeout
            barrier.wait(timeout=2)
            return self.stdout_text, ""

    def factory(command, **kwargs):
        del kwargs
        call_id = command[command.index("--call-id") + 1]
        port = int(command[command.index("--worker-port") + 1])
        execution_id = command[command.index("--execution-id") + 1]
        payload = "VOICEPROBE_CAMPAIGN_CASE_RESULT=" + json.dumps(
            {
                "status": "completed",
                "execution_id": execution_id,
                "artifact_run_id": f"artifact-{port}",
                "duration_seconds": 1.0,
                "call_id": call_id,
                "worker_port": port,
                "selected_media_mode": "v3",
            }
        )
        return OverlappingProcess(stdout=payload)

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=factory,
        )
        from voiceprobe.campaign import run_campaign

        result = run_campaign(auth.plan, executor)

    assert result.maximum_simultaneous_active_workers == 2
    assert executor.active_worker_count == 0


def test_sequential_subprocess_lifetimes_produce_peak_one(tmp_path) -> None:
    auth = authorization(repetitions=2)

    def factory(command, **kwargs):
        del kwargs
        call_id = command[command.index("--call-id") + 1]
        port = int(command[command.index("--worker-port") + 1])
        execution_id = command[command.index("--execution-id") + 1]
        payload = "VOICEPROBE_CAMPAIGN_CASE_RESULT=" + json.dumps(
            {
                "status": "completed",
                "execution_id": execution_id,
                "artifact_run_id": f"artifact-{port}",
                "duration_seconds": 1.0,
                "call_id": call_id,
                "worker_port": port,
                "selected_media_mode": "v3",
            }
        )
        return FakeProcess(stdout=payload)

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=factory,
        )
        executor.execute_case(request_for(auth, 1))
        executor.execute_case(request_for(auth, 2))

    assert executor.maximum_simultaneous_active_workers == 1


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_id", "wrong-execution"),
        ("call_id", "00000000-0000-0000-0000-000000000000"),
        ("worker_port", 9998),
        ("duration_seconds", -1),
        ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("duration_seconds", "not-a-number"),
        ("artifact_run_id", "../escape"),
        ("selected_media_mode", None),
        ("selected_media_mode", "legacy"),
    ),
)
def test_malformed_child_result_is_contained(tmp_path, field, value) -> None:
    auth = authorization()

    def factory(command, **kwargs):
        del kwargs
        payload = {
            "status": "completed",
            "execution_id": command[command.index("--execution-id") + 1],
            "artifact_run_id": "artifact-safe",
            "duration_seconds": 1.0,
            "call_id": command[command.index("--call-id") + 1],
            "worker_port": int(command[command.index("--worker-port") + 1]),
            "selected_media_mode": "v3",
        }
        payload[field] = value
        return FakeProcess(
            stdout="VOICEPROBE_CAMPAIGN_CASE_RESULT=" + json.dumps(payload)
        )

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=factory,
        )
        result = executor.execute_case(request_for(auth))

    assert result.status is CampaignCaseStatus.FAILED
    assert result.evidence_validation_error
    assert result.worker_pid and result.worker_port == 9200 and result.call_uuid
    assert result.subprocess_exit_code == 0
    assert result.attempt_count == 1


@pytest.mark.parametrize(
    "lifecycle_change",
    (
        {"worker_pid": -1},
        {"cpu_user_seconds": -1},
        {"cpu_system_seconds": "bad"},
        {"max_rss": -1},
    ),
)
def test_invalid_lifecycle_metrics_and_pid_are_contained(
    tmp_path, lifecycle_change
) -> None:
    auth = authorization()

    def factory(command, **kwargs):
        del kwargs
        process = FakeProcess()
        call_id = command[command.index("--call-id") + 1]
        port = int(command[command.index("--worker-port") + 1])
        execution_id = command[command.index("--execution-id") + 1]
        path = (
            tmp_path
            / auth.plan.campaign_id
            / "cases"
            / ("001-autonomous-phone-diagnostic-r01.lifecycle.json")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        lifecycle = {
            "worker_pid": process.pid,
            "execution_id": execution_id,
            "call_uuid": call_id,
            "worker_port": port,
            "selected_media_mode": "v3",
        }
        lifecycle.update(lifecycle_change)
        path.write_text(json.dumps(lifecycle))
        process.stdout_text = "VOICEPROBE_CAMPAIGN_CASE_RESULT=" + json.dumps(
            {
                "status": "completed",
                "execution_id": execution_id,
                "artifact_run_id": "artifact-safe",
                "duration_seconds": 1.0,
                "call_id": call_id,
                "worker_port": port,
                "selected_media_mode": "v3",
            }
        )
        return process

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=factory,
        )
        result = executor.execute_case(request_for(auth))

    assert result.status is CampaignCaseStatus.FAILED
    assert result.evidence_validation_error


def test_malformed_lifecycle_json_is_explicit_validation_failure(tmp_path) -> None:
    auth = authorization()
    path = (
        tmp_path
        / auth.plan.campaign_id
        / "cases"
        / ("001-autonomous-phone-diagnostic-r01.lifecycle.json")
    )
    path.parent.mkdir(parents=True)
    path.write_text("{not-json")

    def factory(command, **kwargs):
        del kwargs
        payload = {
            "status": "completed",
            "execution_id": command[command.index("--execution-id") + 1],
            "artifact_run_id": "artifact-safe",
            "duration_seconds": 1.0,
            "call_id": command[command.index("--call-id") + 1],
            "worker_port": int(command[command.index("--worker-port") + 1]),
            "selected_media_mode": "v3",
        }
        return FakeProcess(
            stdout="VOICEPROBE_CAMPAIGN_CASE_RESULT=" + json.dumps(payload)
        )

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_factory=factory,
        )
        result = executor.execute_case(request_for(auth))

    assert result.evidence_validation_error == "malformed lifecycle JSON"


def test_optional_resource_failure_is_nonfatal(monkeypatch) -> None:
    def fail_getrusage(_who):
        raise OSError("metrics unavailable")

    monkeypatch.setattr(resource, "getrusage", fail_getrusage)
    evidence, error = run_campaign_case._optional_resource_evidence()
    assert evidence == {}
    assert error == "OSError: metrics unavailable"


def test_lifecycle_finalization_failure_is_reportable(monkeypatch, tmp_path) -> None:
    def fail_update(_path, _payload):
        raise OSError("evidence unavailable")

    monkeypatch.setattr(run_campaign_case, "update_lifecycle", fail_update)
    error = run_campaign_case._finalize_lifecycle(tmp_path / "lifecycle.json", {})
    assert error == "OSError: evidence unavailable"


def test_resource_and_lifecycle_failures_are_independently_contained(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        resource,
        "getrusage",
        lambda _who: (_ for _ in ()).throw(OSError("metrics unavailable")),
    )
    monkeypatch.setattr(
        run_campaign_case,
        "update_lifecycle",
        lambda _path, _payload: (_ for _ in ()).throw(OSError("write unavailable")),
    )
    _, telemetry_error = run_campaign_case._optional_resource_evidence()
    lifecycle_error = run_campaign_case._finalize_lifecycle(
        tmp_path / "lifecycle.json", {}
    )
    assert telemetry_error and lifecycle_error


def test_setup_failure_best_effort_finalizes_lifecycle(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "voiceprobe.run_campaign_case",
            "--scenario",
            "autonomous-phone-diagnostic",
            "--campaign-id",
            "setup-failure-campaign",
            "--case-id",
            "autonomous-phone-diagnostic-r01",
            "--position",
            "1",
            "--execution-id",
            "setup-failure-campaign-c001",
            "--call-id",
            "00000000-0000-0000-0000-000000000001",
            "--worker-port",
            "9200",
            "--media-mode",
            "v3",
        ],
    )
    attempts = 0

    def fail_settings():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("synthetic setup failure")

    monkeypatch.setattr(run_campaign_case, "Settings", fail_settings)
    with pytest.raises(RuntimeError, match="synthetic setup failure"):
        run_campaign_case.main()

    path = lifecycle_path(
        campaign_id="setup-failure-campaign",
        position=1,
        case_id="autonomous-phone-diagnostic-r01",
    )
    payload = json.loads(path.read_text())
    assert payload["terminal"] is True
    assert payload["status"] == "failed"
    assert attempts == 1
