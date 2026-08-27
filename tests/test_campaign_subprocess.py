from __future__ import annotations

import subprocess
from dataclasses import replace

from voiceprobe.campaign import (
    CAMPAIGN_CONFIRMATION_TOKEN,
    CampaignCaseRequest,
    CampaignCaseStatus,
    CampaignCaseSpec,
    authorize_live_campaign,
    build_campaign_plan,
)
from voiceprobe.campaign_subprocess import (
    WORKER_TEARDOWN_GRACE_SECONDS,
    SubprocessCampaignCaseExecutor,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.telephony.audiosocket_dispatcher import AudioSocketDispatcher

ORIGINATING_NUMBER = "+12025550101"


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


def test_subprocess_worker_command_preserves_original_live_boundary(tmp_path) -> None:
    auth = authorization()
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        payload = (
            'VOICEPROBE_CAMPAIGN_CASE_RESULT={"status":"completed",'
            '"execution_id":"campaign-subprocess-test-c001",'
            '"artifact_run_id":"artifact-1"}\n'
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=payload,
            stderr="",
        )

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_runner=fake_run,
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
    assert "AUTHORIZE_ASSESSMENT_CALLS" in command
    assert "shell" not in kwargs
    assert kwargs["timeout"] == (
        auth.plan.max_call_duration_seconds + WORKER_TEARDOWN_GRACE_SECONDS
    )
    assert result.status is CampaignCaseStatus.COMPLETED
    assert result.artifact_run_id == "artifact-1"


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
            process_runner=fake_run,
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


def test_subprocess_timeout_is_failed_evidence_and_route_is_released(tmp_path) -> None:
    auth = authorization()

    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs["timeout"],
            output="partial worker stdout\n",
            stderr="partial worker stderr\n",
        )

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=auth,
            dispatcher=dispatcher,
            per_call_budget_usd="1.00",
            max_rate_per_minute_usd="0.10",
            log_root=tmp_path,
            process_runner=timed_out,
        )
        result = executor.execute_case(request_for(auth))

        assert dispatcher.registered_count == 0

    assert result.status is CampaignCaseStatus.FAILED
    assert "hard process timeout" in (result.error or "")


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
            process_runner=launch_failure,
        )
        result = executor.execute_case(request_for(auth))

    assert result.status is CampaignCaseStatus.FAILED
    assert "Unable to launch campaign worker" in (result.error or "")
