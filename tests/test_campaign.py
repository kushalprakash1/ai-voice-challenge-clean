from __future__ import annotations

import threading
import time

import pytest

from voiceprobe.campaign import (
    CAMPAIGN_CONFIRMATION_TOKEN,
    MAX_CAMPAIGN_PARALLELISM,
    CampaignCaseRequest,
    CampaignCaseResult,
    CampaignCaseSpec,
    CampaignCaseStatus,
    CampaignSafetyError,
    authorize_live_campaign,
    build_campaign_plan,
    run_campaign,
)
from voiceprobe.policy import CallPolicy
from voiceprobe.safety import ALLOWED_TEST_NUMBER
from voiceprobe.scenarios.catalog import get_scenario

ORIGINATING_NUMBER = "+12025550101"


def policy(*, dry_run: bool = True) -> CallPolicy:
    return CallPolicy(
        originating_number=ORIGINATING_NUMBER,
        dry_run=dry_run,
    )


class ConcurrentFakeExecutor:
    def __init__(self, *, fail_positions: set[int] | None = None) -> None:
        self.fail_positions = set() if fail_positions is None else set(fail_positions)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.attempts: dict[int, int] = {}
        self.requests: list[CampaignCaseRequest] = []

    def execute_case(self, request: CampaignCaseRequest) -> CampaignCaseResult:
        with self.lock:
            self.requests.append(request)
            self.attempts[request.position] = self.attempts.get(request.position, 0) + 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        try:
            # Keep workers overlapped long enough for the concurrency assertion
            # without making the unit suite meaningfully slower.
            time.sleep(0.03)

            if request.position in self.fail_positions:
                raise RuntimeError("synthetic campaign worker failure")

            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.COMPLETED,
                execution_id=f"execution-{request.position}",
                artifact_run_id=f"artifact-{request.position}",
            )
        finally:
            with self.lock:
                self.active -= 1


def test_campaign_expands_repetitions_without_mutating_scenario_truth() -> None:
    scenario = get_scenario("autonomous-phone-diagnostic")
    plan = build_campaign_plan(
        policy(),
        cases=(
            CampaignCaseSpec(
                scenario_id=scenario.scenario_id,
                repetitions=3,
                evaluation_focus="probe booking confirmation consistency",
            ),
        ),
        max_parallel_calls=2,
        campaign_id="campaign-test",
    )

    assert plan.call_count == 3
    assert plan.destination == ALLOWED_TEST_NUMBER
    assert plan.max_parallel_calls == 2
    assert tuple(case.repetition for case in plan.cases) == (1, 2, 3)
    assert all(case.objective == scenario.objective for case in plan.cases)
    assert all(case.test_targets == scenario.test_targets for case in plan.cases)
    assert all(
        case.evaluation_focus == "probe booking confirmation consistency"
        for case in plan.cases
    )


def test_campaign_parallelism_is_hard_capped() -> None:
    with pytest.raises(CampaignSafetyError, match="parallelism"):
        build_campaign_plan(
            policy(),
            cases=(CampaignCaseSpec("autonomous-phone-diagnostic"),),
            max_parallel_calls=MAX_CAMPAIGN_PARALLELISM + 1,
        )


@pytest.mark.parametrize(
    "campaign_id",
    (
        "../escape",
        "Campaign Uppercase",
        "a",
        "campaign/child",
        "campaign.with.dot",
        "x" * 65,
    ),
)
def test_campaign_rejects_unsafe_campaign_ids(campaign_id: str) -> None:
    with pytest.raises(CampaignSafetyError, match="Campaign ID"):
        build_campaign_plan(
            policy(),
            cases=(CampaignCaseSpec("autonomous-phone-diagnostic"),),
            campaign_id=campaign_id,
        )


def test_campaign_rejects_unbounded_evaluation_focus() -> None:
    with pytest.raises(CampaignSafetyError, match="evaluation_focus"):
        build_campaign_plan(
            policy(),
            cases=(
                CampaignCaseSpec(
                    "autonomous-phone-diagnostic",
                    evaluation_focus="x" * 501,
                ),
            ),
        )


def test_dry_run_campaign_cannot_cross_live_boundary() -> None:
    plan = build_campaign_plan(
        policy(dry_run=True),
        cases=(CampaignCaseSpec("autonomous-phone-diagnostic"),),
    )

    with pytest.raises(CampaignSafetyError, match="dry_run"):
        authorize_live_campaign(
            plan,
            live_requested=True,
            confirmation_token=CAMPAIGN_CONFIRMATION_TOKEN,
        )


def test_campaign_live_request_and_exact_token_are_required() -> None:
    plan = build_campaign_plan(
        policy(dry_run=False),
        cases=(CampaignCaseSpec("autonomous-phone-diagnostic"),),
    )

    with pytest.raises(CampaignSafetyError, match="explicit live request"):
        authorize_live_campaign(
            plan,
            live_requested=False,
            confirmation_token=CAMPAIGN_CONFIRMATION_TOKEN,
        )

    with pytest.raises(CampaignSafetyError, match="confirmation token"):
        authorize_live_campaign(
            plan,
            live_requested=True,
            confirmation_token="wrong-token",
        )


def test_valid_campaign_can_cross_live_boundary() -> None:
    plan = build_campaign_plan(
        policy(dry_run=False),
        cases=(CampaignCaseSpec("autonomous-phone-diagnostic", repetitions=2),),
        max_parallel_calls=2,
        campaign_id="campaign-live-test",
    )

    authorization = authorize_live_campaign(
        plan,
        live_requested=True,
        confirmation_token=CAMPAIGN_CONFIRMATION_TOKEN,
    )

    assert authorization.plan is plan
    assert authorization.confirmation_token == CAMPAIGN_CONFIRMATION_TOKEN


def test_campaign_runner_respects_configured_parallelism() -> None:
    plan = build_campaign_plan(
        policy(),
        cases=(
            CampaignCaseSpec("autonomous-phone-diagnostic", repetitions=6),
        ),
        max_parallel_calls=3,
    )
    executor = ConcurrentFakeExecutor()

    result = run_campaign(plan, executor)

    assert result.completed_count == 6
    assert result.failed_count == 0
    assert executor.max_active == 3
    assert sorted(executor.attempts) == [1, 2, 3, 4, 5, 6]
    assert all(attempts == 1 for attempts in executor.attempts.values())


def test_campaign_worker_failure_is_isolated_and_not_retried() -> None:
    plan = build_campaign_plan(
        policy(),
        cases=(
            CampaignCaseSpec("autonomous-phone-diagnostic", repetitions=4),
        ),
        max_parallel_calls=2,
    )
    executor = ConcurrentFakeExecutor(fail_positions={2})

    result = run_campaign(plan, executor)

    assert result.completed_count == 3
    assert result.failed_count == 1
    assert executor.attempts[2] == 1
    assert result.entries[1].status is CampaignCaseStatus.FAILED
    assert "synthetic campaign worker failure" in (result.entries[1].error or "")
    assert result.entries[2].status is CampaignCaseStatus.COMPLETED


def test_campaign_results_are_returned_in_manifest_order() -> None:
    plan = build_campaign_plan(
        policy(),
        cases=(
            CampaignCaseSpec("autonomous-phone-diagnostic", repetitions=5),
        ),
        max_parallel_calls=4,
    )
    executor = ConcurrentFakeExecutor()

    result = run_campaign(plan, executor)

    assert tuple(entry.position for entry in result.entries) == (1, 2, 3, 4, 5)


def test_campaign_revalidates_the_only_authorized_destination() -> None:
    plan = build_campaign_plan(
        policy(),
        cases=(CampaignCaseSpec("autonomous-phone-diagnostic"),),
    )
    executor = ConcurrentFakeExecutor()

    run_campaign(plan, executor)

    assert {request.destination for request in executor.requests} == {
        ALLOWED_TEST_NUMBER,
    }
