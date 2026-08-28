from __future__ import annotations

from voiceprobe.campaign import (
    CampaignCaseResult,
    CampaignCaseSpec,
    CampaignCaseStatus,
    CampaignRunResult,
    build_campaign_plan,
)
from voiceprobe.campaign_report import (
    CampaignReportError,
    build_campaign_report,
)
from voiceprobe.policy import CallPolicy

ORIGINATING_NUMBER = "+12025550101"


def plan():
    return build_campaign_plan(
        CallPolicy(
            originating_number=ORIGINATING_NUMBER,
            dry_run=True,
        ),
        cases=(
            CampaignCaseSpec(
                "autonomous-phone-diagnostic",
                repetitions=2,
                evaluation_focus="baseline scheduling",
            ),
            CampaignCaseSpec(
                "wrong-day-offer",
                evaluation_focus="wrong slot acceptance",
            ),
        ),
        max_parallel_calls=2,
        campaign_id="campaign-report-test",
    )


def test_report_aggregates_scenario_and_bug_focus_failures() -> None:
    campaign = plan()
    result = CampaignRunResult(
        campaign_id=campaign.campaign_id,
        entries=(
            CampaignCaseResult(
                position=1,
                case_id=campaign.cases[0].case_id,
                scenario_id=campaign.cases[0].scenario_id,
                status=CampaignCaseStatus.COMPLETED,
                execution_id="exec-1",
                artifact_run_id="run-1",
            ),
            CampaignCaseResult(
                position=2,
                case_id=campaign.cases[1].case_id,
                scenario_id=campaign.cases[1].scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id="exec-2",
                artifact_run_id="run-2",
                error="booking confirmation missing",
            ),
            CampaignCaseResult(
                position=3,
                case_id=campaign.cases[2].case_id,
                scenario_id=campaign.cases[2].scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id="exec-3",
                artifact_run_id="run-3",
                error="accepted incompatible day",
            ),
        ),
    )

    report = build_campaign_report(campaign, result)

    assert report.call_count == 3
    assert report.completed_count == 1
    assert report.failed_count == 2
    assert report.failure_rate == 2 / 3
    assert len(report.failures) == 2
    assert report.failures[0].artifact_run_id == "run-2"
    assert report.failures[1].evaluation_focus == "wrong slot acceptance"

    by_scenario = {summary.key: summary for summary in report.by_scenario}
    assert by_scenario["autonomous-phone-diagnostic"].call_count == 2
    assert by_scenario["autonomous-phone-diagnostic"].failed_count == 1
    assert by_scenario["wrong-day-offer"].failed_count == 1

    by_focus = {summary.key: summary for summary in report.by_evaluation_focus}
    assert by_focus["baseline scheduling"].call_count == 2
    assert by_focus["wrong slot acceptance"].failure_rate == 1.0


def test_report_rejects_result_from_different_campaign() -> None:
    campaign = plan()
    result = CampaignRunResult(
        campaign_id="different-campaign",
        entries=(),
    )

    try:
        build_campaign_report(campaign, result)
    except CampaignReportError as error:
        assert "ID" in str(error)
    else:
        raise AssertionError("Mismatched campaign result was accepted.")


def test_report_rejects_reordered_or_mismatched_case_evidence() -> None:
    campaign = plan()
    first = campaign.cases[0]
    second = campaign.cases[1]
    third = campaign.cases[2]
    result = CampaignRunResult(
        campaign_id=campaign.campaign_id,
        entries=(
            CampaignCaseResult(
                position=1,
                case_id=second.case_id,
                scenario_id=second.scenario_id,
                status=CampaignCaseStatus.COMPLETED,
            ),
            CampaignCaseResult(
                position=2,
                case_id=first.case_id,
                scenario_id=first.scenario_id,
                status=CampaignCaseStatus.COMPLETED,
            ),
            CampaignCaseResult(
                position=3,
                case_id=third.case_id,
                scenario_id=third.scenario_id,
                status=CampaignCaseStatus.COMPLETED,
            ),
        ),
    )

    try:
        build_campaign_report(campaign, result)
    except CampaignReportError as error:
        assert "case ID" in str(error) or "scenario" in str(error)
    else:
        raise AssertionError("Mismatched campaign case evidence was accepted.")
