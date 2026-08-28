"""Deterministic campaign-level reporting for VoiceProbe evaluation runs.

The reducer consumes only frozen campaign metadata and terminal per-case results.
It never reinterprets call audio or changes assessment outcomes. Detailed oracle
and transcript evidence remains in each referenced per-call artifact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from voiceprobe.campaign import (
    CampaignCaseStatus,
    CampaignPlan,
    CampaignRunResult,
)


class CampaignReportError(ValueError):
    """Raised when terminal results do not match their frozen campaign plan."""


@dataclass(frozen=True, slots=True)
class CampaignGroupSummary:
    """Aggregate pass/fail counts for one scenario or evaluator focus."""

    key: str
    call_count: int
    completed_count: int
    failed_count: int
    failure_rate: float


@dataclass(frozen=True, slots=True)
class CampaignFailureEvidence:
    """Pointer from a campaign failure back to its durable call evidence."""

    position: int
    case_id: str
    scenario_id: str
    evaluation_focus: str
    execution_id: str | None
    artifact_run_id: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Human/dashboard-friendly aggregate over one immutable campaign result."""

    campaign_id: str
    call_count: int
    completed_count: int
    failed_count: int
    failure_rate: float
    by_scenario: tuple[CampaignGroupSummary, ...]
    by_evaluation_focus: tuple[CampaignGroupSummary, ...]
    failures: tuple[CampaignFailureEvidence, ...]


def _summaries(
    grouped_statuses: dict[str, list[CampaignCaseStatus]],
) -> tuple[CampaignGroupSummary, ...]:
    summaries: list[CampaignGroupSummary] = []

    for key in sorted(grouped_statuses):
        statuses = grouped_statuses[key]
        call_count = len(statuses)
        completed_count = sum(
            status is CampaignCaseStatus.COMPLETED for status in statuses
        )
        failed_count = sum(
            status is CampaignCaseStatus.FAILED for status in statuses
        )
        summaries.append(
            CampaignGroupSummary(
                key=key,
                call_count=call_count,
                completed_count=completed_count,
                failed_count=failed_count,
                failure_rate=(failed_count / call_count if call_count else 0.0),
            )
        )

    return tuple(summaries)


def build_campaign_report(
    plan: CampaignPlan,
    result: CampaignRunResult,
) -> CampaignReport:
    """Reduce terminal case evidence without changing any case outcome."""

    if result.campaign_id != plan.campaign_id:
        raise CampaignReportError("Campaign result ID does not match its plan.")

    if len(result.entries) != plan.call_count:
        raise CampaignReportError(
            "Campaign result entry count does not match its plan."
        )

    by_scenario: dict[str, list[CampaignCaseStatus]] = defaultdict(list)
    by_focus: dict[str, list[CampaignCaseStatus]] = defaultdict(list)
    failures: list[CampaignFailureEvidence] = []

    for case, entry in zip(plan.cases, result.entries, strict=True):
        if entry.position != case.position:
            raise CampaignReportError(
                "Campaign result position does not match its planned case."
            )
        if entry.case_id != case.case_id:
            raise CampaignReportError(
                "Campaign result case ID does not match its planned case."
            )
        if entry.scenario_id != case.scenario_id:
            raise CampaignReportError(
                "Campaign result scenario does not match its planned case."
            )

        focus = case.evaluation_focus or "unspecified"
        by_scenario[case.scenario_id].append(entry.status)
        by_focus[focus].append(entry.status)

        if entry.status is CampaignCaseStatus.FAILED:
            failures.append(
                CampaignFailureEvidence(
                    position=case.position,
                    case_id=case.case_id,
                    scenario_id=case.scenario_id,
                    evaluation_focus=focus,
                    execution_id=entry.execution_id,
                    artifact_run_id=entry.artifact_run_id,
                    error=entry.error,
                )
            )

    return CampaignReport(
        campaign_id=plan.campaign_id,
        call_count=plan.call_count,
        completed_count=result.completed_count,
        failed_count=result.failed_count,
        failure_rate=(
            result.failed_count / plan.call_count if plan.call_count else 0.0
        ),
        by_scenario=_summaries(by_scenario),
        by_evaluation_focus=_summaries(by_focus),
        failures=tuple(failures),
    )
