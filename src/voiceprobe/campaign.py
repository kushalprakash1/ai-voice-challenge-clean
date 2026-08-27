"""First-class multi-call evaluation campaigns for VoiceProbe.

A campaign is intentionally one layer above an individual VoiceProbe execution.
The original execution boundary remains the unit that owns patient truth,
telephony authorization, call duration, artifacts, and scenario completion.

This module owns campaign planning, explicit live authorization, and bounded
orchestration. It never implements a telephony provider and never weakens the
single-call safety path. A concrete case executor must be injected, which keeps
concurrency testable without dialing.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import uuid4

from voiceprobe.policy import CallPolicy
from voiceprobe.safety import validate_destination
from voiceprobe.scenarios.catalog import get_scenario

DEFAULT_CAMPAIGN_PARALLELISM = 1
MAX_CAMPAIGN_PARALLELISM = 8
MAX_CAMPAIGN_CALLS = 64
MAX_REPETITIONS_PER_CASE = 16
MAX_EVALUATION_FOCUS_CHARS = 500
# This is a public confirmation phrase, not a password, credential, or secret.
CAMPAIGN_CONFIRMATION_TOKEN = "AUTHORIZE_ASSESSMENT_CAMPAIGN"  # nosec B105
CAMPAIGN_MEDIA_MODE_V3 = "v3"
_CAMPAIGN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class CampaignSafetyError(ValueError):
    """Raised when a campaign would cross a VoiceProbe safety boundary."""


class CampaignExecutionError(RuntimeError):
    """Raised when campaign orchestration cannot safely execute a case."""


class CampaignCaseStatus(StrEnum):
    """Terminal state for one expanded campaign case."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CampaignCaseSpec:
    """Declarative request to exercise one existing VoiceProbe scenario.

    ``evaluation_focus`` is evaluator metadata, not free-form patient truth.
    The patient still receives its authoritative facts/objective from the
    scenario catalog. This prevents a campaign prompt from silently overriding
    scenario-owned state.
    """

    scenario_id: str
    repetitions: int = 1
    evaluation_focus: str = ""


@dataclass(frozen=True, slots=True)
class CampaignCase:
    """One immutable, fully expanded campaign call case."""

    position: int
    case_id: str
    scenario_id: str
    repetition: int
    objective: str
    test_targets: tuple[str, ...]
    evaluation_focus: str


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """Validated campaign description with no provider side effects."""

    campaign_id: str
    originating_number: str
    destination: str
    cases: tuple[CampaignCase, ...]
    max_parallel_calls: int
    max_call_duration_seconds: int
    dry_run: bool
    media_mode: str = CAMPAIGN_MEDIA_MODE_V3

    @property
    def call_count(self) -> int:
        return len(self.cases)


@dataclass(frozen=True, slots=True)
class AuthorizedCampaign:
    """Campaign that crossed the explicit campaign-level live boundary."""

    plan: CampaignPlan
    confirmation_token: str


@dataclass(frozen=True, slots=True)
class CampaignCaseRequest:
    """Minimal information an injected worker may use for one case."""

    campaign_id: str
    position: int
    case_id: str
    scenario_id: str
    originating_number: str
    destination: str
    max_duration_seconds: int
    evaluation_focus: str
    selected_media_mode: str = CAMPAIGN_MEDIA_MODE_V3


@dataclass(frozen=True, slots=True)
class CampaignCaseResult:
    """Terminal evidence returned by one campaign worker."""

    position: int
    case_id: str
    scenario_id: str
    status: CampaignCaseStatus
    execution_id: str | None = None
    artifact_run_id: str | None = None
    error: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    call_duration_seconds: float | None = None
    attempt_count: int = 1
    worker_pid: int | None = None
    worker_port: int | None = None
    call_uuid: str | None = None
    route_registered: bool | None = None
    route_consumed: bool | None = None
    route_released: bool | None = None
    subprocess_exit_code: int | None = None
    timed_out: bool = False
    cpu_user_seconds: float | None = None
    cpu_system_seconds: float | None = None
    max_rss: int | None = None
    max_rss_unit: str | None = None
    telemetry_error: str | None = None
    lifecycle_error: str | None = None
    evidence_validation_error: str | None = None
    worker_connection_established: bool | None = None
    uuid_forwarded: bool | None = None
    selected_media_mode: str | None = None
    pipecat_version: str | None = None
    v3_runtime_dependencies_ready: bool | None = None


@dataclass(frozen=True, slots=True)
class CampaignRunResult:
    """Deterministically ordered terminal state for a campaign."""

    campaign_id: str
    entries: tuple[CampaignCaseResult, ...]
    started_at: str | None = None
    ended_at: str | None = None
    wall_clock_seconds: float | None = None
    maximum_simultaneous_active_workers: int = 0

    @property
    def completed_count(self) -> int:
        return sum(
            entry.status is CampaignCaseStatus.COMPLETED for entry in self.entries
        )

    @property
    def failed_count(self) -> int:
        return sum(entry.status is CampaignCaseStatus.FAILED for entry in self.entries)


@runtime_checkable
class CampaignCaseExecutor(Protocol):
    """Injected boundary responsible for exactly one isolated VoiceProbe call."""

    def execute_case(self, request: CampaignCaseRequest) -> CampaignCaseResult:
        """Execute one case exactly once and return terminal evidence."""


def _normalize_focus(value: str) -> str:
    if not isinstance(value, str):
        raise CampaignSafetyError("evaluation_focus must be text.")

    normalized = " ".join(value.split())

    if len(normalized) > MAX_EVALUATION_FOCUS_CHARS:
        raise CampaignSafetyError(
            f"evaluation_focus cannot exceed {MAX_EVALUATION_FOCUS_CHARS} characters."
        )

    return normalized


def _validate_parallelism(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignSafetyError("Campaign parallelism must be an integer.")

    if not 1 <= value <= MAX_CAMPAIGN_PARALLELISM:
        raise CampaignSafetyError(
            f"Campaign parallelism must be between 1 and {MAX_CAMPAIGN_PARALLELISM}."
        )

    return value


def _validate_campaign_id(value: str) -> str:
    if not isinstance(value, str) or not _CAMPAIGN_ID_PATTERN.fullmatch(value):
        raise CampaignSafetyError(
            "Campaign ID must contain only lowercase letters, numbers, "
            "underscores, and hyphens, and be 3-64 characters long."
        )

    return value


def build_campaign_plan(
    policy: CallPolicy,
    *,
    cases: tuple[CampaignCaseSpec, ...],
    max_parallel_calls: int = DEFAULT_CAMPAIGN_PARALLELISM,
    campaign_id: str | None = None,
) -> CampaignPlan:
    """Expand and validate a campaign without contacting telephony.

    Every case resolves through the existing immutable scenario catalog. The
    campaign can repeat scenarios and attach evaluator focus metadata, but it
    cannot mutate patient facts, the fixed destination, or call-duration caps.
    """

    destination = validate_destination(policy.destination)
    parallelism = _validate_parallelism(max_parallel_calls)

    if not cases:
        raise CampaignSafetyError("Campaign must contain at least one case spec.")

    expanded: list[CampaignCase] = []

    for spec in cases:
        if not isinstance(spec, CampaignCaseSpec):
            raise CampaignSafetyError(
                "Campaign cases must be CampaignCaseSpec values."
            )

        if isinstance(spec.repetitions, bool) or not isinstance(spec.repetitions, int):
            raise CampaignSafetyError("Campaign repetitions must be an integer.")

        if not 1 <= spec.repetitions <= MAX_REPETITIONS_PER_CASE:
            raise CampaignSafetyError(
                "Campaign repetitions must be between 1 and "
                f"{MAX_REPETITIONS_PER_CASE}."
            )

        scenario = get_scenario(spec.scenario_id)
        focus = _normalize_focus(spec.evaluation_focus)

        for repetition in range(1, spec.repetitions + 1):
            position = len(expanded) + 1
            expanded.append(
                CampaignCase(
                    position=position,
                    case_id=f"{scenario.scenario_id}-r{repetition:02d}",
                    scenario_id=scenario.scenario_id,
                    repetition=repetition,
                    objective=scenario.objective,
                    test_targets=scenario.test_targets,
                    evaluation_focus=focus,
                )
            )

    if len(expanded) > MAX_CAMPAIGN_CALLS:
        raise CampaignSafetyError(
            f"Campaign contains {len(expanded)} calls but the hard limit is "
            f"{MAX_CAMPAIGN_CALLS}."
        )

    parallelism = min(parallelism, len(expanded))

    resolved_campaign_id = _validate_campaign_id(
        campaign_id or f"campaign-{uuid4().hex[:12]}"
    )

    return CampaignPlan(
        campaign_id=resolved_campaign_id,
        originating_number=policy.originating_number,
        destination=destination,
        cases=tuple(expanded),
        max_parallel_calls=parallelism,
        max_call_duration_seconds=policy.max_call_duration_seconds,
        dry_run=policy.dry_run,
        media_mode=CAMPAIGN_MEDIA_MODE_V3,
    )


def authorize_live_campaign(
    plan: CampaignPlan,
    *,
    live_requested: bool,
    confirmation_token: str,
) -> AuthorizedCampaign:
    """Cross the explicit boundary from campaign planning to live execution."""

    validate_destination(plan.destination)
    _validate_campaign_id(plan.campaign_id)
    _validate_parallelism(plan.max_parallel_calls)

    if plan.dry_run:
        raise CampaignSafetyError(
            "Live campaign execution is forbidden while dry_run is enabled."
        )

    if plan.media_mode != CAMPAIGN_MEDIA_MODE_V3:
        raise CampaignSafetyError(
            "Live scalable campaigns require media_mode='v3'."
        )

    if not live_requested:
        raise CampaignSafetyError(
            "Live campaign execution requires an explicit live request."
        )

    if confirmation_token != CAMPAIGN_CONFIRMATION_TOKEN:
        raise CampaignSafetyError("Live campaign confirmation token is invalid.")

    if not 1 <= plan.call_count <= MAX_CAMPAIGN_CALLS:
        raise CampaignSafetyError("Campaign call count violates the hard limit.")

    if plan.max_parallel_calls > plan.call_count:
        raise CampaignSafetyError("Campaign parallelism exceeds call count.")

    return AuthorizedCampaign(
        plan=plan,
        confirmation_token=confirmation_token,
    )


def _request_for(
    plan: CampaignPlan,
    case: CampaignCase,
) -> CampaignCaseRequest:
    # Revalidate at the last orchestration boundary. A real worker repeats this
    # again through the existing execution/adapter safety path.
    destination = validate_destination(plan.destination)

    return CampaignCaseRequest(
        campaign_id=plan.campaign_id,
        position=case.position,
        case_id=case.case_id,
        scenario_id=case.scenario_id,
        originating_number=plan.originating_number,
        destination=destination,
        max_duration_seconds=plan.max_call_duration_seconds,
        evaluation_focus=case.evaluation_focus,
        selected_media_mode=plan.media_mode,
    )


def _execute_one(
    executor: CampaignCaseExecutor,
    request: CampaignCaseRequest,
) -> CampaignCaseResult:
    """Execute one attempt and normalize worker failures into evidence."""

    try:
        result = executor.execute_case(request)
    # This boundary deliberately converts every ordinary worker failure into
    # terminal evidence while allowing BaseException control flow to escape.
    except Exception as error:  # noqa: BLE001
        return CampaignCaseResult(
            position=request.position,
            case_id=request.case_id,
            scenario_id=request.scenario_id,
            status=CampaignCaseStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
        )

    if result.position != request.position:
        raise CampaignExecutionError("Campaign worker returned the wrong position.")

    if result.case_id != request.case_id:
        raise CampaignExecutionError("Campaign worker returned the wrong case ID.")

    if result.scenario_id != request.scenario_id:
        raise CampaignExecutionError("Campaign worker returned the wrong scenario ID.")

    return result


def run_campaign(
    plan: CampaignPlan,
    executor: CampaignCaseExecutor,
) -> CampaignRunResult:
    """Run a bounded campaign with one attempt per case and no retries.

    The executor is deliberately injected. Production telephony uses an
    executor that creates an isolated ordinary VoiceProbe execution for every
    request; tests can use an in-memory fake. Results are returned in manifest
    order even though completion order is concurrent.
    """

    validate_destination(plan.destination)
    _validate_campaign_id(plan.campaign_id)
    parallelism = _validate_parallelism(plan.max_parallel_calls)

    if not 1 <= plan.call_count <= MAX_CAMPAIGN_CALLS:
        raise CampaignExecutionError("Campaign call count violates the hard limit.")

    if parallelism > plan.call_count:
        raise CampaignExecutionError("Campaign parallelism exceeds call count.")

    campaign_started_at = datetime.now(UTC).isoformat()
    campaign_started = time.monotonic()
    activity_lock = threading.Lock()
    attempts: dict[int, int] = {}
    results: dict[int, CampaignCaseResult] = {}
    futures: dict[Future[CampaignCaseResult], int] = {}

    def execute_observed(request: CampaignCaseRequest) -> CampaignCaseResult:
        started_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        with activity_lock:
            attempts[request.position] = attempts.get(request.position, 0) + 1
            attempt_count = attempts[request.position]
        result = _execute_one(executor, request)
        return replace(
            result,
            started_at=result.started_at or started_at,
            ended_at=result.ended_at or datetime.now(UTC).isoformat(),
            duration_seconds=(
                result.duration_seconds
                if result.duration_seconds is not None
                else time.monotonic() - started
            ),
            attempt_count=attempt_count,
        )

    with ThreadPoolExecutor(
        max_workers=parallelism,
        thread_name_prefix="voiceprobe-campaign",
    ) as pool:
        for case in plan.cases:
            request = _request_for(plan, case)
            future = pool.submit(execute_observed, request)
            futures[future] = case.position

        for future in as_completed(futures):
            position = futures[future]

            try:
                result = future.result()
            # Preserve completed campaign evidence for any ordinary failure
            # escaping the per-worker normalization boundary.
            except Exception as error:  # noqa: BLE001
                case = plan.cases[position - 1]
                result = CampaignCaseResult(
                    position=position,
                    case_id=case.case_id,
                    scenario_id=case.scenario_id,
                    status=CampaignCaseStatus.FAILED,
                    error=f"{type(error).__name__}: {error}",
                )

            results[position] = result

    ordered = tuple(
        results[position] for position in range(1, plan.call_count + 1)
    )

    return CampaignRunResult(
        campaign_id=plan.campaign_id,
        entries=ordered,
        started_at=campaign_started_at,
        ended_at=datetime.now(UTC).isoformat(),
        wall_clock_seconds=time.monotonic() - campaign_started,
        maximum_simultaneous_active_workers=int(
            getattr(executor, "maximum_simultaneous_active_workers", 0)
        ),
    )
