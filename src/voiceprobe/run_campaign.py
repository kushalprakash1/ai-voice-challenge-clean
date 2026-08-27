"""Bounded concurrent VoiceProbe evaluation-campaign entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from voiceprobe.campaign import (
    MAX_CAMPAIGN_PARALLELISM,
    CampaignCaseSpec,
    authorize_live_campaign,
    build_campaign_plan,
    run_campaign,
)
from voiceprobe.campaign_packs import (
    evaluation_pack_ids,
    get_evaluation_pack,
)
from voiceprobe.campaign_subprocess import SubprocessCampaignCaseExecutor
from voiceprobe.config import Settings
from voiceprobe.execution_state import BudgetLedger, BudgetPolicy
from voiceprobe.policy import DEFAULT_MAX_CALL_DURATION_SECONDS, MAX_CALL_DURATION_SECONDS
from voiceprobe.run_one import DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD
from voiceprobe.safety import require_live_destination
from voiceprobe.scenarios.catalog import list_scenarios
from voiceprobe.telephony.audiosocket_dispatcher import AudioSocketDispatcher

DEFAULT_CAMPAIGN_BUDGET_USD = Decimal("20.00")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(path)


def _create_campaign_root(path: Path) -> None:
    """Create a new evidence root and refuse accidental campaign overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def _merge_focus(primary: str, additional: str) -> str:
    additional = " ".join(additional.split())

    if not additional:
        return primary

    if not primary:
        return additional

    return f"{primary} Additional focus: {additional}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run many isolated VoiceProbe assessment calls with bounded local "
            "parallelism while preserving the original one-call safety path."
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.scenario_id for scenario in list_scenarios()),
        help="Scenario to include; repeat the flag to build a scenario matrix.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Include every scenario in the deterministic catalog.",
    )
    parser.add_argument(
        "--pack",
        choices=(*evaluation_pack_ids(), "full-regression"),
        help=(
            "Curated bug-finding matrix such as booking-integrity, "
            "state-retention, or production-smoke."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat each selected scenario or pack case this many times.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Maximum simultaneous isolated calls. "
            f"Hard cap: {MAX_CAMPAIGN_PARALLELISM}."
        ),
    )
    parser.add_argument(
        "--evaluation-focus",
        default="",
        help=(
            "Optional evaluator-only bug focus recorded in the campaign manifest. "
            "It cannot overwrite patient scenario truth."
        ),
    )
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help="Exact campaign-level live authorization token.",
    )
    parser.add_argument(
        "--max-call-duration-seconds",
        type=int,
        default=DEFAULT_MAX_CALL_DURATION_SECONDS,
    )
    parser.add_argument(
        "--budget-usd",
        default=format(DEFAULT_CAMPAIGN_BUDGET_USD, "f"),
        help="Hard worst-case provider budget for the entire campaign.",
    )
    parser.add_argument(
        "--max-rate-per-minute-usd",
        default=format(DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD, "f"),
        help="Conservative provider-rate ceiling used for campaign reservation.",
    )
    args = parser.parse_args()

    selection_modes = sum(
        (
            bool(args.scenario),
            bool(args.all_scenarios),
            bool(args.pack),
        )
    )

    if selection_modes != 1:
        parser.error(
            "Choose exactly one selection mode: --scenario, --all-scenarios, or --pack."
        )

    selected_pack_id: str | None = None
    selected_pack_description: str | None = None

    if args.pack:
        pack = get_evaluation_pack(args.pack)
        selected_pack_id = pack.pack_id
        selected_pack_description = pack.description
        cases = tuple(
            replace(
                case,
                repetitions=args.repetitions,
                evaluation_focus=_merge_focus(
                    case.evaluation_focus,
                    args.evaluation_focus,
                ),
            )
            for case in pack.cases
        )
    else:
        if args.all_scenarios:
            scenario_ids = tuple(
                scenario.scenario_id for scenario in list_scenarios()
            )
        else:
            scenario_ids = tuple(args.scenario or ())

        cases = tuple(
            CampaignCaseSpec(
                scenario_id=scenario_id,
                repetitions=args.repetitions,
                evaluation_focus=args.evaluation_focus,
            )
            for scenario_id in scenario_ids
        )

    if not 1 <= args.max_call_duration_seconds <= MAX_CALL_DURATION_SECONDS:
        parser.error(
            "--max-call-duration-seconds must be between 1 and "
            f"{MAX_CALL_DURATION_SECONDS}"
        )

    settings = Settings()  # type: ignore[call-arg]
    base_policy = settings.call_policy()
    policy = replace(
        base_policy,
        dry_run=not args.live,
        max_call_duration_seconds=args.max_call_duration_seconds,
    )

    plan = build_campaign_plan(
        policy,
        cases=cases,
        max_parallel_calls=args.parallel,
        campaign_id=args.campaign_id or None,
    )

    try:
        total_budget = Decimal(args.budget_usd)
        max_rate = Decimal(args.max_rate_per_minute_usd)
        budget_policy = BudgetPolicy(
            total_budget_usd=total_budget,
            max_provider_rate_per_minute_usd=max_rate,
        )
    except Exception as error:
        parser.error(f"Invalid campaign budget configuration: {error}")

    budget_probe = BudgetLedger(budget_policy)
    per_call_worst_case = budget_probe.worst_case_call_cost(
        plan.max_call_duration_seconds
    )
    campaign_worst_case = per_call_worst_case * Decimal(plan.call_count)

    if campaign_worst_case > total_budget:
        parser.error(
            "Campaign worst-case reservation exceeds --budget-usd: "
            f"required={campaign_worst_case}; configured={total_budget}."
        )

    campaign_root = Path("artifacts/campaigns") / plan.campaign_id

    try:
        _create_campaign_root(campaign_root)
    except FileExistsError:
        parser.error(
            "Campaign evidence directory already exists; choose a new --campaign-id."
        )

    manifest_path = campaign_root / "manifest.json"
    _write_json_atomic(
        manifest_path,
        {
            **asdict(plan),
            "evaluation_pack_id": selected_pack_id,
            "evaluation_pack_description": selected_pack_description,
            "per_call_worst_case_usd": per_call_worst_case,
            "campaign_worst_case_usd": campaign_worst_case,
            "telephony_enabled": args.live,
        },
    )

    if not args.live:
        print(
            json.dumps(
                {
                    "campaign_id": plan.campaign_id,
                    "manifest_path": str(manifest_path),
                    "evaluation_pack_id": selected_pack_id,
                    "call_count": plan.call_count,
                    "parallel": plan.max_parallel_calls,
                    "destination": plan.destination,
                    "campaign_worst_case_usd": str(campaign_worst_case),
                    "dry_run": True,
                    "telephony_enabled": False,
                },
                indent=2,
            )
        )
        return 0

    require_live_destination()

    try:
        authorization = authorize_live_campaign(
            plan,
            live_requested=args.live,
            confirmation_token=args.confirm,
        )
    except ValueError as error:
        parser.error(str(error))

    authorization_path = campaign_root / "authorization.json"
    _write_json_atomic(
        authorization_path,
        {
            "campaign_id": plan.campaign_id,
            "authorized": True,
            "authorization_boundary": "campaign",
            "confirmation_token_persisted": False,
        },
    )

    with AudioSocketDispatcher() as dispatcher:
        executor = SubprocessCampaignCaseExecutor(
            authorization=authorization,
            dispatcher=dispatcher,
            per_call_budget_usd=format(per_call_worst_case, "f"),
            max_rate_per_minute_usd=format(max_rate, "f"),
            log_root=Path("artifacts/campaigns"),
        )
        result = run_campaign(authorization.plan, executor)

    result_path = campaign_root / "result.json"
    _write_json_atomic(
        result_path,
        {
            "campaign_id": result.campaign_id,
            "evaluation_pack_id": selected_pack_id,
            "completed_count": result.completed_count,
            "failed_count": result.failed_count,
            "entries": [asdict(entry) for entry in result.entries],
        },
    )

    print(
        json.dumps(
            {
                "campaign_id": result.campaign_id,
                "manifest_path": str(manifest_path),
                "authorization_path": str(authorization_path),
                "result_path": str(result_path),
                "evaluation_pack_id": selected_pack_id,
                "call_count": plan.call_count,
                "parallel": plan.max_parallel_calls,
                "completed_count": result.completed_count,
                "failed_count": result.failed_count,
                "campaign_worst_case_usd": str(campaign_worst_case),
            },
            indent=2,
        )
    )

    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
