"""Execute one isolated VoiceProbe campaign case.

This worker intentionally reuses the original production execution chain:
scenario catalog -> CallPolicy -> suite plan -> immutable execution manifest ->
explicit live authorization -> persistent budget/call ledgers -> production
Asterisk adapter -> existing v2/v3 media runtime.

The campaign supplies only an execution ID, call UUID, scenario ID, and an
isolated loopback AudioSocket worker port. Destination authorization and all
other call safety checks are repeated inside this process.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from voiceprobe.config import Settings
from voiceprobe.execution import (
    authorize_live_execution,
    prepare_execution,
    write_execution_manifest,
)
from voiceprobe.execution_state import (
    BudgetPolicy,
    PersistentBudgetLedger,
    PersistentCallLedger,
)
from voiceprobe.policy import (
    DEFAULT_MAX_CALL_DURATION_SECONDS,
    MAX_CALL_DURATION_SECONDS,
)
from voiceprobe.run_one import (
    DEFAULT_AMI_ENV,
    DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD,
    load_ami_config,
)
from voiceprobe.runner import run_persistent_authorized_suite
from voiceprobe.safety import require_live_destination
from voiceprobe.scenarios.catalog import get_scenario, list_scenarios
from voiceprobe.suite import build_suite_plan
from voiceprobe.telephony.asterisk_adapter import AsteriskAssessmentCallAdapter
from voiceprobe.telephony.audiosocket_dispatcher import validate_worker_port

CASE_RESULT_PREFIX = "VOICEPROBE_CAMPAIGN_CASE_RESULT="


def _result_line(payload: dict[str, object]) -> str:
    return CASE_RESULT_PREFIX + json.dumps(payload, sort_keys=True, default=str)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute exactly one isolated VoiceProbe campaign case.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=tuple(scenario.scenario_id for scenario in list_scenarios()),
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--position", required=True, type=int)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--worker-port", required=True, type=int)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--max-call-duration-seconds",
        type=int,
        default=DEFAULT_MAX_CALL_DURATION_SECONDS,
    )
    parser.add_argument("--budget-usd", default="5.00")
    parser.add_argument(
        "--max-rate-per-minute-usd",
        default=format(DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD, "f"),
    )
    parser.add_argument(
        "--ami-env",
        type=Path,
        default=DEFAULT_AMI_ENV,
    )
    args = parser.parse_args()

    if args.position < 1:
        parser.error("--position must be a positive integer")

    worker_port = validate_worker_port(args.worker_port)

    try:
        call_id = UUID(args.call_id)
    except ValueError as error:
        parser.error(f"--call-id must be a UUID: {error}")

    if not 1 <= args.max_call_duration_seconds <= MAX_CALL_DURATION_SECONDS:
        parser.error(
            "--max-call-duration-seconds must be between 1 and "
            f"{MAX_CALL_DURATION_SECONDS}"
        )

    # Campaign workers are processes, so scenario/runtime environment cannot
    # leak between concurrent calls. Explicitly clear run_one-only monitoring
    # behavior because multiple local ffplay monitors are not part of the
    # campaign's call-critical path.
    os.environ["VOICEPROBE_SCENARIO"] = args.scenario
    os.environ["VOICEPROBE_LIVE_MONITOR"] = "0"

    settings = Settings()  # type: ignore[call-arg]
    base_policy = settings.call_policy()
    policy = replace(
        base_policy,
        dry_run=not args.live,
        max_call_duration_seconds=args.max_call_duration_seconds,
    )

    scenario = get_scenario(args.scenario)
    suite = build_suite_plan(policy, scenarios=(scenario,))
    manifest = prepare_execution(
        policy,
        suite,
        execution_id=args.execution_id,
    )

    manifest_path = write_execution_manifest(manifest)
    execution_dir = manifest_path.parent

    if not args.live:
        print(
            _result_line(
                {
                    "campaign_id": args.campaign_id,
                    "case_id": args.case_id,
                    "position": args.position,
                    "scenario_id": args.scenario,
                    "execution_id": manifest.execution_id,
                    "manifest_path": str(manifest_path),
                    "worker_port": worker_port,
                    "call_id": str(call_id),
                    "dry_run": True,
                    "status": "completed",
                    "artifact_run_id": None,
                }
            )
        )
        return 0

    require_live_destination()

    authorization = authorize_live_execution(
        manifest,
        live_requested=True,
        confirmation_token=args.confirm,
    )

    budget_policy = BudgetPolicy(
        total_budget_usd=Decimal(args.budget_usd),
        max_provider_rate_per_minute_usd=Decimal(args.max_rate_per_minute_usd),
    )
    call_ledger = PersistentCallLedger.initialize(
        authorization,
        path=execution_dir / "calls.json",
    )
    budget_ledger = PersistentBudgetLedger.initialize(
        execution_id=manifest.execution_id,
        policy=budget_policy,
        path=execution_dir / "budget.json",
    )

    ami_config = load_ami_config(args.ami_env)
    adapter = AsteriskAssessmentCallAdapter(
        ami_config=ami_config,
        expected_originating_number=manifest.originating_number,
        port=worker_port,
        call_id_factory=lambda: call_id,
    )

    try:
        result = run_persistent_authorized_suite(
            authorization,
            adapter,
            call_ledger=call_ledger,
            budget_ledger=budget_ledger,
        )
    finally:
        adapter.close()

    entry = result.entries[0]
    status = "completed" if result.failed_count == 0 else "failed"
    payload = {
        "campaign_id": args.campaign_id,
        "case_id": args.case_id,
        "position": args.position,
        "scenario_id": args.scenario,
        "execution_id": manifest.execution_id,
        "manifest_path": str(manifest_path),
        "worker_port": worker_port,
        "call_id": str(call_id),
        "status": status,
        "artifact_run_id": entry.artifact_run_id,
        "provider_call_id": entry.provider_call_id,
        "duration_seconds": entry.duration_seconds,
        "error": entry.error,
        "entry": asdict(entry),
    }
    print(_result_line(payload))

    return 1 if result.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
