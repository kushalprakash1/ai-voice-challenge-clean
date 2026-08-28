"""Explicit one-call production entrypoint for VoiceProbe testing.

This module can authorize exactly one immutable patient scenario. It cannot
reuse or execute a multi-call suite manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values

from voiceprobe.autonomous_phone import DEFAULT_VOICE
from voiceprobe.config import Settings
from voiceprobe.execution import (
    ExecutionManifest,
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
from voiceprobe.runner import SuiteRunResult, run_persistent_authorized_suite
from voiceprobe.safety import require_live_destination
from voiceprobe.scenarios.catalog import get_scenario, list_scenarios
from voiceprobe.scenarios.models import PatientScenario
from voiceprobe.suite import build_suite_plan
from voiceprobe.telephony.ami import AsteriskAMIConfig
from voiceprobe.telephony.asterisk_adapter import AsteriskAssessmentCallAdapter
from voiceprobe.v3.accent import accent_mode_from_environment
from voiceprobe.v3.background import background_mode_from_environment
from voiceprobe.v3.personas import (
    ENV_PERSONA,
    ENV_PERSONA_SEED,
    ENV_PERSONA_SEQUENCE,
    list_personas,
    sequence_ids_for,
)
from voiceprobe.v3.production import (
    DEFAULT_PRODUCTION_FLUX_CONFIG,
    resolve_runtime_owner,
)
from voiceprobe.v3.turn_stabilizer import DEFAULT_CONTINUATION_GRACE_MS

DEFAULT_AMI_ENV = Path.home() / ".config/voiceprobe/ami.env"
DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD = Decimal("0.10")


@dataclass(frozen=True, slots=True)
class PreparedOneCall:
    """Behavior-owned input and immutable manifest for exactly one call."""

    scenario: PatientScenario
    manifest: ExecutionManifest


@dataclass(frozen=True, slots=True)
class OneCallTransportOverrides:
    """Transport-only substitutions allowed for an isolated campaign child."""

    ami_config: AsteriskAMIConfig
    port: int | None = None
    call_id_factory: Callable[[], UUID] | None = None
    flux_connect_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class OneCallExecutionResult:
    """Shared one-call execution output for CLI and campaign wrappers."""

    prepared: PreparedOneCall
    manifest_path: Path
    suite_result: SuiteRunResult | None


@dataclass(frozen=True, slots=True)
class OneCallBehaviorContract:
    """Read-only fingerprint of behavior configuration owned below the CLI."""

    scenario_id: str
    patient_facts: dict[str, object]
    objective: str
    runtime_owner: str
    reasoning_mode: str
    semantic_features: tuple[tuple[str, str], ...]
    flux_config: dict[str, object]
    generic_coalescer_grace_ms: float
    continuation_grace_ms: float
    tts_backend: str
    tts_voice: str
    accent_mode: str
    background_mode: str


def describe_one_call_behavior(prepared: PreparedOneCall) -> OneCallBehaviorContract:
    """Describe configuration without constructing or changing patient behavior."""

    flux = asdict(DEFAULT_PRODUCTION_FLUX_CONFIG)
    semantic_keys = (
        "VOICEPROBE_V32_SEMANTIC",
        "VOICEPROBE_V3_QWEN_FALLBACK",
        "VOICEPROBE_V32_MODEL",
        "VOICEPROBE_V32_OLLAMA_ENDPOINT",
    )
    accent_mode = accent_mode_from_environment()
    return OneCallBehaviorContract(
        scenario_id=prepared.scenario.scenario_id,
        patient_facts=prepared.scenario.facts.model_dump(),
        objective=prepared.scenario.objective,
        runtime_owner=resolve_runtime_owner(prepared.scenario.scenario_id),
        reasoning_mode=(
            "v3_live"
            if os.getenv("VOICEPROBE_V3_LIVE", "").strip().casefold()
            in {"1", "true", "on", "yes"}
            else "legacy"
        ),
        semantic_features=tuple((key, os.getenv(key, "")) for key in semantic_keys),
        flux_config=flux,
        generic_coalescer_grace_ms=DEFAULT_CONTINUATION_GRACE_MS,
        continuation_grace_ms=float(flux["continuation_grace_ms"]),
        tts_backend="kokoro" if accent_mode == "none" else "accent_cache",
        tts_voice=DEFAULT_VOICE,
        accent_mode=accent_mode,
        background_mode=background_mode_from_environment(),
    )


def _required_env_value(
    values: dict[str, str | None],
    key: str,
) -> str:
    value = values.get(key)

    if value is None or not value.strip():
        raise ValueError(f"AMI configuration is missing {key!r}.")

    return value.strip()


def load_ami_config(
    path: Path = DEFAULT_AMI_ENV,
) -> AsteriskAMIConfig:
    """Load restricted localhost AMI credentials without printing secrets."""
    if not path.is_file():
        raise FileNotFoundError(f"AMI environment file does not exist: {path}")

    raw = dict(dotenv_values(path))

    username = _required_env_value(raw, "VOICEPROBE_AMI_USERNAME")
    secret = _required_env_value(raw, "VOICEPROBE_AMI_SECRET")
    host = _required_env_value(raw, "VOICEPROBE_AMI_HOST")
    port_text = _required_env_value(raw, "VOICEPROBE_AMI_PORT")

    return AsteriskAMIConfig(
        username=username,
        secret=secret,
        host=host,
        port=int(port_text),
    )


def prepare_one_call_contract(
    *,
    settings: Settings,
    scenario_id: str,
    live_requested: bool = False,
    max_call_duration_seconds: int | None = None,
    execution_id: str | None = None,
) -> PreparedOneCall:
    """Create a fresh execution manifest containing exactly one scenario.

    The CLI's explicit --live request controls whether the prepared manifest
    is live-capable. Authorization still independently requires the live flag
    and exact confirmation token before any dialing side effect is allowed.
    """
    if type(live_requested) is not bool:
        raise TypeError("live_requested must be a boolean.")

    base_policy = settings.call_policy()

    policy = replace(
        base_policy,
        dry_run=not live_requested,
        max_call_duration_seconds=(
            base_policy.max_call_duration_seconds
            if max_call_duration_seconds is None
            else max_call_duration_seconds
        ),
    )

    scenario = get_scenario(scenario_id)

    suite = build_suite_plan(
        policy,
        scenarios=(scenario,),
    )

    if suite.call_count != 1:
        raise RuntimeError("One-call entrypoint produced a suite with call_count != 1.")

    manifest = prepare_execution(
        policy,
        suite,
        execution_id=execution_id,
    )

    if manifest.scenario_ids != (scenario_id,):
        raise RuntimeError("One-call execution manifest contains unexpected scenarios.")

    return PreparedOneCall(scenario=scenario, manifest=manifest)


def prepare_one_call(
    *,
    settings: Settings,
    scenario_id: str,
    live_requested: bool = False,
    max_call_duration_seconds: int | None = None,
):
    """Backward-compatible manifest view of the shared preparation boundary."""

    return prepare_one_call_contract(
        settings=settings,
        scenario_id=scenario_id,
        live_requested=live_requested,
        max_call_duration_seconds=max_call_duration_seconds,
    ).manifest


def execute_one_call(
    prepared: PreparedOneCall,
    *,
    live_requested: bool,
    confirmation_token: str,
    budget_usd: Decimal,
    max_rate_per_minute_usd: Decimal,
    transport: OneCallTransportOverrides | None = None,
) -> OneCallExecutionResult:
    """Execute the sole production one-call contract.

    Campaign callers may replace transport identity and isolation details, but
    scenario, policy, suite construction, authorization, ledgers, adapter
    assembly, and runner invocation remain owned here.
    """

    manifest = prepared.manifest
    manifest_path = write_execution_manifest(manifest)
    execution_dir = manifest_path.parent

    if not live_requested:
        return OneCallExecutionResult(prepared, manifest_path, None)

    require_live_destination()
    authorization = authorize_live_execution(
        manifest,
        live_requested=True,
        confirmation_token=confirmation_token,
    )
    budget_policy = BudgetPolicy(
        total_budget_usd=budget_usd,
        max_provider_rate_per_minute_usd=max_rate_per_minute_usd,
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
    resolved_transport = transport or OneCallTransportOverrides(
        ami_config=load_ami_config()
    )
    adapter_kwargs: dict[str, object] = {
        "ami_config": resolved_transport.ami_config,
        "expected_originating_number": manifest.originating_number,
    }
    if resolved_transport.port is not None:
        adapter_kwargs["port"] = resolved_transport.port
    if resolved_transport.call_id_factory is not None:
        adapter_kwargs["call_id_factory"] = resolved_transport.call_id_factory
    if resolved_transport.flux_connect_timeout_seconds is not None:
        adapter_kwargs["flux_connect_timeout_seconds"] = (
            resolved_transport.flux_connect_timeout_seconds
        )
    adapter = AsteriskAssessmentCallAdapter(**adapter_kwargs)

    try:
        result = run_persistent_authorized_suite(
            authorization,
            adapter,
            call_ledger=call_ledger,
            budget_ledger=budget_ledger,
        )
    finally:
        adapter.close()

    return OneCallExecutionResult(prepared, manifest_path, result)


def _execution_exit_code(failed_count: int) -> int:
    """Map suite outcome to a conventional CLI process status."""

    return 1 if failed_count > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute exactly one authorized VoiceProbe call.",
    )

    parser.add_argument(
        "--scenario",
        required=True,
        choices=tuple(scenario.scenario_id for scenario in list_scenarios()),
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly request live execution.",
    )

    parser.add_argument(
        "--live-monitor",
        action="store_true",
        help=(
            "Listen locally to both sides of the live call through ffplay. "
            "Monitoring uses one local mixed stream and is non-blocking; "
            "it never enters the call-critical media path."
        ),
    )

    parser.add_argument(
        "--persona",
        choices=tuple(
            persona.persona_id
            for persona in list_personas()
            if persona.persona_id != "control"
        ),
        default="",
        help="Optional adversarial patient persona.",
    )

    parser.add_argument(
        "--persona-sequence",
        default="",
        help="Explicit deterministic move sequence for --persona.",
    )

    parser.add_argument(
        "--persona-seed",
        type=int,
        default=6,
        help="Seed used when no explicit persona sequence is selected.",
    )

    parser.add_argument(
        "--confirm",
        default="",
        help="Exact live confirmation token.",
    )

    parser.add_argument(
        "--max-call-duration-seconds",
        type=int,
        default=DEFAULT_MAX_CALL_DURATION_SECONDS,
        help=(
            "Maximum duration for this one call in seconds. "
            f"Default is {DEFAULT_MAX_CALL_DURATION_SECONDS}; "
            f"hard maximum is {MAX_CALL_DURATION_SECONDS}."
        ),
    )

    parser.add_argument(
        "--budget-usd",
        default="5.00",
        help="Budget ceiling for this one-call execution.",
    )

    parser.add_argument(
        "--max-rate-per-minute-usd",
        default=format(
            DEFAULT_MAX_PROVIDER_RATE_PER_MINUTE_USD,
            "f",
        ),
        help=(
            "Conservative provider-rate ceiling used for reservation; "
            "default is $0.10/min."
        ),
    )

    args = parser.parse_args()

    if args.live_monitor and not args.live:
        parser.error("--live-monitor requires --live")

    if args.persona_sequence and not args.persona:
        parser.error("--persona-sequence requires --persona")

    if args.persona:
        available_sequences = sequence_ids_for(args.persona)

        if args.persona_sequence and args.persona_sequence not in available_sequences:
            parser.error(
                f"invalid --persona-sequence for {args.persona!r}; "
                f"choices are: {', '.join(available_sequences)}"
            )

    # Prevent stale shell variables from silently modifying a baseline call.
    os.environ[ENV_PERSONA] = args.persona
    os.environ[ENV_PERSONA_SEQUENCE] = args.persona_sequence
    os.environ[ENV_PERSONA_SEED] = str(args.persona_seed)

    # Keep monitoring explicitly opt-in for run_one. The actual media layer
    # reads this flag independently so the known-good adapter call signatures
    # and safety/budget plumbing remain unchanged.
    os.environ["VOICEPROBE_LIVE_MONITOR"] = "1" if args.live_monitor else "0"
    os.environ["VOICEPROBE_SCENARIO"] = args.scenario

    settings = Settings()  # type: ignore[call-arg]

    prepared = prepare_one_call_contract(
        settings=settings,
        scenario_id=args.scenario,
        live_requested=args.live,
        max_call_duration_seconds=(args.max_call_duration_seconds),
    )

    execution = execute_one_call(
        prepared,
        live_requested=args.live,
        confirmation_token=args.confirm,
        budget_usd=Decimal(args.budget_usd),
        max_rate_per_minute_usd=Decimal(args.max_rate_per_minute_usd),
    )
    manifest = prepared.manifest
    manifest_path = execution.manifest_path

    if not args.live:
        print(
            json.dumps(
                {
                    "execution_id": manifest.execution_id,
                    "manifest_path": str(manifest_path),
                    "call_count": manifest.call_count,
                    "scenario_id": args.scenario,
                    "dry_run": True,
                    "telephony_enabled": False,
                    "provider_call_invoked": False,
                },
                indent=2,
            )
        )
        return 0

    result = execution.suite_result
    if result is None:
        raise RuntimeError("Live one-call execution returned no suite result.")

    print(
        json.dumps(
            {
                "execution_id": result.execution_id,
                "manifest_path": str(manifest_path),
                "call_count": manifest.call_count,
                "scenario_id": args.scenario,
                "destination": manifest.destination,
                "entries": [asdict(entry) for entry in result.entries],
            },
            indent=2,
            default=str,
        )
    )

    return _execution_exit_code(result.failed_count)


if __name__ == "__main__":
    raise SystemExit(main())
