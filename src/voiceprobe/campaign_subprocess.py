"""Process-isolated campaign executor for production VoiceProbe calls.

Each campaign case launches the ordinary single-call production chain in a
separate Python process.  This prevents environment variables, patient state,
Flux sessions, TTS state, and failures from leaking between concurrent calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from voiceprobe.campaign import (
    CampaignCaseRequest,
    CampaignCaseResult,
    CampaignCaseStatus,
    CampaignExecutionError,
)
from voiceprobe.execution import LIVE_CONFIRMATION_TOKEN
from voiceprobe.run_campaign_case import CASE_RESULT_PREFIX
from voiceprobe.telephony.audiosocket_dispatcher import (
    AudioSocketDispatcher,
    validate_worker_port,
)

DEFAULT_CAMPAIGN_WORKER_PORT_BASE = 9200


class SubprocessCampaignCaseExecutor:
    """Execute one campaign case in one isolated VoiceProbe process."""

    def __init__(
        self,
        *,
        dispatcher: AudioSocketDispatcher | None,
        live: bool,
        per_call_budget_usd: str,
        max_rate_per_minute_usd: str,
        worker_port_base: int = DEFAULT_CAMPAIGN_WORKER_PORT_BASE,
        python_executable: str = sys.executable,
        log_root: Path | str = "artifacts/campaigns",
    ) -> None:
        if type(live) is not bool:
            raise TypeError("live must be a boolean.")
        if live and dispatcher is None:
            raise CampaignExecutionError(
                "Live campaign execution requires the shared AudioSocket dispatcher."
            )
        if not python_executable.strip():
            raise CampaignExecutionError("python_executable cannot be blank.")

        self.dispatcher = dispatcher
        self.live = live
        self.per_call_budget_usd = per_call_budget_usd
        self.max_rate_per_minute_usd = max_rate_per_minute_usd
        self.worker_port_base = validate_worker_port(worker_port_base)
        self.python_executable = python_executable
        self.log_root = Path(log_root)

    def execute_case(self, request: CampaignCaseRequest) -> CampaignCaseResult:
        worker_port = validate_worker_port(
            self.worker_port_base + request.position - 1
        )
        call_id = uuid5(
            NAMESPACE_URL,
            f"voiceprobe:{request.campaign_id}:{request.case_id}:{request.position}",
        )
        execution_id = self._execution_id(request)

        dispatcher = self.dispatcher
        registered = False

        if self.live:
            assert dispatcher is not None
            dispatcher.register(call_id, worker_port)
            registered = True

        command = [
            self.python_executable,
            "-m",
            "voiceprobe.run_campaign_case",
            "--scenario",
            request.scenario_id,
            "--campaign-id",
            request.campaign_id,
            "--case-id",
            request.case_id,
            "--position",
            str(request.position),
            "--execution-id",
            execution_id,
            "--call-id",
            str(call_id),
            "--worker-port",
            str(worker_port),
            "--max-call-duration-seconds",
            str(request.max_duration_seconds),
            "--budget-usd",
            self.per_call_budget_usd,
            "--max-rate-per-minute-usd",
            self.max_rate_per_minute_usd,
        ]

        if self.live:
            command.extend(
                [
                    "--live",
                    "--confirm",
                    LIVE_CONFIRMATION_TOKEN,
                ]
            )

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            if registered and dispatcher is not None:
                # Harmless if the UUID was already consumed by a real session.
                dispatcher.unregister(call_id)

        self._write_logs(
            request=request,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

        payload = self._parse_result(completed.stdout)

        if payload is None:
            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id=execution_id,
                error=(
                    "Campaign worker exited without structured terminal evidence; "
                    f"returncode={completed.returncode}"
                ),
            )

        status_text = str(payload.get("status", "failed"))
        status = (
            CampaignCaseStatus.COMPLETED
            if status_text == CampaignCaseStatus.COMPLETED.value
            and completed.returncode == 0
            else CampaignCaseStatus.FAILED
        )

        error = payload.get("error")
        if status is CampaignCaseStatus.FAILED and not error:
            error = f"campaign worker returncode={completed.returncode}"

        return CampaignCaseResult(
            position=request.position,
            case_id=request.case_id,
            scenario_id=request.scenario_id,
            status=status,
            execution_id=str(payload.get("execution_id") or execution_id),
            artifact_run_id=(
                str(payload["artifact_run_id"])
                if payload.get("artifact_run_id")
                else None
            ),
            error=str(error) if error else None,
        )

    @staticmethod
    def _execution_id(request: CampaignCaseRequest) -> str:
        # Execution IDs are intentionally short and independent of long
        # scenario names so they stay inside execution.py's 64-character cap.
        campaign_slug = request.campaign_id[:48].rstrip("-_")
        return f"{campaign_slug}-c{request.position:03d}"

    def _write_logs(
        self,
        *,
        request: CampaignCaseRequest,
        stdout: str,
        stderr: str,
    ) -> None:
        case_root = self.log_root / request.campaign_id / "cases"
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / f"{request.position:03d}-{request.case_id}.stdout.log").write_text(
            stdout
        )
        (case_root / f"{request.position:03d}-{request.case_id}.stderr.log").write_text(
            stderr
        )

    @staticmethod
    def _parse_result(stdout: str) -> dict[str, object] | None:
        for line in reversed(stdout.splitlines()):
            if not line.startswith(CASE_RESULT_PREFIX):
                continue
            raw = line[len(CASE_RESULT_PREFIX) :]
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
        return None
