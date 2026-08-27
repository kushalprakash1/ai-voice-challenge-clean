"""Process-isolated campaign executor for production VoiceProbe calls.

Each campaign case launches the ordinary single-call production chain in a
separate Python process. This prevents environment variables, patient state,
Flux sessions, TTS state, and failures from leaking between concurrent calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from voiceprobe.campaign import (
    AuthorizedCampaign,
    CampaignCaseRequest,
    CampaignCaseResult,
    CampaignCaseStatus,
    CampaignExecutionError,
)
from voiceprobe.execution import LIVE_CONFIRMATION_TOKEN
from voiceprobe.run_campaign_case import CASE_RESULT_PREFIX
from voiceprobe.safety import validate_destination
from voiceprobe.telephony.audiosocket_dispatcher import (
    AudioSocketDispatcher,
    validate_worker_port,
)

DEFAULT_CAMPAIGN_WORKER_PORT_BASE = 9200
WORKER_TEARDOWN_GRACE_SECONDS = 90

_ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class SubprocessCampaignCaseExecutor:
    """Execute one authorized campaign case in one isolated process."""

    def __init__(
        self,
        *,
        authorization: AuthorizedCampaign,
        dispatcher: AudioSocketDispatcher,
        per_call_budget_usd: str,
        max_rate_per_minute_usd: str,
        worker_port_base: int = DEFAULT_CAMPAIGN_WORKER_PORT_BASE,
        python_executable: str = sys.executable,
        log_root: Path | str = "artifacts/campaigns",
        process_runner: _ProcessRunner = subprocess.run,
    ) -> None:
        if not isinstance(authorization, AuthorizedCampaign):
            raise TypeError("authorization must be an AuthorizedCampaign.")

        if not isinstance(dispatcher, AudioSocketDispatcher):
            raise TypeError("dispatcher must be an AudioSocketDispatcher.")

        if not dispatcher.ready:
            raise CampaignExecutionError(
                "AudioSocket dispatcher must be ready before creating live workers."
            )

        if not python_executable.strip():
            raise CampaignExecutionError("python_executable cannot be blank.")

        validate_destination(authorization.plan.destination)

        self.authorization = authorization
        self.dispatcher = dispatcher
        self.per_call_budget_usd = per_call_budget_usd
        self.max_rate_per_minute_usd = max_rate_per_minute_usd
        self.worker_port_base = validate_worker_port(worker_port_base)
        self.python_executable = python_executable
        self.log_root = Path(log_root)
        self._process_runner = process_runner

    def execute_case(self, request: CampaignCaseRequest) -> CampaignCaseResult:
        """Run one authorized case exactly once with a hard process timeout."""

        self._validate_request(request)

        worker_port = validate_worker_port(
            self.worker_port_base + request.position - 1
        )
        call_id = uuid5(
            NAMESPACE_URL,
            f"voiceprobe:{request.campaign_id}:{request.case_id}:{request.position}",
        )
        execution_id = self._execution_id(request)

        self.dispatcher.register(call_id, worker_port)

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
            "--live",
            "--confirm",
            LIVE_CONFIRMATION_TOKEN,
        ]

        timeout_seconds = (
            request.max_duration_seconds + WORKER_TEARDOWN_GRACE_SECONDS
        )
        stdout = ""
        stderr = ""

        try:
            completed = self._process_runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as error:
            stdout = self._coerce_process_text(error.stdout)
            stderr = self._coerce_process_text(error.stderr)
            self._write_logs(
                request=request,
                stdout=stdout,
                stderr=stderr,
            )
            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id=execution_id,
                error=(
                    "Campaign worker exceeded its hard process timeout; "
                    f"timeout_seconds={timeout_seconds}"
                ),
            )
        except OSError as error:
            stderr = f"{type(error).__name__}: {error}\n"
            self._write_logs(
                request=request,
                stdout=stdout,
                stderr=stderr,
            )
            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id=execution_id,
                error=f"Unable to launch campaign worker: {type(error).__name__}: {error}",
            )
        finally:
            # Harmless if the UUID was already consumed by a real session.
            self.dispatcher.unregister(call_id)

        self._write_logs(
            request=request,
            stdout=stdout,
            stderr=stderr,
        )

        payload = self._parse_result(stdout)

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

    def _validate_request(self, request: CampaignCaseRequest) -> None:
        plan = self.authorization.plan

        validate_destination(request.destination)

        if request.campaign_id != plan.campaign_id:
            raise CampaignExecutionError(
                "Campaign request ID does not match live authorization."
            )

        if request.originating_number != plan.originating_number:
            raise CampaignExecutionError(
                "Campaign request originating number does not match authorization."
            )

        if request.destination != plan.destination:
            raise CampaignExecutionError(
                "Campaign request destination does not match authorization."
            )

        if request.max_duration_seconds != plan.max_call_duration_seconds:
            raise CampaignExecutionError(
                "Campaign request duration does not match authorization."
            )

        if not 1 <= request.position <= plan.call_count:
            raise CampaignExecutionError(
                "Campaign request position is outside the authorized plan."
            )

        expected = plan.cases[request.position - 1]

        if request.case_id != expected.case_id:
            raise CampaignExecutionError(
                "Campaign request case ID does not match authorization."
            )

        if request.scenario_id != expected.scenario_id:
            raise CampaignExecutionError(
                "Campaign request scenario does not match authorization."
            )

        if request.evaluation_focus != expected.evaluation_focus:
            raise CampaignExecutionError(
                "Campaign request evaluation focus does not match authorization."
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

        (
            case_root / f"{request.position:03d}-{request.case_id}.stdout.log"
        ).write_text(stdout)
        (
            case_root / f"{request.position:03d}-{request.case_id}.stderr.log"
        ).write_text(stderr)

    @staticmethod
    def _coerce_process_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

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
