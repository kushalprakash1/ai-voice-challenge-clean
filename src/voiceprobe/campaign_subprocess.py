"""Process-isolated campaign executor for production VoiceProbe calls.

Each campaign case launches the ordinary single-call production chain in a
separate Python process. This prevents environment variables, patient state,
Flux sessions, TTS state, and failures from leaking between concurrent calls.
"""

from __future__ import annotations

import json
import math
import os
import re

# Subprocess provides process isolation; execution uses list-form argv, never a shell.
import subprocess  # nosec B404
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from voiceprobe.campaign import (
    CAMPAIGN_MEDIA_MODE_V3,
    AuthorizedCampaign,
    CampaignCaseRequest,
    CampaignCaseResult,
    CampaignCaseStatus,
    CampaignExecutionError,
)
from voiceprobe.campaign_evidence import lifecycle_path
from voiceprobe.execution import LIVE_CONFIRMATION_TOKEN
from voiceprobe.run_campaign_case import CASE_RESULT_PREFIX
from voiceprobe.safety import validate_destination
from voiceprobe.telephony.audiosocket_dispatcher import (
    AudioSocketDispatcher,
    validate_worker_port,
)
from voiceprobe.v3.runtime_dependencies import SUPPORTED_PIPECAT_VERSION

DEFAULT_CAMPAIGN_WORKER_PORT_BASE = 9200
WORKER_TEARDOWN_GRACE_SECONDS = 90

_ProcessFactory = Callable[..., subprocess.Popen[str]]
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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
        process_factory: _ProcessFactory = subprocess.Popen,
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
        self._process_factory = process_factory
        self._process_lock = threading.Lock()
        self._active_pids: set[int] = set()
        self._maximum_active = 0

    @property
    def maximum_simultaneous_active_workers(self) -> int:
        with self._process_lock:
            return self._maximum_active

    @property
    def active_worker_count(self) -> int:
        with self._process_lock:
            return len(self._active_pids)

    def execute_case(self, request: CampaignCaseRequest) -> CampaignCaseResult:
        """Run one authorized case exactly once with a hard process timeout."""

        self._validate_request(request)

        worker_port = validate_worker_port(self.worker_port_base + request.position - 1)
        call_id = uuid5(
            NAMESPACE_URL,
            f"voiceprobe:{request.campaign_id}:{request.case_id}:{request.position}",
        )
        execution_id = self._execution_id(request)
        worker_lifecycle_path = self._lifecycle_path(request)

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
            "--media-mode",
            request.selected_media_mode,
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

        timeout_seconds = request.max_duration_seconds + WORKER_TEARDOWN_GRACE_SECONDS
        stdout = ""
        stderr = ""
        process: subprocess.Popen[str] | None = None
        timed_out = False
        launch_error: OSError | None = None
        try:
            child_environment = self._worker_environment()
            process = self._process_factory(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=child_environment,
            )
            if (
                isinstance(process.pid, bool)
                or not isinstance(process.pid, int)
                or process.pid <= 0
            ):
                raise OSError("campaign worker did not expose a positive PID")
            self._mark_process_started(process.pid)
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                stdout, stderr = process.communicate()
        except OSError as error:
            launch_error = error
            stderr = f"{type(error).__name__}: {error}\n"
        finally:
            if process is not None and isinstance(process.pid, int):
                self._mark_process_stopped(process.pid)
            self.dispatcher.unregister(call_id)

        stdout = stdout or ""
        stderr = stderr or ""

        self._write_logs(
            request=request,
            stdout=stdout,
            stderr=stderr,
        )

        payload = self._parse_result(stdout)
        exit_code = process.returncode if process is not None else None
        known_pid = process.pid if process is not None else None
        evidence, validation_errors = self._validated_evidence(
            lifecycle_path=worker_lifecycle_path,
            payload=payload,
            execution_id=execution_id,
            call_id=call_id,
            worker_port=worker_port,
            worker_pid=known_pid,
            max_duration_seconds=request.max_duration_seconds,
        )
        parent_evidence = {
            "selected_media_mode": request.selected_media_mode,
            "worker_pid": known_pid,
            "worker_port": worker_port,
            "call_uuid": str(call_id),
            "route_registered": True,
            "route_consumed": self.dispatcher.was_consumed(call_id),
            "worker_connection_established": self.dispatcher.was_connected(call_id),
            "uuid_forwarded": self.dispatcher.was_forwarded(call_id),
            "route_released": not self.dispatcher.is_registered(call_id),
            "subprocess_exit_code": exit_code,
            "timed_out": timed_out,
            **evidence,
        }

        if launch_error is not None:
            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id=execution_id,
                **parent_evidence,
                error=f"Unable to launch campaign worker: {type(launch_error).__name__}: {launch_error}",
            )

        if timed_out:
            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id=execution_id,
                **parent_evidence,
                error=f"Campaign worker exceeded its hard process timeout; timeout_seconds={timeout_seconds}",
            )

        if payload is None:
            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id=execution_id,
                **parent_evidence,
                error=(
                    "Campaign worker exited without structured terminal evidence; "
                    f"returncode={exit_code}"
                ),
            )

        validation_error = "; ".join(validation_errors) or None
        if validation_error is not None:
            return CampaignCaseResult(
                position=request.position,
                case_id=request.case_id,
                scenario_id=request.scenario_id,
                status=CampaignCaseStatus.FAILED,
                execution_id=execution_id,
                evidence_validation_error=validation_error,
                **parent_evidence,
                error=f"Child evidence validation failed: {validation_error}",
            )

        status_text = str(payload.get("status", "failed"))
        status = (
            CampaignCaseStatus.COMPLETED
            if status_text == CampaignCaseStatus.COMPLETED.value and exit_code == 0
            else CampaignCaseStatus.FAILED
        )

        error = payload.get("error")

        if status is CampaignCaseStatus.FAILED and not error:
            error = f"campaign worker returncode={exit_code}"

        return CampaignCaseResult(
            position=request.position,
            case_id=request.case_id,
            scenario_id=request.scenario_id,
            status=status,
            execution_id=execution_id,
            telemetry_error=str(payload["telemetry_error"])
            if payload.get("telemetry_error")
            else None,
            lifecycle_error=str(payload["lifecycle_error"])
            if payload.get("lifecycle_error")
            else None,
            **parent_evidence,
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

        if request.selected_media_mode != plan.media_mode:
            raise CampaignExecutionError(
                "Campaign request media mode does not match authorization."
            )

        if request.selected_media_mode != CAMPAIGN_MEDIA_MODE_V3:
            raise CampaignExecutionError(
                "Live scalable campaign workers require media mode v3."
            )

    @staticmethod
    def _execution_id(request: CampaignCaseRequest) -> str:
        # Execution IDs are intentionally short and independent of long
        # scenario names so they stay inside execution.py's 64-character cap.
        campaign_slug = request.campaign_id[:48].rstrip("-_")
        return f"{campaign_slug}-c{request.position:03d}"

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        environment = os.environ.copy()
        environment["VOICEPROBE_V3_LIVE"] = "1"
        return environment

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

    def _lifecycle_path(self, request: CampaignCaseRequest) -> Path:
        expected = lifecycle_path(
            campaign_id=request.campaign_id,
            position=request.position,
            case_id=request.case_id,
        )
        relative = expected.relative_to(Path("artifacts/campaigns"))
        return self.log_root / relative

    def _mark_process_started(self, pid: int) -> None:
        with self._process_lock:
            self._active_pids.add(pid)
            self._maximum_active = max(self._maximum_active, len(self._active_pids))

    def _mark_process_stopped(self, pid: int) -> None:
        with self._process_lock:
            self._active_pids.discard(pid)

    def _validated_evidence(
        self,
        *,
        lifecycle_path: Path,
        payload: dict[str, object] | None,
        execution_id: str,
        call_id: UUID,
        worker_port: int,
        worker_pid: int | None,
        max_duration_seconds: int,
    ) -> tuple[dict[str, object], list[str]]:
        errors: list[str] = []
        lifecycle: dict[str, object] = {}
        try:
            decoded = json.loads(lifecycle_path.read_text())
            if isinstance(decoded, dict):
                lifecycle = decoded
            else:
                errors.append("lifecycle JSON must be an object")
        except json.JSONDecodeError:
            errors.append("malformed lifecycle JSON")
        except OSError:
            errors.append("lifecycle evidence is missing or unreadable")

        if payload is not None:
            self._expect_equal(payload, "execution_id", execution_id, errors)
            self._expect_equal(payload, "call_id", str(call_id), errors)
            self._expect_equal(payload, "worker_port", worker_port, errors)
            self._expect_equal(
                payload, "selected_media_mode", CAMPAIGN_MEDIA_MODE_V3, errors
            )
            self._expect_equal(
                payload, "pipecat_version", SUPPORTED_PIPECAT_VERSION, errors
            )
            self._expect_equal(
                payload, "v3_runtime_dependencies_ready", True, errors
            )

        if lifecycle:
            self._expect_equal(lifecycle, "execution_id", execution_id, errors)
            self._expect_equal(lifecycle, "call_uuid", str(call_id), errors)
            self._expect_equal(lifecycle, "worker_port", worker_port, errors)
            self._expect_equal(
                lifecycle, "selected_media_mode", CAMPAIGN_MEDIA_MODE_V3, errors
            )
            self._expect_equal(
                lifecycle, "pipecat_version", SUPPORTED_PIPECAT_VERSION, errors
            )
            self._expect_equal(
                lifecycle, "v3_runtime_dependencies_ready", True, errors
            )
            lifecycle_pid = lifecycle.get("worker_pid")
            if (
                isinstance(lifecycle_pid, bool)
                or not isinstance(lifecycle_pid, int)
                or lifecycle_pid <= 0
            ):
                errors.append("lifecycle worker_pid must be a positive integer")
            elif worker_pid is not None and lifecycle_pid != worker_pid:
                errors.append("lifecycle worker_pid does not match launched PID")
            for key in ("worker_started_at", "worker_ended_at"):
                if key in lifecycle and not self._valid_utc_timestamp(lifecycle[key]):
                    errors.append(f"lifecycle {key} is not a UTC timestamp")

        artifact_run_id = payload.get("artifact_run_id") if payload else None
        if artifact_run_id is not None and (
            not isinstance(artifact_run_id, str)
            or not _ARTIFACT_ID.fullmatch(artifact_run_id)
        ):
            errors.append("artifact_run_id is not a safe identifier")
            artifact_run_id = None

        duration = self._finite_nonnegative(
            payload.get("duration_seconds") if payload else None
        )
        if (
            payload is not None
            and payload.get("duration_seconds") is not None
            and (
                duration is None
                or duration > max_duration_seconds + WORKER_TEARDOWN_GRACE_SECONDS
            )
        ):
            errors.append("duration_seconds is invalid or outside the sane bound")
            duration = None

        cpu_user = self._metric(lifecycle, "cpu_user_seconds", errors)
        cpu_system = self._metric(lifecycle, "cpu_system_seconds", errors)
        max_rss = lifecycle.get("max_rss")
        if max_rss is not None and (
            isinstance(max_rss, bool)
            or not isinstance(max_rss, (int, float))
            or not math.isfinite(float(max_rss))
            or float(max_rss) < 0
        ):
            errors.append("lifecycle max_rss must be finite and non-negative")
            max_rss = None

        return (
            {
                "artifact_run_id": artifact_run_id,
                "call_duration_seconds": duration,
                "cpu_user_seconds": cpu_user,
                "cpu_system_seconds": cpu_system,
                "max_rss": int(max_rss) if max_rss is not None else None,
                "max_rss_unit": str(lifecycle.get("max_rss_unit"))
                if lifecycle.get("max_rss_unit")
                else None,
                "pipecat_version": str(lifecycle.get("pipecat_version"))
                if lifecycle.get("pipecat_version")
                else None,
                "v3_runtime_dependencies_ready": (
                    lifecycle.get("v3_runtime_dependencies_ready") is True
                ),
            },
            errors,
        )

    @staticmethod
    def _expect_equal(
        payload: dict[str, object], key: str, expected: object, errors: list[str]
    ) -> None:
        if payload.get(key) != expected:
            errors.append(f"{key} does not match parent evidence")

    @staticmethod
    def _finite_nonnegative(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None

    @classmethod
    def _metric(
        cls, payload: dict[str, object], key: str, errors: list[str]
    ) -> float | None:
        value = payload.get(key)
        if value is None:
            return None
        result = cls._finite_nonnegative(value)
        if result is None:
            errors.append(f"lifecycle {key} must be finite and non-negative")
        return result

    @staticmethod
    def _valid_utc_timestamp(value: object) -> bool:
        if not isinstance(value, str):
            return False
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return False
        return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed)

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
