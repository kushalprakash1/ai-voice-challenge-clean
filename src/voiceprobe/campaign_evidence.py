"""Path-safe, atomically replaced campaign worker lifecycle evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path

CAMPAIGN_EVIDENCE_ROOT = Path("artifacts/campaigns")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,127}$")


class CampaignEvidenceError(ValueError):
    """Raised when campaign evidence cannot be safely addressed or persisted."""


def lifecycle_path(*, campaign_id: str, position: int, case_id: str) -> Path:
    """Derive the only permitted lifecycle path from validated identifiers."""

    if not _SAFE_ID.fullmatch(campaign_id):
        raise CampaignEvidenceError("campaign_id is not path-safe")
    if isinstance(position, bool) or not isinstance(position, int) or position < 1:
        raise CampaignEvidenceError("position must be a positive integer")
    if not _SAFE_ID.fullmatch(case_id):
        raise CampaignEvidenceError("case_id is not path-safe")

    return (
        CAMPAIGN_EVIDENCE_ROOT
        / campaign_id
        / "cases"
        / f"{position:03d}-{case_id}.lifecycle.json"
    )


def initialize_lifecycle(path: Path, payload: dict[str, object]) -> None:
    """Reserve a new lifecycle target, then atomically install initial JSON.

    Atomic rename prevents partial JSON visibility. This is consistency for
    concurrent readers, not power-loss durability; no fsync is attempted.
    """

    _ensure_confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_confined(path)
    try:
        with path.open("x"):
            pass
    except FileExistsError as error:
        raise CampaignEvidenceError(f"lifecycle evidence already exists: {path}") from error
    _replace_lifecycle(path, payload)


def update_lifecycle(path: Path, payload: dict[str, object]) -> None:
    """Atomically update an already-reserved lifecycle target."""

    _ensure_confined(path)
    if not path.is_file() or path.is_symlink():
        raise CampaignEvidenceError("lifecycle evidence was not initialized")
    _replace_lifecycle(path, payload)


def _replace_lifecycle(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except FileExistsError as error:
        raise CampaignEvidenceError(
            f"lifecycle temporary path already exists: {temporary}"
        ) from error


def _ensure_confined(path: Path) -> None:
    if path.is_absolute():
        raise CampaignEvidenceError("lifecycle path must be relative")
    try:
        path.relative_to(CAMPAIGN_EVIDENCE_ROOT)
    except ValueError as error:
        raise CampaignEvidenceError("lifecycle path escapes campaign evidence root") from error

    current = Path()
    for part in path.parent.parts:
        current /= part
        if current.exists() and current.is_symlink():
            raise CampaignEvidenceError(
                f"lifecycle path contains a symbolic link: {current}"
            )
