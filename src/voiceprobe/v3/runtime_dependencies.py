"""Deterministic dependency preflight for the production v3 media runtime."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata

SUPPORTED_PIPECAT_VERSION = "1.7.0"
REQUIRED_PIPECAT_IMPORTS = (
    "pipecat.frames.frames",
    "pipecat.pipeline.pipeline",
    "pipecat.pipeline.worker",
    "pipecat.processors.frame_processor",
    "pipecat.services.deepgram.flux.stt",
    "pipecat.workers.runner",
)


@dataclass(frozen=True, slots=True)
class V3RuntimeDependencyStatus:
    """Safe evidence that the installed v3 runtime passed its import contract."""

    pipecat_version: str
    ready: bool = True


def preflight_v3_runtime_dependencies() -> V3RuntimeDependencyStatus:
    """Fail closed unless the supported Pipecat distribution is importable."""

    try:
        pipecat_version = metadata.version("pipecat-ai")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Production v3 dependency preflight failed: pipecat-ai is not installed."
        ) from error
    except Exception as error:
        raise RuntimeError(
            "Production v3 dependency preflight failed while reading pipecat-ai "
            f"metadata ({type(error).__name__})."
        ) from error

    if pipecat_version != SUPPORTED_PIPECAT_VERSION:
        raise RuntimeError(
            "Production v3 dependency preflight failed: unsupported pipecat-ai "
            f"version {pipecat_version!r}; required {SUPPORTED_PIPECAT_VERSION!r}."
        )

    for module_name in REQUIRED_PIPECAT_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            raise RuntimeError(
                "Production v3 dependency preflight failed: required import "
                f"{module_name!r} raised {type(error).__name__}."
            ) from error

    return V3RuntimeDependencyStatus(pipecat_version=pipecat_version)
