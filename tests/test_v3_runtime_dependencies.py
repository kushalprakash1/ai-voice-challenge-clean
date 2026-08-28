from __future__ import annotations

from importlib import metadata

import pytest

from voiceprobe.v3 import runtime_dependencies


def test_preflight_fails_when_pipecat_distribution_is_missing(monkeypatch) -> None:
    def missing_version(distribution: str) -> str:
        raise metadata.PackageNotFoundError(distribution)

    imported: list[str] = []
    monkeypatch.setattr(runtime_dependencies.metadata, "version", missing_version)
    monkeypatch.setattr(
        runtime_dependencies.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(RuntimeError, match="pipecat-ai is not installed"):
        runtime_dependencies.preflight_v3_runtime_dependencies()

    assert imported == []


def test_preflight_fails_closed_on_wrong_pipecat_version(monkeypatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr(runtime_dependencies.metadata, "version", lambda _: "1.7.1")
    monkeypatch.setattr(
        runtime_dependencies.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    with pytest.raises(RuntimeError, match="unsupported pipecat-ai version '1.7.1'"):
        runtime_dependencies.preflight_v3_runtime_dependencies()

    assert imported == []


def test_preflight_imports_every_required_module(monkeypatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr(runtime_dependencies.metadata, "version", lambda _: "1.7.0")
    monkeypatch.setattr(
        runtime_dependencies.importlib,
        "import_module",
        lambda name: imported.append(name),
    )

    status = runtime_dependencies.preflight_v3_runtime_dependencies()

    assert status.pipecat_version == "1.7.0"
    assert status.ready is True
    assert imported == list(runtime_dependencies.REQUIRED_PIPECAT_IMPORTS)


def test_preflight_error_and_safe_evidence_do_not_leak_secret(
    monkeypatch, capsys
) -> None:
    secret = "synthetic-deepgram-secret-never-record"
    monkeypatch.setenv("DEEPGRAM_API_KEY", secret)
    monkeypatch.setattr(runtime_dependencies.metadata, "version", lambda _: "1.7.0")

    def broken_import(module_name: str) -> None:
        raise ImportError(f"cannot import {module_name}")

    monkeypatch.setattr(runtime_dependencies.importlib, "import_module", broken_import)

    with pytest.raises(RuntimeError) as raised:
        runtime_dependencies.preflight_v3_runtime_dependencies()

    assert secret not in str(raised.value)
    assert secret not in capsys.readouterr().out
