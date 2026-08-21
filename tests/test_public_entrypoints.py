from pathlib import Path

from voiceprobe.cli import app


def test_console_script_targets_supported_cli() -> None:
    project = Path(__file__).parents[1] / "pyproject.toml"
    text = project.read_text(encoding="utf-8")

    assert 'voiceprobe = "voiceprobe.cli:app"' in text
    assert callable(app)
