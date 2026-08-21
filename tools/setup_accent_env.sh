#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$REPO_ROOT/tools/accent-requirements-linux-cpu.txt"

ACCENT_VENV="${VOICEPROBE_ACCENT_VENV:-$HOME/.venvs/voiceprobe-accent}"

TORCH_VERSION="2.13.0+cpu"
TORCHAUDIO_VERSION="2.11.0+cpu"
TORCHVISION_VERSION="0.28.0+cpu"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$LOCK_FILE" ]]; then
  echo "ERROR: accent lock file is missing: $LOCK_FILE" >&2
  exit 1
fi

echo "Creating isolated VoiceProbe EN_INDIA environment:"
echo "  $ACCENT_VENV"

uv venv --python 3.11 "$ACCENT_VENV"

# The validated environment uses CPU-specific PyTorch wheels. Install those
# separately because the +cpu builds come from PyTorch's wheel index.
uv pip install \
  --python "$ACCENT_VENV/bin/python" \
  --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==$TORCH_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  "torchvision==$TORCHVISION_VERSION"

# The checked-in lock represents the complete known-good environment.
#
# MeloTTS itself pins some older dependencies upstream, while VoiceProbe's
# validated environment intentionally contains newer versions. Therefore we
# install the frozen package set exactly rather than re-resolving Melo's
# historical dependency graph.
TEMP_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$TEMP_REQUIREMENTS"' EXIT

grep -Ev \
  '^(torch|torchaudio|torchvision)==' \
  "$LOCK_FILE" > "$TEMP_REQUIREMENTS"

uv pip install \
  --python "$ACCENT_VENV/bin/python" \
  --no-deps \
  -r "$TEMP_REQUIREMENTS"

# Melo imports Japanese text support even when VoiceProbe uses EN_INDIA.
# The unidic Python package does not include its full dictionary payload,
# so populate it explicitly before importing Melo.
echo "Downloading UniDic dictionary..."
"$ACCENT_VENV/bin/python" -m unidic download

"$ACCENT_VENV/bin/python" - <<'PY'
from importlib import metadata
import sys

import numpy
import scipy
import soundfile
import torch
import transformers
from melo.api import TTS

expected = {
    "melotts": "0.1.2",
    "numpy": "1.26.4",
    "scipy": "1.17.1",
    "soundfile": "0.14.0",
    "torch": "2.13.0+cpu",
    "torchaudio": "2.11.0+cpu",
    "torchvision": "0.28.0+cpu",
    "transformers": "5.2.0",
}

errors = []

for package, wanted in expected.items():
    actual = metadata.version(package)
    print(f"{package}={actual}")
    if actual != wanted:
        errors.append(f"{package}: expected {wanted}, got {actual}")

if sys.version_info[:2] != (3, 11):
    errors.append(
        f"python: expected 3.11, got "
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )

if errors:
    raise SystemExit(
        "ACCENT_ENV_VALIDATION=FAIL\n" + "\n".join(errors)
    )

print("ACCENT_ENV_IMPORTS=PASS")
print("MELO_TTS_IMPORT=PASS")
print("ACCENT_ENV_VALIDATION=PASS")
PY
