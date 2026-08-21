#!/usr/bin/env python3
"""Build the bounded EN_INDIA VoiceProbe cache outside the live process."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from voiceprobe.v3.accent import (
    DEFAULT_CACHE_ROOT,
    MELO_SPEAKER,
    AccentCache,
    doctor_specialist_phrase_inventory,
    farthest_date_phrase_inventory,
    medication_refill_phrase_inventory,
    metadata_for_wav,
    office_location_insurance_phrase_inventory,
    write_metadata,
)

ACCENT_PYTHON = Path(
    os.environ.get(
        "VOICEPROBE_ACCENT_PYTHON",
        str(Path.home() / ".venvs/voiceprobe-accent/bin/python"),
    )
)
LISTENING_NAMES = {
    "My first name is Chitragupta.": "01_first_name.wav",
    "My last name is Subramnian Singh.": "02_last_name.wav",
    "I'm calling to request a medication refill.": "03_refill_request.wav",
    "Can you add lisinopril to my demo profile so I can continue with the refill request?": "04_add_lisinopril.wav",
    "Is there another way to add a medication to this demo profile?": "05_alternate_setup.wav",
    "The medication is lisinopril.": "06_medication_identity.wav",
    "Ten milligrams.": "07_ten_milligrams.wav",
    "It should be ten milligrams.": "08_dose_correction_listening.wav",
    "Yes, please connect me with someone who can help with the refill.": "09_accept_escalation.wav",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--scenario",
        choices=(
            "medication-refill-correction",
            "office-hours-location-insurance",
            "doctor-specialist-directory",
            "farthest-date-scheduling",
        ),
        default="medication-refill-correction",
    )
    parser.add_argument("--worker-manifest", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--persistent-worker", action="store_true")
    args = parser.parse_args()
    if args.persistent_worker:
        return persistent_worker()
    if args.worker_manifest:
        return worker(args.worker_manifest, args.worker_output)
    return build(args.cache_root, scenario=args.scenario)


def build(cache_root: Path, *, scenario: str) -> int:
    if not ACCENT_PYTHON.is_file():
        raise SystemExit(f"Melo environment is missing: {ACCENT_PYTHON}")
    cache = AccentCache(cache_root, scenario=scenario)
    inventory = {
        "medication-refill-correction": medication_refill_phrase_inventory,
        "office-hours-location-insurance": office_location_insurance_phrase_inventory,
        "doctor-specialist-directory": doctor_specialist_phrase_inventory,
        "farthest-date-scheduling": farthest_date_phrase_inventory,
    }[scenario]
    cache.directory.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, str]] = []
    newly_generated = 0
    for text in inventory():
        lookup = cache.lookup(text)
        if lookup.hit:
            print(f"HIT {lookup.cache_key} {text!r}", flush=True)
            continue
        key, _, wav_path = cache.paths(text)
        status = "MISS" if lookup.metadata is None else "INVALID"
        print(f"{status} {key} {text!r}: {lookup.invalid_reason}", flush=True)
        pending.append({"text": text, "wav_path": str(wav_path)})

    if pending:
        with tempfile.TemporaryDirectory(prefix="voiceprobe-accent-build-") as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps(pending, ensure_ascii=False), encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "NUMBA_CACHE_DIR": str(Path(temp) / "numba"),
            })
            subprocess.run(
                [str(ACCENT_PYTHON), str(Path(__file__).resolve()),
                 "--worker-manifest", str(manifest), "--worker-output", str(cache.directory)],
                check=True, env=env,
            )
        for item in pending:
            text = item["text"]
            key, metadata_path, wav_path = cache.paths(text)
            metadata = metadata_for_wav(text=text, wav_path=wav_path)
            write_metadata(metadata, metadata_path)
            print(f"GENERATED {key} {text!r}", flush=True)
            newly_generated += 1

    if scenario == "medication-refill-correction":
        listening = REPO_ROOT / "artifacts/accent-tests/en_india_current"
        listening.mkdir(parents=True, exist_ok=True)
        for text, name in LISTENING_NAMES.items():
            lookup = cache.lookup(text)
            if not lookup.hit:
                raise SystemExit(f"Listening phrase is absent after build: {text!r}")
            _, _, source = cache.paths(text)
            destination = listening / name
            if not destination.exists():
                shutil.copy2(source, destination)
                print(f"LISTENING {destination}", flush=True)

    required, cached, missing = cache.coverage(inventory())
    print(f"required_phrases={required}", flush=True)
    print(f"cached={cached}", flush=True)
    print(f"newly_generated={newly_generated}", flush=True)
    print(f"missing={len(missing)}", flush=True)
    print(f"coverage={100 * cached / required:.0f}%", flush=True)
    return 0 if not missing else 1


def worker(manifest_path: Path, output: Path | None) -> int:
    if output is None:
        raise SystemExit("--worker-output is required")
    # Melo imports are confined to the isolated interpreter and loaded once.
    import numpy as np
    import soundfile as sf
    from melo.api import TTS
    from scipy.signal import resample_poly

    requests = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"MELO loading language=EN speaker={MELO_SPEAKER}", flush=True)
    model = TTS(language="EN", device="cpu")
    speaker_id = model.hps.data.spk2id[MELO_SPEAKER]
    print(f"MELO loaded speaker_id={speaker_id}", flush=True)
    for item in requests:
        text = item["text"]
        destination = Path(item["wav_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"SYNTHESIZING {text!r}", flush=True)
        native = np.asarray(model.tts_to_file(text, speaker_id, quiet=True), dtype=np.float32)
        source_rate = int(model.hps.data.sampling_rate)
        pcm8 = resample_poly(native, 8000, source_rate).astype(np.float32)
        peak = float(np.max(np.abs(pcm8)))
        if not np.isfinite(peak) or peak == 0:
            raise RuntimeError(f"Melo returned invalid audio for {text!r}")
        # Leave modest headroom; format validation rejects clipped output.
        if peak > 0.98:
            pcm8 *= 0.98 / peak
        sf.write(str(destination), pcm8, 8000, subtype="PCM_16", format="WAV")
        print(f"WROTE {destination}", flush=True)
    return 0


def persistent_worker() -> int:
    """Keep one Melo model warm and serve newline-delimited render requests."""
    import numpy as np
    import soundfile as sf
    from melo.api import TTS
    from scipy.signal import resample_poly

    model = TTS(language="EN", device="cpu")
    speaker_id = model.hps.data.spk2id[MELO_SPEAKER]
    source_rate = int(model.hps.data.sampling_rate)
    print(json.dumps({"event": "ready", "speaker": MELO_SPEAKER}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            text = str(request["text"])
            destination = Path(request["wav_path"])
            native = np.asarray(model.tts_to_file(text, speaker_id, quiet=True), dtype=np.float32)
            pcm8 = resample_poly(native, 8000, source_rate).astype(np.float32)
            peak = float(np.max(np.abs(pcm8)))
            if not np.isfinite(peak) or peak == 0:
                raise RuntimeError(f"Melo returned invalid audio for {text!r}")
            if peak > 0.98:
                pcm8 *= 0.98 / peak
            sf.write(str(destination), pcm8, 8000, subtype="PCM_16", format="WAV")
            print(json.dumps({"event": "rendered", "id": request["id"]}), flush=True)
        except Exception as error:  # noqa: BLE001
            print(json.dumps({
                "event": "error",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
