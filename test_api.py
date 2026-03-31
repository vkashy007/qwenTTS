"""
API tests — run while server.py is up.

Usage:
    python test_api.py                        # all tests
    python test_api.py --mode design          # style-prompt only
    python test_api.py --mode clone  --ref-audio my_voice.wav --ref-text "Hello"
    python test_api.py --mode cached --ref-audio my_voice.wav --ref-text "Hello"
    python test_api.py --host http://localhost:8000
"""

import argparse
import sys
import time
from pathlib import Path

import httpx  # pip install httpx

BASE = "http://localhost:8000"

TEACHER_TEXT = (
    "Welcome everyone. Today we explore machine learning. "
    "First — what is data? "
    "Data is simply information we can measure. "
    "It's the raw material that fuels our models, from customer ages and transaction amounts to images, sounds, or text from the internet. "
    "Think of it as the 'experience' a machine needs to learn, just as humans learn by observing the world. "
    "But raw data alone isn't enough; for a machine to understand it, we often transform this data into numbers or mathematical representations, mapping inputs to outputs. "
    "Quality and quantity matter: the more diverse, clean, and plentiful our data, the better the model can identify hidden patterns and make accurate predictions."
)


def check_server(base: str) -> bool:
    try:
        r = httpx.get(f"{base}/docs", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def test_voice_design(base: str, out_dir: Path) -> None:
    print("\n[1] VoiceDesign — teacher prompt (no reference voice)")
    t0 = time.perf_counter()
    with httpx.stream(
        "POST",
        f"{base}/tts/stream",
        data={
            "text": TEACHER_TEXT,
            "instruct": (
                "A patient teacher speaking slowly with deliberate pauses "
                "between sentences. Warm, encouraging tone."
            ),
        },
        timeout=120,
    ) as r:
        r.raise_for_status()
        out = out_dir / "test_design.wav"
        with open(out, "wb") as f:
            first_chunk = True
            for chunk in r.iter_bytes(chunk_size=4096):
                if first_chunk:
                    print(f"   first bytes in {(time.perf_counter()-t0)*1000:.0f} ms")
                    first_chunk = False
                f.write(chunk)
    size_kb = out.stat().st_size / 1024
    print(f"   saved {out}  ({size_kb:.1f} KB)  total {(time.perf_counter()-t0)*1000:.0f} ms")
    print("   PASS")


def test_voice_clone(base: str, ref_audio: str, ref_text: str, out_dir: Path) -> None:
    print(f"\n[2] VoiceClone + teacher prompt  (ref: {ref_audio})")
    t0 = time.perf_counter()
    with open(ref_audio, "rb") as audio_file:
        with httpx.stream(
            "POST",
            f"{base}/tts/stream",
            data={"text": TEACHER_TEXT, "ref_text": ref_text},
            files={"ref_audio": (Path(ref_audio).name, audio_file, "audio/wav")},
            timeout=120,
        ) as r:
            r.raise_for_status()
            out = out_dir / "test_clone.wav"
            with open(out, "wb") as f:
                first_chunk = True
                for chunk in r.iter_bytes(chunk_size=4096):
                    if first_chunk:
                        print(f"   first bytes in {(time.perf_counter()-t0)*1000:.0f} ms")
                        first_chunk = False
                    f.write(chunk)
    size_kb = out.stat().st_size / 1024
    print(f"   saved {out}  ({size_kb:.1f} KB)  total {(time.perf_counter()-t0)*1000:.0f} ms")
    print("   PASS")


def test_cached_speaker(base: str, ref_audio: str, ref_text: str, out_dir: Path) -> None:
    print(f"\n[3] Pre-cached speaker + teacher prompt")

    # Step 1: register
    print("   registering speaker …")
    with open(ref_audio, "rb") as audio_file:
        r = httpx.post(
            f"{base}/speakers/register",
            data={"speaker_name": "test_speaker", "ref_text": ref_text},
            files={"ref_audio": (Path(ref_audio).name, audio_file, "audio/wav")},
            timeout=120,
        )
    r.raise_for_status()
    token = r.json()["speaker_token"]
    print(f"   speaker_token = {token}")

    # Step 2: verify it appears in /speakers
    speakers = httpx.get(f"{base}/speakers").json()["speakers"]
    assert token in speakers, f"Token not in /speakers: {speakers}"
    print(f"   /speakers lists {len(speakers)} speaker(s)  OK")

    # Step 3: stream using cached token (no re-upload)
    t0 = time.perf_counter()
    with httpx.stream(
        "POST",
        f"{base}/tts/stream",
        data={"text": TEACHER_TEXT, "speaker_token": token},
        timeout=120,
    ) as r:
        r.raise_for_status()
        out = out_dir / "test_cached.wav"
        with open(out, "wb") as f:
            first_chunk = True
            for chunk in r.iter_bytes(chunk_size=4096):
                if first_chunk:
                    print(f"   first bytes in {(time.perf_counter()-t0)*1000:.0f} ms")
                    first_chunk = False
                f.write(chunk)
    size_kb = out.stat().st_size / 1024
    print(f"   saved {out}  ({size_kb:.1f} KB)  total {(time.perf_counter()-t0)*1000:.0f} ms")
    print("   PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=BASE)
    parser.add_argument("--mode", choices=["design", "clone", "cached", "all"], default="all")
    parser.add_argument("--ref-audio", default="ref_audio.wav")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--out-dir", default="test_outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking server at {args.host} …")
    if not check_server(args.host):
        print(f"ERROR: server not reachable at {args.host}")
        print("Start it with:  uvicorn server:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    print("Server OK")

    needs_ref = args.mode in ("clone", "cached", "all")
    if needs_ref and not Path(args.ref_audio).exists():
        if args.mode == "all":
            print(f"\n[warn] {args.ref_audio} not found — skipping clone/cached tests.")
            print("       Pass --ref-audio and --ref-text to run those tests.\n")
            needs_ref = False
        else:
            print(f"ERROR: --ref-audio file not found: {args.ref_audio}")
            sys.exit(1)

    failed = []

    if args.mode in ("design", "all"):
        try:
            test_voice_design(args.host, out_dir)
        except Exception as e:
            msg = str(e)
            if "does not support voice design" in msg or "500" in msg:
                print(f"   SKIP: VoiceDesign not supported by loaded model (use 1.7B instruct model)")
            else:
                print(f"   FAIL: {e}")
                failed.append("design")

    if needs_ref and args.mode in ("clone", "all"):
        try:
            test_voice_clone(args.host, args.ref_audio, args.ref_text, out_dir)
        except Exception as e:
            print(f"   FAIL: {e}")
            failed.append("clone")

    if needs_ref and args.mode in ("cached", "all"):
        try:
            test_cached_speaker(args.host, args.ref_audio, args.ref_text, out_dir)
        except Exception as e:
            print(f"   FAIL: {e}")
            failed.append("cached")

    print("\n" + "="*40)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All tests PASSED")
        print(f"WAV files saved to ./{args.out_dir}/")


if __name__ == "__main__":
    main()
