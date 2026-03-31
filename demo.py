"""
Demo — calls the three TTSEngine.stream() scenarios directly (no server needed).

Usage
-----
  python demo.py --mode design                      # teacher prompt, no ref voice
  python demo.py --mode clone  --ref-audio voice.wav --ref-text "Hello world"
  python demo.py --mode cached --ref-audio voice.wav --ref-text "Hello world"
  python demo.py --no-play                          # save WAVs, skip playback
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_engine import TEACHER_INSTRUCT, TTSEngine

TEXT = (
    "Welcome everyone. Today we explore machine learning. "
    "First — what is data? "
    "Data is simply information we can measure. "
    "A model learns patterns from this data, "
    "much like a child learning to recognise cats from pictures."
)


def make_player():
    try:
        import sounddevice as sd

        class _Player:
            def __init__(self):
                self._stream = None

            def __call__(self, chunk: np.ndarray, sr: int):
                import sounddevice as sd
                if self._stream is None:
                    self._stream = sd.OutputStream(
                        samplerate=sr, channels=1, dtype="float32"
                    )
                    self._stream.start()
                self._stream.write(chunk.astype(np.float32)[:, None])

            def close(self):
                if self._stream:
                    self._stream.stop()
                    self._stream.close()

        return _Player()
    except ImportError:
        print("[warn] sounddevice missing — no live playback.")
        return None


def run_stream(engine, out_path, play, **kwargs):
    player = make_player() if play else None
    chunks, t0 = [], time.perf_counter()
    for i, (chunk, sr) in enumerate(engine.stream(TEXT, **kwargs)):
        ms = (time.perf_counter() - t0) * 1000
        print(f"  chunk {i:02d}  {len(chunk):>6} samples  {ms:6.0f} ms")
        chunks.append((chunk, sr))
        if player:
            player(chunk, sr)
    if player:
        player.close()
    if chunks:
        audio = np.concatenate([c for c, _ in chunks])
        sf.write(str(out_path), audio, chunks[0][1])
        print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["design", "clone", "cached"], default="design")
    parser.add_argument("--ref-audio", default="ref_audio.wav")
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    engine = TTSEngine().load()
    play = not args.no_play

    if args.mode == "design":
        print(f"\n=== VoiceDesign — teacher prompt ===")
        run_stream(engine, f"{args.out_dir}/design_teacher.wav", play,
                   instruct=TEACHER_INSTRUCT)

    elif args.mode == "clone":
        if not Path(args.ref_audio).exists():
            print(f"ref_audio not found: {args.ref_audio}")
            return
        print(f"\n=== VoiceClone + teacher prompt ===")
        run_stream(engine, f"{args.out_dir}/clone_teacher.wav", play,
                   ref_audio=args.ref_audio,
                   ref_text=args.ref_text,
                   instruct=TEACHER_INSTRUCT)

    elif args.mode == "cached":
        if not Path(args.ref_audio).exists():
            print(f"ref_audio not found: {args.ref_audio}")
            return
        print("Pre-caching voice clone prompt …")
        prompt = engine.build_voice_clone_prompt(args.ref_audio, args.ref_text)
        print(f"\n=== Cached VoiceClone + teacher prompt ===")
        run_stream(engine, f"{args.out_dir}/cached_teacher.wav", play,
                   voice_clone_prompt=prompt,
                   instruct=TEACHER_INSTRUCT)


if __name__ == "__main__":
    main()
