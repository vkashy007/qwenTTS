"""
Qwen3-TTS engine — wraps faster-qwen3-tts.

All three requirements work in a single underlying call:
  - Streaming          : always on via _streaming generators
  - Voice clone        : ref_audio + ref_text  (live clone)
                         voice_clone_prompt    (pre-cached clone)
  - Style / instruct   : instruct="speak like a teacher …"

The instruct param is accepted by generate_voice_clone_streaming,
so voice-clone + teacher-prompt + streaming is one API call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

TEACHER_INSTRUCT = (
    "A patient, warm teacher speaking to adult learners. "
    "Slow and deliberate pace. "
    "Clear pauses between sentences and after key concepts. "
    "Slightly emphasised stress on important words. "
    "Encouraging and calm tone throughout."
)


class TTSEngine:
    """
    Load once, call many times.

    Three routing modes (chosen automatically by what you pass):
      1. voice_clone  – ref_audio file  OR  voice_clone_prompt (pre-cached)
      2. custom       – speaker_id string (built-in speaker)
      3. design       – instruct only, no reference voice
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._model = None

    def load(self) -> "TTSEngine":
        from faster_qwen3_tts import FasterQwen3TTS
        print(f"Loading {self.model_id} …")
        self._model = FasterQwen3TTS.from_pretrained(
            self.model_id, device=self.device, dtype=self.dtype
        )
        print("Model ready.")
        return self

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Call TTSEngine.load() first.")
        return self._model

    # ------------------------------------------------------------------
    # Pre-cache a speaker (store raw bytes — avoids re-upload each request)
    # ------------------------------------------------------------------

    def register_speaker(
        self,
        ref_audio: Union[str, Path],
        ref_text: str,
        xvec_only: bool = False,
    ) -> dict:
        """
        Read the reference audio file and return a cache entry dict.
        Store the returned dict; pass it to stream() as speaker_cache_entry=.
        """
        ref_path = Path(ref_audio)
        return {
            "audio_bytes": ref_path.read_bytes(),
            "suffix": ref_path.suffix or ".wav",
            "ref_text": ref_text,
            "xvec_only": xvec_only,
        }

    # ------------------------------------------------------------------
    # Main streaming generator — the single entry point
    # ------------------------------------------------------------------

    def stream(
        self,
        text: str,
        *,
        # Voice clone (live)
        ref_audio: Optional[Union[str, Path]] = None,
        ref_text: str = "",
        # Pre-cached speaker (dict from register_speaker())
        speaker_cache_entry: Optional[dict] = None,
        # Built-in speaker
        speaker_id: Optional[str] = None,
        # Style / role prompt (works with every mode)
        instruct: Optional[str] = None,
        # Common params
        language: str = "Auto",
        chunk_size: int = 8,
        xvec_only: bool = False,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 4096,
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Unified streaming generator.  Yields (audio_chunk, sample_rate).

        Routing logic
        -------------
        ref_audio                →  voice_clone_streaming (live upload)
        speaker_cache_entry      →  voice_clone_streaming (pre-cached bytes)
        speaker_id               →  custom_voice_streaming (built-in speaker)
        neither                  →  voice_design_streaming (instruct required)

        instruct is forwarded to every mode.
        """
        import tempfile, warnings

        _tmp_path = None  # temp file for cached speaker bytes

        # Resolve speaker_cache_entry → temp file
        if speaker_cache_entry is not None and ref_audio is None:
            entry = speaker_cache_entry
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=entry["suffix"])
            tmp.write(entry["audio_bytes"])
            tmp.close()
            _tmp_path = tmp.name
            ref_audio = _tmp_path
            ref_text = entry["ref_text"]
            xvec_only = entry["xvec_only"]

        # ICL mode requires non-empty ref_text; fall back to xvec_only automatically
        if ref_audio is not None and not ref_text and not xvec_only:
            warnings.warn(
                "ref_text is empty — switching to xvec_only=True. "
                "Provide ref_text for higher-quality ICL voice cloning.",
                UserWarning, stacklevel=2,
            )
            xvec_only = True

        common = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
        )

        try:
            if ref_audio is not None:
                gen = self.model.generate_voice_clone_streaming(
                    text=text,
                    language=language,
                    ref_audio=str(ref_audio),
                    ref_text=ref_text,
                    instruct=instruct,
                    xvec_only=xvec_only,
                    **common,
                )
            elif speaker_id is not None:
                gen = self.model.generate_custom_voice_streaming(
                    text=text,
                    speaker=speaker_id,
                    language=language,
                    instruct=instruct,
                    **common,
                )
            else:
                if not instruct:
                    raise ValueError(
                        "VoiceDesign mode requires `instruct`. "
                        "Pass instruct= or provide ref_audio / speaker_id."
                    )
                try:
                    gen = self.model.generate_voice_design_streaming(
                        text=text,
                        instruct=instruct,
                        language=language,
                        **common,
                    )
                except ValueError as e:
                    if "does not support voice design" in str(e):
                        raise ValueError(
                            "VoiceDesign mode is not supported by this model "
                            f"({self.model_id}). Use ref_audio/speaker_token for "
                            "voice clone mode, or switch to the 1.7B instruct model."
                        ) from e
                    raise

            for audio_chunk, sr, _ in gen:
                yield audio_chunk, sr

        finally:
            if _tmp_path:
                import os
                try:
                    os.unlink(_tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Convenience: collect full audio from stream
    # ------------------------------------------------------------------

    def generate(self, text: str, **kwargs) -> Tuple[np.ndarray, int]:
        """Non-streaming wrapper — same kwargs as stream()."""
        chunks, sr_out = [], 0
        for chunk, sr in self.stream(text, **kwargs):
            chunks.append(chunk)
            sr_out = sr
        return np.concatenate(chunks), sr_out

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def save_wav(path: Union[str, Path], audio: np.ndarray, sr: int) -> None:
        sf.write(str(path), audio, sr)
        print(f"Saved → {path}")
