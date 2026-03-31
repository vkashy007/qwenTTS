"""
Single-endpoint TTS API server.

POST /tts/stream
    → StreamingResponse (audio/wav)

All three features in one request:
  - Streaming          : always (chunked WAV over HTTP)
  - Voice clone        : upload ref_audio + ref_text
                         OR send speaker_token (pre-cached)
  - Style prompt       : instruct field (defaults to teacher)

Start:
    uvicorn server:app --host 0.0.0.0 --port 8000

Example calls (httpx / curl):

  # Teacher prompt + voice clone from uploaded file:
  curl -X POST http://localhost:8000/tts/stream \
       -F "text=Welcome to today's lesson on neural networks." \
       -F "ref_audio=@my_voice.wav" \
       -F "ref_text=Hello, this is my reference recording." \
       --output cloned_teacher.wav

  # Teacher prompt + pre-cached speaker (fast):
  curl -X POST http://localhost:8000/tts/stream \
       -F "text=Let us begin." \
       -F "speaker_token=SPEAKER_ABC" \
       --output cached_teacher.wav

  # Pure style design (no reference voice):
  curl -X POST http://localhost:8000/tts/stream \
       -F "text=Welcome everyone." \
       -F "instruct=A calm audiobook narrator with a warm tone." \
       --output design.wav
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tts_server")

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from tts_engine import TEACHER_INSTRUCT, TTSEngine

# ---------------------------------------------------------------------------
# App + model lifecycle
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen3-TTS API", version="1.0")

engine: Optional[TTSEngine] = None

# Speaker token cache: token_string → voice_clone_prompt object
_speaker_cache: dict = {}


@app.on_event("startup")
async def startup():
    global engine
    engine = TTSEngine().load()


# ---------------------------------------------------------------------------
# WAV streaming helpers
# ---------------------------------------------------------------------------

def _wav_header(sample_rate: int, num_channels: int = 1, bits: int = 16) -> bytes:
    """Minimal WAV header for streaming (data size unknown → 0xFFFFFFFF)."""
    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,       # file size (unknown for streaming)
        b"WAVE",
        b"fmt ",
        16,               # PCM chunk size
        1,                # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        0xFFFFFFFF,       # data size (unknown)
    )
    return header


def _to_pcm16(chunk: np.ndarray) -> bytes:
    """Convert float32 numpy array to signed 16-bit PCM bytes."""
    clipped = np.clip(chunk, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Single endpoint
# ---------------------------------------------------------------------------

@app.post("/tts/stream")
async def tts_stream(
    text: str = Form(..., description="Text to synthesise"),
    instruct: Optional[str] = Form(
        None,
        description=(
            "Style/role prompt. Defaults to teacher persona if voice clone is used. "
            "Required when no ref_audio or speaker_token is given."
        ),
    ),
    # Live voice clone
    ref_audio: Optional[UploadFile] = File(
        None, description="WAV file of voice to clone (5–30 s recommended)"
    ),
    ref_text: Optional[str] = Form(
        None, description="Exact transcript of ref_audio"
    ),
    # Pre-cached clone
    speaker_token: Optional[str] = Form(
        None,
        description=(
            "Token returned by POST /speakers/register. "
            "Skips re-processing the reference audio on every request."
        ),
    ),
    # Audio params
    language: str = Form("Auto"),
    chunk_size: int = Form(8, ge=1, le=64),
    temperature: float = Form(0.9, ge=0.0, le=2.0),
    max_new_tokens: int = Form(4096, ge=128, le=8192),
) -> StreamingResponse:
    """
    Stream synthesised speech as chunked WAV audio.

    Priority: ref_audio > speaker_token > voice_design (instruct only)

    The `instruct` field is applied to every mode — you always get
    teacher pacing/tone regardless of which voice source you choose.
    """
    if engine is None:
        raise HTTPException(503, "Model not loaded yet.")

    speaker_cache_entry = None
    tmp_path = None

    # Resolve voice source
    if ref_audio is not None:
        suffix = Path(ref_audio.filename).suffix if ref_audio.filename else ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await ref_audio.read())
        tmp.close()
        tmp_path = tmp.name
        ref_audio_path = tmp_path
        ref_text_str = ref_text or ""
    elif speaker_token is not None:
        if speaker_token not in _speaker_cache:
            raise HTTPException(
                404,
                f"speaker_token '{speaker_token}' not found. "
                "Register it via POST /speakers/register first.",
            )
        speaker_cache_entry = _speaker_cache[speaker_token]
        ref_audio_path = None
        ref_text_str = ""
    else:
        ref_audio_path = None
        ref_text_str = ""

    # Resolve effective instruct — priority: caller > speaker default > global teacher default > None
    effective_instruct = instruct
    if effective_instruct is None and speaker_cache_entry is not None:
        effective_instruct = speaker_cache_entry.get("default_instruct") or TEACHER_INSTRUCT
    elif effective_instruct is None and ref_audio is not None:
        effective_instruct = TEACHER_INSTRUCT

    if ref_audio_path is None and speaker_cache_entry is None and not effective_instruct:
        raise HTTPException(
            422,
            "Provide ref_audio, speaker_token, or instruct (at least one required).",
        )

    async def audio_generator():
        import queue as _queue
        import threading

        q: _queue.Queue = _queue.Queue(maxsize=4)
        _DONE = object()
        header_sent = False

        def producer():
            try:
                for chunk, sr in engine.stream(
                    text=text,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text_str,
                    speaker_cache_entry=speaker_cache_entry,
                    instruct=effective_instruct,
                    language=language,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                ):
                    q.put((chunk, sr))
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(_DONE)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    logger.exception("TTS inference failed: %s", item)
                    raise item
                chunk, sr = item
                if not header_sent:
                    yield _wav_header(sr)
                    header_sent = True
                yield _to_pcm16(chunk.flatten())
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )


# ---------------------------------------------------------------------------
# Speaker registration endpoint (pre-cache a cloned voice)
# ---------------------------------------------------------------------------

@app.post("/speakers/register")
async def register_speaker(
    speaker_name: str = Form(..., description="A label for this speaker"),
    ref_audio: UploadFile = File(..., description="Reference WAV to clone"),
    ref_text: str = Form("", description="Exact transcript of ref_audio (leave empty to use xvec-only mode)"),
    xvec_only: Optional[str] = Form(
        None,
        description="Pass 'true' for faster x-vector-only mode (lower quality)",
    ),
    default_instruct: Optional[str] = Form(
        None,
        description="Default style prompt for this speaker. If omitted, uses the global teacher persona.",
    ),
) -> dict:
    """
    Pre-compute a voice clone prompt and cache it server-side.

    Returns a `speaker_token` you can pass to /tts/stream on every
    subsequent request — no re-upload needed.
    """
    if engine is None:
        raise HTTPException(503, "Model not loaded yet.")

    use_xvec = (xvec_only or "").lower() == "true"
    suffix = Path(ref_audio.filename).suffix if ref_audio.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await ref_audio.read())
    tmp.close()

    try:
        entry = engine.register_speaker(
            ref_audio=tmp.name,
            ref_text=ref_text,
            xvec_only=use_xvec,
            default_instruct=default_instruct,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    token = f"{speaker_name}_{uuid.uuid4().hex[:8]}"
    _speaker_cache[token] = entry
    return {"speaker_token": token, "speaker_name": speaker_name}


@app.get("/speakers")
async def list_speakers() -> dict:
    """List all registered speaker tokens."""
    return {"speakers": list(_speaker_cache.keys())}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
