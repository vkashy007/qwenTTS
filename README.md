# Qwen3-TTS Server

A FastAPI server for real-time text-to-speech using [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), with streaming audio output, voice cloning, and style prompts.

## Features

- **Streaming** — audio chunks delivered to the client as they are generated (~150ms first-byte latency)
- **Voice cloning** — clone any voice from a short WAV reference clip
- **Pre-cached speakers** — register a voice once, reuse by token on every request
- **Style prompts** — control speaking style via natural language (e.g. teacher, narrator, presenter)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (12.x recommended)
- PyTorch 2.5.1+

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/qwenTTS.git
cd qwenTTS
bash setup.sh
source venv/bin/activate
```

## Usage

### Start the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The model downloads automatically on first run (~1.5 GB).

---

### POST /tts/stream

Synthesise speech and receive a streaming WAV response.

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | yes | Text to synthesise |
| `ref_audio` | file | no* | WAV file to clone voice from (5–30s) |
| `ref_text` | string | no | Transcript of ref_audio (improves quality) |
| `speaker_token` | string | no* | Token from `/speakers/register` |
| `instruct` | string | no | Style prompt (defaults to teacher persona when voice clone is used) |
| `language` | string | no | Language hint, default `Auto` |
| `chunk_size` | int | no | Streaming chunk size, default `8` |
| `temperature` | float | no | Sampling temperature, default `0.9` |
| `max_new_tokens` | int | no | Max generation tokens, default `4096` |

*At least one of `ref_audio`, `speaker_token`, or `instruct` is required.

**Priority:** `ref_audio` > `speaker_token` > style-only (`instruct`)

---

### POST /speakers/register

Upload a reference WAV once and get back a `speaker_token` for reuse.

```bash
curl -X POST http://localhost:8000/speakers/register \
     -F "speaker_name=alice" \
     -F "ref_audio=@alice.wav" \
     -F "ref_text=Hello, this is Alice speaking." \
     -F "xvec_only=false"
```

Response:
```json
{ "speaker_token": "alice_3f9a1b2c", "speaker_name": "alice" }
```

### GET /speakers

List all registered speaker tokens.

---

## Example Calls

**Voice clone with teacher style prompt:**
```bash
curl -X POST http://localhost:8000/tts/stream \
     -F "text=Welcome to today's lesson on machine learning." \
     -F "ref_audio=@my_voice.wav" \
     -F "ref_text=Hello this is my reference recording." \
     --output lesson.wav
```

**Pre-cached speaker (fast, no re-upload):**
```bash
curl -X POST http://localhost:8000/tts/stream \
     -F "text=Let us begin with a quick recap." \
     -F "speaker_token=alice_3f9a1b2c" \
     --output recap.wav
```

**Custom style prompt:**
```bash
curl -X POST http://localhost:8000/tts/stream \
     -F "text=And that wraps up today's session." \
     -F "speaker_token=alice_3f9a1b2c" \
     -F "instruct=A calm audiobook narrator with a warm, unhurried tone." \
     --output outro.wav
```

## Running Tests

```bash
# Design mode only (no reference audio needed):
python test_api.py --mode design

# Voice clone:
python test_api.py --mode clone --ref-audio my_voice.wav --ref-text "Hello world"

# Pre-cached speaker:
python test_api.py --mode cached --ref-audio my_voice.wav --ref-text "Hello world"

# All tests:
python test_api.py --ref-audio my_voice.wav --ref-text "Hello world"
```

Output WAV files are saved to `./test_outputs/`.

## Model

Default model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

| Model | VoiceClone | VoiceDesign | Size |
|---|---|---|---|
| `Qwen3-TTS-12Hz-0.6B-Base` | yes | no | ~1.5 GB |
| `Qwen3-TTS-12Hz-1.7B-Instruct` | yes | yes | ~3.5 GB |

To switch models, change `DEFAULT_MODEL` in `tts_engine.py`.

## License

MIT
