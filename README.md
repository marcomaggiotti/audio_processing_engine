# audio_processing_engine
Core processing component with parallel workers, receive audio-processing requests, execute Pydub operations, manage concurrent jobs, and return the processed audio.

## What is here

The audio operations of [voice_embedding_service](https://github.com/marcomaggiotti/voice_embedding_service) that
pydub can do, moved out of their ffmpeg subprocess calls, and a Flask Blueprint that serves them.

| Module | Contents |
| --- | --- |
| `audio_processing_engine/operations.py` | the pydub operations |
| `audio_processing_engine/blueprint.py` | `audio_bp` (Flask Blueprint, `/audio/...`) and `create_app()` |
| `audio_processing_engine/formats.py` | format detection from the header, export settings per format |
| `audio_processing_engine/errors.py` | `AudioProcessingError`, `InvalidAudioError` |

### Operations and where they came from

| Operation | Origin in voice_embedding_service |
| --- | --- |
| `probe_duration`, `info` | `app/audio/audio.py` (ffprobe) |
| `frame_levels_db` (RMS dBFS per 50 ms) | `app/audio/audio.py` |
| `cut_clip` (m4a, mp3, flac, wav, ogg), `cut_audio` (FLAC) | `app/audio/audio.py` |
| `cut_with_margin` (segment ± 0.3 s) | `app/openai_transcribe.py` |
| `voice_clip` (mono OGG/Opus 32k) | `app/interfaces/telegram.py` |
| `split` (parts by time, all checked first) | `app/cutting.py` |
| `cut_points` (≤ 30 s pieces cut at the quietest 20 ms), `export_wav16k` | `app/chunked.py` |
| `decode_pcm`, `window_audio` (30 s, 16 kHz, padded) | `app/audio/audio.py` `_pcm_chunks`, `app/whisper_view.py` |
| `band`, `apply_eq`, `band_filter`, `adjust_audio` | `app/audio/audio.py`, `app/schemas.py` `EqBand` |
| `highpass` (rumble), `remove_hum`, `shorten_pauses` | `app/schemas.py` `AudioProcessing` (steps of the clean service) |
| `normalize_loudness` | ffmpeg `loudnorm` in `band_filter` / `adjust_audio` |
| `webm_to_ogg`, `detect_format` | `app/audio/audio.py` |

**Differences from the ffmpeg originals**

- Loudness: pydub has no EBU R128 meter, so `loudnorm=I=-16:TP=-1.5` became an RMS target of -16 dBFS with a
  -1.5 dBFS peak ceiling.
- Filters use `pydub.scipy_effects` (Butterworth). The equalizer's peaks and shelves are pydub's band-based `eq`
  (bandwidth = frequency / Q), close to the RBJ biquads but not identical; notches subtract a narrow band-pass.
- `webm_to_ogg` always re-encodes to Opus 48k (pydub decodes everything; the original stream-copied Opus).

**Not ported** (pydub has no equivalent): `afftdn` noise reduction (`denoise_db`), declip, declick. The
spectrogram and Whisper's log-mel are numpy DSP; `decode_pcm` and `window_audio` provide their input.

## The Blueprint

```python
from flask import Flask
from audio_processing_engine.blueprint import audio_bp

app = Flask(__name__)
app.register_blueprint(audio_bp)  # routes under /audio
```

Each endpoint takes a multipart upload in `file`, and its parameters as form fields (or a JSON object in the
form field `options`).

| Endpoint | Parameters | Returns |
| --- | --- | --- |
| `POST /audio/info` | | JSON: duration, frame_rate, channels, sample_width, dbfs, max_dbfs, format |
| `POST /audio/levels` | `frame_seconds`, `limit_seconds` | JSON: levels (dBFS) |
| `POST /audio/pieces` | `max_seconds`, `search_seconds` | JSON: pieces `{index, start, end}` |
| `POST /audio/cut` | `start`, `end`, `format` | the clip |
| `POST /audio/voice-clip` | `start`, `end` | OGG/Opus |
| `POST /audio/convert` | `format`, `transcription` | the whole audio (16 kHz mono with `transcription`) |
| `POST /audio/filter` | `low_hz`, `high_hz`, `eq` | 16 kHz mono FLAC |
| `POST /audio/adjust` | `low_hz`, `high_hz`, `eq`, `gain_db`, `loudnorm`, `highpass_hz`, `dehum_hz` (50/60), `remove_silence`, `min_silence_seconds` | 16 kHz mono FLAC |

`eq` is a JSON list of `{type, frequency, gain, q, enabled}`, `type` one of peaking, lowshelf, highshelf,
highpass, lowpass, notch, bandpass. Errors are JSON `{error}`: 400 bad parameters, 415 unsupported format,
422 audio that can't be decoded or encoded.

```bash
curl -F file=@talk.m4a -F start=12.5 -F end=40 -F format=mp3 localhost:5000/audio/cut -o clip.mp3
```

## Setup

ffmpeg must be on `PATH`.

```bash
pip install -r requirements.txt pytest
```

```bash
python -m pytest
```

```bash
flask --app "audio_processing_engine.blueprint:create_app()" run
```
