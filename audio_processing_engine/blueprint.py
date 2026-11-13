"""Flask Blueprint exposing the pydub operations over HTTP.

Every endpoint takes a multipart upload in the field `file`, plus form fields (or JSON in the form field
`options`) for its parameters, and returns either JSON or the processed audio.

    POST /audio/info             -> {duration, frame_rate, channels, sample_width, dbfs, max_dbfs, format}
    POST /audio/levels           -> {frame_seconds, levels}            frame_seconds, limit_seconds
    POST /audio/pieces           -> {pieces: [{index, start, end}]}    max_seconds, search_seconds
    POST /audio/cut              -> audio                              start, end, format (m4a|mp3|flac|wav|ogg)
    POST /audio/voice-clip       -> audio/ogg                          start, end
    POST /audio/convert          -> audio                              format, transcription (16 kHz mono)
    POST /audio/filter           -> audio/flac (16 kHz mono)           low_hz, high_hz, eq (JSON list of bands)
    POST /audio/adjust           -> audio/flac (16 kHz mono)           low_hz, high_hz, eq, gain_db, loudnorm,
                                                                       highpass_hz, dehum_hz, remove_silence,
                                                                       min_silence_seconds

Register it with `app.register_blueprint(audio_bp)` (url_prefix /audio), or use create_app().
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict

from flask import Blueprint, Flask, jsonify, request, send_file
from werkzeug.exceptions import BadRequest

from audio_processing_engine import operations as ops
from audio_processing_engine.errors import AudioProcessingError, InvalidAudioError
from audio_processing_engine.formats import CLIP_FORMATS, detect_format

audio_bp = Blueprint("audio", __name__, url_prefix="/audio")


# ------------------------------------------------------------------------------------------- inputs

def _options() -> dict:
    opts = dict(request.form)
    if "options" in opts:
        try:
            extra = json.loads(opts.pop("options"))
        except ValueError as e:
            raise BadRequest("options: invalid JSON") from e
        if not isinstance(extra, dict):
            raise BadRequest("options: expected a JSON object")
        opts.update(extra)
    return opts


def _upload():
    """The uploaded audio (seekable, checked by its header) and its detected format."""
    f = request.files.get("file")
    if f is None:
        raise BadRequest("Missing the audio file (multipart field 'file')")
    data = f.read()
    if not data:
        raise BadRequest("The audio file is empty")
    return io.BytesIO(data), detect_format(data)


def _float(opts: dict, name: str, default: float | None = None, required: bool = False) -> float | None:
    value = opts.get(name)
    if value in (None, ""):
        if required:
            raise BadRequest(f"Missing '{name}'")
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise BadRequest(f"'{name}' must be a number") from e


def _bool(opts: dict, name: str, default: bool = False) -> bool:
    value = opts.get(name)
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def _eq(opts: dict) -> list[ops.EqBand]:
    value = opts.get("eq")
    if value in (None, ""):
        return []
    try:
        bands = json.loads(value) if isinstance(value, str) else value
        return [ops.EqBand.from_dict(b) for b in bands]
    except (ValueError, TypeError) as e:
        raise BadRequest("'eq' must be a JSON list of bands {type, frequency, gain, q, enabled}") from e


def _format(opts: dict, default: str) -> str:
    fmt = str(opts.get("format") or default).lower()
    if fmt not in CLIP_FORMATS:
        raise BadRequest(f"'format' must be one of {', '.join(CLIP_FORMATS)}")
    return fmt


def _send(data: bytes, fmt: str, name: str):
    spec = CLIP_FORMATS[fmt]
    return send_file(io.BytesIO(data), mimetype=spec.media_type, as_attachment=True,
                     download_name=f"{name}.{spec.extension}")


def _load(buf, fmt):
    return ops.load(buf, None if fmt == "mp4" else fmt)  # let ffmpeg probe M4A/MP4


# ------------------------------------------------------------------------------------------ routes

@audio_bp.post("/info")
def info():
    buf, fmt = _upload()
    return jsonify({**ops.info(_load(buf, fmt)), "format": fmt})


@audio_bp.post("/levels")
def levels():
    opts = _options()
    buf, fmt = _upload()
    frame = _float(opts, "frame_seconds", ops.LEVEL_FRAME_SECONDS)
    if frame <= 0:
        raise BadRequest("'frame_seconds' must be positive")
    return jsonify({"frame_seconds": frame,
                    "levels": ops.frame_levels_db(_load(buf, fmt), _float(opts, "limit_seconds"), frame)})


@audio_bp.post("/pieces")
def pieces():
    opts = _options()
    buf, fmt = _upload()
    max_seconds = _float(opts, "max_seconds", 30.0)
    if max_seconds <= 0:
        raise BadRequest("'max_seconds' must be positive")
    result = ops.cut_points(_load(buf, fmt), max_seconds, _float(opts, "search_seconds", 5.0))
    return jsonify({"pieces": [asdict(p) for p in result]})


@audio_bp.post("/cut")
def cut():
    opts = _options()
    buf, fmt = _upload()
    out = _format(opts, "m4a")
    start, end = _float(opts, "start", required=True), _float(opts, "end", required=True)
    return _send(ops.cut_clip(_load(buf, fmt), None, start, end, out), out, "clip")


@audio_bp.post("/voice-clip")
def voice_clip():
    opts = _options()
    buf, fmt = _upload()
    start, end = _float(opts, "start", required=True), _float(opts, "end", required=True)
    return _send(ops.voice_clip(_load(buf, fmt), start, end), "ogg", "voice")


@audio_bp.post("/convert")
def convert():
    opts = _options()
    buf, fmt = _upload()
    out = _format(opts, "flac")
    return _send(ops.convert(_load(buf, fmt), None, out, _bool(opts, "transcription")), out, "audio")


@audio_bp.post("/filter")
def band_filter():
    opts = _options()
    buf, fmt = _upload()
    data = ops.band_filter(_load(buf, fmt), None, _float(opts, "low_hz"), _float(opts, "high_hz"), _eq(opts))
    return _send(data, "flac", "filtered")


@audio_bp.post("/adjust")
def adjust():
    opts = _options()
    buf, fmt = _upload()
    dehum = _float(opts, "dehum_hz")
    if dehum is not None and dehum not in (50, 60):
        raise BadRequest("'dehum_hz' must be 50 or 60")
    data = ops.adjust_audio(
        _load(buf, fmt), None, _float(opts, "low_hz"), _float(opts, "high_hz"), _eq(opts),
        gain_db=_float(opts, "gain_db", 0.0), loudnorm=_bool(opts, "loudnorm", True),
        highpass_hz=_float(opts, "highpass_hz"), dehum_hz=int(dehum) if dehum else None,
        remove_silence=_bool(opts, "remove_silence"), min_silence_seconds=_float(opts, "min_silence_seconds", 1.0))
    return _send(data, "flac", "adjusted")


# ------------------------------------------------------------------------------------------ errors

@audio_bp.errorhandler(InvalidAudioError)
def _invalid(e):
    return jsonify({"error": str(e)}), 415


@audio_bp.errorhandler(AudioProcessingError)
def _processing(e):
    return jsonify({"error": str(e)}), 422


@audio_bp.errorhandler(ValueError)
def _value(e):
    return jsonify({"error": str(e)}), 400


@audio_bp.errorhandler(BadRequest)
def _bad_request(e):
    return jsonify({"error": e.description}), 400


def create_app(max_upload_mb: int = 200) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.register_blueprint(audio_bp)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
