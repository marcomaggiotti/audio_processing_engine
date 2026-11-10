"""Container formats: detection from the file header, and the export settings of downloadable clips.

Ported from voice_embedding_service (app/audio/audio.py).
"""

from dataclasses import dataclass

from audio_processing_engine.errors import InvalidAudioError

# Container formats accepted, with the media type used when serving a file back.
MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",  # Telegram voice notes (Opus)
    "mp4": "audio/mp4",  # M4A / AAC, Telegram video notes
    "flac": "audio/flac",
    "webm": "audio/webm",  # browser microphone recordings (Chrome, Edge)
}


def detect_format(data: bytes) -> str:
    """Return the container format from the file header, or raise InvalidAudioError."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "mp4"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "mp3"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    raise InvalidAudioError("Unsupported audio format: expected WAV, MP3, OGG/Opus, M4A, FLAC or WebM")


@dataclass(frozen=True)
class ExportFormat:
    """How AudioSegment.export writes a format: ffmpeg container, codec, bitrate, extra ffmpeg args, media type."""

    container: str
    codec: str | None
    bitrate: str | None
    parameters: tuple[str, ...]
    media_type: str
    extension: str


# A piece of a recording to download.
CLIP_FORMATS: dict[str, ExportFormat] = {
    "m4a": ExportFormat("ipod", "aac", "96k", (), "audio/mp4", "m4a"),  # plays everywhere, small
    "mp3": ExportFormat("mp3", "libmp3lame", None, ("-q:a", "4"), "audio/mpeg", "mp3"),
    "flac": ExportFormat("flac", "flac", None, (), "audio/flac", "flac"),  # lossless
    "wav": ExportFormat("wav", None, None, (), "audio/wav", "wav"),
    # Telegram-style voice message: small and fast to send.
    "ogg": ExportFormat("ogg", "libopus", "32k", (), "audio/ogg", "ogg"),
}
