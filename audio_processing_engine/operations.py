"""Audio operations done with pydub, ported from voice_embedding_service.

There each one was an ffmpeg subprocess call; here the audio is an AudioSegment (pydub still uses ffmpeg to
decode and encode files). Where the original came from:

    probe_duration, frame_levels_db, cut_clip, cut_audio,
    band_filter, adjust_audio, webm_to_ogg          app/audio/audio.py
    decode_pcm, cut_points, export_wav16k           app/chunked.py (and _pcm_chunks in app/audio/audio.py)
    window_audio                                    app/whisper_view.py
    voice_clip                                      app/interfaces/telegram.py (cut_clip)
    cut_with_margin                                 app/openai_transcribe.py
    apply_eq                                        app/schemas.py (EqBand.ffmpeg)
    remove_hum, shorten_pauses, highpass (rumble)   app/schemas.py (AudioProcessing: the steps pydub can do)

Not ported, because pydub has no equivalent: ffmpeg's afftdn noise reduction (adjust_audio's denoise_db),
declip and declick. EBU R128 loudness (ffmpeg's loudnorm) becomes an RMS level target with a peak ceiling,
see normalize_loudness. The spectrogram and Whisper's log-mel are numpy DSP on decoded samples: decode_pcm
and window_audio give them their input.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError, CouldntEncodeError
from pydub.silence import detect_silence

from audio_processing_engine.errors import AudioProcessingError, InvalidAudioError
from audio_processing_engine.formats import CLIP_FORMATS

Source = str | Path | BinaryIO | AudioSegment

TRANSCRIPTION_RATE = 16_000  # what the transcription uses: 16 kHz mono, 16-bit
LEVEL_FRAME_SECONDS = 0.05
QUIET_FRAME_MS = 20  # the step at which the quietest cut is looked for
WHISPER_WINDOW_SECONDS = 30
MARGIN_SECONDS = 0.3  # around a segment sent for transcription, so words at its edges aren't clipped
BUTTERWORTH_ORDER = 4  # two 2-pole Butterworth sections in the original = 24 dB/octave
LOUDNESS_TARGET_DB = -16.0
TRUE_PEAK_DB = -1.5


# ------------------------------------------------------------------------------------------- loading

def load(src: Source, fmt: str | None = None) -> AudioSegment:
    """The audio as an AudioSegment, from a path, a file object or an AudioSegment (returned as it is)."""
    if isinstance(src, AudioSegment):
        return src
    try:
        return AudioSegment.from_file(str(src) if isinstance(src, Path) else src, format=fmt)
    except (CouldntDecodeError, IndexError, OSError) as e:
        message = str(e).strip().splitlines()
        raise AudioProcessingError(f"Could not decode audio: {message[-1] if message else e.__class__.__name__}") from e


def for_transcription(seg: AudioSegment, rate: int = TRANSCRIPTION_RATE) -> AudioSegment:
    """Mono, 16-bit, at `rate`."""
    return seg.set_channels(1).set_sample_width(2).set_frame_rate(rate)


def samples(seg: AudioSegment) -> np.ndarray:
    """The segment's samples as int16 (the segment should be 16-bit; channels are interleaved)."""
    return np.array(seg.get_array_of_samples(), dtype=np.int16)


def from_samples(data: np.ndarray, rate: int = TRANSCRIPTION_RATE) -> AudioSegment:
    """Mono 16-bit int16 samples as an AudioSegment."""
    return AudioSegment(data.astype(np.int16).tobytes(), frame_rate=rate, sample_width=2, channels=1)


def export(seg: AudioSegment, dst: str | Path | BinaryIO | None, fmt: str) -> bytes | None:
    """Write `seg` in one of CLIP_FORMATS to `dst` (a path or file object), or return the bytes if `dst` is None."""
    spec = CLIP_FORMATS[fmt]
    out = io.BytesIO() if dst is None else dst
    try:
        handle = seg.export(str(out) if isinstance(out, Path) else out, format=spec.container, codec=spec.codec,
                            bitrate=spec.bitrate, parameters=list(spec.parameters) or None)
    except (CouldntEncodeError, OSError) as e:
        if isinstance(dst, (str, Path)):
            Path(dst).unlink(missing_ok=True)
        raise AudioProcessingError(f"Could not encode the audio as {fmt}: {e}") from e
    if dst is None:
        handle.seek(0)
        return handle.read()
    if isinstance(dst, (str, Path)):
        handle.close()
        if not Path(dst).exists() or Path(dst).stat().st_size == 0:
            Path(dst).unlink(missing_ok=True)
            raise AudioProcessingError(f"Could not encode the audio as {fmt}: empty file")
    return None


def seconds(seg: AudioSegment) -> float:
    return len(seg) / 1000.0


# ------------------------------------------------------------------------------------- measurements

def probe_duration(src: Source) -> float | None:
    """Length in seconds, or None if the audio can't be read."""
    try:
        return round(seconds(load(src)), 3)
    except AudioProcessingError:
        return None


def info(src: Source) -> dict:
    """Duration, sample rate, channels, sample width and loudness of the audio."""
    seg = load(src)
    return {"duration": round(seconds(seg), 3), "frame_rate": seg.frame_rate, "channels": seg.channels,
            "sample_width": seg.sample_width, "dbfs": _finite(seg.dBFS), "max_dbfs": _finite(seg.max_dBFS)}


def frame_levels_db(src: Source, limit_seconds: float | None = None,
                    frame_seconds: float = LEVEL_FRAME_SECONDS) -> list[float]:
    """Loudness (RMS, dBFS) of every `frame_seconds` of audio (a partial last frame is dropped); silence is -100."""
    seg = load(src).set_channels(1)
    if limit_seconds:
        seg = seg[: int(limit_seconds * 1000)]
    step = max(1, int(round(frame_seconds * 1000)))
    return [max(round(seg[i:i + step].dBFS, 2), -100.0) for i in range(0, len(seg) - step + 1, step)]


def _finite(x: float) -> float | None:
    return None if math.isinf(x) or math.isnan(x) else round(x, 2)


# ------------------------------------------------------------------------------------------ cutting

def cut(src: Source, start: float, end: float) -> AudioSegment:
    """`start`-`end` seconds of the audio (sample-exact: pydub slices the decoded audio)."""
    if start < 0 or end <= start:
        raise ValueError(f"Invalid range {start}-{end}")
    seg = load(src)
    part = seg[int(round(start * 1000)): int(round(end * 1000))]
    if len(part) == 0:
        raise AudioProcessingError("Could not cut the audio: the range is outside the recording")
    return part


def cut_clip(src: Source, dst: str | Path | BinaryIO | None, start: float, end: float, fmt: str = "m4a") -> bytes | None:
    """`start`-`end` seconds as a file to download, in one of CLIP_FORMATS (m4a, mp3, flac, wav, ogg)."""
    return export(cut(src, start, end), dst, fmt)


def cut_audio(src: Source, dst: str | Path | BinaryIO | None, start: float, end: float) -> bytes | None:
    """`start`-`end` seconds as lossless FLAC, at the original rate and channels."""
    return cut_clip(src, dst, start, end, "flac")


def cut_with_margin(src: Source, dst: str | Path | BinaryIO | None, start: float, end: float,
                    margin: float = MARGIN_SECONDS) -> tuple[float, float, bytes | None]:
    """A segment plus `margin` seconds on each side (inside the recording), as FLAC: (cut_from, cut_to, data)."""
    seg = load(src)
    cut_from = round(max(0.0, start - margin), 3)
    cut_to = round(min(seconds(seg), end + margin), 3)
    return cut_from, cut_to, cut_audio(seg, dst, cut_from, cut_to)


def voice_clip(src: Source, start: float, end: float) -> bytes:
    """`start`-`end` (at least 0.5 s) as a mono OGG/Opus voice message at 32 kb/s."""
    seg = load(src)
    end = max(end, start + 0.5)
    return export(cut(seg, max(0.0, start), end).set_channels(1), None, "ogg")


def split(src: Source, parts: Iterable[tuple[float, float]], min_seconds: float = 0.5) -> list[AudioSegment]:
    """One segment per (start, end); every range is checked before anything is cut."""
    seg = load(src)
    total = seconds(seg)
    ranges = []
    for i, (start, end) in enumerate(parts, 1):
        end = min(end, total)
        if start < 0 or end - start < min_seconds:
            raise ValueError(f"Part {i} ({start:g}-{end:g} s) is shorter than {min_seconds} s")
        ranges.append((start, end))
    return [cut(seg, start, end) for start, end in ranges]


# ------------------------------------------------------------------- pieces for parallel transcription

@dataclass(frozen=True)
class Piece:
    index: int
    start: float  # seconds into the recording
    end: float


def cut_points(src: Source, max_seconds: float = 30.0, search_seconds: float = 5.0) -> list[Piece]:
    """Pieces of at most `max_seconds`, each cut at the quietest 20 ms of its last `search_seconds` when that is a
    real pause (under a quarter of the window's median energy); otherwise at the limit, so a word is rarely cut."""
    seg = for_transcription(load(src))
    n = len(seg)
    limit = int(max_seconds * 1000)
    search = int(min(search_seconds, max_seconds / 2) * 1000)
    pieces, start = [], 0
    while start < n:
        if n - start <= limit:
            end = n
        else:
            lo, hi = start + limit - search, start + limit
            energy = [seg[t:t + QUIET_FRAME_MS].rms ** 2 for t in range(lo, hi - QUIET_FRAME_MS + 1, QUIET_FRAME_MS)]
            end = hi
            if energy:
                quietest = int(np.argmin(energy))
                if energy[quietest] < 0.25 * float(np.median(energy)):
                    end = lo + quietest * QUIET_FRAME_MS + QUIET_FRAME_MS // 2
        pieces.append(Piece(len(pieces), round(start / 1000, 3), round(end / 1000, 3)))
        start = end
    return pieces


def export_wav16k(seg: AudioSegment, dst: str | Path | BinaryIO | None) -> bytes | None:
    """The segment as a 16 kHz mono 16-bit WAV (what the transcription services take)."""
    return export(for_transcription(seg), dst, "wav")


def decode_pcm(src: Source, rate: int = TRANSCRIPTION_RATE, limit_seconds: float | None = None) -> np.ndarray:
    """The audio as mono int16 samples at `rate` (input for numpy analysis: spectrogram, log-mel...)."""
    seg = for_transcription(load(src), rate)
    if limit_seconds:
        seg = seg[: int(limit_seconds * 1000)]
    return samples(seg)


def window_audio(src: Source, start: float, window_seconds: int = WHISPER_WINDOW_SECONDS) -> np.ndarray:
    """`window_seconds` of audio from `start`, 16 kHz mono float32 in [-1, 1), padded with silence (as Whisper pads)."""
    seg = for_transcription(load(src))[int(max(start, 0) * 1000): int((max(start, 0) + window_seconds) * 1000)]
    audio = samples(seg).astype(np.float32) / 32768.0
    out = np.zeros(window_seconds * TRANSCRIPTION_RATE, dtype=np.float32)
    out[: min(audio.shape[0], out.shape[0])] = audio[: out.shape[0]]
    return out


# ------------------------------------------------------------------------------------------ filters

EqType = Literal["peaking", "lowshelf", "highshelf", "highpass", "lowpass", "notch", "bandpass"]


@dataclass(frozen=True)
class EqBand:
    """One band of the parametric equalizer (as voice_embedding_service's app.schemas.EqBand)."""

    type: EqType = "peaking"
    frequency: float = 1000.0  # Hz
    gain: float = 0.0  # dB: peaking and shelves only
    q: float = 0.7071  # bandwidth (peaking, notch, band-pass)
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "EqBand":
        return cls(**{k: d[k] for k in ("type", "frequency", "gain", "q", "enabled") if k in d})

    def active(self) -> bool:
        """False when it does nothing (off, or a peak or shelf at 0 dB)."""
        return self.enabled and not (self.type in ("peaking", "lowshelf", "highshelf") and abs(self.gain) < 0.05)


def _scipy_effects():
    """pydub's scipy filters (Butterworth, band-pass, eq); importing registers them on AudioSegment."""
    try:
        from pydub import scipy_effects  # noqa: F401
    except ImportError as e:
        raise AudioProcessingError("Filters need scipy (pip install scipy)") from e
    return scipy_effects


def _nyquist_cap(seg: AudioSegment, hz: float) -> float:
    return min(hz, seg.frame_rate / 2 * 0.99)


def highpass(seg: AudioSegment, hz: float, order: int = BUTTERWORTH_ORDER) -> AudioSegment:
    """Remove what is below `hz` (rumble)."""
    return _scipy_effects().high_pass_filter(seg, _nyquist_cap(seg, hz), order=order)


def lowpass(seg: AudioSegment, hz: float, order: int = BUTTERWORTH_ORDER) -> AudioSegment:
    return _scipy_effects().low_pass_filter(seg, _nyquist_cap(seg, hz), order=order)


def bandpass(seg: AudioSegment, low_hz: float, high_hz: float, order: int = 2) -> AudioSegment:
    return _scipy_effects().band_pass_filter(seg, max(low_hz, 1.0), _nyquist_cap(seg, high_hz), order=order)


def notch(seg: AudioSegment, hz: float, q: float = 10.0) -> AudioSegment:
    """Remove a narrow band around `hz`: the audio minus its band-pass around `hz` (width hz / q)."""
    width = hz / q
    band = bandpass(seg, hz - width / 2, hz + width / 2)
    return seg.overlay(band.invert_phase())


def band(seg: AudioSegment, low_hz: float | None, high_hz: float | None) -> AudioSegment:
    """Keep only `low_hz`-`high_hz` (24 dB/octave each side); a missing edge is left open."""
    if low_hz:
        seg = highpass(seg, low_hz)
    if high_hz:
        seg = lowpass(seg, high_hz)
    return seg


def apply_eq(seg: AudioSegment, bands: Iterable[EqBand | dict] | None) -> AudioSegment:
    """The equalizer's bands, in order. Peaks and shelves use pydub's scipy eq (bandwidth = frequency / q)."""
    fx = None
    for b in bands or []:
        b = EqBand.from_dict(b) if isinstance(b, dict) else b
        if not b.active():
            continue
        fx = fx or _scipy_effects()
        f = _nyquist_cap(seg, b.frequency)
        if b.type == "highpass":
            seg = fx.high_pass_filter(seg, f, order=2)
        elif b.type == "lowpass":
            seg = fx.low_pass_filter(seg, f, order=2)
        elif b.type == "bandpass":
            width = f / b.q
            seg = bandpass(seg, f - width / 2, f + width / 2)
        elif b.type == "notch":
            seg = notch(seg, f, b.q)
        else:
            mode = {"peaking": "peak", "lowshelf": "low_shelf", "highshelf": "high_shelf"}[b.type]
            seg = fx.eq(seg, f, bandwidth=max(f / b.q, 1.0), filter_mode=mode, gain_dB=b.gain)
    return seg


def remove_hum(seg: AudioSegment, hz: int = 50, harmonics: int = 4, q: float = 30.0) -> AudioSegment:
    """Notch out mains hum at `hz` (50 or 60) and its first harmonics below Nyquist."""
    for k in range(1, harmonics + 1):
        if hz * k < seg.frame_rate / 2 * 0.95:
            seg = notch(seg, hz * k, q)
    return seg


def normalize_loudness(seg: AudioSegment, target_db: float = LOUDNESS_TARGET_DB,
                       peak_db: float = TRUE_PEAK_DB) -> AudioSegment:
    """Bring the average (RMS) level to `target_db` dBFS without letting the peak go over `peak_db`.

    pydub has no EBU R128 meter, so this stands in for ffmpeg's loudnorm=I=-16:TP=-1.5 (for speech, RMS dBFS
    and LUFS are within a few dB). Silence is returned as it is.
    """
    if math.isinf(seg.dBFS):
        return seg
    gain = target_db - seg.dBFS
    if seg.max_dBFS + gain > peak_db:
        gain = peak_db - seg.max_dBFS
    return seg.apply_gain(gain)


def shorten_pauses(seg: AudioSegment, min_silence_seconds: float = 1.0, silence_offset_db: float = 16.0) -> AudioSegment:
    """Shorten every pause longer than `min_silence_seconds` to that length (the middle of it is removed).

    A pause is audio quieter than the recording's average level minus `silence_offset_db`.
    """
    if math.isinf(seg.dBFS):
        return seg[: int(min_silence_seconds * 1000)]
    keep = int(min_silence_seconds * 1000)
    silences = detect_silence(seg, min_silence_len=keep + 1, silence_thresh=seg.dBFS - silence_offset_db)
    if not silences:
        return seg
    out, last = AudioSegment.empty(), 0
    for start, end in silences:
        out += seg[last:start + keep // 2]
        last = end - (keep - keep // 2)
    return out + seg[last:]


# ------------------------------------------------------------------------------------ whole pipelines

def band_filter(src: Source, dst: str | Path | BinaryIO | None, low_hz: float | None, high_hz: float | None,
                eq: Iterable[EqBand | dict] | None = None) -> bytes | None:
    """Keep only `low_hz`-`high_hz`, apply `eq` at the original rate, normalize the loudness, and write a 16 kHz mono
    FLAC (what the transcription uses). A band holds only part of the energy, hence the normalization."""
    return adjust_audio(src, dst, low_hz, high_hz, eq, gain_db=0.0, loudnorm=True)


def adjust_audio(src: Source, dst: str | Path | BinaryIO | None, low_hz: float | None = None,
                 high_hz: float | None = None, eq: Iterable[EqBand | dict] | None = None, gain_db: float = 0.0,
                 loudnorm: bool = True, highpass_hz: float | None = None, dehum_hz: int | None = None,
                 remove_silence: bool = False, min_silence_seconds: float = 1.0) -> bytes | None:
    """The audio with the given changes, as a 16 kHz mono FLAC, in this order: rumble high-pass, hum removal, the
    band, the equalizer, the gain, shortened pauses, loudness normalization. Nothing asked for: just converted."""
    seg = load(src).set_channels(1)
    if highpass_hz:
        seg = highpass(seg, highpass_hz)
    if dehum_hz:
        seg = remove_hum(seg, dehum_hz)
    seg = apply_eq(band(seg, low_hz, high_hz), eq)
    if gain_db:
        seg = seg.apply_gain(gain_db)
    if remove_silence:
        seg = shorten_pauses(seg, min_silence_seconds)
    if loudnorm:
        seg = normalize_loudness(seg)
    return export(for_transcription(seg), dst, "flac")


def convert(src: Source, dst: str | Path | BinaryIO | None, fmt: str, transcription: bool = False) -> bytes | None:
    """The whole audio in another of CLIP_FORMATS; with `transcription`, as 16 kHz mono first."""
    seg = load(src)
    return export(for_transcription(seg) if transcription else seg, dst, fmt)


def webm_to_ogg(path: str | Path) -> None:
    """A browser recording (WebM) as OGG/Opus in place, which the transcription service reads.

    pydub always decodes, so the audio is re-encoded to Opus at 48 kb/s (the original copied the stream when it
    was Opus already).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".ogg")
    try:
        seg = load(path, "webm")
        seg.export(str(tmp), format="ogg", codec="libopus", bitrate="48k").close()
    except (AudioProcessingError, CouldntEncodeError, OSError) as e:
        tmp.unlink(missing_ok=True)
        raise InvalidAudioError("Could not read the WebM recording") from e
    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise InvalidAudioError("Could not read the WebM recording")
    tmp.replace(path)
