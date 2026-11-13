import io
import math
import wave

import numpy as np
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from audio_processing_engine import operations as ops
from audio_processing_engine.errors import AudioProcessingError, InvalidAudioError
from audio_processing_engine.formats import detect_format


def tone(hz=440, ms=1000, db=-20.0, rate=44_100):
    return Sine(hz, sample_rate=rate).to_audio_segment(duration=ms, volume=db)


def speech_like():
    """Three tones with pauses of 0.3 s and 2 s between them."""
    silent = AudioSegment.silent
    return (tone(300, 1000) + silent(300, 44_100) + tone(500, 1000) + silent(2000, 44_100) + tone(700, 1000))


def band_power(seg, hz):
    x = ops.samples(seg.set_channels(1).set_sample_width(2)).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / seg.frame_rate)
    return spectrum[(freqs > hz * 0.9) & (freqs < hz * 1.1)].sum()


def test_detect_format():
    buf = io.BytesIO()
    tone().export(buf, format="wav")
    assert detect_format(buf.getvalue()) == "wav"
    assert detect_format(b"fLaC....") == "flac"
    with pytest.raises(InvalidAudioError):
        detect_format(b"not audio")


def test_load_rejects_garbage():
    with pytest.raises(AudioProcessingError):
        ops.load(io.BytesIO(b"garbage" * 100), "wav")


def test_info_and_duration(tmp_path):
    path = tmp_path / "a.wav"
    tone(ms=1500).export(path, format="wav")
    assert ops.probe_duration(path) == pytest.approx(1.5, abs=0.01)
    info = ops.info(path)
    assert info["frame_rate"] == 44_100 and info["channels"] == 1
    assert ops.probe_duration(tmp_path / "missing.wav") is None


def test_frame_levels():
    seg = tone(ms=500) + AudioSegment.silent(500, 44_100)
    levels = ops.frame_levels_db(seg, frame_seconds=0.05)
    assert len(levels) == 20
    assert levels[0] > -30 and levels[-1] == -100.0


@pytest.mark.parametrize("fmt", ["m4a", "mp3", "flac", "wav", "ogg"])
def test_cut_clip_formats(fmt):
    data = ops.cut_clip(tone(ms=3000), None, 0.5, 2.0, fmt)
    back = AudioSegment.from_file(io.BytesIO(data), format=None if fmt == "m4a" else fmt)
    assert ops.seconds(back) == pytest.approx(1.5, abs=0.1)


def test_cut_audio_is_exact_flac(tmp_path):
    dst = tmp_path / "part.flac"
    ops.cut_audio(tone(ms=3000), dst, 1.0, 2.25)
    back = AudioSegment.from_file(dst)
    assert len(back) == 1250 and back.frame_rate == 44_100


def test_cut_invalid_range():
    with pytest.raises(ValueError):
        ops.cut(tone(), 1.0, 0.5)
    with pytest.raises(AudioProcessingError):
        ops.cut(tone(ms=1000), 5.0, 6.0)


def test_cut_with_margin_stays_inside():
    cut_from, cut_to, data = ops.cut_with_margin(tone(ms=2000), None, 0.1, 1.9)
    assert (cut_from, cut_to) == (0.0, 2.0) and data


def test_voice_clip_is_mono_ogg():
    data = ops.voice_clip(AudioSegment.from_mono_audiosegments(tone(), tone()), 0.0, 0.2)
    back = AudioSegment.from_file(io.BytesIO(data), format="ogg")
    assert back.channels == 1 and ops.seconds(back) == pytest.approx(0.5, abs=0.05)


def test_split_checks_all_ranges_first():
    parts = ops.split(tone(ms=3000), [(0, 1), (1, 3)])
    assert [len(p) for p in parts] == [1000, 2000]
    with pytest.raises(ValueError):
        ops.split(tone(ms=3000), [(0, 1), (2.8, 3.1)])


def test_cut_points_prefers_pauses():
    seg = tone(ms=8000) + AudioSegment.silent(400, 44_100) + tone(ms=8000)
    pieces = ops.cut_points(seg, max_seconds=10, search_seconds=5)
    assert len(pieces) == 2
    assert 8.0 <= pieces[0].end <= 8.4  # inside the pause, not at 10 s
    assert pieces[-1].end == pytest.approx(16.4, abs=0.01)
    continuous = ops.cut_points(tone(ms=25_000), max_seconds=10)
    assert [p.end for p in continuous] == [10.0, 20.0, 25.0]


def test_export_wav16k():
    data = ops.export_wav16k(AudioSegment.from_mono_audiosegments(tone(), tone()), None)
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)


def test_decode_pcm_and_window():
    pcm = ops.decode_pcm(tone(ms=2000), 8000)
    assert pcm.dtype == np.int16 and len(pcm) == 16_000
    window = ops.window_audio(tone(ms=2000), 1.0)
    assert window.shape == (30 * 16_000,) and window.dtype == np.float32
    assert np.abs(window[:16_000]).max() > 0.05 and not window[16_000:].any()


def test_band_removes_outside_frequencies():
    seg = tone(100, 2000).overlay(tone(1000, 2000)).overlay(tone(6000, 2000))
    out = ops.band(seg, 300, 3000)
    assert band_power(out, 1000) > 0.5 * band_power(seg, 1000)
    assert band_power(out, 100) < 0.05 * band_power(seg, 100)
    assert band_power(out, 6000) < 0.05 * band_power(seg, 6000)


def test_remove_hum():
    seg = tone(50, 2000).overlay(tone(100, 2000)).overlay(tone(1000, 2000))
    out = ops.remove_hum(seg, 50)
    assert band_power(out, 50) < 0.2 * band_power(seg, 50)
    assert band_power(out, 1000) > 0.5 * band_power(seg, 1000)


def test_eq_peak_boosts_its_band():
    seg = tone(1000, 2000).overlay(tone(3000, 2000))
    out = ops.apply_eq(seg, [{"type": "peaking", "frequency": 1000, "gain": 12, "q": 2}])
    assert band_power(out, 1000) / band_power(out, 3000) > band_power(seg, 1000) / band_power(seg, 3000)
    assert ops.apply_eq(seg, [{"type": "peaking", "frequency": 1000, "gain": 0}]) is seg  # inactive band


def test_normalize_loudness_respects_peak():
    quiet = tone(db=-40)
    out = ops.normalize_loudness(quiet)
    assert out.max_dBFS <= ops.TRUE_PEAK_DB + 0.1
    assert out.dBFS > quiet.dBFS + 20
    silence = AudioSegment.silent(500)
    assert ops.normalize_loudness(silence) is silence


def test_shorten_pauses():
    seg = speech_like()
    out = ops.shorten_pauses(seg, min_silence_seconds=1.0)
    assert len(out) == pytest.approx(len(seg) - 1000, abs=30)  # only the 2 s pause shrinks, to 1 s


def test_adjust_audio_outputs_transcription_flac(tmp_path):
    dst = tmp_path / "out.flac"
    ops.adjust_audio(speech_like(), dst, 200, 4000, [{"type": "lowshelf", "frequency": 200, "gain": -6}],
                     gain_db=3, highpass_hz=80, dehum_hz=50, remove_silence=True)
    back = AudioSegment.from_file(dst)
    assert (back.frame_rate, back.channels) == (16_000, 1)
    assert not math.isinf(back.dBFS)


def test_band_filter(tmp_path):
    data = ops.band_filter(tone(db=-45), None, 300, 3400)
    back = AudioSegment.from_file(io.BytesIO(data), format="flac")
    assert back.frame_rate == 16_000 and back.dBFS > -30


def test_webm_to_ogg(tmp_path):
    path = tmp_path / "rec.webm"
    tone().export(path, format="webm", codec="libopus")
    ops.webm_to_ogg(path)
    assert path.read_bytes()[:4] == b"OggS"
    bad = tmp_path / "bad.webm"
    bad.write_bytes(b"\x1a\x45\xdf\xa3 not really")
    with pytest.raises(InvalidAudioError):
        ops.webm_to_ogg(bad)
