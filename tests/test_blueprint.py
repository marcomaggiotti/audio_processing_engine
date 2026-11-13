import io
import json

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from audio_processing_engine.blueprint import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def wav(ms=3000):
    buf = io.BytesIO()
    Sine(440).to_audio_segment(duration=ms, volume=-20).export(buf, format="wav")
    return buf.getvalue()


def post(client, path, data=None, audio=None):
    form = dict(data or {})
    form["file"] = (io.BytesIO(wav() if audio is None else audio), "a.wav")
    return client.post(f"/audio/{path}", data=form, content_type="multipart/form-data")


def test_info(client):
    res = post(client, "info")
    assert res.status_code == 200
    assert res.json["format"] == "wav" and res.json["duration"] == pytest.approx(3.0, abs=0.01)


def test_levels_and_pieces(client):
    assert len(post(client, "levels", {"frame_seconds": "0.1"}).json["levels"]) == 30
    pieces = post(client, "pieces", {"max_seconds": "1"}).json["pieces"]
    assert [p["end"] for p in pieces] == [1.0, 2.0, 3.0]


def test_cut_returns_audio(client):
    res = post(client, "cut", {"start": "0.5", "end": "1.5", "format": "flac"})
    assert res.status_code == 200 and res.mimetype == "audio/flac"
    assert len(AudioSegment.from_file(io.BytesIO(res.data), format="flac")) == 1000


def test_voice_clip_and_convert(client):
    assert post(client, "voice-clip", {"start": "0", "end": "1"}).mimetype == "audio/ogg"
    res = post(client, "convert", {"format": "wav", "transcription": "true"})
    assert AudioSegment.from_file(io.BytesIO(res.data), format="wav").frame_rate == 16_000


def test_filter_and_adjust_with_options_json(client):
    eq = [{"type": "peaking", "frequency": 440, "gain": 6, "q": 1}]
    assert post(client, "filter", {"low_hz": "100", "high_hz": "4000", "eq": json.dumps(eq)}).status_code == 200
    res = post(client, "adjust", {"options": json.dumps({"gain_db": 3, "dehum_hz": 50, "eq": eq, "loudnorm": False})})
    assert res.status_code == 200 and res.mimetype == "audio/flac"


def test_errors(client):
    assert post(client, "info", audio=b"not audio at all").status_code == 415
    assert post(client, "cut", {"start": "1"}).status_code == 400
    assert post(client, "cut", {"start": "2", "end": "1"}).status_code == 400
    assert post(client, "cut", {"start": "0", "end": "1", "format": "aiff"}).status_code == 400
    assert post(client, "adjust", {"dehum_hz": "55"}).status_code == 400
    assert client.post("/audio/info").status_code == 400
