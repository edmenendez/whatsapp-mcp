"""Tests for whisper model routing and transcript storage."""

import sqlite3
import subprocess
from pathlib import Path

import pytest

import transcription
from transcription import SCHEMA_SQL, SHORT_CLIP_SECONDS, record_transcript, transcribe_audio

MAIN = "/models/ggml-large-v3-turbo.bin"
ACCURATE = "/models/ggml-large-v3.bin"


@pytest.fixture
def fake_tools(monkeypatch, tmp_path):
    """Fake ffmpeg (writes a WAV of a chosen length) and whisper (returns a chosen language per model)."""
    state = {"seconds": 30.0, "languages": {MAIN: "es", ACCURATE: "es"}, "calls": []}

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(b"\0" * (44 + int(state["seconds"] * 32000)))
        return subprocess.CompletedProcess(cmd, 0)

    def fake_whisper(wav, model_path):
        state["calls"].append(model_path)
        return f"text from {Path(model_path).name}", state["languages"][model_path], state["seconds"]

    monkeypatch.setattr(transcription.subprocess, "run", fake_run)
    monkeypatch.setattr(transcription, "_run_whisper", fake_whisper)
    monkeypatch.setattr(transcription, "require_binary", lambda name: name)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: str(self) in (MAIN, ACCURATE) or real_exists(self))
    return state


def _transcribe(tmp_path):
    audio = tmp_path / "a.ogg"
    audio.write_bytes(b"ogg")
    return transcribe_audio(audio, MAIN, ACCURATE)


class TestModelRouting:
    def test_long_clip_in_expected_language_uses_main_model_only(self, fake_tools, tmp_path):
        assert _transcribe(tmp_path)[3] == MAIN
        assert fake_tools["calls"] == [MAIN]

    def test_short_clip_goes_straight_to_accurate_model(self, fake_tools, tmp_path):
        fake_tools["seconds"] = SHORT_CLIP_SECONDS - 0.5
        assert _transcribe(tmp_path)[3] == ACCURATE
        assert fake_tools["calls"] == [ACCURATE]

    def test_clip_at_cutoff_uses_main_model(self, fake_tools, tmp_path):
        fake_tools["seconds"] = SHORT_CLIP_SECONDS
        assert _transcribe(tmp_path)[3] == MAIN

    def test_unexpected_language_is_rerun_on_accurate_model(self, fake_tools, tmp_path):
        """Turbo tagged short Spanish as Icelandic; the accurate model decides the language itself."""
        fake_tools["languages"] = {MAIN: "is", ACCURATE: "es"}
        text, language, _, used = _transcribe(tmp_path)
        assert (language, used) == ("es", ACCURATE)
        assert fake_tools["calls"] == [MAIN, ACCURATE]

    def test_accurate_model_result_is_kept_even_if_unexpected(self, fake_tools, tmp_path):
        """A genuinely Italian note stays Italian; nothing forces Spanish."""
        fake_tools["languages"] = {MAIN: "it", ACCURATE: "it"}
        assert _transcribe(tmp_path)[1] == "it"

    def test_missing_accurate_model_falls_back_to_main(self, fake_tools, tmp_path):
        fake_tools["seconds"] = 2.0
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg")
        assert transcribe_audio(audio, MAIN, "/models/missing.bin")[3] == MAIN


class TestRecordTranscript:
    def test_rewrite_replaces_old_text_in_search_index(self, tmp_path):
        conn = sqlite3.connect(tmp_path / "t.db")
        conn.executescript(SCHEMA_SQL.read_text())
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"ogg")
        record_transcript(conn, "m1", "c1", audio, "hacer billy", "is", 3.0, "turbo")
        record_transcript(conn, "m1", "c1", audio, "estamos para servirle", "es", 3.0, "large")

        rows = conn.execute("SELECT transcription, language, model FROM transcriptions").fetchall()
        assert rows == [("estamos para servirle", "es", "large")]
        fts = lambda q: conn.execute(  # noqa: E731
            "SELECT count(*) FROM transcriptions_fts WHERE transcriptions_fts MATCH ?", (q,)
        ).fetchone()[0]
        assert fts("servirle") == 1
        assert fts("billy") == 0
