"""Transcribe WhatsApp audio messages with whisper-cpp and search across text + transcripts.

CLI:
    uv run python transcription.py backfill [--model PATH] [--chat JID] [--limit N] [--dry-run]
    uv run python transcription.py search "query" [--chat JID] [--limit N] [--only text|audio]

Environment:
    WHISPER_MODEL_PATH   Path to a ggml-*.bin model file
    WHATSAPP_DB_PATH     Path to messages.db (default: ../whatsapp-bridge/store/messages.db)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_MESSAGES_DB = REPO_ROOT / "whatsapp-bridge" / "store" / "messages.db"
DEFAULT_TRANSCRIPTIONS_DB = REPO_ROOT / "whatsapp-bridge" / "store" / "transcriptions.db"
DEFAULT_STORE_DIR = REPO_ROOT / "whatsapp-bridge" / "store"
SCHEMA_SQL = HERE / "schema_transcriptions.sql"

DEFAULT_MODEL = os.environ.get(
    "WHISPER_MODEL_PATH",
    "/Volumes/Crucial X10/whisper-models/ggml-large-v3-turbo.bin",
)

BRIDGE_API_URL = os.environ.get("WHATSAPP_API_URL", "http://localhost:8080/api")


def messages_db_path() -> Path:
    return Path(os.environ.get("WHATSAPP_DB_PATH", str(DEFAULT_MESSAGES_DB)))


def transcriptions_db_path() -> Path:
    return messages_db_path().parent / "transcriptions.db"


def store_dir() -> Path:
    return messages_db_path().parent


def ensure_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        with open(SCHEMA_SQL) as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        sys.exit(f"error: {name} not found in PATH")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def download_via_bridge(message_id: str, chat_jid: str) -> Path | None:
    """Ask the Go bridge to download media. Returns local path or None."""
    import requests  # local import so search has no hard dep on network libs

    try:
        resp = requests.post(
            f"{BRIDGE_API_URL}/download",
            json={"message_id": message_id, "chat_jid": chat_jid},
            timeout=60,
        )
    except requests.RequestException as e:
        print(f"       bridge error: {e}")
        return None
    if resp.status_code != 200:
        print(f"       bridge HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    data = resp.json()
    if not data.get("success"):
        return None
    path = data.get("path")
    return Path(path) if path else None


def find_audio_file(chat_jid: str, timestamp: str) -> Path | None:
    """Locate the audio file for a message. Names are audio_YYYYMMDD_HHMMSS.ogg in local time."""
    chat_dir = store_dir() / chat_jid
    if not chat_dir.exists():
        return None
    ts = datetime.fromisoformat(timestamp)
    expected = chat_dir / f"audio_{ts.strftime('%Y%m%d_%H%M%S')}.ogg"
    if expected.exists():
        return expected
    # Fall back to any audio file within the same second (handles edge cases).
    stem = f"audio_{ts.strftime('%Y%m%d_%H%M%S')}"
    for candidate in chat_dir.glob(f"{stem}*"):
        if candidate.suffix in (".ogg", ".opus", ".m4a", ".mp3"):
            return candidate
    return None


def transcribe_audio(ogg_path: Path, model_path: str) -> tuple[str, str | None, float | None]:
    """Convert to 16kHz mono WAV and run whisper-cli. Returns (text, language, duration_sec)."""
    require_binary("ffmpeg")
    require_binary("whisper-cli")

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(ogg_path),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(wav),
            ],
            check=True,
            capture_output=True,
        )

        out_prefix = Path(td) / "out"
        subprocess.run(
            [
                "whisper-cli",
                "-m", model_path,
                "-f", str(wav),
                "-l", "auto",
                "-oj",
                "-of", str(out_prefix),
                "--no-prints",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        json_path = Path(str(out_prefix) + ".json")
        data = json.loads(json_path.read_text())

    segments = data.get("transcription", [])
    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
    language = data.get("result", {}).get("language")
    duration = None
    if segments:
        last = segments[-1].get("offsets", {}).get("to")
        if isinstance(last, (int, float)):
            duration = last / 1000.0
    return text, language, duration


def record_transcript(
    conn: sqlite3.Connection,
    message_id: str,
    chat_jid: str,
    audio_path: Path,
    text: str,
    language: str | None,
    duration: float | None,
    model_name: str,
) -> None:
    """Insert or replace a transcription row."""
    conn.execute(
        """INSERT OR REPLACE INTO transcriptions
           (message_id, chat_jid, transcription, language, model, duration_sec, audio_sha256)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (message_id, chat_jid, text, language, model_name, duration, sha256_file(audio_path)),
    )
    conn.commit()


def lookup_message_for_audio_file(audio_path: Path) -> tuple[str, str] | None:
    """Given a store/<chat_jid>/audio_YYYYMMDD_HHMMSS.ogg path, return (message_id, chat_jid).

    The bridge names files by timestamp, so two messages arriving in the same
    second collide on a single filename. Disambiguate by matching file_length
    to the actual file size on disk.
    """
    chat_jid = audio_path.parent.name
    stem = audio_path.stem
    if not stem.startswith("audio_"):
        return None
    ts_token = stem[len("audio_"):]

    try:
        actual_size = audio_path.stat().st_size
    except OSError:
        actual_size = None

    conn = sqlite3.connect(f"file:{messages_db_path()}?mode=ro", uri=True)
    try:
        candidates: list[tuple[str, int | None]] = []
        for mid, ts_str, file_length in conn.execute(
            "SELECT id, timestamp, file_length FROM messages WHERE media_type='audio' AND chat_jid=?",
            (chat_jid,),
        ):
            try:
                dt = datetime.fromisoformat(ts_str)
            except (TypeError, ValueError):
                continue
            if dt.strftime("%Y%m%d_%H%M%S") == ts_token:
                candidates.append((mid, file_length))
    finally:
        conn.close()

    if not candidates:
        return None
    if actual_size is not None:
        for mid, file_length in candidates:
            if file_length == actual_size:
                return mid, chat_jid
    return candidates[0][0], chat_jid


def already_transcribed(conn: sqlite3.Connection, message_id: str, chat_jid: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM transcriptions WHERE message_id=? AND chat_jid=? LIMIT 1",
        (message_id, chat_jid),
    ).fetchone()
    return row is not None


def transcribe_and_record(
    conn: sqlite3.Connection,
    message_id: str,
    chat_jid: str,
    audio_path: Path,
    model_path: str,
) -> tuple[str, str | None, float | None] | None:
    """Transcribe audio and write to DB. Returns (text, language, duration) on success, None if empty."""
    text, language, duration = transcribe_audio(audio_path, model_path)
    if not text:
        return None
    record_transcript(
        conn,
        message_id,
        chat_jid,
        audio_path,
        text,
        language,
        duration,
        Path(model_path).name,
    )
    return text, language, duration


def iter_untranscribed(
    msgs_conn: sqlite3.Connection,
    trans_conn: sqlite3.Connection,
    chat: str | None,
    limit: int | None,
):
    done = {
        (mid, jid)
        for mid, jid in trans_conn.execute(
            "SELECT message_id, chat_jid FROM transcriptions"
        )
    }
    # Skip audios missing the fields whatsmeow needs to decrypt the media.
    # History-sync rows from before the bridge was paired only carry a stub
    # and can never be fetched; no point wasting bridge calls on them.
    sql = (
        "SELECT id, chat_jid, timestamp, sender FROM messages "
        "WHERE media_type = 'audio' "
        "AND url != '' "
        "AND length(media_key) > 0 "
        "AND length(file_sha256) > 0 "
        "AND length(file_enc_sha256) > 0 "
        "AND file_length > 0"
    )
    params: list = []
    if chat:
        sql += " AND chat_jid = ?"
        params.append(chat)
    sql += " ORDER BY timestamp DESC"
    cursor = msgs_conn.execute(sql, params)
    count = 0
    for row in cursor:
        if (row[0], row[1]) in done:
            continue
        yield row
        count += 1
        if limit and count >= limit:
            return


def cmd_backfill(args: argparse.Namespace) -> int:
    model_path = args.model or DEFAULT_MODEL
    if not Path(model_path).exists():
        sys.exit(f"error: model not found: {model_path}")

    ensure_schema(transcriptions_db_path())
    msgs_conn = sqlite3.connect(f"file:{messages_db_path()}?mode=ro", uri=True)
    trans_conn = sqlite3.connect(transcriptions_db_path())

    targets = list(iter_untranscribed(msgs_conn, trans_conn, args.chat, args.limit))
    print(f"Found {len(targets)} untranscribed audio messages.")
    if args.dry_run:
        return 0

    ok = skipped = failed = 0
    for message_id, chat_jid, timestamp, sender in targets:
        short_id = message_id[:12]
        audio = find_audio_file(chat_jid, timestamp)
        if not audio and args.download:
            print(f"[dl]   {short_id} fetching from bridge...")
            audio = download_via_bridge(message_id, chat_jid)
        if not audio:
            skipped += 1
            print(f"[skip] {short_id} no audio file for {chat_jid} @ {timestamp}")
            continue
        try:
            result = transcribe_and_record(trans_conn, message_id, chat_jid, audio, model_path)
        except subprocess.CalledProcessError as e:
            failed += 1
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")[:200] if isinstance(e.stderr, bytes) else (e.stderr or "")[:200]
            print(f"[fail] {short_id} whisper/ffmpeg: {stderr}")
            continue
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[fail] {short_id} {type(e).__name__}: {e}")
            continue

        if result is None:
            skipped += 1
            print(f"[skip] {short_id} empty transcript ({audio.name})")
            continue

        text, language, duration = result
        ok += 1
        dur = f"{duration:.1f}s" if duration else "?s"
        print(f"[ok] {short_id} {language or '?'} {dur} {len(text)}ch")

    print(f"\nDone: {ok} ok, {skipped} skipped, {failed} failed.")
    return 0 if failed == 0 else 1


def cmd_search(args: argparse.Namespace) -> int:
    ensure_schema(transcriptions_db_path())
    conn = sqlite3.connect(f"file:{transcriptions_db_path()}?mode=ro", uri=True)
    conn.execute(f"ATTACH DATABASE 'file:{messages_db_path()}?mode=ro' AS msgs")

    results: list[dict] = []

    if args.only != "audio":
        like = f"%{args.query}%"
        sql = (
            "SELECT m.id, m.chat_jid, m.timestamp, m.sender, m.content, "
            "       c.name AS chat_name, 'text' AS kind, NULL AS snippet "
            "FROM msgs.messages m "
            "LEFT JOIN msgs.chats c ON c.jid = m.chat_jid "
            "WHERE m.content IS NOT NULL AND m.content != '' "
            "  AND LOWER(m.content) LIKE LOWER(?)"
        )
        params: list = [like]
        if args.chat:
            sql += " AND m.chat_jid = ?"
            params.append(args.chat)
        sql += " ORDER BY m.timestamp DESC LIMIT ?"
        params.append(args.limit)
        for row in conn.execute(sql, params):
            results.append(dict(zip(
                ("id", "chat_jid", "timestamp", "sender", "content", "chat_name", "kind", "snippet"),
                row,
            )))

    if args.only != "text":
        sql = (
            "SELECT t.message_id, t.chat_jid, m.timestamp, m.sender, t.transcription, "
            "       c.name AS chat_name, 'audio' AS kind, "
            "       snippet(transcriptions_fts, 0, '[', ']', '...', 16) AS snippet "
            "FROM transcriptions_fts "
            "JOIN transcriptions t ON t.rowid = transcriptions_fts.rowid "
            "JOIN msgs.messages m ON m.id = t.message_id AND m.chat_jid = t.chat_jid "
            "LEFT JOIN msgs.chats c ON c.jid = t.chat_jid "
            "WHERE transcriptions_fts MATCH ?"
        )
        params = [args.query]
        if args.chat:
            sql += " AND t.chat_jid = ?"
            params.append(args.chat)
        sql += " ORDER BY bm25(transcriptions_fts), m.timestamp DESC LIMIT ?"
        params.append(args.limit)
        for row in conn.execute(sql, params):
            results.append(dict(zip(
                ("id", "chat_jid", "timestamp", "sender", "content", "chat_name", "kind", "snippet"),
                row,
            )))

    results.sort(key=lambda r: r["timestamp"] or "", reverse=True)
    results = results[: args.limit]
    if not results:
        print("(no matches)")
        return 0

    for r in results:
        display = r["chat_name"] or r["chat_jid"]
        marker = "[audio]" if r["kind"] == "audio" else "[text]"
        body = r["snippet"] or r["content"] or ""
        body = body.replace("\n", " ")
        if len(body) > 240:
            body = body[:240] + "..."
        print(f"{r['timestamp']}  {marker}  {display}  ({r['sender']})")
        print(f"    {body}")
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Transcribe WhatsApp audio and search across text + transcripts.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bf = sub.add_parser("backfill", help="Transcribe audio messages missing from transcriptions.db")
    bf.add_argument("--model", default=None, help="Path to whisper ggml model (default: env WHISPER_MODEL_PATH)")
    bf.add_argument("--chat", default=None, help="Restrict to one chat JID")
    bf.add_argument("--limit", type=int, default=None, help="Max messages to transcribe")
    bf.add_argument("--dry-run", action="store_true", help="Only list what would be transcribed")
    bf.add_argument("--download", action="store_true", help="Fetch missing audio from the Go bridge")
    bf.set_defaults(func=cmd_backfill)

    s = sub.add_parser("search", help="Search text messages and audio transcripts")
    s.add_argument("query", help="Search query (FTS5 syntax for audio, LIKE for text)")
    s.add_argument("--chat", default=None, help="Restrict to one chat JID")
    s.add_argument("--limit", type=int, default=20, help="Max results (default 20)")
    s.add_argument("--only", choices=["text", "audio"], default=None, help="Restrict to one kind")
    s.set_defaults(func=cmd_search)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
