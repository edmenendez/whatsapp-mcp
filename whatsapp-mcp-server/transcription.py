"""Transcribe WhatsApp audio messages with whisper-cpp and search across text + transcripts.

CLI:
    uv run python transcription.py backfill [--model PATH] [--chat JID] [--limit N] [--dry-run]
    uv run python transcription.py redo [--model PATH] [--chat JID] [--limit N] [--dry-run]
    uv run python transcription.py search "query" [--chat JID] [--limit N] [--only text|audio]

Model routing: the main model (turbo) handles most clips. Clips shorter than
SHORT_CLIP_SECONDS go straight to the accurate model, and any clip the main
model tags with a language outside EXPECTED_LANGUAGES is re-run on it. Turbo
misdetects the language on very short clips (3 s Spanish came back as
Icelandic); large-v3 does not. Language is always auto-detected, never forced.

Environment:
    WHISPER_MODEL_PATH           Path to the main ggml-*.bin model file
    WHISPER_ACCURATE_MODEL_PATH  Path to the model for short or odd-language clips
    WHATSAPP_DB_PATH             Path to messages.db (default: ../whatsapp-bridge/store/messages.db)
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

from image_description import within_max_age

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
ACCURATE_MODEL = os.environ.get(
    "WHISPER_ACCURATE_MODEL_PATH",
    "/Volumes/Crucial X10/whisper-models/ggml-large-v3.bin",
)
SHORT_CLIP_SECONDS = 6.0
EXPECTED_LANGUAGES = {"es", "en"}

BRIDGE_API_URL = os.environ.get("WHATSAPP_API_URL", "http://localhost:8080/api")


def _bridge_headers() -> dict[str, str]:
    """Bearer header for bridge REST calls, or {} when no token is configured.

    Delegates to whatsapp.py so token resolution (WHATSAPP_BRIDGE_TOKEN, else
    the generated .bridge-token file) stays defined in exactly one place.
    Returns {} on import failure so a partial checkout degrades to the
    pre-auth behavior rather than crashing the backfill.
    """
    try:
        from whatsapp import _bridge_headers as _resolve
    except Exception:
        return {}
    return _resolve()


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
            headers=_bridge_headers(),
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


def find_audio_file(chat_jid: str, timestamp: str, message_id: str | None = None) -> Path | None:
    """Locate the audio file for a message.

    New format (after bridge filename-collision fix): audio_YYYYMMDD_HHMMSS_MSGID.ogg
    Legacy format (pre-fix): audio_YYYYMMDD_HHMMSS.ogg
    Both are accepted; new format preferred.
    """
    chat_dir = store_dir() / chat_jid
    if not chat_dir.exists():
        return None
    ts = datetime.fromisoformat(timestamp)
    ts_token = ts.strftime('%Y%m%d_%H%M%S')

    # Preferred: new format with message ID
    if message_id:
        new_fmt = chat_dir / f"audio_{ts_token}_{message_id}.ogg"
        if new_fmt.exists():
            return new_fmt

    # Legacy: timestamp only
    legacy = chat_dir / f"audio_{ts_token}.ogg"
    if legacy.exists():
        return legacy

    # Defensive: any same-second audio file
    for candidate in chat_dir.glob(f"audio_{ts_token}*"):
        if candidate.suffix in (".ogg", ".opus", ".m4a", ".mp3"):
            return candidate
    return None


def transcribe_audio(
    ogg_path: Path,
    model_path: str,
    accurate_model_path: str | None = ACCURATE_MODEL,
) -> tuple[str, str | None, float | None, str]:
    """Convert to 16kHz mono WAV and run whisper-cli with model routing.

    Returns (text, language, duration_sec, model_path_used). Routing is skipped
    when the accurate model is missing or is the main model.
    """
    require_binary("ffmpeg")
    require_binary("whisper-cli")

    use_accurate = bool(
        accurate_model_path and accurate_model_path != model_path and Path(accurate_model_path).exists()
    )

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
        # 16 kHz mono 16-bit PCM is 32,000 bytes per second after the 44-byte header.
        clip_seconds = max(wav.stat().st_size - 44, 0) / 32000

        if use_accurate and clip_seconds < SHORT_CLIP_SECONDS:
            return (*_run_whisper(wav, accurate_model_path), accurate_model_path)

        text, language, duration = _run_whisper(wav, model_path)
        if use_accurate and language not in EXPECTED_LANGUAGES:
            return (*_run_whisper(wav, accurate_model_path), accurate_model_path)
        return text, language, duration, model_path


def _run_whisper(wav: Path, model_path: str) -> tuple[str, str | None, float | None]:
    """Run whisper-cli on a prepared WAV with language auto-detection."""
    out_prefix = wav.with_name(f"out_{Path(model_path).stem}")
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
    """Insert or update a transcription row.

    An upsert rather than INSERT OR REPLACE: REPLACE deletes the old row without
    firing the delete trigger, which would leave the old text in the FTS index.
    """
    conn.execute(
        """INSERT INTO transcriptions
           (message_id, chat_jid, transcription, language, model, duration_sec, audio_sha256)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (message_id, chat_jid) DO UPDATE SET
               transcription = excluded.transcription,
               language = excluded.language,
               model = excluded.model,
               duration_sec = excluded.duration_sec,
               audio_sha256 = excluded.audio_sha256,
               transcribed_at = CURRENT_TIMESTAMP""",
        (message_id, chat_jid, text, language, model_name, duration, sha256_file(audio_path)),
    )
    conn.commit()


def lookup_message_for_audio_file(audio_path: Path) -> tuple[str, str] | None:
    """Given an audio file path, return (message_id, chat_jid).

    New format (post filename-collision fix): audio_YYYYMMDD_HHMMSS_MSGID.ogg
      - message ID is in the filename; return directly.

    Legacy format: audio_YYYYMMDD_HHMMSS.ogg
      - Disambiguate by matching messages.file_length to actual file size.
    """
    chat_jid = audio_path.parent.name
    stem = audio_path.stem
    if not stem.startswith("audio_"):
        return None
    rest = stem[len("audio_"):]  # "YYYYMMDD_HHMMSS" or "YYYYMMDD_HHMMSS_MSGID"
    parts = rest.split("_", 2)
    if len(parts) < 2:
        return None

    if len(parts) == 3:
        # New format: message ID embedded in filename.
        return parts[2], chat_jid

    # Legacy format: resolve via messages DB + file_length.
    ts_token = f"{parts[0]}_{parts[1]}"
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
    text, language, duration, used_model = transcribe_audio(audio_path, model_path)
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
        Path(used_model).name,
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
        audio = find_audio_file(chat_jid, timestamp, message_id)
        if not audio and args.download and within_max_age(timestamp, args.max_age_days):
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


def cmd_redo(args: argparse.Namespace) -> int:
    """Re-run routing on existing rows that the accurate model would now handle.

    Candidates are rows not already from the accurate model whose stored
    duration is under the cutoff (plus slack, since whisper's last segment ends
    before the audio does) or whose language is unexpected. Only rows the
    router actually sends to the accurate model are rewritten.
    """
    model_path = args.model or DEFAULT_MODEL
    if not Path(ACCURATE_MODEL).exists():
        sys.exit(f"error: accurate model not found: {ACCURATE_MODEL}")

    ensure_schema(transcriptions_db_path())
    msgs_conn = sqlite3.connect(f"file:{messages_db_path()}?mode=ro", uri=True)
    trans_conn = sqlite3.connect(transcriptions_db_path())

    sql = (
        "SELECT message_id, chat_jid, language, duration_sec FROM transcriptions "
        "WHERE model != ? AND (duration_sec IS NULL OR duration_sec < ? "
        f"OR language IS NULL OR language NOT IN ({','.join('?' * len(EXPECTED_LANGUAGES))}))"
    )
    params: list = [Path(ACCURATE_MODEL).name, SHORT_CLIP_SECONDS + 2, *sorted(EXPECTED_LANGUAGES)]
    if args.chat:
        sql += " AND chat_jid = ?"
        params.append(args.chat)
    sql += " ORDER BY duration_sec"
    if args.limit:
        sql += " LIMIT ?"
        params.append(args.limit)
    candidates = trans_conn.execute(sql, params).fetchall()
    print(f"Found {len(candidates)} candidate rows.")
    if args.dry_run:
        return 0

    redone = kept = skipped = failed = 0
    for message_id, chat_jid, old_language, _ in candidates:
        short_id = message_id[:12]
        row = msgs_conn.execute(
            "SELECT timestamp FROM messages WHERE id = ? AND chat_jid = ?", (message_id, chat_jid)
        ).fetchone()
        audio = find_audio_file(chat_jid, row[0], message_id) if row else None
        if not audio:
            skipped += 1
            print(f"[skip] {short_id} no audio file")
            continue
        try:
            text, language, duration, used_model = transcribe_audio(audio, model_path)
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"[fail] {short_id} whisper/ffmpeg: {str(e.stderr)[:200]}")
            continue
        if used_model != ACCURATE_MODEL:
            kept += 1
            continue
        if not text:
            skipped += 1
            print(f"[skip] {short_id} empty transcript")
            continue
        record_transcript(
            trans_conn, message_id, chat_jid, audio, text, language, duration, Path(used_model).name
        )
        redone += 1
        print(f"[redo] {short_id} {old_language or '?'}->{language or '?'} {len(text)}ch")

    print(f"\nDone: {redone} redone, {kept} kept main model, {skipped} skipped, {failed} failed.")
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
    bf.add_argument(
        "--max-age-days",
        type=int,
        default=None,
        help="With --download, only fetch media newer than N days (older CDN URLs have expired)",
    )
    bf.set_defaults(func=cmd_backfill)

    rd = sub.add_parser("redo", help="Re-transcribe existing short or odd-language rows with the accurate model")
    rd.add_argument("--model", default=None, help="Path to the main whisper ggml model (default: env WHISPER_MODEL_PATH)")
    rd.add_argument("--chat", default=None, help="Restrict to one chat JID")
    rd.add_argument("--limit", type=int, default=None, help="Max rows to consider")
    rd.add_argument("--dry-run", action="store_true", help="Only count the candidate rows")
    rd.set_defaults(func=cmd_redo)

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
