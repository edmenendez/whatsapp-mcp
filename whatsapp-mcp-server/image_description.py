"""Describe WhatsApp image messages with a local Ollama VLM (Qwen3-VL) and
back-fill descriptions + verbatim OCR into a searchable FTS5 index.

CLI:
    uv run python image_description.py backfill [--model NAME] [--chat JID] [--limit N] [--dry-run] [--download] [--skip-status]
    uv run python image_description.py describe MESSAGE_ID CHAT_JID

Environment:
    IMAGE_MODEL_NAME     Ollama model tag (default: qwen3-vl:8b-instruct)
    OLLAMA_API_URL       Ollama REST base (default: http://localhost:11434)
    WHATSAPP_DB_PATH     messages.db (default: ../whatsapp-bridge/store/messages.db)
    WHATSAPP_API_URL     Bridge REST base (default: http://localhost:8080/api)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_MESSAGES_DB = REPO_ROOT / "whatsapp-bridge" / "store" / "messages.db"
SCHEMA_SQL = HERE / "schema_image_descriptions.sql"

DEFAULT_MODEL = os.environ.get("IMAGE_MODEL_NAME", "qwen3-vl:8b-instruct")
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
BRIDGE_API_URL = os.environ.get("WHATSAPP_API_URL", "http://localhost:8080/api")

DESCRIBE_PROMPT = (
    'Describe this image in 1-2 sentences. Then, if any text is visible, '
    'add a "Text:" line with the text transcribed verbatim. Be concise.'
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def messages_db_path() -> Path:
    return Path(os.environ.get("WHATSAPP_DB_PATH", str(DEFAULT_MESSAGES_DB)))


def descriptions_db_path() -> Path:
    return messages_db_path().parent / "image_descriptions.db"


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def download_via_bridge(message_id: str, chat_jid: str) -> Path | None:
    """Ask the Go bridge to download media. Returns local path or None."""
    import requests

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


def find_image_file(chat_jid: str, timestamp: str, message_id: str | None = None) -> Path | None:
    """Locate the image file for a message.

    New format (post filename-collision fix): image_YYYYMMDD_HHMMSS_MSGID.ext
    Legacy format: image_YYYYMMDD_HHMMSS.ext
    """
    chat_dir = store_dir() / chat_jid
    if not chat_dir.exists():
        return None

    try:
        ts = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return None
    ts_token = ts.strftime("%Y%m%d_%H%M%S")

    if message_id:
        for ext in IMAGE_EXTENSIONS:
            candidate = chat_dir / f"image_{ts_token}_{message_id}{ext}"
            if candidate.exists():
                return candidate

    for ext in IMAGE_EXTENSIONS:
        legacy = chat_dir / f"image_{ts_token}{ext}"
        if legacy.exists():
            return legacy

    for candidate in chat_dir.glob(f"image_{ts_token}*"):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Return (width, height) by reading JPEG/PNG headers. Returns (None, None) on failure."""
    try:
        with open(path, "rb") as f:
            header = f.read(32)
        if header.startswith(b"\xff\xd8"):  # JPEG
            return _jpeg_dimensions(path)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(header) >= 24:
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                return width, height
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return _webp_dimensions(path)
    except OSError:
        pass
    return None, None


def _jpeg_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Walk JPEG markers to find SOF0..SOF3 and read height, width."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None, None
        while True:
            byte = f.read(1)
            while byte and byte != b"\xff":
                byte = f.read(1)
            while byte == b"\xff":
                byte = f.read(1)
            if not byte:
                return None, None
            marker = byte[0]
            if 0xC0 <= marker <= 0xC3:
                f.read(3)
                height = int.from_bytes(f.read(2), "big")
                width = int.from_bytes(f.read(2), "big")
                return width, height
            size = int.from_bytes(f.read(2), "big")
            if size < 2:
                return None, None
            f.seek(size - 2, 1)


def _webp_dimensions(path: Path) -> tuple[int | None, int | None]:
    with open(path, "rb") as f:
        f.seek(12)
        chunk = f.read(4)
        if chunk == b"VP8 ":
            f.seek(26)
            width = int.from_bytes(f.read(2), "little") & 0x3FFF
            height = int.from_bytes(f.read(2), "little") & 0x3FFF
            return width, height
        if chunk == b"VP8L":
            f.seek(21)
            b = f.read(4)
            width = 1 + (((b[1] & 0x3F) << 8) | b[0])
            height = 1 + (((b[3] & 0x0F) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
            return width, height
        if chunk == b"VP8X":
            f.seek(24)
            width = 1 + int.from_bytes(f.read(3), "little")
            height = 1 + int.from_bytes(f.read(3), "little")
            return width, height
    return None, None


def check_ollama_ready(model: str) -> str | None:
    """Verify the Ollama daemon is reachable and the requested model is pulled.

    Returns None on success, or a human-readable error describing what to fix.
    """
    import requests

    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
    except requests.RequestException as e:
        return (
            f"cannot reach Ollama at {OLLAMA_API_URL} ({e.__class__.__name__}). "
            "Start it with `brew services start ollama` "
            "(or `ollama serve`), then retry."
        )
    if resp.status_code != 200:
        return f"Ollama responded with HTTP {resp.status_code}: {resp.text[:200]}"

    installed = {m.get("name") for m in resp.json().get("models", [])}
    if model not in installed:
        return (
            f"model '{model}' is not pulled. "
            f"Run: ollama pull {model}"
        )
    return None


def describe_image(
    path: Path,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str | None]:
    """Call Ollama to describe an image. Returns (description, ocr_text).

    ocr_text is None when the model returns no "Text:" section.
    """
    import requests

    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "prompt": DESCRIBE_PROMPT,
        "images": [b64],
        "stream": False,
        "keep_alive": "30m",
    }
    resp = requests.post(f"{OLLAMA_API_URL}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    text = (resp.json().get("response") or "").strip()
    return _split_description_and_ocr(text)


def _split_description_and_ocr(text: str) -> tuple[str, str | None]:
    """Split the model output on the first 'Text:' marker (case-insensitive, bol).

    Everything before the marker (stripped) is the description.
    Everything after (stripped) is OCR. Returns (desc, None) if no marker.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("text:"):
            head = line.strip()[5:].strip()
            tail_lines = lines[i + 1:]
            ocr_parts = [head] if head else []
            ocr_parts.extend(line for line in tail_lines)
            description = "\n".join(lines[:i]).strip()
            ocr_text = "\n".join(ocr_parts).strip() or None
            return description, ocr_text
    return text.strip(), None


def record_description(
    conn: sqlite3.Connection,
    message_id: str,
    chat_jid: str,
    image_path: Path,
    description: str,
    ocr_text: str | None,
    width: int | None,
    height: int | None,
    model_name: str,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO image_descriptions
           (message_id, chat_jid, description, ocr_text, model, image_sha256, width, height)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            message_id,
            chat_jid,
            description,
            ocr_text,
            model_name,
            sha256_file(image_path),
            width,
            height,
        ),
    )
    conn.commit()


def already_described(conn: sqlite3.Connection, message_id: str, chat_jid: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM image_descriptions WHERE message_id=? AND chat_jid=? LIMIT 1",
        (message_id, chat_jid),
    ).fetchone()
    return row is not None


def describe_and_record(
    conn: sqlite3.Connection,
    message_id: str,
    chat_jid: str,
    image_path: Path,
    model_name: str,
) -> tuple[str, str | None, int | None, int | None] | None:
    description, ocr_text = describe_image(image_path, model_name)
    if not description and not ocr_text:
        return None
    width, height = image_dimensions(image_path)
    record_description(
        conn,
        message_id,
        chat_jid,
        image_path,
        description,
        ocr_text,
        width,
        height,
        model_name,
    )
    return description, ocr_text, width, height


def iter_undescribed(
    msgs_conn: sqlite3.Connection,
    img_conn: sqlite3.Connection,
    chat: str | None,
    limit: int | None,
    skip_status: bool,
):
    done = {
        (mid, jid)
        for mid, jid in img_conn.execute(
            "SELECT message_id, chat_jid FROM image_descriptions"
        )
    }
    sql = (
        "SELECT id, chat_jid, timestamp, sender FROM messages "
        "WHERE media_type = 'image' "
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
    if skip_status:
        sql += " AND chat_jid != 'status@broadcast'"
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
    model_name = args.model or DEFAULT_MODEL
    if err := check_ollama_ready(model_name):
        print(f"error: {err}", file=sys.stderr)
        return 2
    ensure_schema(descriptions_db_path())
    msgs_conn = sqlite3.connect(f"file:{messages_db_path()}?mode=ro", uri=True)
    img_conn = sqlite3.connect(descriptions_db_path())

    targets = list(iter_undescribed(msgs_conn, img_conn, args.chat, args.limit, args.skip_status))
    print(f"Found {len(targets)} image messages missing a description.")
    if args.dry_run:
        return 0

    ok = skipped = failed = 0
    for message_id, chat_jid, timestamp, _sender in targets:
        short_id = message_id[:12]
        image = find_image_file(chat_jid, timestamp, message_id)
        if not image and args.download:
            print(f"[dl]   {short_id} fetching from bridge...")
            image = download_via_bridge(message_id, chat_jid)
        if not image:
            skipped += 1
            print(f"[skip] {short_id} no image file for {chat_jid} @ {timestamp}")
            continue
        try:
            result = describe_and_record(img_conn, message_id, chat_jid, image, model_name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[fail] {short_id} {type(e).__name__}: {e}")
            continue

        if result is None:
            skipped += 1
            print(f"[skip] {short_id} empty description ({image.name})")
            continue

        description, ocr_text, _w, _h = result
        ok += 1
        ocr_chars = len(ocr_text) if ocr_text else 0
        print(f"[ok] {short_id} desc={len(description)}ch ocr={ocr_chars}ch")

    print(f"\nDone: {ok} ok, {skipped} skipped, {failed} failed.")
    return 0 if failed == 0 else 1


def cmd_describe(args: argparse.Namespace) -> int:
    model_name = args.model or DEFAULT_MODEL
    if err := check_ollama_ready(model_name):
        print(f"error: {err}", file=sys.stderr)
        return 2
    ensure_schema(descriptions_db_path())

    msgs_conn = sqlite3.connect(f"file:{messages_db_path()}?mode=ro", uri=True)
    row = msgs_conn.execute(
        "SELECT timestamp FROM messages WHERE id = ? AND chat_jid = ? AND media_type = 'image'",
        (args.message_id, args.chat_jid),
    ).fetchone()
    msgs_conn.close()
    if not row:
        print(f"error: no image message {args.message_id} in chat {args.chat_jid}", file=sys.stderr)
        return 1
    timestamp = row[0]

    image = find_image_file(args.chat_jid, timestamp, args.message_id)
    if not image:
        image = download_via_bridge(args.message_id, args.chat_jid)
    if not image:
        print("error: image file not found and bridge download failed", file=sys.stderr)
        return 1

    img_conn = sqlite3.connect(descriptions_db_path())
    try:
        result = describe_and_record(img_conn, args.message_id, args.chat_jid, image, model_name)
    finally:
        img_conn.close()

    if result is None:
        print("(empty description)")
        return 1

    description, ocr_text, width, height = result
    dims = f"{width}x{height}" if width and height else "?x?"
    print(f"[{dims}] {description}")
    if ocr_text:
        print(f"\nText: {ocr_text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Describe WhatsApp image messages with a local VLM and search results.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bf = sub.add_parser("backfill", help="Describe image messages missing from image_descriptions.db")
    bf.add_argument("--model", default=None, help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    bf.add_argument("--chat", default=None, help="Restrict to one chat JID")
    bf.add_argument("--limit", type=int, default=None, help="Max images to describe")
    bf.add_argument("--dry-run", action="store_true", help="Only list what would be described")
    bf.add_argument("--download", action="store_true", help="Fetch missing images from the Go bridge")
    bf.add_argument("--skip-status", action="store_true", help="Skip status@broadcast messages")
    bf.set_defaults(func=cmd_backfill)

    ds = sub.add_parser("describe", help="Describe a single image message by id/chat")
    ds.add_argument("message_id", help="Message ID")
    ds.add_argument("chat_jid", help="Chat JID")
    ds.add_argument("--model", default=None, help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    ds.set_defaults(func=cmd_describe)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
