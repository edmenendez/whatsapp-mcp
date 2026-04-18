"""Rename legacy audio filenames to the new msgID-suffixed format.

Before: audio_YYYYMMDD_HHMMSS.ogg
After:  audio_YYYYMMDD_HHMMSS_MSGID.ogg

The Go bridge used to produce timestamp-only filenames, which collided
when two messages arrived in the same second. This script walks the
store, matches each legacy audio file to its source message via
(chat_jid, timestamp) with file_length as a tiebreaker, renames the
file, and updates the messages.filename column to match.

Usage:
    uv run python scripts/migrate_audio_filenames.py --dry-run
    uv run python scripts/migrate_audio_filenames.py

Environment:
    WHATSAPP_DB_PATH  Path to messages.db
                      (default: ../whatsapp-bridge/store/messages.db)
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_DB = REPO_ROOT / "whatsapp-bridge" / "store" / "messages.db"

MESSAGES_DB_PATH = Path(os.environ.get("WHATSAPP_DB_PATH", str(DEFAULT_DB)))
STORE_DIR = MESSAGES_DB_PATH.parent

LEGACY_AUDIO_RE = re.compile(r"^audio_(\d{8})_(\d{6})\.ogg$")


def build_plan(conn: sqlite3.Connection) -> list[tuple[Path, Path, str, str]]:
    """Return (src_path, dst_path, message_id, chat_jid) for each file to migrate."""
    plans: list[tuple[Path, Path, str, str]] = []
    for chat_dir in sorted(STORE_DIR.iterdir()):
        if not chat_dir.is_dir():
            continue
        chat_jid = chat_dir.name
        for audio in sorted(chat_dir.glob("audio_*.ogg")):
            m = LEGACY_AUDIO_RE.match(audio.name)
            if not m:
                continue  # already migrated or unexpected name
            date_str, time_str = m.groups()
            ts_token = f"{date_str}_{time_str}"

            candidates: list[tuple[str, int | None]] = []
            for mid, ts_str, file_length in conn.execute(
                "SELECT id, timestamp, file_length FROM messages "
                "WHERE media_type='audio' AND chat_jid=?",
                (chat_jid,),
            ):
                try:
                    dt = datetime.fromisoformat(ts_str)
                except (TypeError, ValueError):
                    continue
                if dt.strftime("%Y%m%d_%H%M%S") == ts_token:
                    candidates.append((mid, file_length))

            if not candidates:
                print(f"[no-match] {chat_jid}/{audio.name} — no message at {ts_token}")
                continue

            chosen_mid: str | None = None
            if len(candidates) == 1:
                chosen_mid = candidates[0][0]
            else:
                try:
                    actual_size = audio.stat().st_size
                except OSError:
                    actual_size = None
                if actual_size is not None:
                    for mid, file_length in candidates:
                        if file_length == actual_size:
                            chosen_mid = mid
                            break
                if chosen_mid is None:
                    ids = ", ".join(c[0][:12] for c in candidates)
                    print(f"[ambiguous] {chat_jid}/{audio.name} — {len(candidates)} candidates ({ids}), no size match")
                    continue

            dst = audio.parent / f"audio_{ts_token}_{chosen_mid}.ogg"
            plans.append((audio, dst, chosen_mid, chat_jid))
    return plans


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate legacy audio filenames to include message ID.")
    ap.add_argument("--dry-run", action="store_true", help="Show plan; do not rename")
    args = ap.parse_args()

    if not MESSAGES_DB_PATH.exists():
        sys.exit(f"error: messages.db not found at {MESSAGES_DB_PATH}")

    # Use read-only connection for dry-run, writable otherwise.
    if args.dry_run:
        conn = sqlite3.connect(f"file:{MESSAGES_DB_PATH}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(MESSAGES_DB_PATH)

    try:
        plans = build_plan(conn)

        print(f"\n{len(plans)} file(s) to migrate.\n")
        for src, dst, _mid, _jid in plans:
            print(f"  {src.parent.name}/{src.name}  ->  {dst.name}")

        if args.dry_run or not plans:
            return 0

        applied = 0
        skipped = 0
        for src, dst, mid, chat_jid in plans:
            if dst.exists():
                print(f"[skip] {dst.name} already exists")
                skipped += 1
                continue
            try:
                src.rename(dst)
            except OSError as e:
                print(f"[fail] rename {src.name}: {e}")
                continue
            conn.execute(
                "UPDATE messages SET filename=? WHERE id=? AND chat_jid=?",
                (dst.name, mid, chat_jid),
            )
            applied += 1
        conn.commit()
        print(f"\nMigrated {applied}/{len(plans)}, skipped {skipped}.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
