"""Watch the WhatsApp bridge store for new audio files and transcribe them.

Single-worker queue: concurrent whisper processes would fight for GPU/RAM,
so we serialize. Events fire on file close (FSEvents), with a size-stable
fallback in case close events are missed.

Usage:
    uv run python watch.py
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from transcription import (
    DEFAULT_MODEL,
    already_transcribed,
    ensure_schema,
    lookup_message_for_audio_file,
    store_dir,
    transcribe_and_record,
    transcriptions_db_path,
)

logger = logging.getLogger("whatsapp-watch")

STABLE_CHECKS = 3  # number of consecutive size-equal reads before considering the file stable
STABLE_INTERVAL = 0.5  # seconds between size checks


def is_audio_path(p: Path) -> bool:
    return p.suffix.lower() == ".ogg" and p.stem.startswith("audio_")


def wait_until_stable(path: Path, timeout: float = 30.0) -> bool:
    """Return True when the file size stops changing (writer done)."""
    deadline = time.monotonic() + timeout
    last_size = -1
    equal_count = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size:
            equal_count += 1
            if equal_count >= STABLE_CHECKS and size > 0:
                return True
        else:
            equal_count = 0
            last_size = size
        time.sleep(STABLE_INTERVAL)
    logger.warning("file did not stabilize within %.0fs: %s", timeout, path)
    return False


def process_audio(path: Path, model_path: str) -> None:
    """Transcribe one audio file. Idempotent — skips if already recorded."""
    if not wait_until_stable(path):
        return
    lookup = lookup_message_for_audio_file(path)
    if not lookup:
        logger.warning("no message row matches %s — skipping", path)
        return
    message_id, chat_jid = lookup

    ensure_schema(transcriptions_db_path())
    conn = sqlite3.connect(transcriptions_db_path())
    try:
        if already_transcribed(conn, message_id, chat_jid):
            logger.info("already transcribed: %s", message_id[:12])
            return
        logger.info("transcribing %s (%s)", path.name, message_id[:12])
        result = transcribe_and_record(conn, message_id, chat_jid, path, model_path)
        if result is None:
            logger.info("empty transcript: %s", message_id[:12])
            return
        text, language, duration = result
        dur = f"{duration:.1f}s" if duration else "?s"
        logger.info("[ok] %s %s %s %dch", message_id[:12], language or "?", dur, len(text))
    finally:
        conn.close()


class AudioEventHandler(FileSystemEventHandler):
    def __init__(self, work_queue: queue.Queue[Path]):
        self._queue = work_queue
        self._seen: set[Path] = set()
        self._lock = threading.Lock()

    def _enqueue(self, path: Path) -> None:
        with self._lock:
            if path in self._seen:
                return
            self._seen.add(path)
        logger.debug("queued %s", path)
        self._queue.put(path)

    def forget(self, path: Path) -> None:
        """Remove a path from the dedup set once processing is done."""
        with self._lock:
            self._seen.discard(path)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if is_audio_path(path):
            self._enqueue(path)

    def on_closed(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if is_audio_path(path):
            self._enqueue(path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        dest = Path(getattr(event, "dest_path", ""))
        if is_audio_path(dest):
            self._enqueue(dest)


def worker_loop(
    work_queue: queue.Queue[Path],
    model_path: str,
    stop: threading.Event,
    handler: AudioEventHandler,
) -> None:
    while not stop.is_set():
        try:
            path = work_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            process_audio(path, model_path)
        except Exception:  # noqa: BLE001
            logger.exception("failed to process %s", path)
        finally:
            handler.forget(path)
            work_queue.task_done()


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    model_path = os.environ.get("WHISPER_MODEL_PATH", DEFAULT_MODEL)
    if not Path(model_path).exists():
        logger.error("model not found: %s", model_path)
        return 1

    watch_dir = store_dir()
    if not watch_dir.exists():
        logger.error("store directory not found: %s", watch_dir)
        return 1

    work_queue: queue.Queue[Path] = queue.Queue()
    stop = threading.Event()
    handler = AudioEventHandler(work_queue)
    worker = threading.Thread(
        target=worker_loop,
        args=(work_queue, model_path, stop, handler),
        name="transcribe-worker",
        daemon=True,
    )
    worker.start()
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()
    logger.info("watching %s (model=%s)", watch_dir, Path(model_path).name)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        observer.stop()
        observer.join(timeout=5)
        stop.set()
        worker.join(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
