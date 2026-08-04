#!/usr/bin/env python3
"""WhatsApp bridge health check.

The bridge can die silently: the process keeps running and reconnecting,
but WhatsApp rejects it (e.g. "Client outdated 405") and no new messages
land in messages.db. Message traffic is normally constant, so a stale
newest-message timestamp is a reliable death signal.

Alerts via macOS notification banner + email (AWS SES SMTP).

Usage:
    python3 bridge_health.py          # check, alert only if stale
    python3 bridge_health.py --test   # force an alert to verify delivery
"""

import sqlite3
import subprocess
import smtplib
import sys
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

DB_PATH = Path.home() / "projects/mcp/whatsapp-mcp/whatsapp-bridge/store/messages.db"
BRIDGE_LOG = Path("/tmp/whatsapp-bridge-stdout.log")
STALE_HOURS = 24
ENV_FILE = Path.home() / ".config/whatsapp-bridge-health.env"

ERROR_SIGNATURES = [
    "Client outdated",
    "connect failure",
    "Logged out",
    "client version",
]


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def hours_since_last_message() -> tuple[float, str]:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT max(timestamp),"
            " (julianday('now') - julianday(max(timestamp))) * 24"
            " FROM messages"
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[1] is None:
        return float("inf"), "no messages found"
    return float(row[1]), str(row[0])


def log_diagnosis() -> str:
    if not BRIDGE_LOG.exists():
        return "bridge log not found"
    tail = BRIDGE_LOG.read_text(errors="replace").splitlines()[-200:]
    for sig in ERROR_SIGNATURES:
        hits = [l for l in tail if sig in l]
        if hits:
            return hits[-1].strip()
    return "no known error signature in recent log (bridge may be down entirely)"


def banner(title: str, text: str) -> None:
    script = f'display notification "{text}" with title "{title}" sound name "Basso"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def email(subject: str, body: str, env: dict) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = env["ALERT_FROM"]
    msg["To"] = env["ALERT_TO"]
    msg.set_content(body)
    with smtplib.SMTP(env["SMTP_HOST"], int(env["SMTP_PORT"]), timeout=30) as s:
        s.starttls()
        s.login(env["SMTP_USER"], env["SMTP_PASSWORD"])
        s.send_message(msg)


def main() -> int:
    test = "--test" in sys.argv
    age_hours, newest = hours_since_last_message()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if age_hours < STALE_HOURS and not test:
        print(f"[{now}] OK: newest message {age_hours:.1f}h old ({newest})")
        return 0

    diagnosis = log_diagnosis()
    label = "TEST ALERT" if test and age_hours < STALE_HOURS else "ALERT"
    subject = f"[whatsapp-bridge] {label}: no messages for {age_hours:.1f}h"
    body = (
        f"WhatsApp bridge on {subprocess.getoutput('hostname')} looks dead.\n\n"
        f"Newest message in messages.db: {newest} ({age_hours:.1f} hours ago)\n"
        f"Threshold: {STALE_HOURS}h\n\n"
        f"Log diagnosis:\n  {diagnosis}\n\n"
        f"Likely fix (the June 2026 outage was 'Client outdated 405'):\n"
        f"  cd ~/projects/mcp/whatsapp-mcp/whatsapp-bridge\n"
        f"  go get -u go.mau.fi/whatsmeow@latest && go build -o whatsapp-bridge .\n"
        f"  launchctl kickstart -k gui/$(id -u)/com.whatsapp-bridge\n"
    )

    print(f"[{now}] {subject}")
    print(body)
    banner("WhatsApp bridge dead", f"No messages for {age_hours:.1f}h. Check email for fix steps.")
    try:
        email(subject, body, load_env(ENV_FILE))
        print("email sent")
    except Exception as e:
        print(f"email FAILED: {e}", file=sys.stderr)
        banner("WhatsApp bridge alert email failed", str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
