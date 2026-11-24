"""Shared HTTP mode: bearer token, browser refusal, startup checks, bridge timeouts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

import audio
import http_server
import whatsapp

PORT = 8783
TOKEN = "t" * 40
LIST_TOOLS = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
SERVER_DIR = Path(__file__).resolve().parents[1]


def make_client(host="127.0.0.1"):
    # A fresh server per test: the SDK's session manager can only run once.
    mcp = FastMCP("test")

    @mcp.tool()
    def ping() -> str:
        """Reply pong."""
        return "pong"

    app = http_server.build_app(mcp, TOKEN, PORT)
    return TestClient(app, base_url=f"http://{host}:{PORT}")


def post(client, auth=f"Bearer {TOKEN}", **headers):
    headers = {"accept": "application/json, text/event-stream", **headers}
    if auth:
        headers["authorization"] = auth
    return client.post("/mcp", headers=headers, json=LIST_TOOLS)


def test_right_token_lists_tools():
    with make_client() as client:
        resp = post(client)
    assert resp.status_code == 200
    assert "ping" in resp.text


@pytest.mark.parametrize("auth", [None, "Bearer " + "x" * 40, TOKEN, f"Basic {TOKEN}"])
def test_missing_or_wrong_token_is_refused(auth):
    with make_client() as client:
        resp = post(client, auth=auth)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_browser_origin_is_refused_even_from_localhost():
    with make_client() as client:
        assert post(client, origin="http://localhost:3000").status_code == 403


def test_foreign_host_is_refused():
    with make_client(host="evil.example") as client:
        assert post(client).status_code == 421


def test_short_token_file_is_refused(tmp_path):
    path = tmp_path / "token"
    path.write_text("short\n")
    with pytest.raises(SystemExit):
        http_server.load_token(str(path))


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"WHATSAPP_MCP_TRANSPORT": "http"}, "WHATSAPP_MCP_HTTP_TOKEN_FILE"),
        ({"WHATSAPP_MCP_TRANSPORT": "sse"}, "not supported"),
        ({"WHATSAPP_MCP_TRANSPORT": "http", "WHATSAPP_MCP_HOST": "0.0.0.0"}, "only listens on 127.0.0.1"),
    ],
)
def test_network_transports_refuse_to_start_unsafely(env, expected):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("WHATSAPP_MCP_")}
    clean.update(env, WHATSAPP_MCP_PORT="1")
    result = subprocess.run(
        [sys.executable, "main.py"], cwd=SERVER_DIR, env=clean, capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0
    assert expected in result.stderr


# --- bridge timeouts ----------------------------------------------------------


SENDS = [
    lambda: whatsapp.send_message("123", "hi"),
    lambda: whatsapp.send_file("123", "/tmp/x.jpg"),
    lambda: whatsapp.send_audio_message("123", "/tmp/x.ogg"),
    lambda: whatsapp.send_reaction("123", "msgid", "👍"),
]


def test_every_bridge_call_has_a_timeout(monkeypatch):
    seen = []

    def fake_post(url, **kwargs):
        seen.append(kwargs.get("timeout"))
        raise requests.ConnectionError("down")

    monkeypatch.setattr(whatsapp.requests, "post", fake_post)
    monkeypatch.setattr(whatsapp.os.path, "isfile", lambda path: True)
    for send in SENDS:
        send()
    whatsapp.download_media("msgid", "123@s.whatsapp.net")
    assert seen == [whatsapp.BRIDGE_TIMEOUT_S] * 5


@pytest.mark.parametrize("send", SENDS)
def test_send_timeout_says_it_may_have_been_sent(monkeypatch, send):
    def fake_post(url, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(whatsapp.requests, "post", fake_post)
    monkeypatch.setattr(whatsapp.os.path, "isfile", lambda path: True)
    ok, message = send()
    assert not ok
    assert "may have been sent" in message


def test_ffmpeg_has_a_timeout(monkeypatch, tmp_path):
    source = tmp_path / "in.mp3"
    source.write_bytes(b"x")

    def fake_run(cmd, **kwargs):
        assert kwargs.get("timeout") == 120
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="longer than 120s"):
        audio.convert_to_opus_ogg(str(source), str(tmp_path / "out.ogg"))
