"""Shared local HTTP mode: one server for many MCP clients, token required.

Every request needs ``Authorization: Bearer <token>``, the token being the
content of WHATSAPP_MCP_HTTP_TOKEN_FILE. Only localhost Host headers are
accepted and any Origin header (a web page) is refused. Stateless, so clients
keep working across a server restart.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings


class BearerAuth:
    """ASGI wrapper: every HTTP request needs the bearer token. Lifespan passes through."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and not hmac.compare_digest(
            dict(scope["headers"]).get(b"authorization", b""), self.expected
        ):
            body = json.dumps({"error": "Missing or invalid token"}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


def load_token(path: str) -> str:
    token = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise SystemExit(f"Token in {path} must be at least 32 characters")
    return token


def build_app(mcp: FastMCP, token: str, port: int):
    # The SDK reads these when building the app. Its localhost default allows any
    # localhost Origin, which would let a local web page in; allow none.
    mcp.settings.stateless_http = True
    mcp.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}"], allowed_origins=[]
    )
    return BearerAuth(mcp.streamable_http_app(), token)


def serve(mcp: FastMCP, host: str, port: int) -> None:
    if host not in ("127.0.0.1", "localhost"):
        raise SystemExit(f"HTTP mode only listens on 127.0.0.1 (got WHATSAPP_MCP_HOST={host})")
    token_file = os.getenv("WHATSAPP_MCP_HTTP_TOKEN_FILE", "").strip()
    if not token_file:
        raise SystemExit("HTTP mode needs WHATSAPP_MCP_HTTP_TOKEN_FILE (this server can send messages)")
    app = build_app(mcp, load_token(token_file), port)

    import uvicorn

    # Clients hold connections open: cap the shutdown wait or a restart hangs.
    uvicorn.run(app, host=host, port=port, lifespan="on", log_level="info", timeout_graceful_shutdown=5)
