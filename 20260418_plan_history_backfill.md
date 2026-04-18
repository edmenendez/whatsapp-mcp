# History Backfill for whatsapp-mcp — 2026-04-18

## Goal

Add an on-demand WhatsApp message history backfill capability to the bridge so Claude (via MCP) can request older messages for a specific chat. Response flows through the existing `handleHistorySync` path and lands in the local SQLite DB.

Upstream repo: `verygoodplugins/whatsapp-mcp` (fork of `lharries/whatsapp-mcp`).

## Why

The bridge today only stores messages WhatsApp pushes to it (initial sync on pairing + live messages). Older messages that pre-date pairing aren't retrievable without a new code path. `whatsmeow` exposes `Client.BuildHistorySyncRequest(lastKnownMessageInfo, count)` which asks the primary device (phone) for up to `count` messages before a given anchor. Response arrives asynchronously as `*events.HistorySync` with `Type == ON_DEMAND` and is processed by the bridge's existing `handleHistorySync` handler at `main.go:1670` — so storage is already wired; only the *request* side is missing.

Verified clear of duplicate work:
- `verygoodplugins/whatsapp-mcp`: no issues or PRs for history/backfill/older-messages/BuildHistorySyncRequest.
- `lharries/whatsapp-mcp` (upstream): PR #59's "history sync" refers to read-receipt history, not message backfill. No matches on backfill either.

## Architecture recap

```
Claude ──MCP──► Python FastMCP server ──HTTP──► Go bridge ──whatsmeow──► phone
                (thin HTTP wrapper)              (real logic)
```

- Go bridge: `whatsapp-bridge/main.go` — holds the `whatsmeow` client and the REST API
- Python MCP server: `whatsapp-mcp-server/main.py` — exposes `@mcp.tool()` functions that `httpx.post()` to the Go bridge

## Branching strategy

Two published feature branches based on `main`, one private integration branch for day-to-day running:

```
main ─────────────────────────────── (tracks upstream/main)
  ├── fix/contact-sync-from-whatsmeow ── PR #30 (already open)
  ├── feat/history-backfill ─────────── new, will become PR
  └── local/running ─────────────────── private merge of both
```

- Bridge runs from `local/running`
- Feature work happens on the feature branches
- `local/running` is never pushed

## Phase 1 — Go bridge endpoint

Branch: `feat/history-backfill` (off clean `main`, not off the contacts branch).

### 1.1 Handler: `/api/history/backfill`

Add alongside existing handlers around `main.go:1100–1250`.

Request body:
```json
{
  "chat_jid": "12025551234@s.whatsapp.net",
  "count": 50
}
```

Response: `202 Accepted` immediately, because the sync response is async and arrives as an event.

### 1.2 Helper: find oldest message for a chat

`findOldestMessageInfo(chatJID string) (*types.MessageInfo, error)` — reads the oldest row for the chat from `messages.db` and reconstructs the minimum `types.MessageInfo` whatsmeow needs:
- `Chat` (parsed JID)
- `Sender` (parsed from stored sender value)
- `ID` (message ID)
- `Timestamp`
- `IsFromMe`

### 1.3 Send the request

```go
req := client.BuildHistorySyncRequest(info, count)
_, err := client.SendPeerMessage(ctx, client.Store.ID.ToNonAD(), nil, req)
```

Note: peer target is the user's own primary device JID, **not** the chat JID.

### 1.4 Logging

Log the outbound request (chat, anchor message ID, count) so we can correlate with the later `Received history sync event` log line from `handleHistorySync`.

### 1.5 Validation

- Reject if `chat_jid` empty or unparseable
- Clamp `count` to `[1, 50]` (50 is whatsmeow's recommended max per request)
- Return 404-style error if no prior messages exist for the chat (can't anchor without one)

## Phase 2 — Python MCP tool wrapper

Same branch.

### 2.1 New tool: `backfill_history`

In `whatsapp-mcp-server/main.py`, add a `@mcp.tool()` following the existing pattern (`send_message`, `download_media`).

```python
@mcp.tool()
async def backfill_history(chat_jid: str, count: int = 50) -> dict:
    """Request older messages for a chat from the primary device.

    Best-effort and asynchronous: the response arrives seconds to minutes
    later and is stored in the local SQLite DB. Poll list_messages to
    observe new rows. Phone must be online. Rate-limited; pause ~60s
    between repeated calls for the same chat.
    """
    ...
```

Keep the docstring tight but explicit about async/best-effort/rate-limit so the LLM doesn't hammer it.

### 2.2 No other Python changes needed

`list_messages` already returns everything in the DB, so newly backfilled rows appear on the next query.

## Phase 3 — Local running branch

```bash
git checkout main && git pull upstream main
git checkout -b local/running
git merge --no-ff fix/contact-sync-from-whatsmeow
git merge --no-ff feat/history-backfill
cd whatsapp-bridge && go build -o whatsapp-bridge
# restart the bridge process; restart Claude Desktop
```

## Phase 4 — Testing

Before committing to the PR path, prove the endpoint works locally.

1. Pick a chat with ~sparse history — `list_messages` shows oldest row on record
2. `curl -XPOST http://localhost:8080/api/history/backfill -d '{"chat_jid": "...", "count": 50}'`
3. Expected immediate: `202 Accepted` + log line from the bridge
4. Expected within seconds to minutes: `Received history sync event with N conversations` in bridge logs
5. Expected in SQLite: new rows with `timestamp` older than the previous oldest

Failure modes to confirm handling of:
- Phone offline → no response event; endpoint still returned 202 (correct — best-effort)
- Chat with no prior messages → 400 from endpoint
- Invalid JID → 400 from endpoint
- Called twice in rapid succession → second call still 202'd; WhatsApp may drop or delay second request (document, don't fight)

## Phase 5 — Upstream PR

Recommended order, given the current review lag (PR #30 has sat untouched for 2 days, dependabot PRs from mid-April are piling up):

1. **Open an issue first** on `verygoodplugins/whatsapp-mcp` describing the feature and asking if they'd accept it. Cheaper than a cold PR.
2. If green-lit: open PR from `feat/history-backfill` → upstream `main`. Include:
   - Go endpoint + Python tool
   - README update under the "What we changed" list
   - Short docstring notes on best-effort / rate-limit / phone-online requirements
   - Test stub if appropriate (CI uses pytest + golangci-lint)
3. Do **not** bundle with the contacts fix (PR #30). Keep each PR one concern.

## Risks and gotchas

- **Best-effort.** WhatsApp may return nothing. Not a bug.
- **Phone dependency.** Primary device must be online when the request fires.
- **Rate limiting.** Enforced server-side as a per-chat cooldown (default 60s, matches wacli's guidance). Callers get `429 Too Many Requests` with `Retry-After`. Overridable via `HISTORY_BACKFILL_COOLDOWN` env var. NOT global — parallel backfills across many distinct chats are not throttled; a future "backfill all chats" loop must pace itself (1–2s between distinct chats).
- **Hard wall.** WhatsApp doesn't retain history indefinitely on its servers; chats already backfilled to 2022 likely won't yield more.
- **LID chats.** Already handled by `handleHistorySync` via the existing LID store mapping (line 1692). No extra work unless tests reveal otherwise.
- **Go binary is not hot-reloaded.** Every Go change needs `go build` + bridge restart. Python MCP reloads on its own via `uv run`.
- **Squash-merge drift.** When PR #30 merges upstream, its SHAs change. Rebuild `local/running` by `reset --hard main` + re-merge remaining feature branches; don't try to rebase the old merges.
- **Plan file hygiene.** This file sits at the repo root. It's not in `.gitignore` — keep it out of commits or add it before pushing.

## Out of scope (explicit non-goals)

- Multi-chat "backfill all" endpoint. Caller can loop if needed; avoids adding pacing logic to the bridge.
- Configurable `FullSyncDaysLimit` on initial pairing. Only helps fresh pairings; doesn't solve the existing-session case we care about.
- Automatic backfill on startup. Too aggressive; stays opt-in per request.

## Locked decisions (2026-04-18)

- **Endpoint path: `/api/history/backfill`.** Introduces a `/api/history/*` namespace for future siblings (e.g. `/api/history/status`). Existing endpoints are flat single-verb (`/api/send`, `/api/download`), and a second history endpoint under a flat name would look orphaned.
- **Python tool name: `backfill_history`.** Matches the repo's `verb_noun` convention (`send_message`, `list_messages`, `search_contacts`) and wacli's naming so the concept is recognizable across tools.
- **No correlation token in the 202.** whatsmeow's `events.HistorySync` doesn't correlate back to the triggering request, so any status endpoint would be heuristic and fragile. The natural completion check is observable: "did a message older than my previous oldest appear?" — which `list_messages` already answers. Response body is `{"accepted": true, "chat_jid": "...", "anchor_message_id": "..."}`. Revisit only if testing reveals a real need.
- **Server-side per-chat cooldown (60s default).** Rate limiting is enforced in the Go bridge, not just documented in the Python tool's docstring. The bridge is the shared chokepoint — every caller (MCP, curl, future scripts) must go through it — so defending only at the docstring layer is paper armor against an eager LLM or a buggy loop. Keyed per chat (matches wacli's guidance; different chats aren't the same rate-limit vector). In-memory state only (restart clears the map, acceptable because restarts are rare). Returns `429 Too Many Requests` with a `Retry-After` header. Cooldown value is overridable via `HISTORY_BACKFILL_COOLDOWN` env var (accepts any `time.ParseDuration` format, e.g. `60s`, `2m`). Known gaps documented inline in `main.go` near the `backfillCooldown` var: first request per chat always passes, parallel floods across many distinct chats are not throttled, cooldown timer doesn't account for the phone's processing time.
