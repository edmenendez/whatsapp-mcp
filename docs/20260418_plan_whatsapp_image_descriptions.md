# 20260418 — Plan: WhatsApp image descriptions (Qwen3-VL local)

## Goal

Mirror the existing audio-transcription pipeline for images. Every incoming
WhatsApp image gets a natural-language description + verbatim OCR of any text
it contains, stored in a searchable FTS5 index that joins seamlessly with the
existing text-message and audio-transcript search.

## Decisions already locked in

- **Model**: `qwen3-vl:8b-instruct` (non-thinking variant) via Ollama
- **Runtime**: Ollama 0.21 daemon on `:11434`, MLX backend, models stored on
  `/Volumes/Crucial X10/ollama-models` via `$OLLAMA_MODELS`
- **Measured cost**: ~12 s/image, ~40 tok/s decode, 6.1 GB on disk, ~8–10 GB
  active RAM. Backfill of 3,869 images ≈ 13 h; skipping `status@broadcast`
  brings it to ≈ 10 h overnight.

## Architecture — mirrors transcriptions

```
store/<chat_jid>/image_*.jpg|png|webp
             │
             ▼
     watch_images.py  (fsevents → single-worker queue)
             │
             ▼
   image_description.py
             │   POST /api/generate
             ▼
      Ollama (qwen3-vl:8b-instruct)
             │
             ▼
   image_descriptions.db  (sibling of transcriptions.db)
             │
             ▼
    FTS5 ← joined via ATTACH at search time
```

## Components

### 1. Go bridge caption fix (small, do first)

**File**: `whatsapp-bridge/main.go`

`extractTextContent` currently only reads `ConversationMessage` and
`ExtendedTextMessage`. `ImageMessage.GetCaption()`, `VideoMessage.GetCaption()`,
and `DocumentMessage.GetCaption()` are silently dropped — 3,869 image rows
in the DB, zero with content. Patch the function to also surface captions on
media messages.

Effect: captions become searchable via the existing `messages.content` column
with no schema change. Independent of the Qwen pipeline — ship it regardless.

### 2. Schema: `image_descriptions.db`

**New file**: `whatsapp-mcp-server/schema_image_descriptions.sql`

Structure mirrors `schema_transcriptions.sql` with one extra dimension
(a description column distinct from any OCR'd text, for search-quality
tuning later).

```sql
CREATE TABLE IF NOT EXISTS image_descriptions (
    message_id TEXT NOT NULL,
    chat_jid TEXT NOT NULL,
    description TEXT NOT NULL,          -- the "what is this" sentence
    ocr_text TEXT,                      -- verbatim text if present, else NULL
    model TEXT NOT NULL,                -- e.g. qwen3-vl:8b-instruct
    image_sha256 TEXT,
    width INTEGER,
    height INTEGER,
    described_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (message_id, chat_jid)
);

CREATE INDEX IF NOT EXISTS idx_image_descriptions_chat
    ON image_descriptions(chat_jid);

-- FTS5 over description + OCR combined, for unified search
CREATE VIRTUAL TABLE IF NOT EXISTS image_descriptions_fts USING fts5(
    description,
    ocr_text,
    message_id UNINDEXED,
    chat_jid UNINDEXED,
    content=image_descriptions,
    content_rowid=rowid,
    tokenize='unicode61 remove_diacritics 2'
);

-- Standard insert/update/delete triggers to keep FTS in sync
-- (same three-trigger pattern as transcriptions)
```

### 3. Core module: `image_description.py`

**New file**: `whatsapp-mcp-server/image_description.py`

Responsibility: describe one image, or backfill many. Subcommands mirror
`transcription.py`.

CLI surface:
```
uv run python image_description.py backfill [--model NAME] [--chat JID] [--limit N] [--dry-run] [--download] [--skip-status]
uv run python image_description.py describe MESSAGE_ID CHAT_JID
```

Key functions (same shape as `transcription.py`):
- `find_image_file(chat_jid, timestamp, message_id) -> Path | None`
- `download_via_bridge(message_id, chat_jid) -> Path | None`  (reuse logic)
- `describe_image(path, model) -> tuple[str, str | None, tuple[int, int]]`
  returning `(description, ocr_text, (width, height))`
- `record_description(conn, ...)`
- `lookup_message_for_image_file(path) -> (message_id, chat_jid) | None`
- `already_described(conn, message_id, chat_jid) -> bool`
- `iter_undescribed(msgs_conn, img_conn, chat, limit, skip_status)`

Filename format (matches audio after 2026-04-18 bridge fix):
- **New**: `image_YYYYMMDD_HHMMSS_MSGID.jpg` — message ID embedded
- **Legacy**: `image_YYYYMMDD_HHMMSS.jpg` — disambiguate via
  `messages.file_length` vs on-disk size (same dual-mode pattern
  `transcription.py:lookup_message_for_audio_file` uses)
- An optional `scripts/migrate_image_filenames.py` parallel to
  `migrate_audio_filenames.py` can rename legacy files to the new format.
  Low priority — most legacy image files are `status@broadcast` which we
  skip in backfill anyway.

Prompt (proven in benchmark):
```
Describe this image in 1-2 sentences. Then, if any text is visible,
add a "Text:" line with the text transcribed verbatim. Be concise.
```

Parse the response: split on `\nText:` — everything before is `description`,
everything after is `ocr_text` (NULL if absent).

API call: `POST http://localhost:11434/api/generate` with
`{"model": "qwen3-vl:8b-instruct", "prompt": PROMPT, "images": [b64], "stream": false}`.

### 4. Watcher: `watch_images.py`

**New file**: `whatsapp-mcp-server/watch_images.py`

Near-clone of `watch.py`. Only differences:
- `is_image_path` checks for `image_*.{jpg,jpeg,png,webp}`
- Calls `describe_and_record` instead of `transcribe_and_record`
- Uses `IMAGE_MODEL = "qwen3-vl:8b-instruct"` from env `IMAGE_MODEL_NAME`

Single-worker queue same as audio — concurrent vision inference would thrash
Metal/unified memory; serialize.

Can run alongside the existing audio watcher (separate process, separate
queue, separate DB).

### 5. Extend search — unified multimodal results

**File**: `whatsapp-mcp-server/transcription.py` (or rename to `search.py`)

`cmd_search` currently unions text + audio. Add `image` as a third kind:
```sql
SELECT i.message_id, i.chat_jid, m.timestamp, m.sender,
       COALESCE(i.description, '') || ' ' || COALESCE(i.ocr_text, '') AS content,
       c.name AS chat_name, 'image' AS kind,
       snippet(image_descriptions_fts, 0, '[', ']', '...', 16) AS snippet
FROM image_descriptions_fts ...
```

Extend `--only` to accept `text|audio|image`. Default remains all three.

Marker in output: `[image]` next to `[text]`/`[audio]`.

### 6. Backfill strategy

**First run** (overnight):
```bash
uv run python image_description.py backfill --skip-status --download
```
- Skips `status@broadcast` (ephemeral, rarely worth searching)
- `--download` tells the bridge to fetch any missing .jpg files first
- ~3,000 real images × 12 s = ~10 h

**Status chats** (optional, later): same command without `--skip-status`.

Idempotent — re-running skips already-described images via
`already_described()` check.

### 7. Startup

Add to whatever runs the audio watcher today (launchd plist, tmux session,
etc.). Two processes:
- `watch.py` — audio
- `watch_images.py` — images

Both depend on `ollama serve` being up (brew service, already configured to
start on login).

## Optional: MCP tool

**File**: `whatsapp-mcp-server/main.py`

Add a `describe_image_message` tool:
```python
@mcp.tool()
def describe_image_message(message_id: str, chat_jid: str) -> dict:
    """Generate or retrieve a description of an image message."""
```
- If already in `image_descriptions` table, return it
- Otherwise download (if needed), describe, record, return

This lets Claude ask "what's in the photo Mom sent yesterday?" without
waiting for the watcher. Ad-hoc path for interactive use, batch path for
storage.

## Risks / gotchas

1. **Disk space**: each run keeps the .jpg on the 2TB drive indefinitely.
   Current image count is 3,869, typical WhatsApp compression keeps these
   ~50–200 KB → ~500 MB worst case. Non-issue.
2. **Ollama cold starts**: first call after idle is ~5 s longer as the model
   reloads. Keep-alive in the API call: `"keep_alive": "30m"`.
3. **Prompt injection via OCR**: descriptions get shoved into FTS5 and later
   into agent context. The instruct variant and our prompt both treat the
   image content as data, not instructions, but worth a sanity pass on
   malicious status/meme injection later.
4. **Mixed languages**: Qwen3-VL handled English + Spanish cleanly in the
   benchmark. FTS5's `unicode61 remove_diacritics 2` tokenizer is already
   what transcriptions uses — same config covers both.
5. **Captions vs descriptions**: after the bridge fix, images have *two*
   text sources — user-supplied caption (`messages.content`) and model
   description (`image_descriptions.description` + `ocr_text`). Search
   unions both naturally; no dedup needed.

## Open questions

- Should we downsample large images (> 1600 px) before calling Ollama?
  Qwen3-VL handles any resolution but vision encode time scales with pixels.
  Benchmark showed 1.9–3.6 s prompt eval — acceptable. Defer optimization.
- `qwen3-vl:4b-instruct` for a second tier (faster, slightly weaker) if we
  want live feedback while the 8B handles backfill? Probably overkill.
- Rotate/expire old descriptions? Not needed — append-only table, small rows.

## Branch / worktree strategy

This repo is the `verygoodplugins/whatsapp-mcp` fork. Upstream review is slow,
so we keep multiple in-flight PRs and stack them locally on `local/running`
for day-to-day use. The image-description work has to fit that pattern —
it can't block on upstream approval before we benefit from it, and it
shouldn't complicate whatever else is in flight.

### Current branch landscape (2026-04-18)

- `main` → tracks `upstream/main` (verygoodplugins), at release 0.1.0
- `local/running` → integration branch we actually run from; stacks our open
  PRs on top of `main`
- Open PRs pending upstream review:
  - `feat/call-events` (PR #39)
  - `feat/full-history-pair` (PR #37)
  - `feat/transcription` (audio pipeline — sibling of this work, has its
    own worktree at `/Users/edmenendez/projects/mcp/whatsapp-mcp-txn`)
  - `fix/audio-filename-collision` (worktree at `…-gofix`)
  - `fix/contact-sync-from-whatsmeow` (PR #30)
- `feat/history-backfill` — local-only, explicitly "not shipping"

### Two branches for this work — not one

The bridge caption fix and the image-description pipeline are independent.
Bundling them into one PR makes review harder and ties a tiny bridge fix to
a large Python+DB change. Split them:

1. **`fix/media-captions`** (small, standalone, upstream-bound)
   - Single change: `extractTextContent` also reads
     `ImageMessage.GetCaption()`, `VideoMessage.GetCaption()`,
     `DocumentMessage.GetCaption()`
   - Useful on its own even if the VLM work never ships upstream
   - Should merge quickly — mirror the style of other `fix/*` PRs

2. **`feat/image-descriptions`** (larger, Python side, upstream-bound but
   less urgent)
   - Schema, `image_description.py`, `watch_images.py`, search extension,
     optional MCP tool
   - Mirrors `feat/transcription`'s shape so reviewers see a familiar
     pattern

### Worktree

Create a dedicated worktree alongside the audio pair, so `local/running`
stays untouched while we develop:

```bash
git worktree add ../whatsapp-mcp-images feat/image-descriptions
```

`feat/image-descriptions` should branch from `main`, not `local/running` —
that keeps the eventual PR clean (no unrelated commits from other in-flight
branches).

### Integrating into `local/running` for daily use

After each commit on either branch, merge it into `local/running` so we
get the benefit locally without waiting for upstream:

```bash
# on local/running
git merge --no-ff feat/image-descriptions
git merge --no-ff fix/media-captions
```

Keep `local/running` merge-based (not rebase) — merges are easy to undo
if upstream ships a conflicting change, and the merge commits make it
obvious which feature each chunk belongs to. This matches the existing
`local/running` convention (the current tip is `merge: call events (Phase 3
only) on top of contacts + full-history-pair`).

### When upstream finally merges something

When one of our PRs lands upstream:
1. `git checkout main && git pull upstream main`
2. Rebase the still-open branches onto new `main`
3. Rebuild `local/running` = `main` + all still-open feature branches
4. Drop any local-only hacks that are now redundant

The worktrees for merged branches can be removed (`git worktree remove`).

### Naming the model/config file to avoid conflict

The schema file `schema_image_descriptions.sql` and module
`image_description.py` share a directory with `schema_transcriptions.sql`
and `transcription.py`. No conflict. If someone upstream adds their own
image work while our PR is pending, we rebase normally — the names are
specific enough that collisions are unlikely.

## Rollout order

1. Branch `fix/media-captions` off `main`, patch `extractTextContent`,
   open PR, merge into `local/running`
2. Branch `feat/image-descriptions` off `main`, create worktree
3. Add schema + `image_description.py` + 5-image smoke test
4. Add `watch_images.py`, run for a day on live messages only
5. Kick off overnight backfill with `--skip-status`
6. Extend `cmd_search` to include images
7. (Optional) MCP tool for interactive describe
8. Open PR for `feat/image-descriptions`, merge into `local/running`
   independent of upstream review timeline
