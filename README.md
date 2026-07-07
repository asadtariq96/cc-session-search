# cc-session-search

Semantic (meaning-based, **not** keyword) search over your old Claude Code chat
sessions. Describe what you remember a session was *about* and get its **session ID**
and a ready-to-run `claude --resume` command — even if you don't recall the title.

Everything runs **locally and offline**. Session content never leaves your Mac.
See [`DECISIONS.md`](DECISIONS.md) for design rationale.

## Use it

```bash
~/Developer/cc-session-search/ccsearch
```

This reindexes new/changed sessions, then opens the GUI in your browser
(`http://127.0.0.1:8422`). Type what you remember; results rank by meaning.

Optional alias:

```bash
echo "alias ccsearch='~/Developer/cc-session-search/ccsearch'" >> ~/.zshrc
```

## What you get per result
- The Claude-generated **title** + a snippet of your opening prompt
- **Project** dir, git branch, date, message count
- **⧉ Copy resume command** → `cd "<cwd>" && claude --resume <id>` (paste in a terminal)
- **⧉ Copy ID** → just the session ID
- **▸ Open in iTerm2** → opens a new tab and resumes it (best-effort)

## Commands
```bash
uv run python app.py index                 # (re)build the index only
uv run python app.py serve                 # serve GUI (no reindex)
uv run python app.py serve --reindex       # reindex then serve
uv run python app.py serve --no-open       # don't auto-open browser
uv run python app.py serve --port 9000     # custom port
```

## How it works (two-stage, multi-vector — all local)
1. Parse every `~/.claude/projects/**/*.jsonl`; strip harness/skill noise.
2. **Chunk** each session (your prompts + Claude's replies) into overlapping windows
   and embed *every* chunk locally with `BAAI/bge-large-en-v1.5` (1024-dim, fastembed).
3. **Stage 1 — recall:** query is scored by dense cosine **and** a numpy BM25, fused
   with Reciprocal Rank Fusion; a session scores by its best-matching chunk.
4. **Stage 2 — precision:** the top ~50 sessions are reranked by a cross-encoder
   (`jina-reranker-v2`) that reads query + passage together.
5. Results show the matching passage; embeddings are cached by content hash so reindex
   only embeds genuinely new text.

See [`DECISIONS.md`](DECISIONS.md) for why each piece exists.

## Requirements
- [`uv`](https://github.com/astral-sh/uv) (already installed)
- First run downloads models (~2.3GB total: bge-large + reranker), one time, cached.
  Indexing embeds a few thousand chunks (minutes); queries take ~1–2s.
