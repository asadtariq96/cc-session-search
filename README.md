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

## How it works
`~/.claude/projects/**/*.jsonl` → parse each session's gist (AI title + key prompts,
harness noise stripped) → embed locally with `BAAI/bge-small-en-v1.5` (via fastembed)
→ store one normalized numpy matrix → query = cosine similarity top-k. Incremental:
only new/changed sessions get re-embedded.

## Requirements
- [`uv`](https://github.com/astral-sh/uv) (already installed)
- First run downloads the embedding model (~130MB, one time, cached)
