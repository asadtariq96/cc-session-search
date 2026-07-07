# Design Decisions — cc-session-search

Every choice here was made autonomously with a bias toward **local/offline** and
**safe defaults**, because the corpus (your Claude Code sessions) contains internal
Unity code, prompts, and data, and this runs on a managed/monitored Mac.

## Goal
Semantic (meaning-based, not text-matching) search over old Claude Code chat
sessions so you can find a session by what you *remember* it was about — even
without the exact title — and get its **session ID** to `claude --resume`.

## Decisions

### 1. Embeddings run 100% locally & offline
- **Model:** `BAAI/bge-small-en-v1.5` (384-dim) via **`fastembed`** (ONNX runtime).
- **Why:** No API key, and **no session content ever leaves the machine** — the
  decisive factor given internal content on a monitored device. `fastembed` avoids
  a heavyweight PyTorch install (uses onnxruntime, ~ tens of MB) and is purpose-built
  for embeddings. Model (~130MB) is downloaded once from HuggingFace, then cached.
- **Alternative rejected:** hosted embedding APIs (OpenAI/Voyage) — better quality
  but exfiltrates session text to a third party. Not acceptable here.
- **Alternative rejected:** Ollama — also local, but not installed; more moving parts
  than a self-contained Python model.

### 2. Asymmetric retrieval (BGE instruction prefix)
- Queries are prefixed with `"Represent this sentence for searching relevant
  passages: "`; documents are not. This is how BGE was trained and it matters here:
  your query ("that VPN routing thing") looks nothing like the stored document, so
  symmetric matching would underperform.

### 3. What gets embedded: the *gist*, not the whole transcript
- Per session we build one compact "document" from: the Claude-generated
  `ai-title` + the first user prompt + a few sampled prompts + the last prompt
  (capped at 4000 chars).
- **Why:** `ai-title` is a high-signal one-line summary Claude already wrote.
  Embedding whole 4k-line transcripts would drown the gist in tool output and noise.

### 4. Harness noise is stripped
- Lines like `<command-name>`, `<local-command-*>`, `<system-reminder>`, caveats,
  and sidechain/subagent messages are filtered out. Only human-authored prose and
  the AI title feed the index.
- **Consequence:** sessions with *no* human prose (e.g. only `/context`, `/mobile`
  slash-commands, or aborted) are skipped — they have nothing to search on. On the
  current corpus this is 181 searchable of 534 files; verified that **zero**
  prose-containing sessions are wrongly dropped.

### 5. Brute-force cosine, no vector DB
- Embeddings stored as a single normalized numpy matrix; search = one matrix-vector
  product + top-k. Sub-millisecond at hundreds/thousands of sessions.
- **Why:** FAISS/Chroma would add dependencies and setup risk for zero perceptible
  speedup at this scale. YAGNI.

### 6. Incremental indexing
- A manifest tracks each file's `(mtime, size)`. Reindex only re-embeds new/changed
  sessions and reuses existing vectors — so "↻ Reindex" is fast after the first run.

### 7. GUI: stdlib `http.server` + one static HTML file
- No web framework, no build step, no npm. The server binds **127.0.0.1 only**
  (never off-machine). The whole UI is one auditable `index.html` with vanilla JS
  (debounced search-as-you-type).
- **Why:** most robust option on a managed Mac — nothing to compile or keep running
  except the Python process.

### 8. Runtime isolation via `uv`, pinned to Python 3.12
- **Why:** the system Python is 3.14 (very new); pinning 3.12 guarantees prebuilt
  wheels for onnxruntime/tokenizers. `uv` keeps the whole thing in a self-contained
  `.venv` with zero impact on system Python.

### 9. Getting you back into the session
- Each result shows the **session ID** and a copy-ready
  `cd "<cwd>" && claude --resume <id>` command (Claude Code scopes sessions by
  project dir, so the `cd` matters).
- Bonus **"Open in iTerm2"** button: best-effort AppleScript that opens a new iTerm2
  tab and runs the resume command. If iTerm2 isn't available it fails gracefully and
  tells you to use the copy button.

## Not done (deliberately, YAGNI)
- No fuzzy/keyword hybrid ranking — pure semantic, as requested.
- No auth (localhost, single user).
- No background auto-reindex daemon — reindex is a button / `--reindex` flag.
