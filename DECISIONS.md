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
- No auth (localhost, single user).
- No background auto-reindex daemon — reindex is a button / `--reindex` flag.
- No LLM query-expansion / HyDE — powerful, but needs an LLM; none local, and I
  won't send queries to an external API without asking. Candidate future upgrade.

---

# v2 — Quality upgrade (compute is explicitly not a concern)

Reworked from single-vector to **multi-vector + two-stage retrieval**, the pattern
production search engines use. Rationale for each change:

### 10. Chunk whole sessions; index every chunk (multi-vector)
- **Was:** one embedding per session from a 4000-char gist — the rest of the session
  was **truncated away** and unsearchable.
- **Now:** each session is sliced into overlapping ~1500-char windows (250 overlap)
  over *both* your prompts and Claude's replies; every window is embedded. A session
  scores by its **best-matching chunk**. On the real corpus this went from 181 vectors
  to ~5000 chunks — the thing you half-remember is now in the index even if it was at
  message 200. The matching passage is shown as the result snippet so you recognise it.
- Per-session chunk cap (80, evenly sampled) bounds runaway 500-message sessions.

### 11. Cross-encoder reranker (stage 2)
- **Model:** `jinaai/jina-reranker-v2-base-multilingual` (handles code + mixed text).
- Stage 1 (dense+BM25) returns ~50 candidate sessions; the cross-encoder reads
  `(query, best-passage)` **jointly** and reorders them. Cross-encoders are far more
  accurate than cosine but too slow to run on everything — hence candidates-then-rerank.
- Falls back to the fused stage-1 score if the reranker errors (never hard-fails).

### 12. Bigger embedding model
- `bge-small-en-v1.5` (384-dim) → **`bge-large-en-v1.5` (1024-dim)**. Strictly better
  stage-1 recall; download/compute cost irrelevant per the brief.

### 13. Hybrid dense + BM25 (RRF fusion)
- Added a **pure-numpy Okapi BM25** (no new dependency, fully offline) over the chunks,
  fused with dense ranking via **Reciprocal Rank Fusion**. Rescues cases where you
  remember an exact token (a repo, an error string, "renovate") that the dense model
  ranks poorly. It augments — never replaces — semantic ranking.

### 14. Content-hash embedding cache
- Every chunk's vector is cached by `sha1(text)`. Reindex re-embeds only genuinely new
  chunk text; unchanged sessions are free. Cache is pruned to the current chunk set on
  every build so it can't grow unbounded.

### 15. Stronger cleaning
- Also strips injected skill dumps (`Base directory for this skill:`) and file/CLAUDE.md
  dumps (`Contents of `), which previously polluted the gist embedding.

**Cost of all this:** first index build downloads ~2.3GB of models and embeds a few
thousand chunks (minutes); each query now runs dense + BM25 + a cross-encoder over ~50
candidates (~1–2s). Accepted deliberately in exchange for accuracy.
