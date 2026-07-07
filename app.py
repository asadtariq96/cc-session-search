#!/usr/bin/env python3
"""Semantic search over Claude Code chat sessions.

Two-stage, multi-vector retrieval — all local/offline:

    parse  ->  chunk whole sessions (your prompts + Claude's replies)
           ->  embed every chunk (bge-large-en-v1.5, 1024-dim)
    query  ->  dense cosine  +  BM25 (numpy)  ->  RRF fuse   [stage 1: recall]
           ->  cross-encoder rerank the best chunk per session [stage 2: precision]

Session content never leaves the machine. See DECISIONS.md for rationale.

Subcommands:
    index   Scan ~/.claude/projects, (re)build the local index.
    serve   Load the index and serve the local web GUI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECTS_DIR = Path.home() / ".claude" / "projects"
INDEX_DIR = Path(__file__).resolve().parent / ".index"
CHUNK_EMB_PATH = INDEX_DIR / "chunk_emb.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
SESSIONS_PATH = INDEX_DIR / "sessions.json"
CACHE_EMB_PATH = INDEX_DIR / "vec_cache.npy"
CACHE_KEYS_PATH = INDEX_DIR / "vec_cache_keys.json"

# Bigger model — compute is not a concern; recall is. bge-large is 1024-dim.
DENSE_MODEL = "BAAI/bge-large-en-v1.5"
# Cross-encoder for stage-2 reranking (reads query+passage jointly).
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
# BGE is asymmetric: instruction on the *query* only; passages are embedded raw.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

CHUNK_CHARS = 1500          # ~380 tokens per chunk
CHUNK_OVERLAP = 250         # sliding-window overlap so ideas aren't split
MAX_CHUNKS_PER_SESSION = 80 # cap runaway sessions (evenly sampled beyond this)

# Retrieval widths (generous — compute is free)
STAGE1_PER_LIST = 400       # top chunks pulled from each of dense / bm25
RERANK_SESSIONS = 50        # sessions handed to the cross-encoder
RRF_K = 60                  # reciprocal-rank-fusion constant

# Harness / injected-content markers to strip (not human-authored prose).
_NOISE_PREFIXES = (
    "<local-command", "<command-name", "<command-message", "<command-args",
    "</command", "<system-reminder", "</system-reminder",
    "<user-prompt-submit-hook", "Caveat:",
    "This session is being continued from a previous",
    "Base directory for this skill:",   # skill injection dumps
    "Contents of ",                     # CLAUDE.md / file dumps
    "<system>", "</system>", "[Request interrupted",
)


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------
@dataclass
class Session:
    session_id: str
    path: str
    ai_title: str
    first_prompt: str
    turns: list[str]        # ordered, cleaned user+assistant prose
    cwd: str
    project: str
    git_branch: str
    first_ts: str
    last_ts: str
    msg_count: int
    version: str

    def meta(self) -> dict:
        """Lightweight metadata stored per session (no bulky turns)."""
        d = asdict(self)
        d.pop("turns", None)
        return d


def _clean_text(text: str) -> str:
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(_NOISE_PREFIXES):
            continue
        kept.append(s)
    return "\n".join(kept).strip()


def _extract_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return _clean_text(content)
    if isinstance(content, list):
        chunks = [b.get("text", "") for b in content
                  if isinstance(b, dict) and b.get("type") == "text"]
        return _clean_text("\n".join(chunks))
    return ""


def parse_session(path: Path) -> Session | None:
    session_id = path.stem
    ai_title = cwd = git_branch = version = first_ts = last_ts = ""
    msg_count = 0
    turns: list[str] = []
    last_prompt = ""

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = obj.get("type")

                if t == "ai-title" and obj.get("aiTitle"):
                    ai_title = obj["aiTitle"]
                    continue
                if t == "last-prompt" and obj.get("lastPrompt"):
                    last_prompt = _clean_text(obj["lastPrompt"])
                    continue

                if obj.get("timestamp"):
                    first_ts = first_ts or obj["timestamp"]
                    last_ts = obj["timestamp"]
                cwd = obj.get("cwd") or cwd
                git_branch = obj.get("gitBranch") or git_branch
                version = obj.get("version") or version

                if t in ("user", "assistant") and not obj.get("isSidechain"):
                    msg_count += 1
                    if obj.get("userType", "external") != "external" and t == "user":
                        continue
                    text = _extract_text(obj.get("message") or {})
                    if text and len(text) > 2:
                        turns.append(text)
    except OSError:
        return None

    first_prompt = turns[0] if turns else ""
    if last_prompt and last_prompt not in turns:
        turns.append(last_prompt)

    if not (ai_title or turns):
        return None

    return Session(
        session_id=session_id, path=str(path), ai_title=ai_title,
        first_prompt=first_prompt, turns=turns, cwd=cwd,
        project=_project_label(cwd, path), git_branch=git_branch,
        first_ts=first_ts, last_ts=last_ts, msg_count=msg_count, version=version,
    )


def _project_label(cwd: str, path: Path) -> str:
    if cwd:
        return cwd
    name = path.parent.name
    return "/" + name[1:].replace("-", "/") if name.startswith("-") else name


def iter_session_files():
    if PROJECTS_DIR.exists():
        yield from PROJECTS_DIR.rglob("*.jsonl")


# ---------------------------------------------------------------------------
# Chunking — one session becomes many overlapping passages
# ---------------------------------------------------------------------------
def chunk_session(sess: Session) -> list[str]:
    chunks: list[str] = []

    # A synthetic header chunk so title/project/branch are directly matchable.
    proj = Path(sess.project).name if sess.project else ""
    header_bits = [b for b in (sess.ai_title, proj, sess.git_branch) if b and b != "HEAD"]
    if header_bits:
        chunks.append(" — ".join(header_bits))

    body = "\n".join(sess.turns)
    step = max(1, CHUNK_CHARS - CHUNK_OVERLAP)
    windows = []
    for start in range(0, len(body), step):
        piece = body[start:start + CHUNK_CHARS].strip()
        if len(piece) > 30:
            windows.append(piece)
        if start + CHUNK_CHARS >= len(body):
            break

    if len(windows) > MAX_CHUNKS_PER_SESSION:
        idxs = np.linspace(0, len(windows) - 1, MAX_CHUNKS_PER_SESSION).astype(int)
        windows = [windows[i] for i in sorted(set(idxs))]

    chunks.extend(windows)
    return chunks or [sess.ai_title or sess.first_prompt[:CHUNK_CHARS]]


# ---------------------------------------------------------------------------
# Embeddings (+ persistent content-hash cache so reindex only embeds new text)
# ---------------------------------------------------------------------------
_MODEL = None
_RERANKER = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(model_name=DENSE_MODEL)
    return _MODEL


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _RERANKER = TextCrossEncoder(model_name=RERANK_MODEL)
    return _RERANKER


def _embed_raw(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(get_model().embed(texts)), dtype=np.float32)
    return _normalize(vecs)


def embed_query(query: str) -> np.ndarray:
    vec = np.array(list(get_model().embed([QUERY_INSTRUCTION + query]))[0], dtype=np.float32)
    return _normalize(vec.reshape(1, -1))[0]


def _normalize(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, np.ndarray]:
    if CACHE_EMB_PATH.exists() and CACHE_KEYS_PATH.exists():
        keys = json.loads(CACHE_KEYS_PATH.read_text())
        vecs = np.load(CACHE_EMB_PATH)
        return {k: vecs[i] for i, k in enumerate(keys)}
    return {}


def embed_passages_cached(texts: list[str], cache: dict, verbose=False) -> np.ndarray:
    out: list[np.ndarray | None] = [None] * len(texts)
    misses, miss_idx = [], []
    hashes = [_sha(t) for t in texts]
    for i, h in enumerate(hashes):
        v = cache.get(h)
        if v is not None:
            out[i] = v
        else:
            misses.append(texts[i]); miss_idx.append(i)
    if verbose:
        print(f"[index] {len(texts)} chunks; embedding {len(misses)} new "
              f"(cache hit {len(texts)-len(misses)})…", flush=True)
    if misses:
        # batch to keep memory sane on huge corpora
        B = 256
        vecs_all = []
        for s in range(0, len(misses), B):
            vecs_all.append(_embed_raw(misses[s:s + B]))
            if verbose and len(misses) > B:
                print(f"[index]   …{min(s+B,len(misses))}/{len(misses)}", flush=True)
        vecs = np.vstack(vecs_all)
        for j, i in enumerate(miss_idx):
            out[i] = vecs[j]; cache[hashes[i]] = vecs[j]
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------
def build_index(verbose: bool = True) -> dict:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()

    sessions: dict[str, dict] = {}
    chunk_sids: list[str] = []
    chunk_texts: list[str] = []
    scanned = 0

    for path in iter_session_files():
        scanned += 1
        sess = parse_session(path)
        if sess is None:
            continue
        sessions[sess.session_id] = sess.meta()
        for c in chunk_session(sess):
            chunk_sids.append(sess.session_id)
            chunk_texts.append(c)

    if verbose:
        print(f"[index] scanned {scanned} files → {len(sessions)} sessions, "
              f"{len(chunk_texts)} chunks", flush=True)

    emb = embed_passages_cached(chunk_texts, cache, verbose=verbose)

    np.save(CHUNK_EMB_PATH, emb)
    CHUNKS_PATH.write_text(json.dumps(
        [{"sid": s, "text": t} for s, t in zip(chunk_sids, chunk_texts)],
        ensure_ascii=False))
    SESSIONS_PATH.write_text(json.dumps(sessions, ensure_ascii=False))
    # persist cache pruned to exactly the current chunks
    keys = list(cache.keys())
    np.save(CACHE_EMB_PATH, np.array([cache[k] for k in keys], dtype=np.float32))
    CACHE_KEYS_PATH.write_text(json.dumps(keys))

    if verbose:
        print(f"[index] done: {len(sessions)} sessions / {len(chunk_texts)} chunks "
              f"→ {INDEX_DIR}", flush=True)
    return {"sessions": len(sessions), "chunks": len(chunk_texts)}


# ---------------------------------------------------------------------------
# BM25 (pure numpy, no deps) over chunks
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 with postings lists for fast query-time scoring."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = np.zeros(self.N, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            toks = _tok(d)
            self.dl[i] = len(toks)
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, c in tf.items():
                postings[t].append((i, c))
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        self.postings = {}
        self.idf = {}
        for t, plist in postings.items():
            ids = np.array([p[0] for p in plist], dtype=np.int64)
            tfs = np.array([p[1] for p in plist], dtype=np.float32)
            self.postings[t] = (ids, tfs)
            df = len(plist)
            self.idf[t] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.N, dtype=np.float32)
        if self.avgdl == 0:
            return out
        for t in set(_tok(query)):
            post = self.postings.get(t)
            if post is None:
                continue
            ids, tfs = post
            idf = self.idf[t]
            denom = tfs + self.k1 * (1 - self.b + self.b * self.dl[ids] / self.avgdl)
            out[ids] += idf * (tfs * (self.k1 + 1)) / denom
        return out


def _rrf(*ranked_id_lists, k=RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked lists of item indices."""
    fused: dict[int, float] = defaultdict(float)
    for lst in ranked_id_lists:
        for rank, idx in enumerate(lst):
            fused[int(idx)] += 1.0 / (k + rank + 1)
    return fused


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class SearchIndex:
    def __init__(self):
        self.reload()

    def reload(self):
        self.emb = np.load(CHUNK_EMB_PATH) if CHUNK_EMB_PATH.exists() \
            else np.zeros((0, 1024), np.float32)
        chunks = json.loads(CHUNKS_PATH.read_text()) if CHUNKS_PATH.exists() else []
        self.chunk_sids = [c["sid"] for c in chunks]
        self.chunk_texts = [c["text"] for c in chunks]
        self.sessions = json.loads(SESSIONS_PATH.read_text()) if SESSIONS_PATH.exists() else {}
        self.bm25 = BM25(self.chunk_texts) if self.chunk_texts else None

    def search(self, query: str, k: int = 25) -> list[dict]:
        if not query.strip() or not self.chunk_texts:
            return []

        # --- stage 1: dense + BM25, fused with RRF ---------------------------
        qv = embed_query(query)
        dense = self.emb @ qv
        dtop = np.argsort(-dense)[:STAGE1_PER_LIST]
        lists = [dtop]
        if self.bm25 is not None:
            bm = self.bm25.scores(query)
            btop = np.argsort(-bm)[:STAGE1_PER_LIST]
            lists.append(btop)
        fused = _rrf(*lists)

        # best chunk per session (by fused score)
        best: dict[str, tuple[float, int]] = {}
        for idx, sc in fused.items():
            sid = self.chunk_sids[idx]
            if sid not in best or sc > best[sid][0]:
                best[sid] = (sc, idx)
        ranked_sids = sorted(best, key=lambda s: best[s][0], reverse=True)
        cand = ranked_sids[:RERANK_SESSIONS]

        # --- stage 2: cross-encoder rerank the best passage per session ------
        passages = [self.chunk_texts[best[s][1]] for s in cand]
        order = list(range(len(cand)))
        rr_scores = [best[cand[i]][0] for i in order]  # fallback = fused score
        try:
            rr_scores = list(get_reranker().rerank(query, passages))
        except Exception:
            pass
        order.sort(key=lambda i: rr_scores[i], reverse=True)

        results = []
        for i in order[:k]:
            sid = cand[i]
            s = self.sessions.get(sid, {})
            confidence = 1.0 / (1.0 + math.exp(-float(rr_scores[i])))  # logit → 0..1
            results.append({
                "score": round(confidence, 4),
                "logit": round(float(rr_scores[i]), 3),
                "session_id": sid,
                "title": s.get("ai_title") or "(untitled session)",
                "snippet": _trim(passages[i]),
                "project": s.get("project", ""),
                "cwd": s.get("cwd", ""),
                "git_branch": s.get("git_branch", ""),
                "last_ts": s.get("last_ts", ""),
                "first_ts": s.get("first_ts", ""),
                "msg_count": s.get("msg_count", 0),
                "resume_cmd": _resume_cmd(sid, s),
            })
        return results


def _trim(text: str, n: int = 320) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t[:n] + ("…" if len(t) > n else "")


def _resume_cmd(sid: str, s: dict) -> str:
    cwd = s.get("cwd") or s.get("project") or ""
    return f'cd "{cwd}" && claude --resume {sid}' if cwd else f"claude --resume {sid}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="build/refresh the local index")
    sp = sub.add_parser("serve", help="serve the local web GUI")
    sp.add_argument("--port", type=int, default=8422)
    sp.add_argument("--no-open", action="store_true")
    sp.add_argument("--reindex", action="store_true")
    args = ap.parse_args()

    if args.cmd == "index":
        build_index()
    elif args.cmd == "serve":
        from server import serve
        serve(port=args.port, open_browser=not args.no_open, reindex=args.reindex)


if __name__ == "__main__":
    main()
