#!/usr/bin/env python3
"""Semantic search over Claude Code chat sessions.

Subcommands:
    index   Scan ~/.claude/projects, build/refresh the local embedding index.
    serve   Load the index and serve a local web GUI for semantic search.

Everything runs locally/offline. Session content never leaves the machine.
See DECISIONS.md for the rationale behind the design choices.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECTS_DIR = Path.home() / ".claude" / "projects"
INDEX_DIR = Path(__file__).resolve().parent / ".index"
EMB_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "sessions.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, ONNX, ~130MB, CPU-friendly
# BGE is an asymmetric retriever: prepend this instruction to *queries* only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

DOC_CHAR_CAP = 4000  # max chars of a session "document" fed to the embedder

# Tags/markers that are harness noise rather than something the user typed.
_NOISE_PREFIXES = (
    "<local-command",
    "<command-name",
    "<command-message",
    "<command-args",
    "</command",
    "<system-reminder",
    "</system-reminder",
    "<user-prompt-submit-hook",
    "Caveat:",
    "This session is being continued from a previous",
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
    last_prompt: str
    sampled_prompts: list[str]
    cwd: str
    project: str          # human-friendly project label
    git_branch: str
    first_ts: str
    last_ts: str
    msg_count: int
    version: str

    def document(self) -> str:
        """The text we embed: the gist of the session (title + key prompts)."""
        parts: list[str] = []
        if self.ai_title:
            parts.append(self.ai_title)
        if self.first_prompt:
            parts.append(self.first_prompt[:1500])
        for p in self.sampled_prompts[:3]:
            parts.append(p[:500])
        if self.last_prompt and self.last_prompt != self.first_prompt:
            parts.append(self.last_prompt[:500])
        return "\n".join(parts)[:DOC_CHAR_CAP].strip()

    def snippet(self) -> str:
        base = self.first_prompt or self.last_prompt or self.ai_title
        base = re.sub(r"\s+", " ", base).strip()
        return base[:280]


def _clean_text(text: str) -> str:
    """Drop harness-injected wrapper lines, keep human-authored prose."""
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_NOISE_PREFIXES):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def _extract_user_text(message: dict) -> str:
    """Pull human-authored text out of a user message (string or block array)."""
    content = message.get("content")
    if isinstance(content, str):
        return _clean_text(content)
    if isinstance(content, list):
        chunks = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                chunks.append(block.get("text", ""))
            # skip tool_result / image / etc. — not user prose
        return _clean_text("\n".join(chunks))
    return ""


def parse_session(path: Path) -> Session | None:
    """Parse one session .jsonl into a Session (or None if it has no content)."""
    session_id = path.stem
    ai_title = ""
    last_prompt = ""
    cwd = ""
    git_branch = ""
    version = ""
    first_ts = ""
    last_ts = ""
    msg_count = 0
    user_prompts: list[str] = []

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

                ts = obj.get("timestamp")
                if ts:
                    if not first_ts:
                        first_ts = ts
                    last_ts = ts
                if obj.get("cwd"):
                    cwd = obj["cwd"]
                if obj.get("gitBranch"):
                    git_branch = obj["gitBranch"]
                if obj.get("version"):
                    version = obj["version"]

                if t in ("user", "assistant"):
                    if not obj.get("isSidechain"):
                        msg_count += 1
                    # Collect main-thread, external user prose for the gist.
                    if (
                        t == "user"
                        and not obj.get("isSidechain")
                        and obj.get("userType", "external") == "external"
                    ):
                        msg = obj.get("message") or {}
                        text = _extract_user_text(msg)
                        if text and len(text) > 2:
                            user_prompts.append(text)
    except OSError:
        return None

    first_prompt = user_prompts[0] if user_prompts else ""
    # sample a few prompts from across the conversation (skip the first)
    sampled = user_prompts[1:]
    if len(sampled) > 6:
        idxs = np.linspace(0, len(sampled) - 1, 6).astype(int)
        sampled = [sampled[i] for i in sorted(set(idxs))]

    if not (ai_title or first_prompt or last_prompt):
        return None  # empty/meta-only session, nothing to search on

    project = _project_label(cwd, path)
    return Session(
        session_id=session_id,
        path=str(path),
        ai_title=ai_title,
        first_prompt=first_prompt,
        last_prompt=last_prompt,
        sampled_prompts=sampled,
        cwd=cwd,
        project=project,
        git_branch=git_branch,
        first_ts=first_ts,
        last_ts=last_ts,
        msg_count=msg_count,
        version=version,
    )


def _project_label(cwd: str, path: Path) -> str:
    if cwd:
        return cwd
    # decode the encoded folder name, e.g. -Users-asad-tariq-Documents-GitHub
    name = path.parent.name
    if name.startswith("-"):
        return "/" + name[1:].replace("-", "/")
    return name


def iter_session_files():
    if not PROJECTS_DIR.exists():
        return
    yield from PROJECTS_DIR.rglob("*.jsonl")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding  # lazy: heavy import
        _MODEL = TextEmbedding(model_name=MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_model()
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    return _normalize(vecs)


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    vec = np.array(list(model.embed([QUERY_INSTRUCTION + query]))[0], dtype=np.float32)
    return _normalize(vec.reshape(1, -1))[0]


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# Index build (incremental)
# ---------------------------------------------------------------------------
def build_index(verbose: bool = True) -> dict:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    old_manifest = _load_json(MANIFEST_PATH, {})
    old_meta = _load_json(META_PATH, [])
    old_emb = np.load(EMB_PATH) if EMB_PATH.exists() else None
    old_by_id = {s["session_id"]: (i, s) for i, s in enumerate(old_meta)}

    new_manifest: dict[str, dict] = {}
    meta: list[dict] = []
    reuse_rows: list[int] = []   # index into old_emb, or -1 if must embed
    to_embed: list[str] = []
    to_embed_slots: list[int] = []

    scanned = 0
    reused = 0
    for path in iter_session_files():
        scanned += 1
        try:
            st = path.stat()
        except OSError:
            continue
        key = str(path)
        sig = {"mtime": st.st_mtime, "size": st.st_size}
        sid = path.stem

        prev = old_manifest.get(key)
        if (
            prev
            and prev.get("mtime") == sig["mtime"]
            and prev.get("size") == sig["size"]
            and sid in old_by_id
            and old_emb is not None
        ):
            # unchanged → reuse embedding + metadata
            old_i, old_s = old_by_id[sid]
            meta.append(old_s)
            reuse_rows.append(old_i)
            new_manifest[key] = sig
            reused += 1
            continue

        sess = parse_session(path)
        if sess is None:
            continue
        meta.append(asdict(sess))
        reuse_rows.append(-1)
        to_embed_slots.append(len(meta) - 1)
        to_embed.append(sess.document())
        new_manifest[key] = sig

    if verbose:
        print(f"[index] scanned {scanned} files, reusing {reused}, "
              f"embedding {len(to_embed)} new/changed…", flush=True)

    dim = old_emb.shape[1] if old_emb is not None else 384
    emb = np.zeros((len(meta), dim), dtype=np.float32)
    for row, old_i in enumerate(reuse_rows):
        if old_i >= 0:
            emb[row] = old_emb[old_i]
    if to_embed:
        new_vecs = embed_texts(to_embed)
        dim = new_vecs.shape[1]
        if emb.shape[1] != dim:  # first-ever build
            emb = np.zeros((len(meta), dim), dtype=np.float32)
            for row, old_i in enumerate(reuse_rows):
                if old_i >= 0 and old_emb is not None:
                    emb[row] = old_emb[old_i]
        for slot, vec in zip(to_embed_slots, new_vecs):
            emb[slot] = vec

    np.save(EMB_PATH, emb)
    _save_json(META_PATH, meta)
    _save_json(MANIFEST_PATH, new_manifest)
    if verbose:
        print(f"[index] done: {len(meta)} sessions indexed → {INDEX_DIR}", flush=True)
    return {"sessions": len(meta), "embedded": len(to_embed), "reused": reused}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class SearchIndex:
    def __init__(self):
        self.emb = np.load(EMB_PATH) if EMB_PATH.exists() else np.zeros((0, 384), np.float32)
        self.meta = _load_json(META_PATH, [])

    def reload(self):
        self.__init__()

    def search(self, query: str, k: int = 20) -> list[dict]:
        if not query.strip() or len(self.meta) == 0:
            return []
        qv = embed_query(query)
        scores = self.emb @ qv  # cosine (all rows unit-normalized)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results = []
        for i in top:
            s = self.meta[int(i)]
            results.append({
                "score": round(float(scores[i]), 4),
                "session_id": s["session_id"],
                "title": s.get("ai_title") or "(untitled session)",
                "snippet": Session(**s).snippet() if _is_session_dict(s) else s.get("first_prompt", "")[:280],
                "project": s.get("project", ""),
                "cwd": s.get("cwd", ""),
                "git_branch": s.get("git_branch", ""),
                "last_ts": s.get("last_ts", ""),
                "first_ts": s.get("first_ts", ""),
                "msg_count": s.get("msg_count", 0),
                "resume_cmd": _resume_cmd(s),
            })
        return results


def _is_session_dict(d: dict) -> bool:
    return "sampled_prompts" in d and "first_prompt" in d


def _resume_cmd(s: dict) -> str:
    cwd = s.get("cwd") or s.get("project") or ""
    sid = s["session_id"]
    if cwd:
        return f'cd "{cwd}" && claude --resume {sid}'
    return f"claude --resume {sid}"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="build/refresh the local embedding index")
    sp = sub.add_parser("serve", help="serve the local web GUI")
    sp.add_argument("--port", type=int, default=8422)
    sp.add_argument("--no-open", action="store_true", help="don't auto-open browser")
    sp.add_argument("--reindex", action="store_true", help="refresh index before serving")
    args = ap.parse_args()

    if args.cmd == "index":
        build_index()
    elif args.cmd == "serve":
        from server import serve
        serve(port=args.port, open_browser=not args.no_open, reindex=args.reindex)


if __name__ == "__main__":
    main()
