#!/usr/bin/env python3
"""Local web GUI for semantic Claude Code session search.

Serves a single-page UI plus a tiny JSON API from Python's stdlib http.server —
no web framework. Binds to 127.0.0.1 only (never exposed off-machine).
"""
from __future__ import annotations

import json
import subprocess
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import app as core

HTML_PATH = Path(__file__).resolve().parent / "index.html"

_index: core.SearchIndex | None = None
_index_lock = threading.Lock()


def _get_index() -> core.SearchIndex:
    global _index
    if _index is None:
        _index = core.SearchIndex()
    return _index


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/search":
            qs = parse_qs(parsed.query)
            q = (qs.get("q", [""])[0]).strip()
            k = int(qs.get("k", ["20"])[0])
            results = _get_index().search(q, k=k)
            self._json({"query": q, "count": len(results), "results": results})
            return
        if parsed.path == "/api/stats":
            idx = _get_index()
            self._json({"sessions": len(idx.sessions), "chunks": len(idx.chunk_texts)})
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if parsed.path == "/api/reindex":
            with _index_lock:
                stats = core.build_index(verbose=False)
                _get_index().reload()
            self._json({"ok": True, **stats})
            return

        if parsed.path == "/api/resume":
            cwd = payload.get("cwd", "")
            sid = payload.get("session_id", "")
            ok, msg = _launch_iterm(cwd, sid)
            self._json({"ok": ok, "message": msg})
            return

        self._json({"error": "not found"}, 404)


def _launch_iterm(cwd: str, session_id: str) -> tuple[bool, str]:
    """Best-effort: open a new iTerm2 tab and run `claude --resume` there."""
    if not session_id:
        return False, "missing session id"
    cd = f'cd {json.dumps(cwd)} && ' if cwd else ""
    cmd = f"{cd}claude --resume {session_id}"
    applescript = f'''
    tell application "iTerm2"
      activate
      tell current window
        create tab with default profile
        tell current session to write text {json.dumps(cmd)}
      end tell
    end tell
    '''
    try:
        subprocess.run(["osascript", "-e", applescript], check=True,
                       capture_output=True, timeout=10)
        return True, "Opened in iTerm2"
    except Exception as e:  # noqa: BLE001 — best-effort; report reason to UI
        return False, f"Could not open iTerm2 ({e}). Use the copy button instead."


def serve(port: int = 8422, open_browser: bool = True, reindex: bool = False):
    if reindex or not core.CHUNK_EMB_PATH.exists():
        core.build_index()
    # warm models + index up front so the first search is instant
    print("[serve] loading index + models (dense + reranker)…", flush=True)
    _get_index()
    core.get_model()
    core.get_reranker()

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    idx = _get_index()
    print(f"[serve] cc-session-search ready → {url}  "
          f"({len(idx.sessions)} sessions / {len(idx.chunk_texts)} chunks)", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] bye", flush=True)
