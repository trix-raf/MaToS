"""FastAPI app: serves ./server/frontend untouched (app.html gets the bridge script injected on the fly) and
talks to the page over a WebSocket, with a plain-HTTP fallback for phones that refuse wss:// to a self-signed cert."""
from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.engine import Engine
from app.session import Session
from app.settings import STATIC_DIR, Settings

HTTP_SESSION_TTL_S = 120.0
MAX_MSGS_PER_POST = 400
KEY_COOKIE = "mt_key"
KEY_COOKIE_MAX_AGE = 30 * 24 * 3600

_DENIED = ("<!doctype html><meta name=viewport content=\"width=device-width,initial-scale=1\">"
           "<body style=\"font:16px system-ui;padding:2em;max-width:30em;margin:auto\">"
           "<h2>Access key needed</h2><p>Open the link (or scan the QR code) that the server printed in its console - "
           "it ends with <code>?key=...</code>.  The key changes every run unless you start the server with "
           "<code>--key</code>.</p></body>")


def _inject(html: str) -> str:
    v = int((STATIC_DIR / "bridge.js").stat().st_mtime)
    head = f'<link rel="stylesheet" href="/bridge/bridge.css?v={v}">\n'
    body = f'<script type="module" src="/bridge/bridge.js?v={v}"></script>\n'
    html = html.replace("</head>", head + "</head>", 1) if "</head>" in html else head + html
    return html.replace("</body>", body + "</body>", 1) if "</body>" in html else html + body


def create_app(engine: Engine, cfg: Settings, log=print) -> FastAPI:
    frontend = Path(cfg.frontend).resolve()
    page = frontend / "app.html"
    if not page.is_file():
        raise SystemExit(f"frontend not found: {page}\n  expected ./server/frontend/app.html (use --frontend DIR to point elsewhere)")

    app = FastAPI(title="MaToS bridge", docs_url=None, redoc_url=None)
    http_sessions: Dict[str, Session] = {}

    key = (cfg.key or "").strip() or None

    def _key_ok(value) -> bool:
        return key is None or (bool(value) and secrets.compare_digest(str(value), key))

    @app.middleware("http")
    async def access_key(request: Request, call_next):
        """No key configured -> open (LAN use).  With a key (the default behind ngrok) every route needs it: the first
        visit comes with ?key=... in the link, which is swapped for a cookie, so the page's own requests, the
        WebSocket and /api/msg are all covered without the page knowing about it."""
        if key is None:
            return await call_next(request)
        if _key_ok(request.query_params.get("key")):
            rest = request.url.remove_query_params("key")
            target = rest.path + (f"?{rest.query}" if rest.query else "")       # relative: behind ngrok our own scheme looks like http
            resp = RedirectResponse(target, status_code=303)
            secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
            resp.set_cookie(KEY_COOKIE, key, max_age=KEY_COOKIE_MAX_AGE, httponly=True, samesite="lax", secure=secure)
            return resp
        if _key_ok(request.cookies.get(KEY_COOKIE)):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "access key required"}, status_code=401)
        return HTMLResponse(_DENIED, status_code=401)

    @app.middleware("http")
    async def no_cache(request: Request, call_next):           # a prototype: edits must show up on the phone on reload
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/app.html", response_class=HTMLResponse)
    @app.get("/app", response_class=HTMLResponse)
    async def workspace():
        return HTMLResponse(_inject(page.read_text(encoding="utf-8")))

    @app.get("/api/health")
    async def health():
        return {"ok": True, **engine.info(), "http_sessions": len(http_sessions)}

    # -- WebSocket transport ---------------------------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        if not _key_ok(sock.cookies.get(KEY_COOKIE)):         # http middleware does not run for websockets
            await sock.close(code=1008)
            return
        await sock.accept()
        who = sock.client.host if sock.client else "?"
        session = Session(engine, log)
        log(f"[bridge] websocket connected: {who}")
        try:
            while True:
                try:
                    msg = json.loads(await sock.receive_text())
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                for out in await session.handle(msg):
                    await sock.send_text(json.dumps(out, separators=(",", ":")))
        except WebSocketDisconnect:
            pass
        except RuntimeError:                                    # client sent a binary frame / closed mid-send
            pass
        finally:
            log(f"[bridge] websocket closed: {who} ({session.total_frames} frames)")

    # -- HTTP fallback transport -------------------------------------------------------------------------------------
    @app.post("/api/msg")
    async def http_msg(request: Request):
        try:
            body = await request.json()
            sid, msgs = str(body["sid"])[:64], body["msgs"]
            if not isinstance(msgs, list):
                raise ValueError
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "expected {sid, msgs:[...]}"}, status_code=400)
        now = time.monotonic()
        for k in [k for k, s in http_sessions.items() if now - s.touched > HTTP_SESSION_TTL_S]:
            del http_sessions[k]
        session = http_sessions.get(sid)
        if session is None:
            session = http_sessions[sid] = Session(engine, log)
            log(f"[bridge] http session started: {request.client.host if request.client else '?'}")
        out = []
        for msg in msgs[:MAX_MSGS_PER_POST]:
            if isinstance(msg, dict):
                out += await session.handle(msg)
        return {"msgs": out}

    # -- static: bridge assets, then the untouched frontend -----------------------------------------------------------
    app.mount("/bridge", StaticFiles(directory=str(STATIC_DIR)), name="bridge")
    app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")
    return app
