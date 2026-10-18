"""ngrok tunnel: gives the bridge a public https address that works from any network (phone on mobile data, another
wifi, ...).  ngrok terminates TLS with a real certificate, so the phone gets camera access without the self-signed
certificate warning, and the local server can stay on plain http (see __main__.py).

Needs:  pip install pyngrok      and a free ngrok account -> https://dashboard.ngrok.com/get-started/your-authtoken
"""
from __future__ import annotations

import atexit
import os
import shutil
from typing import Optional

TOKEN_URL = "https://dashboard.ngrok.com/get-started/your-authtoken"


def _hint(err: BaseException) -> str:
    text = str(err)
    if "4018" in text or "authtoken" in text.lower():
        return (f"ngrok needs your authtoken (free account).  Get it at {TOKEN_URL}, then either run\n"
                f"      python -m app --ngrok --ngrok-token <token>\n"
                f"    or set it once:  ngrok config add-authtoken <token>   (or the NGROK_AUTHTOKEN environment variable)")
    if "108" in text or "simultaneous" in text.lower():
        return ("your ngrok account already has an agent session running somewhere (the free plan allows one).\n"
                "    Close the other `ngrok` window/process or stop it at https://dashboard.ngrok.com/agents")
    if "334" in text or "already online" in text.lower():
        return "that ngrok domain is already online in another agent - stop the other one first"
    return text.strip().splitlines()[0][:300] if text.strip() else type(err).__name__


def start_tunnel(port: int, token: Optional[str] = None, domain: Optional[str] = None, log=print) -> str:
    """Open an https tunnel to 127.0.0.1:<port>; returns the public base URL (no trailing slash).  Closed at exit."""
    try:
        from pyngrok import conf, ngrok
    except ImportError:
        raise SystemExit("--ngrok needs the 'pyngrok' package:  pip install pyngrok")

    token = token or os.environ.get("NGROK_AUTHTOKEN")
    # use the ngrok you already installed if there is one; otherwise pyngrok downloads its own copy on first use
    exe = shutil.which("ngrok")
    cfg = conf.PyngrokConfig(ngrok_path=exe) if exe else conf.get_default()
    if token:
        ngrok.set_auth_token(token, pyngrok_config=cfg)

    opts = {"domain": domain} if domain else {}
    try:
        tunnel = ngrok.connect(addr=f"127.0.0.1:{port}", proto="http", pyngrok_config=cfg, **opts)
    except Exception as e:  # noqa: BLE001  (PyngrokNgrokError, network, bad token, ...)
        raise SystemExit(f"[ngrok] could not start the tunnel:\n    {_hint(e)}")

    def _close():
        try:
            ngrok.kill(pyngrok_config=cfg)
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_close)
    url = str(tunnel.public_url).rstrip("/")
    if url.startswith("http://"):                      # older ngrok agents report the plain-http twin first
        url = "https://" + url[len("http://"):]
    log(f"[ngrok] tunnel up: {url}  ->  127.0.0.1:{port}")
    return url
