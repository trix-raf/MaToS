"""python -m app            (run it from the project root, the folder that contains app/, server/ and training/)"""
from __future__ import annotations

import argparse
import os
import secrets
import sys

from app.settings import DEFAULT_FRONTEND, Settings, add_matos_to_path


def _banner(cfg: Settings, engine, ips, public_url=None):
    scheme = "https" if cfg.https else "http"
    q = f"?key={cfg.key}" if cfg.key else ""
    if public_url:
        urls = [f"{public_url}/app.html{q}"]
    else:
        urls = [f"{scheme}://{ip}:{cfg.port}/app.html{q}" for ip in ips] or [f"{scheme}://localhost:{cfg.port}/app.html{q}"]
    print("\n" + "=" * 72)
    print(f"  MaToS bridge ready   bundle: {engine.bundle_name}   engine: {engine.bundle.engine.name}")
    print(f"  text: mode={engine.text_mode}  stage={engine.text_stage}   repeats={'on' if engine.squeeze else 'off'}   mirror={engine.mirror}")
    if public_url:
        print("  open on the phone (ANY network - mobile data works too):")
    else:
        print("  open on the phone (same wifi):")
    for u in urls:
        print(f"      {u}")
    print(f"  on this PC:  {scheme}://localhost:{cfg.port}/app.html{q}")
    if public_url:
        print("  ngrok free plan: the first visit shows an ngrok notice page -> tap 'Visit Site'.  Not an error.")
        if cfg.key:
            print("  the link contains the access key: anyone who has it can use this server, so do not post it.")
    elif cfg.https:
        print("  first visit: the browser warns the certificate is not trusted - that is the self-signed")
        print("  certificate; choose Advanced -> proceed / Visit this website.  (needed for camera access)")
    print("=" * 72 + "\n", flush=True)
    if cfg.qr and (ips or public_url):
        try:
            import qrcode

            sys.stdout.reconfigure(encoding="utf-8")
            code = qrcode.QRCode(border=1)
            code.add_data(urls[0])
            code.make()
            code.print_ascii(invert=True)
            print()
        except Exception:  # noqa: BLE001  (qrcode not installed / terminal cannot draw blocks: the URL above is enough)
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app", description="MaToS phone bridge (FastAPI)")
    ap.add_argument("--host", default=None, help="default 0.0.0.0 (LAN); 127.0.0.1 with --ngrok, so only the tunnel can reach it")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bundle", help="bundle name/path as for `python -m matos live --bundle` (default: newest under training/models)")
    ap.add_argument("--engine", default="auto", choices=["auto", "opencv", "ort"])
    ap.add_argument("--text-mode", default="sentences", choices=["sentences", "words"],
                    help="sentences: autocorrected sentence after a ~1.5 s pause (default) | words: each word as soon as it ends, uncorrected")
    ap.add_argument("--polish", action="store_true", help="T5 grammar polish on sentences (needs models/asl_t5_onnx_opencv)")
    ap.add_argument("--no-squeeze", action="store_true",
                    help="do not collapse repeated letters (PAAINT -> PAINT, ROOMY -> ROMY via names.txt); plain spelling correction only")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror landmarks (live.py --no-mirror)")
    ap.add_argument("--no-https", action="store_true", help="plain http: camera then only works on localhost, NOT on a phone")
    ap.add_argument("--no-qr", action="store_true")
    ap.add_argument("--ngrok", action="store_true",
                    help="public https URL through ngrok: works from any network (needs `pip install pyngrok` + a free ngrok authtoken)")
    ap.add_argument("--ngrok-token", help="ngrok authtoken (default: NGROK_AUTHTOKEN env var, or the one saved by `ngrok config add-authtoken`)")
    ap.add_argument("--ngrok-domain", help="your fixed ngrok domain (free account: dashboard -> Domains) so the link is the same every run")
    ap.add_argument("--key", help="access key required to open the page (default with --ngrok: random per run; or set MATOS_KEY)")
    ap.add_argument("--no-auth", action="store_true", help="no access key, even with --ngrok (anyone with the URL can use the server)")
    ap.add_argument("--frontend", default=str(DEFAULT_FRONTEND))
    ap.add_argument("--training", help="folder that contains matos/ (default ./training)")
    a = ap.parse_args(argv)

    key = None if a.no_auth else (a.key or os.environ.get("MATOS_KEY") or (secrets.token_urlsafe(12) if a.ngrok else None))
    cfg = Settings(host=a.host or ("127.0.0.1" if a.ngrok else "0.0.0.0"), port=a.port, bundle=a.bundle, engine=a.engine,
                   mirror=False if a.no_mirror else None, text_mode=a.text_mode, polish=a.polish, squeeze=not a.no_squeeze,
                   https=not (a.no_https or a.ngrok),       # ngrok brings its own valid certificate: plain http locally
                   frontend=a.frontend, training=a.training, qr=not a.no_qr,
                   ngrok=a.ngrok, ngrok_token=a.ngrok_token, ngrok_domain=a.ngrok_domain, key=key)
    add_matos_to_path(cfg.training)

    from app.certs import ensure_cert, lan_ips
    from app.engine import Engine
    from app.server import create_app

    engine = Engine(cfg)                           # fail fast, before the server starts
    app = create_app(engine, cfg)
    ips = lan_ips()
    ssl = {}
    if cfg.https:
        cert, key_file = ensure_cert(ips)
        ssl = dict(ssl_certfile=str(cert), ssl_keyfile=str(key_file))
    public_url = None
    if cfg.ngrok:
        from app.tunnel import start_tunnel

        public_url = start_tunnel(cfg.port, cfg.ngrok_token, cfg.ngrok_domain)
        ips = []                                     # LAN addresses are of no use now: the phone goes through the tunnel
    _banner(cfg, engine, ips, public_url)

    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning", access_log=False,
                proxy_headers=True, forwarded_allow_ips="127.0.0.1", **ssl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
