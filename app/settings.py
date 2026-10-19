"""Where things live.  The bridge never copies or edits ./server/frontend or ./training/matos - it only reads them.

    <root>/
        app/                    this bridge
        server/frontend/        app.html, index.html, js/, css/, image/   (served as-is)
        training/matos/         the python package   (training/models, training/data live next to it)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
CERT_DIR = APP_DIR / ".certs"

DEFAULT_FRONTEND = ROOT / "server" / "frontend"
DEFAULT_TRAINING = ROOT / "training"


def add_matos_to_path(training_dir: Optional[Path] = None) -> Path:
    """Make `import matos` work no matter where the server is launched from (matos/paths.py anchors itself on
    the folder that contains `matos/`, so models/ and data/ are found next to it - nothing to configure)."""
    d = Path(training_dir or os.environ.get("MATOS_TRAINING_DIR") or DEFAULT_TRAINING).resolve()
    if not (d / "matos" / "__init__.py").is_file():
        raise SystemExit(f"matos package not found in {d}\n  expected {d / 'matos'}  "
                         f"(use --training DIR or set MATOS_TRAINING_DIR to the folder that contains matos/)")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    bundle: Optional[str] = None        # None -> newest bundle found under training/models
    engine: str = "auto"                # auto | opencv | ort   (same meaning as `python -m matos live --engine`)
    mirror: Optional[bool] = None       # None -> matos.config.CAMERA_MIRROR (what the model was trained with)
    text_mode: str = "sentences"        # sentences: autocorrected sentence after a pause | words: every finished word at once
    polish: bool = False                # T5 grammar polish on sentences (needs models/asl_t5_onnx_opencv)
    squeeze: bool = True                # repeated letters: PAAINT -> PAINT, ROOMY -> ROMY if ROMY is in names.txt (False = old behaviour)
    https: bool = True                  # phones only allow camera access on https (or localhost)
    frontend: Path = DEFAULT_FRONTEND
    training: Optional[Path] = None
    qr: bool = True
    ngrok: bool = False                 # public https address through an ngrok tunnel (reachable from any network)
    ngrok_token: Optional[str] = None   # else NGROK_AUTHTOKEN / the token saved by `ngrok config add-authtoken`
    ngrok_domain: Optional[str] = None  # a fixed domain from your ngrok dashboard, so the link survives restarts
    key: Optional[str] = None           # access key required on every request; None = open
