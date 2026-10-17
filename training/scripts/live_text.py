"""
live_text.py - jembatan  matos/live.py  ->  matos/text/pipeline.py

    frames --(MediaPipe + bundle + LetterDecoder, matos/live.py)--> event "sentence", e.words = ["NME", "IS", "LOKY"]
           --(matos.text.pipeline.process_letter_chunks)-->          "MY NAME IS LOKY"

Bisa dijalankan dari folder mana pun. --bundle = nama run di models/ (run1 -> models/bundles/run1, base -> bundle dasar
di models/base/) atau path lengkap:

    python scripts/live_text.py --bundle run1                    # webcam, koreksi ejaan saja (cepat, tanpa model teks)
    python scripts/live_text.py --bundle run1 --polish           # + T5 (ONNX) gloss -> kalimat Inggris
    python scripts/live_text.py --bundle run1 --polish --t5-model D:/path/ke/asl_t5_onnx_opencv
    python scripts/live_text.py --bundle run2 --video clip.mp4 --no-show
    python scripts/live_text.py --bundle run1 --save-chunks real.jsonl   # simpan chunk asli untuk stress test

Batas kalimat ditentukan LetterDecoder (event "sentence"), bukan SentenceAccumulator di pipeline.py.
Pipeline jalan di worker thread supaya kamera / decoder tidak tersendat saat T5 sedang berpikir.
Model T5 dicari di models/asl_t5_onnx_opencv (atau $MATOS_T5_MODEL, atau --t5-model).
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

# supaya `matos` (termasuk matos.text.*) ter-import dari cwd mana pun
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matos.text.pipeline import process_letter_chunks  # noqa: E402


class SentenceWorker(threading.Thread):
    """Antrian kalimat mentah -> process_letter_chunks -> teks akhir. Satu thread saja,
    jadi model T5 tidak pernah dipakai dari dua thread sekaligus."""

    def __init__(self, pipeline_kw: dict, save_path: str | None = None, log=print):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.kw, self.save_path, self.log = pipeline_kw, save_path, log
        self.results: list[dict] = []

    def submit(self, chunks: list[str]):
        self.q.put(chunks)

    def close(self):
        """Tunggu semua kalimat di antrian selesai (kalimat terakhir baru keluar saat live.run() berhenti)."""
        pending = self.q.qsize()
        if pending:
            self.log(f"[bridge] menyelesaikan {pending} kalimat yang masih antri...")
        self.q.put(None)
        self.join()

    def run(self):
        while True:
            chunks = self.q.get()
            if chunks is None:
                return
            self._process(chunks)

    def _process(self, chunks: list[str]):
        t0 = time.perf_counter()
        try:
            corrected, final, _combos, info = process_letter_chunks(chunks, **self.kw)
            text = final if self.kw.get("polish") and final else corrected   # sama seperti stress_test.evaluate
        except Exception as e:                       # satu kalimat error tidak boleh mematikan sesi
            text, info = " ".join(chunks), {"error": repr(e)}
            self.log(f"[bridge] pipeline error, pakai teks mentah: {e!r}")
        ms = 1000 * (time.perf_counter() - t0)
        self.results.append({"chunks": chunks, "text": text, "info": info, "ms": round(ms)})
        self.log(f"[text] {text}    <- raw: {' '.join(chunks)}   ({ms:.0f} ms)")
        if self.save_path:
            with open(self.save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"chunks": chunks, "output": text}, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # sama dengan `python -m matos live`
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--camera", default="0")
    ap.add_argument("--video", help="pakai file video, bukan webcam")
    ap.add_argument("--hand-model")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--engine", default="auto", choices=["auto", "opencv", "ort"])
    ap.add_argument("--max-seconds", type=float)
    # pipeline
    ap.add_argument("--polish", action="store_true", help="T5 gloss -> English (butuh onnxruntime + tokenizers + folder model T5)")
    ap.add_argument("--t5-model", metavar="FOLDER", help="folder export T5 (default: models/asl_t5_onnx_opencv)")
    ap.add_argument("--save-chunks", metavar="FILE.jsonl",
                    help="tambahkan {chunks, output} tiap kalimat ke file ini (bahan test dari data asli)")
    a = ap.parse_args(argv)

    from matos.live import run

    if a.polish:
        print("[bridge] memuat T5 (sekali, sebelum kamera mulai)...")
        from matos.text.t5_polish import get_t5

        t5 = get_t5(a.t5_model)
        print(f"[bridge] T5 dari {t5.dir}")
        process_letter_chunks(["HELLO", "YOU"], polish=True)   # warm-up, kalimat pertama tidak lambat

    worker = SentenceWorker(dict(polish=a.polish), save_path=a.save_chunks)
    worker.start()

    def on_event(e):
        if e.kind != "sentence":
            return
        chunks = [w.strip().upper() for w in e.words if w and w.strip()]
        if chunks:
            worker.submit(chunks)

    cam = int(a.camera) if a.camera.isdigit() else a.camera
    stats = None
    try:
        stats = run(a.bundle, cam, a.video, a.hand_model, mirror=not a.no_mirror, show=not a.no_show,
                    prefer=a.engine, on_event=on_event, max_seconds=a.max_seconds)
    finally:
        worker.close()

    print("\n=== ringkasan ===")
    if stats:
        print(f"frames {stats['frames']}, dengan tangan {stats['hands']}, "
              f"huruf mentah '{stats['letters']}', inferensi {stats['infer_ms_mean']:.1f} ms/frame")
    for r in worker.results:
        print(f"  {' '.join(r['chunks'])}  ->  {r['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())