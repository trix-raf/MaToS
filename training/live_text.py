"""
live_text.py - jembatan  matos/live.py  ->  pipeline.py

    frames --(MediaPipe + bundle + LetterDecoder, matos/live.py)--> event "sentence", e.words = ["NME", "IS", "LOKY"]
           --(pipeline.process_letter_chunks)-->                     "MY NAME IS LOKY"

Jalankan dari root project (folder yang berisi matos/, pipeline.py, autocorrect.py, models/):

    python live_text.py --bundle bundles/main                    # webcam, koreksi ejaan saja (cepat, tanpa LLM)
    python live_text.py --bundle bundles/main --rank             # + LM rerank
    python live_text.py --bundle bundles/main --rank --polish    # + grammar polish (menambah kata yang hilang)
    python live_text.py --bundle bundles/main --video clip.mp4 --no-show
    python live_text.py --bundle bundles/main --save-chunks real.jsonl   # simpan chunk asli untuk stress test

Batas kalimat ditentukan LetterDecoder (event "sentence"), bukan SentenceAccumulator di pipeline.py.
Pipeline jalan di worker thread supaya kamera / decoder tidak tersendat saat LLM sedang berpikir.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

# supaya `pipeline`, `autocorrect`, `llama_ranker`, `matos` ter-import dari cwd mana pun
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import process_letter_chunks  # noqa: E402


class SentenceWorker(threading.Thread):
    """Antrian kalimat mentah -> process_letter_chunks -> teks akhir. Satu thread saja,
    jadi llama.cpp tidak pernah dipakai dari dua thread sekaligus."""

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
    ap.add_argument("--rank", action="store_true", help="LM rerank kandidat ejaan (butuh llama_cpp + model gguf)")
    ap.add_argument("--polish", action="store_true", help="grammar polish (butuh llama_cpp + model gguf)")
    ap.add_argument("--lm-weight", type=float, default=0.15)
    ap.add_argument("--lm-margin", type=float, default=3.0)
    ap.add_argument("--save-chunks", metavar="FILE.jsonl",
                    help="tambahkan {chunks, output} tiap kalimat ke file ini (bahan test dari data asli)")
    a = ap.parse_args(argv)

    from matos.live import run

    if a.rank or a.polish:
        print("[bridge] memuat LLM (sekali, sebelum kamera mulai)...")
        from llama_ranker import get_llm

        get_llm()
        process_letter_chunks(["HELLO", "YOU"], rank=a.rank, polish=a.polish)   # warm-up, kalimat pertama tidak lambat

    worker = SentenceWorker(
        dict(rank=a.rank, polish=a.polish, lm_weight=a.lm_weight, lm_margin=a.lm_margin),
        save_path=a.save_chunks,
    )
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