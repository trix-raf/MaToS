"""
Download SIGN-LEVEL (word) data from Kaggle.   Needs:  pip install kagglehub pyarrow   and Kaggle credentials
(kaggle.com -> Settings -> API -> Create New Token, or the KAGGLE_USERNAME / KAGGLE_KEY environment variables).

  python scripts/kaggle_words.py --per-sign 40       A FIRST TRY: ~40 sequences per sign, file by file (a few GB at most), resumable
  python scripts/kaggle_words.py                     the whole Google ISLR competition `asl-signs`: ~40 GB, 57 GB unpacked.  Asks first.
  python scripts/kaggle_words.py --dataset OWNER/SLUG   any other Kaggle DATASET (kagglehub.dataset_download)

`asl-signs` ("Google - Isolated Sign Language Recognition", the PopSign signs) is a COMPETITION, not a dataset, so it is fetched with
kagglehub.competition_download - and you must have accepted its rules once, at https://www.kaggle.com/competitions/asl-signs/rules.
It holds MediaPipe landmarks (face, both hands, POSE) of 250 signs from 21 signers, as parquet files - exactly what the word engine needs.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
COMPETITION = "asl-signs"
RULES = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"


def _hub():
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed:   pip install kagglehub")
        raise SystemExit(1)
    return kagglehub


def _next(root, per_sign: int = 0, seed: int = 0) -> str:
    extra = f" --max-per-class {per_sign} --seed {seed}" if per_sign else ""
    return (f'\nNext:\n  pip install pyarrow          (reads the parquet files; not in requirements-freeze.txt)\n'
            f'  python -m matos words-extract --src islr="{root}" --out wshards{extra}\n'
            f'  python -m matos words-train --shards wshards --out words1')


def _fail(e: Exception) -> None:
    print(f"\nKaggle said: {type(e).__name__}: {e}")
    print(f"  - a 401/403 usually means: no credentials, or the competition rules are not accepted yet ({RULES})")


def _dataset(hub, slug: str) -> int:
    try:
        path = hub.dataset_download(slug)
    except Exception as e:  # noqa: BLE001
        print(f"Could not download the dataset '{slug}': {type(e).__name__}: {e}")
        print(f"  - '{slug}' was not found (or you have no access).  Check it against the address of the page: kaggle.com/datasets/OWNER/NAME")
        print(f"  - Google's sign-level data is a COMPETITION called '{COMPETITION}', not a dataset - fetch it with:\n"
              f"      python scripts/kaggle_words.py --per-sign 40")
        return 1
    print("Path to dataset files:", path)
    print(f'\nIf it holds a train.csv + train_landmark_files/ (Google format) or <word>/*.npz folders:\n'
          f'  python -m matos words-extract --src NAME="{path}" --out wshards')
    return 0


def _full(hub, a) -> int:
    print(f"The competition '{a.competition}' is about 40 GB to download (57 GB unpacked).  For a first experiment use:  --per-sign 40")
    if not a.yes and input("Continue with the full download? [y/N] ").strip().lower() not in ("y", "yes"):
        return 1
    try:
        path = hub.competition_download(a.competition)
    except Exception as e:  # noqa: BLE001
        _fail(e)
        return 1
    print("Path to competition files:", path)
    print(_next(path))
    return 0


def _subset(hub, a) -> int:
    from matos.words.sources import discover_islr

    try:
        root = Path(hub.competition_download(a.competition, path="train.csv")).parent
    except Exception as e:  # noqa: BLE001
        _fail(e)
        return 1
    items, info = discover_islr(str(root), max_per_class=a.per_sign, seed=a.seed)
    rels = [Path(i[1]).relative_to(root).as_posix() for i in items]
    print(f"{len(rels)} files ({info['signs']} signs, {info['signers']} signers, up to {a.per_sign} per sign, spread over the signers) -> {root}")
    failed = []
    for k, rel in enumerate(rels, 1):
        dst = root / rel
        if not (dst.is_file() and dst.stat().st_size > 0):                        # resumable: what is already here is kept
            try:
                got = Path(hub.competition_download(a.competition, path=rel))
                if got.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(got, dst)
            except Exception as e:  # noqa: BLE001
                failed.append((rel, f"{type(e).__name__}: {e}"))
                if len(failed) >= 3 and len(failed) > 0.2 * len(rels):
                    print(f"\nToo many files failed ({len(failed)} of the first {k}); stopping.  Last error: {failed[-1][1]}")
                    _fail(Exception(failed[-1][1]))
                    return 1
        if k % 50 == 0 or k == len(rels):
            print(f"  {k}/{len(rels)} files", flush=True)
    if failed:
        print(f"\n{len(failed)} file(s) could not be downloaded (extraction skips them); run again to retry:")
        for rel, err in failed[:5]:
            print(f"   {rel}: {err}")
    print(_next(root, a.per_sign, a.seed))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download sign-level (word) data from Kaggle.")
    ap.add_argument("--per-sign", type=int, default=0, help="download only N sequences per sign (a few GB instead of 40)")
    ap.add_argument("--dataset", help="OWNER/SLUG of a Kaggle dataset")
    ap.add_argument("--competition", default=COMPETITION)
    ap.add_argument("--seed", type=int, default=0, help="which subset (must match --seed of words-extract; printed for you)")
    ap.add_argument("--yes", action="store_true", help="do not ask before the full download")
    a = ap.parse_args(argv)
    hub = _hub()
    if a.dataset:
        return _dataset(hub, a.dataset)
    return _subset(hub, a) if a.per_sign else _full(hub, a)


if __name__ == "__main__":
    raise SystemExit(main())
