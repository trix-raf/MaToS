"""
sources.py - find the images of an alphabet dataset and infer each image's label.

Works for the three Kaggle sets you use (and your own capture folder `signer2/<class>/*.jpg`) without
hard-coding their layout:
  grassknoted/asl-alphabet               asl_alphabet_train/asl_alphabet_train/A/A1.jpg ...
  rupaul007/american-sign-language-...   <letter>/<image>
  dorukdemirci/asl-alphabet-dataset      <letter>[/<left|right>_<light|nolight>]/<image>   (A-Y, no J / Z)

Label inference ("auto"): look at every directory level; the level whose folder names are the most
class-like (A..Z, space, del, nothing) is the class level - so a "left/right" or "L/R" level next to the
letter level cannot be mistaken for a letter.  If no level looks like classes, fall back to the file name
("A_001.jpg", "space12.png").
"""
from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from matos.config import CLASSES, LABEL_ALIASES

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_LETTER = re.compile(r"(?:letter[_\- ]?|class[_\- ]?)?([a-z])")


def canonical(token: str) -> Optional[str]:
    t = token.strip().lower()
    if t in LABEL_ALIASES:
        return LABEL_ALIASES[t]
    m = _LETTER.fullmatch(t)
    return m.group(1).upper() if m else None


def label_from_filename(stem: str) -> Optional[str]:
    m = re.match(r"^([A-Za-z]+)(?=[\d_\-. ]|$)", stem)
    return canonical(m.group(1)) if m else None


def discover(root: str, label_from: str = "auto", exclude: Tuple[str, ...] = ("test",), limit_per_class: int = 0,
             seed: int = 0) -> Tuple[List[Tuple[str, int]], dict]:
    """Returns ([(path, class_index)], info).  `exclude`: skip folders whose name contains one of these."""
    root_p = Path(root)
    if not root_p.is_dir():
        raise SystemExit(f"dataset folder not found: {root}")
    files = []
    for dp, dn, fn in os.walk(root_p):
        dn[:] = sorted(d for d in dn if not any(x in d.lower() for x in exclude))
        files += [Path(dp) / f for f in sorted(fn) if Path(f).suffix.lower() in IMG_EXT]
    if not files:
        raise SystemExit(f"no images under {root}")
    rel_dirs = [f.relative_to(root_p).parts[:-1] for f in files]

    depth = None
    if label_from in ("auto", "parent"):
        seen = defaultdict(set)
        for parts in rel_dirs:
            for d, name in enumerate(parts):
                c = canonical(name)
                if c:
                    seen[d].add(c)
        if seen:
            best = max(seen, key=lambda d: len(seen[d]))
            if len(seen[best]) >= 3 or label_from == "parent":
                depth = best
        if depth is None and label_from == "parent":
            raise SystemExit(f"{root}: cannot infer labels from folder names")
    items, skipped = [], Counter()
    for f, parts in zip(files, rel_dirs):
        lab = canonical(parts[depth]) if depth is not None and len(parts) > depth else None
        if lab is None and label_from in ("auto", "filename"):
            lab = label_from_filename(f.stem)
        if lab is None or lab not in CLASSES:
            skipped["unlabelled"] += 1
            continue
        items.append((str(f), CLASSES.index(lab)))
    if limit_per_class:
        by = defaultdict(list)
        for it in items:
            by[it[1]].append(it)
        rng = random.Random(seed)
        items = [it for k in sorted(by) for it in (rng.sample(by[k], limit_per_class) if len(by[k]) > limit_per_class else by[k])]
    cnt = Counter(CLASSES[y] for _, y in items)
    info = dict(root=str(root), class_depth=depth, images=len(items), skipped=dict(skipped),
                classes_present=len(cnt), per_class=dict(sorted(cnt.items())))
    return items, info