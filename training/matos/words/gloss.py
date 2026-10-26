"""
gloss.py - dataset label -> the word that goes into the transcript.

The Google/PopSign labels are lower-case concept names, some of them glued together ('thankyou', 'frenchfries', 'minemy').
A signed word enters the sentence as an upper-case ASL gloss, which is exactly what the T5 stage was fine-tuned on
(gloss -> English).  Override or extend the table with  data/gloss_map.tsv   (label <TAB> gloss, '#' comments).
"""
from __future__ import annotations

from typing import Dict, Optional

from matos.paths import LAYOUT

DEFAULTS: Dict[str, str] = {
    "callonphone": "CALL", "frenchfries": "FRENCH FRIES", "glasswindow": "WINDOW", "haveto": "HAVE TO", "hesheit": "IT",
    "icecream": "ICE CREAM", "minemy": "MY", "thankyou": "THANK YOU", "weus": "WE", "yourself": "YOURSELF", "tv": "TV",
    "stuffedanimal": "STUFFED ANIMAL", "shhh": "QUIET", "owie": "HURT",
}
_cache: Optional[Dict[str, str]] = None


def load_map(force: bool = False) -> Dict[str, str]:
    global _cache
    if _cache is not None and not force:
        return _cache
    m = dict(DEFAULTS)
    p = LAYOUT.data / "gloss_map.tsv"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "\t" in line:
                k, v = line.split("\t", 1)
                m[k.strip().lower()] = v.strip().upper()
    _cache = m
    return m


def display(label: str) -> str:
    return load_map().get(label.lower(), label.replace("_", " ").upper())
