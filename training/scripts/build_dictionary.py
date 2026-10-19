# build_dictionary.py  ->  data/dictionary.tsv   (needs: pip install wordfreq)
import csv
import sys
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project root, so `matos` imports from any cwd
from matos.paths import LAYOUT  # noqa: E402

TARGET = 7500          # total words in the final dictionary (extras included)
POOL = 100_000         # how deep to search for J/Z words
JZ_QUOTA = 1200        # max slots reserved for words containing J or Z
JZ_MIN_ZIPF = 2.0      # skip very rare/junk J/Z words below this frequency

# Keep alphabetic ASCII only, uppercased
pool = [w.upper() for w in top_n_list("en", POOL) if w.isalpha() and w.isascii()]

def has_jz(w: str) -> bool:
    return "J" in w or "Z" in w

# Add names, days, months, numbers
extra = [
    "ALICE","BOB","CAROL","DAVID","EMMA","FRANK","GRACE","HENRY",
    "JOHN","LOKY","MARY","PETER","SARAH","TOM",
    "MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY",
    "JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE","JULY",
    "AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER",
    "ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN",
]

selected = set(extra)

# 1) Priority: most frequent J/Z words first, up to the quota
jz_added = 0
for w in pool:
    if jz_added >= JZ_QUOTA:
        break
    if has_jz(w) and w not in selected and zipf_frequency(w, "en") >= JZ_MIN_ZIPF:
        selected.add(w)
        jz_added += 1

# 2) Fill the remaining slots with the most frequent other words
for w in pool:
    if len(selected) >= TARGET:
        break
    selected.add(w)

words = sorted(selected)

OUT = LAYOUT.data / "dictionary.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["word", "freq"])
    for word in words:
        # zipf_frequency returns log10 scale, higher = more frequent
        w.writerow([word, zipf_frequency(word, "en")])

n_jz = sum(has_jz(x) for x in words)
print(f"wrote {len(words)} words ({n_jz} with J/Z) -> {OUT}")