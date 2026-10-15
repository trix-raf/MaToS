# build_dictionary.py  ->  data/dictionary.tsv   (needs: pip install wordfreq)
import csv
import sys
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project root, so `matos` imports from any cwd
from matos.paths import LAYOUT  # noqa: E402

words = top_n_list("en", 5000)
# Filter: keep alphabetic, length >= 1
words = [w.upper() for w in words if w.isalpha()]

# Add names, days, months, numbers
extra = [
    "ALICE","BOB","CAROL","DAVID","EMMA","FRANK","GRACE","HENRY",
    "JOHN","LOKY","MARY","PETER","SARAH","TOM",
    "MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY",
    "JANUARY","FEBRUARY","MARCH","APRIL","MAY","JUNE","JULY",
    "AUGUST","SEPTEMBER","OCTOBER","NOVEMBER","DECEMBER",
    "ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN",
]
words.extend(extra)
words = sorted(set(words))

OUT = LAYOUT.data / "dictionary.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["word", "freq"])
    for word in words:
        # zipf_frequency returns log10 scale, higher = more frequent
        w.writerow([word, zipf_frequency(word, "en")])
print(f"wrote {len(words)} words -> {OUT}")