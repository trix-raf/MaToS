# build_dictionary.py
from wordfreq import top_n_list, zipf_frequency
import csv

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

with open("dictionary.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["word", "freq"])
    for word in words:
        # zipf_frequency returns log10 scale, higher = more frequent
        w.writerow([word, zipf_frequency(word, "en")])
print(f"wrote {len(words)} words")