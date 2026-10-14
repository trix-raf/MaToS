"""
Stress test the pipeline with synthetic letter errors.
Generates raw chunks from target sentences, runs through the pipeline,
compares to target.
"""

import random
import time
from autocorrect import candidates, build_combinations
from pipeline import process_letter_chunks


# ---------- test corpus ----------
TARGETS = [
    "MY NAME IS LOKY",
    "HELLO HOW ARE YOU",
    "I WANT TO GO TO THE STORE",
    "HER NAME IS SARAH",
    "THANK YOU VERY MUCH",
    "MY MOM IS NICE",
    "I AM TWENTY ONE",
    "SEE YOU TOMORROW",
    "WHERE IS THE BATHROOM",
    "PLEASE HELP ME",
    "GOOD MORNING",
    "I LOVE YOU",
    "WHAT IS YOUR NAME",
    "NICE TO MEET YOU",
    "I AM FROM INDONESIA",
    "CAN YOU HELP ME",
    "I DONT UNDERSTAND",
    "PLEASE SPEAK SLOWLY",
    "HOW MUCH DOES IT COST",
    "SEE YOU LATER",
]


# ---------- error simulation ----------
def confuse_mn(c):
    return {"M": "N", "N": "M"}.get(c, c)


def confuse_uv(c):
    return {"U": "V", "V": "U"}.get(c, c)


def apply_errors(word, error_rate, mode="random"):
    """Apply character-level errors. Returns a single erroneous string."""
    if not word:
        return word
    out = []
    for c in word:
        r = random.random()
        if r < error_rate * 0.5:
            continue                          # deletion
        elif r < error_rate * 0.75:
            out.append(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        elif r < error_rate:
            out.append(c)
            out.append(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        else:
            if mode == "mn" and random.random() < 0.5:
                out.append(confuse_mn(c))
            elif mode == "uv" and random.random() < 0.5:
                out.append(confuse_uv(c))
            else:
                out.append(c)
    return "".join(out) if out else word


def simulate_sentence(target, error_rate, mode="random"):
    return [apply_errors(w, error_rate, mode) for w in target.split()]


# ---------- metrics ----------
def word_accuracy(target, predicted):
    tw = target.split()
    pw = predicted.split()
    correct = sum(1 for a, b in zip(tw, pw) if a == b)
    return correct / max(len(tw), len(pw))

import re

def normalize_text(s):
    s = s.upper()
    s = re.sub(r"[.,!?;:]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sentence_exact(target, predicted):
    return normalize_text(target) == normalize_text(predicted)

def char_error_rate(target, predicted):
    # simple Levenshtein distance / target length
    a, b = target, predicted
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            )
        prev = curr
    return prev[-1] / max(len(a), 1)


# ---------- driver ----------
def run(error_rate, mode="random", n_trials=20, polish=False, rank=False):
    exact = 0
    word_accs = []
    cers = []
    times = []
    failures = []

    for trial in range(n_trials):
        for target in TARGETS:
            chunks = simulate_sentence(target, error_rate, mode)
            t0 = time.time()
            corrected, final, combos, info = process_letter_chunks(
                chunks,
                polish=True,
                rank=rank,          # <— let the LLM pick from combos
            )
            times.append(time.time() - t0)

            output = final if polish and final else corrected
            if sentence_exact(target, output): exact += 1
            word_accs.append(word_accuracy(target, output))
            cers.append(char_error_rate(target, output))

            if not sentence_exact(target, output) and len(failures) < 5:
                failures.append((chunks, target, output))

    total = n_trials * len(TARGETS)
    print(f"\n--- error_rate={error_rate:.0%}  mode={mode}  polish={polish} ---")
    print(f"  trials:              {total}")
    print(f"  sentence exact:      {exact}/{total}  ({exact/total:.1%})")
    print(f"  word accuracy:       {sum(word_accs)/len(word_accs):.1%}")
    print(f"  char error rate:     {sum(cers)/len(cers):.1%}")
    print(f"  avg latency:         {1000*sum(times)/len(times):.0f} ms")
    if failures:
        print("  sample failures:")
        for chunks, tgt, out in failures:
            print(f"    chunks: {chunks}")
            print(f"    target: {tgt}")
            print(f"    output: {out}")


if __name__ == "__main__":
    random.seed(42)

    # ---- FAST: deterministic only, no LLM at all ----
    print("Stress test — deterministic (no LLM)")
    for rate in [0.0, 0.05, 0.10, 0.15, 0.20]:
        run(error_rate=rate, mode="random", n_trials=5,
            polish=False, rank=False)          # <— rank=False

    # ---- SLOW: LLM polish, tiny n ----
    print("\n\nStress test — with LLM polish (Qwen 3B)")
    for rate in [0.0, 0.10, 0.20]:
        run(error_rate=rate, mode="random", n_trials=1,   # <— n_trials=1
            polish=True, rank=True)

    # ---- SLOW: M/N confusion with LLM ----
    print("\n\nM/N confusion stress test")
    for rate in [0.10, 0.20]:
        run(error_rate=rate, mode="mn", n_trials=1,
            polish=True, rank=True)