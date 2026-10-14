"""
Stress test the pipeline with synthetic letter errors.

    python stress_test.py                  # deterministic only (no LLM, fast)
    python stress_test.py --llm            # + LM rerank (+polish), paired vs deterministic
    python stress_test.py --polish-suite   # grammar polish on clean gloss input (needs LLM)

Every configuration in a block is run on the SAME simulated inputs, so the
"+fixed / -broken" counts are a paired comparison, not noise between samples.
"""

import argparse
import random
import re
import time
from collections import Counter

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

# For the polish suite: ASL-gloss style input (function words dropped) -> full English.
GLOSS_CASES = [
    ("ME NAME LOKY", "MY NAME IS LOKY"),
    ("I WANT GO STORE", "I WANT TO GO TO THE STORE"),
    ("HELLO HOW YOU", "HELLO HOW ARE YOU"),
    ("WHERE BATHROOM", "WHERE IS THE BATHROOM"),
    ("WHAT YOUR NAME", "WHAT IS YOUR NAME"),
    ("I FROM INDONESIA", "I AM FROM INDONESIA"),
]


# ---------- error simulation ----------
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CONFUSE = {
    "mn": {"M": "N", "N": "M"},
    "uv": {"U": "V", "V": "U"},
}


def apply_errors(word, error_rate, rng, mode="random", confuse_p=0.5):
    """Character-level errors: r/2 deletion, r/4 substitution, r/4 insertion.
    mode 'mn' / 'uv' additionally swaps those letter pairs with prob confuse_p
    on every letter that did not already get a random error."""
    out = []
    for c in word:
        r = rng.random()
        if r < error_rate * 0.5:
            continue
        elif r < error_rate * 0.75:
            out.append(rng.choice(LETTERS))
        elif r < error_rate:
            out.append(c)
            out.append(rng.choice(LETTERS))
        else:
            if mode in _CONFUSE and rng.random() < confuse_p:
                out.append(_CONFUSE[mode].get(c, c))
            else:
                out.append(c)
    return "".join(out) if out else word


def simulate_sentence(target, error_rate, rng, mode="random"):
    return [apply_errors(w, error_rate, rng, mode) for w in target.split()]


def make_cases(rate, mode, n_trials, seed):
    rng = random.Random(f"{seed}-{mode}-{rate}")
    return [(t, simulate_sentence(t, rate, rng, mode))
            for _ in range(n_trials) for t in TARGETS]


# ---------- metrics (all computed on the SAME normalised text) ----------
def normalize_text(s):
    s = s.upper()
    s = re.sub(r"[.,!?;:]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _lev(a, b):
    """Levenshtein distance between two sequences (strings or word lists)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def sentence_exact(target, predicted):
    return normalize_text(target) == normalize_text(predicted)


def word_accuracy(target, predicted):
    # 1 - WER. The old version zipped words by position, so a single inserted
    # word shifted everything after it, and punctuation was not normalised
    # (while sentence_exact did normalise) -> 95% exact but 88% word accuracy.
    t = normalize_text(target).split()
    p = normalize_text(predicted).split()
    return max(0.0, 1 - _lev(t, p) / max(len(t), 1))


def char_error_rate(target, predicted):
    t, p = normalize_text(target), normalize_text(predicted)
    return _lev(t, p) / max(len(t), 1)


# ---------- driver ----------
def evaluate(cases, **kw):
    results = []
    for target, chunks in cases:
        t0 = time.perf_counter()
        corrected, final, _combos, info = process_letter_chunks(chunks, **kw)
        ms = 1000 * (time.perf_counter() - t0)
        # BUG FIX: old run() always called process_letter_chunks(polish=True)
        # even for the "no LLM" section, so the deterministic section was
        # loading and calling the LLM and its latency numbers were polish time.
        out = final if kw.get("polish") and final else corrected
        results.append({
            "target": target, "chunks": chunks, "output": out, "info": info,
            "ok": sentence_exact(target, out),
            "wacc": word_accuracy(target, out),
            "cer": char_error_rate(target, out),
            "ms": ms,
        })
    return results


def report(name, res, baseline=None, show_ms=True, show_failures=0):
    n = len(res)
    exact = sum(r["ok"] for r in res)
    line = (f"  {name:<22} exact {exact:>4}/{n} ({exact/n:5.1%})"
            f"  word-acc {sum(r['wacc'] for r in res)/n:5.1%}"
            f"  CER {sum(r['cer'] for r in res)/n:5.1%}")
    if show_ms:
        line += f"  {sum(r['ms'] for r in res)/n:6.0f} ms"
    if baseline is not None:
        fixed = sum(a["ok"] and not b["ok"] for a, b in zip(res, baseline))
        broke = sum(b["ok"] and not a["ok"] for a, b in zip(res, baseline))
        line += f"   (+{fixed} fixed / -{broke} broken)"
    print(line)
    fails = [r for r in res if not r["ok"]][:show_failures]
    for r in fails:
        print(f"      chunks: {r['chunks']}\n      target: {r['target']}\n"
              f"      output: {r['output']}   [{r['info'].get('rank_source', '')}"
              f" {r['info'].get('polish_source', '')} {r['info'].get('reason', '')}]")


def run_fast(n_trials, seed):
    print("Deterministic (no LLM)")
    settings = ([("random", r) for r in (0.0, 0.05, 0.10, 0.15, 0.20)]
                + [("mn", 0.10), ("mn", 0.20), ("uv", 0.10)])
    for mode, rate in settings:
        cases = make_cases(rate, mode, n_trials, seed)
        print(f"\n--- mode={mode}  error_rate={rate:.0%}  ({len(cases)} sentences) ---")
        report("deterministic", evaluate(cases), show_failures=3)


def run_llm(n_trials, seed, weights):
    from llama_ranker import get_llm, sentence_logprob
    print("\n\nLM rerank / polish (Qwen) vs deterministic, same inputs")
    get_llm()
    process_letter_chunks(["HELLO", "YOU"], rank=True, polish=True)   # warm-up, not timed

    settings = [("random", 0.0), ("random", 0.10), ("random", 0.20),
                ("mn", 0.10), ("mn", 0.20)]
    for mode, rate in settings:
        cases = make_cases(rate, mode, n_trials, seed)
        print(f"\n--- mode={mode}  error_rate={rate:.0%}  ({len(cases)} sentences) ---")
        base = evaluate(cases)
        report("deterministic", base)
        sentence_logprob.cache_clear()
        for i, w in enumerate(weights):
            # sentence_logprob is cached, so only the first weight pays for LM
            # calls; later weights are nearly free (latency shown only for the first).
            res = evaluate(cases, rank=True, lm_weight=w)
            report(f"lm rerank w={w}", res, base, show_ms=(i == 0))
            lm_err = sum(r["info"].get("rank_source") == "lm_error" for r in res)
            if lm_err:
                print(f"      !! {lm_err} LM scoring errors, e.g. "
                      f"{next(r['info']['lm_error'] for r in res if 'lm_error' in r['info'])}")
        best_w = weights[len(weights) // 2]
        res = evaluate(cases, rank=True, polish=True, lm_weight=best_w)
        report(f"rerank w={best_w}+polish", res, base, show_failures=4)
        print("      polish outcomes:", dict(Counter(r["info"].get("polish_source") for r in res)))


def run_polish_suite():
    """Polish on CLEAN input, so it is measured on its own job (adding missing
    function words) instead of being judged on already-complete sentences,
    where it can only break things."""
    from llama_ranker import grammar_polish, get_llm
    get_llm()
    print("\n\nPolish suite (clean input, no spelling errors)")

    ok = 0
    for gloss, want in GLOSS_CASES:
        got, src, info = grammar_polish(gloss, gloss.split())
        good = got == want
        ok += good
        print(f"  {'OK ' if good else 'BAD'} {gloss!r} -> {got!r} [{src}"
              f"{': ' + info['reason'] if 'reason' in info else ''}]"
              + ("" if good else f"  want {want!r}"))
    print(f"  adds missing words correctly: {ok}/{len(GLOSS_CASES)}")

    same = 0
    reasons = Counter()
    for t in TARGETS:
        got, src, info = grammar_polish(t, t.split())
        same += (got == t)
        if got != t:
            print(f"  CHANGED already-correct sentence: {t!r} -> {got!r}")
        if src == "fallback":
            reasons[info.get("reason", "?")] += 1
    print(f"  leaves correct sentences alone: {same}/{len(TARGETS)}"
          f"   fallback reasons: {dict(reasons)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run LM rerank/polish")
    ap.add_argument("--polish-suite", action="store_true")
    ap.add_argument("--trials", type=int, default=5, help="repeats of the 20 targets (deterministic)")
    ap.add_argument("--llm-trials", type=int, default=3, help="repeats of the 20 targets (LLM runs)")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.15, 0.3],
                    help="LM weights to sweep; the middle one is used for the polish run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_fast(args.trials, args.seed)
    if args.llm:
        run_llm(args.llm_trials, args.seed, args.weights)
    if args.polish_suite:
        run_polish_suite()
