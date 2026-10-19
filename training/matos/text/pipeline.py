import warnings

from matos.text.autocorrect import candidates, build_combinations
from matos.text.repeats import has_repeats


def process_letter_chunks(
    chunks,
    polish=False,
    top_k_per_chunk=6,
    top_n_combos=20,
    rank=False,
    squeeze=True,
):
    """
    Returns (corrected, final, combos, info)
      corrected : best sentence after spelling correction (edit score + word frequency)
      final     : T5 grammar-polished sentence (None unless polish=True)
      combos    : candidate sentences, best edit-score first
      info      : rank_source / polish_source / reasons, for debugging;  "repeat_fixes" = {typed: chosen} for every chunk
                  whose repeated letters were collapsed (PAAINT -> PAINT, ROOMY -> ROMY)

    squeeze   : handle repeated letters (matos/text/repeats.py).  False = plain edit-distance spelling correction.

    rank      : REMOVED with the Qwen stage (it needed a causal LM to score sentences; the fine-tuned T5 is a
                gloss -> English model, not that).  Accepted so old callers do not crash; it only warns.
    """
    if rank:
        warnings.warn("process_letter_chunks(rank=True) is ignored: the Qwen LM rerank was removed, candidates are "
                      "ranked by the spelling score only.", stacklevel=2)
    chunks = [c.strip() for c in chunks if c and c.strip()]
    if not chunks:
        return "", None, [], {"reason": "no input"}

    per_chunk = [candidates(ch, top_k=top_k_per_chunk, squeeze=squeeze) for ch in chunks]
    scored = build_combinations(per_chunk, top_n=top_n_combos, with_scores=True)
    if not scored:
        return "", None, [], {"reason": "no candidates"}

    combos = [s for s, _ in scored]
    corrected = combos[0]
    info = {"rank_source": "score"}
    if squeeze:
        fixes = {ch.upper(): c[0][0] for ch, c in zip(chunks, per_chunk) if has_repeats(ch.upper()) and c[0][0] != ch.upper()}
        if fixes:
            info["repeat_fixes"] = fixes

    if not polish:
        return corrected, None, combos, info

    from matos.text.t5_polish import grammar_polish        # only import onnxruntime / tokenizers if needed
    final, source, pinfo = grammar_polish(corrected, chunks)
    return corrected, final, combos, {**info, "polish_source": source, **pinfo}


class SentenceAccumulator:
    def __init__(self, pause_threshold_s=1.5):
        self.words = []
        self.last_word_time = None
        self.pause_threshold_s = pause_threshold_s

    def add_word(self, word, timestamp_s):
        finished = None
        if (self.last_word_time is not None
                and timestamp_s - self.last_word_time > self.pause_threshold_s):
            finished = self.flush()
        self.words.append(word)
        self.last_word_time = timestamp_s
        return finished

    def flush(self):
        if not self.words:
            return None
        s = " ".join(self.words)
        self.words = []
        self.last_word_time = None
        return s


def stream_sentences(word_stream, pause_threshold_s=1.5):
    """
    word_stream: iterable of (word, timestamp_s).
    Yields raw chunk strings when a sentence boundary is detected.
    """
    acc = SentenceAccumulator(pause_threshold_s)
    for word, t in word_stream:
        finished = acc.add_word(word, t)
        if finished:
            yield finished
    last = acc.flush()
    if last:
        yield last


if __name__ == "__main__":
    print("=== word candidates ===")
    for ch in ["NME", "THNK", "MY", "LOKY", "HELO", "WRD", "T", "A", "PAAINT", "HELLO", "HEEELLO"]:
        print(ch, "->", [c[0] for c in candidates(ch, top_k=4)])

    print("\n=== sentence tests (deterministic) ===")
    for chunks in [
        ["NME", "IS", "LOKY"],
        ["MY", "NME", "IS", "LOKY"],
        ["HELO", "MY", "NME", "IS", "LOKY"],
        ["ME", "NAME", "LOKY"],
    ]:
        corrected, _, combos, _ = process_letter_chunks(chunks, polish=False)
        print(f"in:        {chunks}")
        print(f"corrected: {corrected}\n")

    print("=== stream with pause ===")
    stream = [("NME", 0.0), ("IS", 0.5), ("LOKY", 1.0),
              ("HELO", 3.0), ("WRD", 3.5)]
    for raw in stream_sentences(stream, pause_threshold_s=1.5):
        corrected, _, _, _ = process_letter_chunks(raw.split(), polish=False)
        print(f"raw:       {raw}")
        print(f"corrected: {corrected}")