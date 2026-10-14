from autocorrect import candidates, build_combinations

def process_letter_chunks(
    chunks,
    polish=False,
    rank=False,
    top_k_per_chunk=6,
    top_n_combos=12,
):
    per_chunk = [candidates(ch, top_k=top_k_per_chunk) for ch in chunks]
    per_chunk_top3 = [c[:3] for c in per_chunk]
    combos = build_combinations(per_chunk_top3, top_n=8)


    if not combos:
        return "", None, [], {"reason": "no candidates"}

    # ONLY import the LLM module if we actually need it
    if rank:
        from llama_ranker import rank_sentences
        idx, _ = rank_sentences(chunks, combos)
        corrected = combos[idx]
    else:
        corrected = combos[0]

    if not polish:
        return corrected, None, combos, {"source": "rank" if rank else "score"}

    from llama_ranker import grammar_polish
    final, source, info = grammar_polish(corrected, chunks)
    return corrected, final, combos, {"source": source, **info}


class SentenceAccumulator:
    def __init__(self, pause_threshold_s=1.5):
        self.words = []
        self.last_word_time = None
        self.pause_threshold_s = pause_threshold_s

    def add_word(self, word, timestamp_s):
        if self.last_word_time is not None:
            if timestamp_s - self.last_word_time > self.pause_threshold_s:
                finished = self.flush()
                self.words = [word]
                self.last_word_time = timestamp_s
                return finished
        self.words.append(word)
        self.last_word_time = timestamp_s
        return None

    def flush(self):
        if not self.words:
            return None
        s = " ".join(self.words)
        self.words = []
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
    for ch in ["NME", "THNK", "MY", "LOKY", "HELO", "WRD"]:
        print(ch, "→", [c[0] for c in candidates(ch, top_k=4)])

    print("\n=== sentence tests (deterministic) ===")
    for chunks in [
        ["NME", "IS", "LOKY"],
        ["MY", "NME", "IS", "LOKY"],
        ["HELO", "MY", "NME", "IS", "LOKY"],
        ["ME", "NAME", "LOKY"],
    ]:
        corrected, _, combos, _ = process_letter_chunks(chunks, polish=False)
        print(f"in:        {chunks}")
        print(f"corrected: {corrected}")
        print()

    print("=== stream with pause ===")
    stream = [
        ("NME", 0.0), ("IS", 0.5), ("LOKY", 1.0),
        ("HELO", 3.0), ("WRD", 3.5),
    ]
    for raw in stream_sentences(stream, pause_threshold_s=1.5):
        corrected, _, _, _ = process_letter_chunks(raw.split(), polish=False)
        print(f"raw:       {raw}")
        print(f"corrected: {corrected}")