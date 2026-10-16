"""
T5 ONNX runner (matos/text/t5_polish.py) against tiny hand-made encoder/decoder graphs - no real model needed.
    python tests/test_t5_onnx.py   |   pytest tests

The fake graphs have the same input/output names as the optimum export (encoder_model.onnx: input_ids, attention_mask
-> last_hidden_state;  decoder_model.onnx: input_ids, encoder_attention_mask, encoder_hidden_states -> logits) and a
next-token table that is written out below, so the expected outputs can be worked out by hand.
Skipped when `onnx` is not installed (only this test needs it; the app itself does not).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import onnx
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
    from onnx import TensorProto as TP, helper as H, numpy_helper as NH
    HAVE = True
except ImportError:
    HAVE = False

import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(not HAVE, reason="needs onnx + onnxruntime + tokenizers")

# ids: 0 <pad>=decoder start, 1 </s>, 2 <unk>, 3 my, 4 name, 5 is, 6 loky, 7 x, 8.. the words of the prompt prefix
WORDS = ["<pad>", "</s>", "<unk>", "my", "name", "is", "loky", "x", "translate", "asl", "gloss", "to", "english", ":"]
V = len(WORDS)
START, EOS, MY, NAME, IS, LOKY, X = 0, 1, 3, 4, 5, 6, 7


def table(rows, default_eos=0.0):
    """rows: {token: {next_token: prob}} -> [V,V] log-probs; the unspecified rest of a row shares the missing mass."""
    t = np.full((V, V), np.log(1.0 / V), np.float32)
    for tok, probs in rows.items():
        left = 1.0 - sum(probs.values())
        rest = [i for i in range(V) if i not in probs]
        row = np.full(V, np.log(left / len(rest)) if left > 1e-9 else -30.0, np.float32)
        for i, p in probs.items():
            row[i] = np.log(p)
        t[tok] = row
    return t


def write_model(d: Path, tab, enc_scale=0.0, int32_ids=False, with_past=False):
    d.mkdir(parents=True, exist_ok=True)
    idt = TP.INT32 if int32_ids else TP.INT64
    opset = [H.make_opsetid("", 13)]

    def save(graph, name):
        m = H.make_model(graph, opset_imports=opset)
        m.ir_version = 8
        onnx.checker.check_model(m)
        onnx.save(m, str(d / name))

    # encoder: one-hot embedding of every source token
    save(H.make_graph(
        [H.make_node("Gather", ["E", "input_ids"], ["last_hidden_state"], axis=0)], "enc",
        [H.make_tensor_value_info("input_ids", idt, ["b", "s"]), H.make_tensor_value_info("attention_mask", idt, ["b", "s"])],
        [H.make_tensor_value_info("last_hidden_state", TP.FLOAT, ["b", "s", V])],
        [NH.from_array(np.eye(V, dtype=np.float32), "E")]), "encoder_model.onnx")

    # decoder: logits = TABLE[current token] + enc_scale * mean(encoder one-hots)
    inputs = [H.make_tensor_value_info("input_ids", idt, ["b", "l"]),
              H.make_tensor_value_info("encoder_attention_mask", idt, ["b", "s"]),
              H.make_tensor_value_info("encoder_hidden_states", TP.FLOAT, ["b", "s", V])]
    if with_past:
        inputs.append(H.make_tensor_value_info("past_key_values.0.decoder.key", TP.FLOAT, ["b", 1, "p", 4]))
    save(H.make_graph(
        [H.make_node("Gather", ["T", "input_ids"], ["bigram"], axis=0),
         H.make_node("ReduceMean", ["encoder_hidden_states"], ["mean"], axes=[1], keepdims=1),
         H.make_node("Mul", ["mean", "scale"], ["enc_term"]),
         H.make_node("Add", ["bigram", "enc_term"], ["logits"])], "dec", inputs,
        [H.make_tensor_value_info("logits", TP.FLOAT, ["b", "l", V])],
        [NH.from_array(tab.astype(np.float32), "T"), NH.from_array(np.float32(enc_scale), "scale")]), "decoder_model.onnx")

    # tokenizer.json (lower-casing word-level tokenizer that appends </s>, like T5's) + the two config files
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<pad>", "</s>"])
    tok.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", EOS)])
    tok.save(str(d / "tokenizer.json"))
    (d / "config.json").write_text(json.dumps({"decoder_start_token_id": 0, "eos_token_id": 1, "pad_token_id": 0}))
    (d / "generation_config.json").write_text(json.dumps({"decoder_start_token_id": 0, "eos_token_id": 1, "pad_token_id": 0}))
    return d


# start -> MY .5 / NAME .4;   MY -> EOS .34 / IS .33 / LOKY .33;   NAME -> IS .97;   IS -> EOS .97
#   greedy:  MY EOS               = "my"         logp -1.77 , /2 tokens = -0.89
#   beam 2:  NAME IS EOS          = "name is"    logp -0.98 , /3 tokens = -0.33   <- better per token, greedy can't see it
BEAM_TABLE = table({START: {MY: .5, NAME: .4}, MY: {EOS: .34, IS: .33, LOKY: .33}, NAME: {IS: .97}, IS: {EOS: .97},
                    LOKY: {EOS: .97}, X: {EOS: .97}})


def load(d):
    from matos.text.t5_polish import T5Gloss
    return T5Gloss(d)


def test_greedy_follows_the_argmax_chain():
    with tempfile.TemporaryDirectory() as d:
        m = load(write_model(Path(d), BEAM_TABLE))
        assert m.generate("ME NAME LOKY", num_beams=1) == "my"


def test_beam_search_finds_the_better_sentence_greedy_misses():
    with tempfile.TemporaryDirectory() as d:
        m = load(write_model(Path(d), BEAM_TABLE))
        assert m.generate("ME NAME LOKY", num_beams=2) == "name is"
        assert m.generate("ME NAME LOKY", num_beams=4) == "name is"


def test_translate_adds_the_training_prompt():
    from matos.text.t5_polish import PROMPT_PREFIX
    with tempfile.TemporaryDirectory() as d:
        m = load(write_model(Path(d), BEAM_TABLE))
        seen = []
        real = m.tok.encode
        m.tok.encode = lambda text, *a, **k: (seen.append(text), real(text, *a, **k))[1]
        m.translate("ME NAME LOKY")
        assert seen == [PROMPT_PREFIX + "ME NAME LOKY"]
        assert PROMPT_PREFIX == "translate ASL gloss to English: "


def test_encoder_output_reaches_the_decoder():
    # bigram part is flat except "after the first token, stop"; the first token then depends only on the source
    rows = {t: {EOS: 1.0} for t in range(1, V)}                  # everything except EOS gets logit -30
    tab = table(rows)
    tab[START] = np.log(1.0 / V)                                 # flat: the source alone decides the first token
    with tempfile.TemporaryDirectory() as d:
        m = load(write_model(Path(d), tab, enc_scale=50.0))
        assert m.translate("LOKY LOKY LOKY", num_beams=1) == "loky"
        assert m.translate("NAME NAME NAME", num_beams=1) == "name"
        assert m.translate("LOKY LOKY LOKY", num_beams=2) == "loky"
        assert m.translate("NAME NAME NAME", num_beams=2) == "name"


def test_stops_at_max_new_tokens_when_eos_never_comes():
    tab = table({START: {MY: .99}, MY: {NAME: .99}, NAME: {MY: .99}})
    with tempfile.TemporaryDirectory() as d:
        m = load(write_model(Path(d), tab))
        for beams in (1, 2):
            assert m.generate("x", num_beams=beams, max_new_tokens=5) == "my name my name my"


def test_int32_graph_inputs_are_cast():
    with tempfile.TemporaryDirectory() as d:
        m = load(write_model(Path(d), BEAM_TABLE, int32_ids=True))
        assert m.generate("ME NAME LOKY", num_beams=2) == "name is"


def test_decoder_with_past_is_refused_clearly():
    with tempfile.TemporaryDirectory() as d:
        m = load(write_model(Path(d), BEAM_TABLE, with_past=True))
        with pytest.raises(RuntimeError, match="past_key_values"):
            m.generate("ME NAME LOKY")


def test_missing_files_are_named():
    from matos.text.t5_polish import T5Gloss
    with tempfile.TemporaryDirectory() as d:
        write_model(Path(d), BEAM_TABLE)
        (Path(d) / "decoder_model.onnx").unlink()
        with pytest.raises(FileNotFoundError, match="decoder_model.onnx"):
            T5Gloss(d)


def test_grammar_polish_end_to_end():
    """grammar_polish with a model whose answer is known: accepted, unchanged, rejected, too short."""
    import matos.text.t5_polish as tp
    with tempfile.TemporaryDirectory() as d:
        write_model(Path(d), BEAM_TABLE)
        t5 = load(d)
        tp._t5 = t5
        try:
            canned = {"ME NAME LOKY": "My name is Loky.", "MY NAME IS": "My name is.", "NAME LOKY": "Lucky, my name"}
            t5.translate = lambda gloss, **kw: canned[gloss]                 # the graph itself is covered above
            assert tp.grammar_polish("ME NAME LOKY") == ("MY NAME IS LOKY", "t5", {"t5_output": "My name is Loky."})
            out, src, info = tp.grammar_polish("MY NAME IS")
            assert (out, src) == ("MY NAME IS", "unchanged")
            out, src, info = tp.grammar_polish("NAME LOKY")                  # LOKY is a name -> must survive; here it did, but
            assert src in ("fallback", "t5")                                  # coverage / order decide; just never crash
            assert tp.grammar_polish("HELLO") == ("HELLO", "skipped", {"reason": "too short"})
            strict_out, strict_src, why = tp.grammar_polish("ME NAME LOKY", strict=True)
            assert (strict_out, strict_src) == ("MY NAME IS LOKY", "t5")
        finally:
            tp._t5 = None


def test_grammar_polish_survives_a_broken_model():
    import matos.text.t5_polish as tp
    with tempfile.TemporaryDirectory() as d:
        t5 = load(write_model(Path(d), BEAM_TABLE))
        tp._t5 = t5

        def boom(*a, **k):
            raise RuntimeError("onnx exploded")
        t5.translate = boom
        try:
            out, src, info = tp.grammar_polish("ME NAME LOKY")
            assert (out, src) == ("ME NAME LOKY", "fallback") and "onnx exploded" in info["reason"]
        finally:
            tp._t5 = None


def test_pipeline_polish_uses_t5_and_rank_only_warns():
    import warnings

    import matos.text.t5_polish as tp
    from matos.text.pipeline import process_letter_chunks
    with tempfile.TemporaryDirectory() as d:
        t5 = load(write_model(Path(d), BEAM_TABLE))
        t5.translate = lambda gloss, **kw: "My name is Loky."
        tp._t5 = t5
        try:
            corrected, final, combos, info = process_letter_chunks(["ME", "NAME", "LOKY"], polish=True)
            assert corrected == "ME NAME LOKY" and final == "MY NAME IS LOKY"
            assert info["polish_source"] == "t5" and info["rank_source"] == "score"
            c2, f2, _, _ = process_letter_chunks(["ME", "NAME", "LOKY"])      # no polish -> T5 is not touched
            assert (c2, f2) == ("ME NAME LOKY", None)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                process_letter_chunks(["ME", "NAME", "LOKY"], rank=True)
            assert any("rank" in str(x.message) for x in w)
        finally:
            tp._t5 = None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
