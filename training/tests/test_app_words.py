"""The phone bridge (app/) with the word engine: real Engine + Session + FastAPI app, trained letter and word bundles from the session
fixture.  Needs the folders side by side (app/ next to training/); skipped when app/ or fastapi is missing."""
import asyncio
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
app_settings = pytest.importorskip("app.settings", reason="app/ is not next to training/")

from app.engine import Engine  # noqa: E402
from app.session import Session  # noqa: E402
from app.settings import Settings  # noqa: E402
from matos.words import synth as S  # noqa: E402

TPLS = S.make_templates(8)                       # the words of the fixture's word bundle
quiet = lambda *a, **k: None                     # noqa: E731


def settings(env, **kw):
    base = dict(bundle=env.letters, word_bundle=env.words, mode="auto", text_mode="sentences", polish=False, https=False)
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="module")
def engine(word_env):
    return Engine(settings(word_env), log=quiet)


@pytest.fixture
def nowords(monkeypatch, tmp_path):
    """an environment in which models/words does not exist, whatever is on the developer's disk"""
    from matos.paths import LAYOUT

    monkeypatch.setattr(LAYOUT, "models", tmp_path)


def stream(items, signer=0, seed=0, pose_every=1, left_share=0.0):
    sg = S.make_signers(6, np.random.default_rng(3), left_handed_share=left_share)[signer]
    return S.stream(items, sg, np.random.default_rng(seed), TPLS, 30.0, pose_every=pose_every)


def play(engine, frames, extra=None, strip_pose=False):
    """a whole session through the same Session object the websocket and the HTTP fallback use -> every message the phone would get"""
    async def go():
        ses = Session(engine, quiet)
        out = list(await ses.handle({"type": "start"}))
        for m in (extra or []):
            out += await ses.handle(m)
        for f in frames:
            msg = {"type": "f", "t": f["t"], "w": f["w"], "h": f["h"], "hands": f["hands"]}
            if f["pose"] is not None and not strip_pose:
                msg["pose"] = f["pose"]
            out += await ses.handle(msg)
        out += await ses.handle({"type": "stop"})
        return out
    return asyncio.run(go())


def of(out, kind):
    return [m for m in out if m["type"] == kind]


# -- the engine -----------------------------------------------------------------------------------------------------------------------
def test_the_engine_reports_both_levels(engine):
    info = engine.info()
    assert info["words"] is True and info["mode"] == "auto" and info["word_bundle"] == "wb" and info["vocab"] == 8
    assert info["pose_ms"] > 0 and info["word_engine"] == "onnxruntime"
    d = engine.new_decoder()
    assert type(d).__name__ == "HybridDecoder" and d.mode == "auto"


def test_without_a_word_bundle_everything_is_exactly_as_before(word_env, nowords):
    eng = Engine(settings(word_env, word_bundle=None), log=quiet)
    assert eng.words is None and eng.info()["words"] is False and "word_bundle" not in eng.info()
    assert type(eng.new_decoder()).__name__ == "LetterDecoder"
    out = play(eng, stream([("rest", .5), ("spell", "HI"), ("rest", 2.4)]))
    assert out[0]["type"] == "ready" and out[0]["words"] is False
    assert not of(out, "sign") and all("seg" not in m for m in of(out, "state"))
    assert [m["text"] for m in of(out, "letter")], "the letter engine still works"


def test_mode_letters_never_touches_the_word_engine(word_env):
    eng = Engine(settings(word_env, mode="letters"), log=quiet)
    assert eng.words is None and type(eng.new_decoder()).__name__ == "LetterDecoder"


def test_a_missing_word_bundle_is_fatal_only_when_it_was_asked_for(word_env, nowords):
    assert Engine(settings(word_env, word_bundle=None, mode="auto"), log=quiet).words is None
    with pytest.raises(SystemExit, match="no word bundle"):
        Engine(settings(word_env, word_bundle=None, mode="words"), log=quiet)
    with pytest.raises(SystemExit, match="not found"):
        Engine(settings(word_env, word_bundle="nope"), log=quiet)


def test_a_broken_word_bundle_degrades_to_letters_unless_asked_for(word_env, tmp_path, monkeypatch):
    from matos.paths import LAYOUT

    monkeypatch.setattr(LAYOUT, "models", tmp_path)
    shutil.copytree(word_env.words, tmp_path / "words" / "broken")
    (tmp_path / "words" / "broken" / "word_model.onnx").write_bytes(b"this is not an onnx file")
    logs = []
    eng = Engine(settings(word_env, word_bundle=None), log=logs.append)                      # found by itself, then fails to load
    assert eng.words is None and any("letters only" in l for l in logs)
    with pytest.raises(SystemExit):
        Engine(settings(word_env, word_bundle="broken"), log=quiet)


def test_the_newest_word_bundle_is_picked_by_default(word_env, tmp_path, monkeypatch):
    import os
    import time
    from matos.paths import LAYOUT

    monkeypatch.setattr(LAYOUT, "models", tmp_path)
    for name, age in (("old", 100), ("newest", 0), ("middle", 50)):
        shutil.copytree(word_env.words, tmp_path / "words" / name)
        t = time.time() - age
        os.utime(tmp_path / "words" / name / "config.json", (t, t))
    assert Engine(settings(word_env, word_bundle=None), log=quiet).word_bundle.name == "newest"


# -- sessions ----------------------------------------------------------------------------------------------------------------------------
def test_a_mixed_stream_comes_out_as_signs_and_spelling_in_order(engine):
    ok, n = 0, 12
    for k in range(n):
        w1, w2 = k % 8, (k * 3 + 1) % 8
        items = [("rest", .6), ("word", w1), ("rest", .6), ("spell", "HI"), ("rest", 1.2), ("word", w2), ("rest", 2.6)]
        out = play(engine, stream(items, signer=k % 6, seed=k, pose_every=2 if k % 3 == 0 else 1, left_share=0.34))
        sent = of(out, "sentence")
        ok += len(sent) == 1 and sent[0]["raw"] == f"W{w1:02d} HI W{w2:02d}"
    assert ok >= 9, f"{ok}/{n} exact"


def test_sign_events_carry_what_the_ui_needs(engine):
    out = play(engine, stream([("rest", .6), ("word", 2), ("rest", 2.6)]))
    s = of(out, "sign")
    assert len(s) == 1 and s[0]["text"] == "W02" and s[0]["label"] == "w02" and 0.5 < s[0]["conf"] <= 1
    assert s[0]["alts"] and all(len(a) == 2 for a in s[0]["alts"])
    states = of(out, "state")
    assert states and all(set(st) >= {"hand", "top", "conf", "progress", "word", "seg", "mode"} for st in states)
    assert any(st["seg"] for st in states), "the UI must be told when a sign is in progress"
    t = of(out, "text")
    assert t and t[-1]["kind"] == "sentence" and t[-1]["text"] == "W02"


def test_text_mode_words_appends_each_sign_at_once(word_env):
    eng = Engine(settings(word_env, text_mode="words"), log=quiet)
    out = play(eng, stream([("rest", .6), ("word", 4), ("rest", .8), ("word", 6), ("rest", .4)]))
    texts = [(m["text"], m["kind"]) for m in of(out, "text")]
    assert texts == [("W04", "sign"), ("W06", "sign")], texts          # each sign at once, and the sentence is NOT appended a second time


def test_no_pose_means_the_letter_engine_only(engine):
    out = play(engine, stream([("rest", .5), ("spell", "HI"), ("rest", 2.4)]), strip_pose=True)
    assert [m["text"] for m in of(out, "letter")] == ["H", "I"] and not of(out, "sign")
    out = play(engine, stream([("rest", .6), ("word", 2), ("rest", 2.4)]), strip_pose=True)
    assert not of(out, "sign")                                          # a sign needs the body


def test_the_client_can_switch_modes_live(engine):
    words = stream([("rest", .6), ("word", 5), ("rest", 2.4)])
    spelled = stream([("rest", .5), ("spell", "HI"), ("rest", 2.4)])
    out = play(engine, words, extra=[{"type": "mode", "mode": "letters"}])
    assert {"type": "mode", "mode": "letters"} in out and not of(out, "sign")
    assert all(st["mode"] == "letters" for st in of(out, "state"))
    out = play(engine, spelled, extra=[{"type": "mode", "mode": "words"}])
    assert {"type": "mode", "mode": "words"} in out and not of(out, "letter") and not of(out, "word")
    out = play(engine, words, extra=[{"type": "mode", "mode": "words"}])
    assert [m["text"] for m in of(out, "sign")] == ["W05"]
    out = play(engine, words, extra=[{"type": "mode", "mode": "nonsense"}])
    assert [m for m in out if m["type"] == "mode"] == [{"type": "mode", "mode": "auto"}] and of(out, "sign")


def test_the_chosen_mode_survives_a_reset(engine):
    async def go():
        ses = Session(engine, quiet)
        await ses.handle({"type": "start"})
        await ses.handle({"type": "mode", "mode": "words"})
        await ses.handle({"type": "reset"})
        return ses.dec.mode
    assert asyncio.run(go()) == "words"


def test_garbage_input_does_not_crash_the_session(engine):
    async def go():
        ses = Session(engine, quiet)
        await ses.handle({"type": "start"})
        good = stream([("rest", .3), ("word", 1)])[40]
        junk = [
            {"type": "f", "t": 0.1, "w": 640, "h": 480, "hands": "nope", "pose": 12},
            {"type": "f", "t": 0.2, "w": 640, "h": 480, "hands": [[1, 2, 3]], "pose": [0.1] * 74},
            {"type": "f", "t": 0.3, "w": 640, "h": 480, "hands": [[float("nan")] * 63], "pose": [float("nan")] * 75},
            {"type": "f", "t": 0.4, "w": 0, "h": 0, "hands": [], "pose": None},
            {"type": "f", "t": "x", "w": 640, "h": 480, "hands": []},
            {"type": "f", "t": 0.1, "w": 640, "h": 480, "hands": [], "pose": [1e9] * 75},          # time goes backwards: ignored
            {"type": "mode"}, {"type": "mode", "mode": 5}, {"type": "unknown"}, {},
            {"type": "f", "t": 0.5, **{k: good[k] for k in ("w", "h", "hands", "pose")}},
        ]
        for m in junk:
            out = await ses.handle(m)
            assert isinstance(out, list)
        return True
    assert asyncio.run(go())


def test_signs_and_spelling_are_corrected_together(engine):
    text, info = engine.correct_tokens([("sign", "THANK YOU"), ("spell", "LOKY")])
    assert text.startswith("THANK YOU ") and len(text.split()) == 3
    text, _ = engine.correct_tokens([("sign", "ME"), ("sign", "HUNGRY")])
    assert text == "ME HUNGRY"
    only_spelling = engine.correct_tokens([("spell", "HI"), ("spell", "BEN")])
    assert only_spelling == engine.correct(["HI", "BEN"]), "pure fingerspelling takes the old path unchanged"
    text, _ = engine.correct_tokens([("spell", "HI"), ("sign", "ME"), ("spell", "BEN"), ("spell", "TOM")])
    assert text.split()[1] == "ME" and len(text.split()) == 4


# -- the real transports ------------------------------------------------------------------------------------------------------------------
def test_the_websocket_carries_signs_and_the_new_messages(engine, word_env):
    from fastapi.testclient import TestClient
    from app.server import create_app

    client = TestClient(create_app(engine, settings(word_env), log=quiet))
    health = client.get("/api/health").json()
    assert health["ok"] and health["words"] is True and health["word_bundle"] == "wb"
    frames = stream([("rest", .6), ("word", 3), ("rest", 2.6)])
    got = []
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_text("this is not json")                                  # ignored, the connection stays up
        ws.send_text(json.dumps({"type": "mode", "mode": "auto"}))
        for f in frames:
            ws.send_text(json.dumps({"type": "f", "t": f["t"], "w": f["w"], "h": f["h"], "hands": f["hands"],
                                     **({"pose": f["pose"]} if f["pose"] else {})}))
        ws.send_text(json.dumps({"type": "stop"}))
        while True:
            m = json.loads(ws.receive_text())
            got.append(m)
            if m["type"] == "text" and m["kind"] == "sentence":
                break
    assert got[0]["type"] == "ready" and got[0]["words"] is True and got[0]["mode"] == "auto"
    assert [m["text"] for m in got if m["type"] == "sign"] == ["W03"]
    assert [m["raw"] for m in got if m["type"] == "sentence"] == ["W03"]


def test_the_http_fallback_works_with_pose(engine, word_env):
    from fastapi.testclient import TestClient
    from app.server import create_app

    client = TestClient(create_app(engine, settings(word_env), log=quiet))
    frames = stream([("rest", .6), ("word", 3), ("rest", 2.6)])
    msgs = [{"type": "start"}] + [{"type": "f", "t": f["t"], "w": f["w"], "h": f["h"], "hands": f["hands"],
                                   **({"pose": f["pose"]} if f["pose"] else {})} for f in frames] + [{"type": "stop"}]
    out = []
    for i in range(0, len(msgs), 40):
        r = client.post("/api/msg", json={"sid": "t1", "msgs": msgs[i:i + 40]})
        assert r.status_code == 200
        out += r.json()["msgs"] if isinstance(r.json(), dict) and "msgs" in r.json() else r.json()
    assert [m["text"] for m in out if m["type"] == "sign"] == ["W03"]
