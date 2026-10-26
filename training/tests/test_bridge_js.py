"""app/static/bridge.js under Node with a mocked browser (DOM, WebSocket, MediaPipe).  This is the one part of the word engine that cannot be
exercised against a real phone here, so what it SENDS and how it REACTS is pinned down message by message.  Skipped without node / app/."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
BRIDGE = Path(__file__).resolve().parents[2] / "app" / "static" / "bridge.js"
JS = Path(__file__).resolve().parent / "js"
pytestmark = [pytest.mark.skipif(NODE is None, reason="node is not installed"),
              pytest.mark.skipif(not BRIDGE.is_file(), reason="app/static/bridge.js is not next to training/")]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("bridge_js")
    src = BRIDGE.read_text(encoding="utf-8")
    assert "import { HandTracker } from '/js/handtracker.js';" in src
    (d / "bridge.mjs").write_text(src.replace("'/js/handtracker.js'", "'./handtracker_stub.mjs'"), encoding="utf-8")
    for f in ("handtracker_stub.mjs", "bridge_harness.mjs"):
        shutil.copy(JS / f, d / f)
    return d


def run(work, scenario, query=""):
    p = subprocess.run([NODE, "bridge_harness.mjs"], cwd=work, capture_output=True, text=True, timeout=60,
                       env={"SCENARIO": scenario, "TEST_QUERY": query, "PATH": "/usr/bin:/bin"})
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, f"no result from the harness:\nSTDOUT {p.stdout[-1500:]}\nSTDERR {p.stderr[-1500:]}"
    return json.loads(line[-1][7:])


def test_letters_only_the_bridge_behaves_exactly_as_before(work):
    r = run(work, "letters_only")
    assert r["n_frames"] == 4 and not r["any_pose"]
    assert r["keys"] == ["h", "hands", "t", "type", "w"]                       # no new fields on a letters-only server
    assert r["hand_len"] == 63 and r["first_hand"][:3] == [0.5, 0.4, 0.0]
    assert r["first_hand"][3:6] == [0.51235, 0.4, 0.01]                         # rounded to 5 decimals like before
    assert r["sent_types"][:3] == ["hello", "start", "f"] and r["mode_hidden"] is True
    assert r["pose_state"] == {"loading": False, "failed": False, "lm": False}   # the pose model is never even loaded


def test_when_the_pose_model_cannot_load_nothing_breaks(work):
    r = run(work, "words_pose_unavailable")
    assert r["mode_hidden"] is False and r["n_frames"] == 3 and not r["any_pose"]
    assert r["pose_state"]["failed"] is True and r["pose_state"]["lm"] is False
    assert any("staying on the letter engine" in w for w in r["warnings"])


def test_the_pose_flag_off_never_loads_it(work):
    r = run(work, "pose_off_flag", query="?pose=off")
    assert r["pose_state"] == {"loading": False, "failed": False, "lm": False, "off": True} and r["warned"] == 0
    assert r["mode_hidden"] is False, "the mode buttons still appear: the word engine itself is on the server"


def test_pose_points_ride_along_in_the_frame_message(work):
    r = run(work, "pose_ok")
    f = r["first"]
    assert f["len"] == 75 and f["x0"] == 0.3 and f["y0"] == 0.2
    assert f["v0"] == 1, "a landmark without visibility counts as visible"
    assert f["x1"] == 0.31 and f["v1"] == 0.988, "visibility is rounded to 3 decimals"
    assert r["input"] == {"w": 320, "h": 240}                                  # 320 px wide, the video's aspect
    assert r["burst_poses"] == 0, "pose is throttled: the frames right after the first carry none"
    assert r["after_wait"] is True and r["ts_increasing"] is True
    assert r["calls"] == 3 and r["hands_still_there"]


def test_a_slow_pose_model_slows_itself_down(work):
    r = run(work, "pose_slow_throttles")
    assert r["every_ms"] >= 110, r                                              # 40 ms per run -> at most ~1/3 of the time budget


def test_no_person_or_too_few_points_means_no_pose_field(work):
    r = run(work, "pose_none_or_short")
    assert r["n_frames"] == 2 and not r["any_pose"]


def test_a_model_that_keeps_failing_is_dropped_not_retried_forever(work):
    r = run(work, "pose_throws")
    assert r["calls"] <= 6, "retrying a failing model on every frame would burn the phone's CPU"
    assert r["failed"] and r["lm_dropped"] and r["gave_up_warning"]
    assert r["n_frames"] == 40 and not r["any_pose"]                            # the hand frames keep flowing regardless


def test_server_messages_drive_the_hud_and_the_transcript(work):
    r = run(work, "server_messages")
    assert r["mode_visible"] and r["on_after_ready"] == ["words"]
    assert r["sign"] == {"text": "THANK YOU", "kind": "ok", "pending": "THANK YOU "}
    assert r["miss"] == {"text": "MOM?", "kind": "miss"}
    assert r["pending_mixed"] == "THANK YOU LOKY "
    assert r["pending_after_text"] == "" and r["appended"] == ["THANK YOU"]
    assert r["on_after_mode"] == ["letters"]
    assert r["seg_class"] is True and r["seg_cleared"] is True and r["letter_shown"] == "-"
    assert r["pending_after_sentence"] == "" and r["survived_unknown"]


def test_the_mode_buttons_send_the_mode_message(work):
    r = run(work, "mode_click")
    assert r["sent_modes"] == [{"type": "mode", "mode": "words"}]


def test_the_session_still_starts_when_the_camera_is_ready(work):
    r = run(work, "session_lifecycle")
    assert r["start_sent"] >= 1
