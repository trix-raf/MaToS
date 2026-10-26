"""matos.words.segmenter: when does a sign start and end (wrist motion in torso units, time based, fps independent)."""
import numpy as np

from matos.words import holistic as H
from matos.words import synth as S
from matos.words.segmenter import SegParams, SignSegmenter

TPLS = S.make_templates(12)
SIGNERS = S.make_signers(4, np.random.default_rng(3), left_handed_share=0.0)


def run(rows, fps, params=None):
    seg, out = SignSegmenter(params), []
    for i, w in enumerate(rows):
        r = seg.update(i / fps, w)
        if r:
            out.append(r)
    return out


def wrists_of(hands, hmask, tail_s=0.0, fps=30.0):
    rows = [[hands[i, s, 0, :2] if hmask[i, s] else None for s in (0, 1)] for i in range(len(hands))]
    return rows + [[None, None]] * int(tail_s * fps)                # the stream goes on after the sign: nothing in view


def still(y, n, x=-0.6):
    return [[None, np.array([x, y], np.float32)]] * n


def test_every_isolated_sign_is_one_segment_at_any_frame_rate():
    rng = np.random.default_rng(0)
    cover, n = [], 0
    for fps in (15.0, 24.0, 30.0, 60.0):
        for tpl in TPLS:
            for sg in SIGNERS:
                pose, hands, hm = S.word_sequence(tpl, sg, rng, fps)
                segs = [s for s in run(wrists_of(hands, hm, 1.0, fps), fps) if s.ok]
                n += 1
                assert len(segs) == 1, (fps, tpl["name"], sg.name, segs)
                s = segs[0]
                raised = np.where(hands[:, 1, 0, 1] < 1.9)[0] / fps if not sg.left_handed else np.array([])
                raised = raised[hm[:len(hands), 1][(raised * fps).astype(int)]] if len(raised) else raised
                if len(raised) > 3:
                    cover.append(((raised >= s.t0) & (raised <= s.t1 + 0.1)).mean())
                assert s.reason == "drop"
    cover = np.array(cover)
    assert n == 192 and cover.mean() > 0.95 and np.percentile(cover, 5) > 0.85, (cover.mean(), np.percentile(cover, 5), cover.min())


def test_two_signs_with_a_pause_are_two_segments():
    sg = SIGNERS[0]
    rng = np.random.default_rng(1)
    frames = S.stream([("rest", .5), ("word", 2), ("rest", .7), ("word", 5), ("rest", .8)], sg, rng, TPLS)
    tt, rows = H.TorsoTracker(), []
    for f in frames:
        pose = H.parse_pose(f["pose"]); tt.update(f["t"], pose, sg.w, sg.h)
        sl = H.assign_slots([np.array(x, np.float32).reshape(21, 3) for x in f["hands"]], pose, sg.w, sg.h)
        rows.append([tt.norm(sl[s, 0, :2] * np.array([sg.w, sg.h])) if np.isfinite(sl[s]).all() else None for s in (0, 1)])
    segs = [s for s in run(rows, 30.0) if s.ok]
    assert len(segs) == 2 and segs[0].t1 < segs[1].t0


def test_fingerspelling_makes_one_segment_for_the_hand_coming_up_and_no_more():
    sg = SIGNERS[0]
    rng = np.random.default_rng(2)
    frames = S.stream([("rest", .5), ("spell", "HELLO"), ("rest", .6)], sg, rng, TPLS)
    tt, rows = H.TorsoTracker(), []
    for f in frames:
        pose = H.parse_pose(f["pose"]); tt.update(f["t"], pose, sg.w, sg.h)
        sl = H.assign_slots([np.array(x, np.float32).reshape(21, 3) for x in f["hands"]], pose, sg.w, sg.h)
        rows.append([tt.norm(sl[s, 0, :2] * np.array([sg.w, sg.h])) if np.isfinite(sl[s]).all() else None for s in (0, 1)])
    segs = run(rows, 30.0)
    assert len(segs) <= 2, segs                                    # the hand rising (+ at most the hand lowering); never one per letter
    assert all(s.t1 - s.t0 < 1.5 for s in segs)


def test_a_resting_hand_in_view_never_triggers():
    assert run(still(2.4, 300), 30.0) == []
    jitter = np.random.default_rng(0)
    rows = [[None, np.array([-0.6, 1.4], np.float32) + jitter.normal(0, .004, 2).astype(np.float32)] for _ in range(300)]
    assert run(rows, 30.0) == [], "landmark jitter on a raised, still hand must not look like motion"


def test_a_quick_twitch_is_returned_as_not_ok_so_the_caller_can_release_what_it_held():
    rows = still(1.0, 10) + [[None, np.array([-0.6 + 0.35 * k, 1.0], np.float32)] for k in range(3)] + still(1.0, 40, x=0.45) + [[None, None]] * 30
    segs = run(rows, 30.0)
    assert segs and all(not s.ok and s.reason == "short" for s in segs), segs


def test_a_raised_hand_that_stops_closes_with_still_and_needs_new_motion_to_restart():
    p = SegParams()
    move = [[None, np.array([-0.5 + 0.12 * np.sin(i / 2.0) * 3, 0.5 + 0.12 * np.cos(i / 2.0) * 3], np.float32)] for i in range(24)]
    rows = move + still(0.5, 40) + move + [[None, None]] * 20
    segs = run(rows, 30.0, p)
    assert [s.reason for s in segs[:2]] == ["still", "drop"] and all(s.ok for s in segs[:2]), segs
    a = segs[0]
    assert a.t1 < 24 / 30 + 0.2                                     # closed when the motion stopped, not when the hold ended


def test_a_very_long_gesture_is_cut_at_max_s():
    p = SegParams(max_s=1.5)
    rows = [[None, np.array([-0.5 + 0.4 * np.sin(i / 3.0), 0.5 + 0.4 * np.cos(i / 3.0)], np.float32)] for i in range(150)]
    segs = run(rows, 30.0, p)
    assert segs and segs[0].reason == "max" and 1.4 < segs[0].t1 - segs[0].t0 < 1.7, segs


def test_a_hand_lost_for_a_moment_inside_a_sign_does_not_end_it():
    rng = np.random.default_rng(3)
    pose, hands, hm = S.word_sequence(TPLS[4], SIGNERS[1], rng, 30.0)
    rows = wrists_of(hands, hm, 1.0)
    mid = int(np.where(hm[:, 1])[0].mean())
    for i in range(mid, mid + 4):                                   # four frames of tracking loss
        rows[i] = [None, None]
    assert len([s for s in run(rows, 30.0) if s.ok]) == 1


def test_a_slow_float_is_not_a_sign():
    rows = [[None, np.array([-0.6 + 0.0012 * i, 0.6], np.float32)] for i in range(400)]         # ~0.04 shoulder widths / s
    assert run(rows, 30.0) == []
    seg = SignSegmenter()
    for i, r in enumerate(rows):
        seg.update(i / 30, r)
    assert not seg.active and seg.speed < SegParams().v_off


def test_reset_forgets_everything():
    seg = SignSegmenter()
    for i in range(12):                                             # a clear sweep: 3 shoulder widths / second at chest height
        seg.update(i / 30, [None, np.array([-0.8 + 0.1 * i, 0.5], np.float32)])
    assert seg.active
    seg.reset()
    assert not seg.active and seg.t_start is None and seg.sm == [None, None]
