"""matos.words.holistic: torso normalisation, left/right hand assignment, gap filling, augmentation, features."""
import numpy as np
import pytest

from matos.words import holistic as H

NEUTRAL = {0: (0, -.62), 2: (.12, -.72), 5: (-.12, -.72), 7: (.28, -.66), 8: (-.28, -.66), 9: (.1, -.45), 10: (-.1, -.45),
           11: (.5, 0), 12: (-.5, 0), 13: (.62, 1), 14: (-.62, 1), 15: (.7, 2), 16: (-.7, 2)}


def person(T, w, h, origin=(0.5, 0.45), sw_frac=0.30):
    """a waving right hand; sw_frac = shoulder width as a share of the frame WIDTH, y uses the same pixel scale (aspect-correct)"""
    pose = np.full((T, 25, 3), np.nan, np.float32)
    hands = np.full((T, 2, 21, 3), np.nan, np.float32)
    px = sw_frac * w
    for t in range(T):
        for j, (x, y) in NEUTRAL.items():
            pose[t, j] = (origin[0] + x * px / w, origin[1] + y * px / h, 1.0)
        c = np.array([-0.5 + 0.2 * np.sin(t / 3), -0.3])
        for k in range(21):
            off = np.array([np.cos(k * 0.7), np.sin(k * 0.7)]) * 0.05 * (1 + k / 21)
            hands[t, 1, k] = (origin[0] + (c[0] + off[0]) * px / w, origin[1] + (c[1] + off[1]) * px / h, 0.01 * k * sw_frac / 0.3)
        pose[t, 16, :2] = hands[t, 1, 0, :2]
    return pose, hands


def test_feature_sets_have_the_documented_widths():
    assert H.WORD_FEATURE_SETS == {"pose_hands": 210, "pose_hands_geo": 298}
    rng = np.random.default_rng(0)
    for fs, n in H.WORD_FEATURE_SETS.items():
        x = H.frame_features(rng.normal(size=(3, 5, 13, 2)), rng.normal(size=(3, 5, 2, 21, 3)), rng.random((3, 5, 2)) < .7, fs)
        assert x.shape == (3, 5, n) and x.dtype == np.float32 and np.isfinite(x).all()
    with pytest.raises(ValueError):
        H.frame_features(np.zeros((1, 2, 13, 2)), np.zeros((1, 2, 2, 21, 3)), np.zeros((1, 2, 2), bool), "nope")


def test_torso_normalisation_ignores_where_you_stand_and_how_far():
    a = H.torso_normalise(*person(30, 640, 480, (0.5, 0.45), 0.30), 640, 480)
    b = H.torso_normalise(*person(30, 640, 480, (0.35, 0.60), 0.18), 640, 480)       # elsewhere in the frame, farther from the camera
    assert np.abs(a.pose - b.pose).max() < 1e-4 and np.abs(a.hands - b.hands).max() < 1e-3
    assert np.allclose(a.pose[0, H.I_L_SH], (0.5, 0), atol=1e-5) and np.allclose(a.pose[0, H.I_R_SH], (-0.5, 0), atol=1e-5)
    assert a.hmask[:, 1].all() and not a.hmask[:, 0].any()


def test_the_aspect_ratio_matters():
    p, h = person(10, 640, 480)
    right = H.torso_normalise(p, h, 640, 480)
    wrong = H.torso_normalise(p, h, 480, 640)
    assert np.abs(right.pose - wrong.pose).max() > 0.05


def test_no_shoulders_no_torso_and_tiny_shoulders_are_refused():
    p, h = person(10, 640, 480)
    p[:, [11, 12]] = np.nan
    assert H.torso_normalise(p, h, 640, 480) is None
    p, h = person(10, 640, 480, sw_frac=0.005)
    assert H.torso_normalise(p, h, 640, 480) is None
    p, h = person(10, 640, 480)
    p[:, 11, 2] = 0.05                                          # a visibility MediaPipe does not believe
    assert H.torso_normalise(p, h, 640, 480) is None


def test_frames_without_shoulders_reuse_the_nearest_torso_reference():
    p, h = person(20, 640, 480)
    want = H.torso_normalise(p, h, 640, 480)
    p[5:9, [11, 12]] = np.nan                                   # pose model dropped the shoulders for 4 frames (or: pose rate < video rate)
    got = H.torso_normalise(p, h, 640, 480)
    assert np.abs(got.hands - want.hands).max() < 1e-4


def test_mirror_twice_is_the_identity_and_swaps_the_hands():
    p, h = person(30, 640, 480)
    n = H.torso_normalise(p, h, 640, 480)
    ap = H.AugParams(p_mirror=1.0, rot_deg=0, scale=0, iso_scale=0, shift=0, jitter_pose=0, jitter_hand=0, z_scale=0, crop=0, warp=0,
                     p_hand_loss=0, p_frame_loss=0)
    rng = np.random.default_rng(0)
    b = (n.pose[None], n.hands[None], n.hmask[None])
    m1 = H.augment_batch(*b, rng, 30, ap)
    m2 = H.augment_batch(*m1, rng, 30, ap)
    assert np.allclose(m2[0], b[0], atol=1e-5) and np.allclose(m2[1], b[1], atol=1e-5) and (m2[2] == b[2]).all()
    assert m1[2][0, :, 0].all() and not m1[2][0, :, 1].any()                           # right hand -> left slot
    assert np.sign(m1[1][0, 0, 0, 0, 0]) == -np.sign(b[1][0, 0, 1, 0, 0])             # and on the other side of the body
    assert np.allclose(m1[0][0, :, H.I_L_SH, 0], 0.5, atol=1e-5)                       # the shoulders keep their sides


def test_augmentation_output_is_finite_masked_and_deterministic_per_seed():
    p, h = person(30, 640, 480)
    n = H.torso_normalise(p, h, 640, 480)
    big = tuple(np.repeat(x, 32, 0) for x in (n.pose[None], n.hands[None], n.hmask[None]))
    a = H.augment_batch(*big, np.random.default_rng(1), 32, H.AugParams())
    b = H.augment_batch(*big, np.random.default_rng(1), 32, H.AugParams())
    assert a[0].shape == (32, 32, 13, 2) and a[1].shape == (32, 32, 2, 21, 3) and a[2].shape == (32, 32, 2)
    assert all(np.isfinite(x).all() for x in a[:2])
    assert (a[1][~a[2]] == 0).all()                                                    # a missing hand is exact zeros
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    c = H.augment_batch(*big, np.random.default_rng(2), 32, H.AugParams())
    assert not np.array_equal(a[0], c[0])


def test_an_absent_hand_contributes_exact_zeros_to_every_feature():
    x = H.frame_features(np.zeros((1, 5, 13, 2), np.float32), np.zeros((1, 5, 2, 21, 3), np.float32), np.zeros((1, 5, 2), bool),
                         "pose_hands_geo")
    hand_cols = np.r_[26:26 + 121, 26 + 121:26 + 242]
    assert (x[0][:, hand_cols] == 0).all()


def test_features_do_not_depend_on_the_batch_they_are_computed_in():
    rng = np.random.default_rng(3)
    pose, hands = rng.normal(0, 1, (6, 8, 13, 2)), rng.normal(0, .5, (6, 8, 2, 21, 3))
    m = rng.random((6, 8, 2)) < .6
    hands = np.where(m[..., None, None], hands, 0)
    full = H.frame_features(pose, hands, m)
    for i in (0, 3, 5):
        assert np.allclose(H.frame_features(pose[i:i + 1], hands[i:i + 1], m[i:i + 1])[0], full[i], atol=1e-6)


# -- which detected hand is the signer's left / right ------------------------------------------------------------------------------------
def _hand(cx, cy, w=640, h=480, s=0.03):
    k = np.arange(21)
    a = np.stack([np.cos(k * 0.7), np.sin(k * 0.7)], 1) * s * (1 + k / 21)[:, None]
    return np.concatenate([np.array([cx, cy]) + a * np.array([1, w / h]), 0.01 * k[:, None]], 1).astype(np.float32)


def _body():
    pose = np.full((25, 3), np.nan, np.float32)
    for j, (x, y) in NEUTRAL.items():
        pose[j] = (0.5 + x * 0.3, 0.45 + y * 0.3 * 640 / 480, 1.0)
    return pose


def _slots(hands, pose):
    s = H.assign_slots(hands, pose, 640, 480)
    return [i for i in range(2) if np.isfinite(s[i]).all()]


def test_assign_slots_follows_the_pose_wrist():
    left_up, right_up = _hand(0.5 + 0.15, 0.35), _hand(0.5 - 0.15, 0.35)         # the signer's LEFT is on the image's right
    pose = _body(); pose[15, :2] = left_up[0, :2]
    assert _slots([left_up], pose) == [0]
    pose = _body(); pose[16, :2] = right_up[0, :2]
    assert _slots([right_up], pose) == [1]
    pose[15, :2] = left_up[0, :2]
    for order in ([left_up, right_up], [right_up, left_up]):
        s = H.assign_slots(order, pose, 640, 480)
        assert np.allclose(s[0], left_up) and np.allclose(s[1], right_up)


def test_assign_slots_survives_a_pose_glitch_and_a_missing_pose():
    left_up, right_up = _hand(0.65, 0.35), _hand(0.35, 0.35)
    assert _slots([left_up], _body()) == [0]                                       # left wrist of the pose stuck at the hip
    assert _slots([right_up], _body()) == [1]
    assert _slots([left_up], None) == [0] and _slots([right_up], None) == [1]      # no pose: the side of the image decides
    pose = _body(); pose[15, 2] = 0.05                                             # an invisible pose wrist is not evidence
    assert _slots([left_up], pose) == [0]
    assert _slots([], _body()) == []
    assert H.assign_slots([np.full((21, 3), np.nan)], None, 640, 480).shape == (2, 21, 3)
    assert len(_slots([_hand(.3, .3), _hand(.5, .3), _hand(.7, .3)], None)) == 2   # three hands in view: two slots at most


# -- gaps and resampling ---------------------------------------------------------------------------------------------------------------
def test_short_gaps_are_interpolated_long_ones_are_not():
    n = H.torso_normalise(*person(30, 640, 480), 640, 480)
    hh, mm = n.hands.copy(), n.hmask.copy()
    mm[10:13, 1] = False; hh[10:13, 1] = 0
    f, fm = H.fill_hand_gaps(hh, mm)
    assert fm[10:13, 1].all() and np.abs(f[10:13, 1] - n.hands[10:13, 1]).max() < 0.04   # linear interpolation of a curved path
    mm[10:20, 1] = False; hh[10:20, 1] = 0
    _, fm = H.fill_hand_gaps(hh, mm)
    assert (~fm[10:20, 1]).all()
    mm[:] = False; mm[20:, 1] = True
    _, fm = H.fill_hand_gaps(hh, mm)
    assert not fm[:20, 1].any(), "a gap at the START has no earlier sighting to interpolate from"


def test_resampling_never_blends_a_hand_with_nothing():
    T = 10
    pose = np.zeros((1, T, 13, 2), np.float32)
    hands = np.ones((1, T, 2, 21, 3), np.float32)
    m = np.zeros((1, T, 2), bool)
    m[0, :5, 1] = True                                         # the hand is there for the first 5 frames, then gone
    hands = np.where(m[..., None, None], hands, 0)
    _, h, mo = H.gather_time(pose, hands, m, np.linspace(0, T - 1, 40, dtype=np.float32)[None])
    assert set(np.unique(h[0, mo[0, :, 1], 1])) == {1.0}       # present frames are exactly 1.0: never 0.5 from blending with zeros
    assert (h[0, ~mo[0, :, 1], 1] == 0).all()


def test_trim_keeps_only_the_frames_with_a_hand():
    n = H.torso_normalise(*person(30, 640, 480), 640, 480)
    n.hmask[:6], n.hmask[24:] = False, False
    t = H.trim_active(n, margin=2)
    assert len(t) == 18 + 4
    n.hmask[:] = False
    assert H.trim_active(n) is None


def test_the_wire_formats_are_validated():
    assert H.parse_pose(None) is None and H.parse_pose([1, 2, 3]) is None and H.parse_pose("x") is None
    ok = np.zeros(75); ok[3 * 11:3 * 11 + 2] = (0.6, 0.5); ok[3 * 12:3 * 12 + 2] = (0.4, 0.5)
    assert H.parse_pose(ok.tolist()).shape == (25, 3)
    bad = ok.copy(); bad[3 * 11] = np.nan
    assert H.parse_pose(bad.tolist()) is None                  # the shoulders are the reference: without them the pose is useless
    assert H.hand_array([0.0] * 62) is None and H.hand_array([float("nan")] * 63) is None and H.hand_array([0.5] * 63).shape == (21, 3)


def test_the_torso_tracker_smooths_and_expires():
    tt = H.TorsoTracker()
    pose = _body()
    assert not tt.ready(0.0)
    tt.update(0.0, pose, 640, 480)
    assert tt.ready(0.5) and not tt.ready(3.0)
    xy = np.array([0.5 * 640, 0.45 * 480], np.float32)
    assert np.allclose(tt.norm(xy), (0, 0), atol=1e-3)
    tt.update(0.1, None, 640, 480)                             # a frame without pose changes nothing
    assert tt.ready(0.2)
