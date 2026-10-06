import numpy as np
from mchoi.multiview import CAMERAS, Rig, prepare_observations
from mchoi.fusion_quality import StableViewWeights, visibility_weights, triangulation_reliability


def synthetic_rig():
    cameras = {}
    for name, center in zip(CAMERAS, [[0, 0, 0], [.065, 0, 0], [-.3, .1, 0], [.2, -.2, 0]]):
        T = np.eye(4)
        T[:3, 3] = -np.asarray(center)
        cameras[name] = dict(camera_matrix=[[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
                             distortion_coefficients=[0, 0, 0, 0, 0], T_camera_from_reference=T)
    return Rig(dict(reference_camera='zed_left', units='metres', cameras=cameras))


def test_triangulation_recovers_metric_points_and_rejects_bad_camera():
    rig = synthetic_rig()
    points = np.array([[.03, .02, .7], [-.04, -.02, .8]])
    uv = np.array([rig.project(points, c) for c in range(4)])
    uv[3] += [150, -100]
    recovered, mask, errors = rig.triangulate(uv, np.ones((4, 2)), threshold=5)
    np.testing.assert_allclose(recovered, points, atol=1e-8)
    assert np.all(mask[:3]) and not np.any(mask[3])
    assert np.max(errors[:3]) < 1e-6


def test_single_view_cannot_claim_triangulation():
    rig = synthetic_rig()
    uv = np.array([rig.project([[0, 0, .7]], c) for c in range(4)])
    weights = np.array([[1.], [0.], [0.], [0.]])
    recovered, mask, _ = rig.triangulate(uv, weights)
    assert np.isnan(recovered).all()
    assert not mask.any()


def test_distortion_roundtrip_and_epipolar_relation():
    rig = synthetic_rig()
    rig.D[0] = np.array([.1, -.03, .002, -.001, 0.])
    point = np.array([[.05, -.05, .6], [.02, .1, .8]])
    raw_a = rig.project(point, 0, distorted=True)
    raw_b = rig.project(point, 2, distorted=True)
    np.testing.assert_allclose(rig.undistort(raw_a, 0), rig.project(point, 0), atol=1e-5)
    assert np.max(rig.epipolar_distance(raw_a, raw_b, 0, 2)) < 1e-5
    assert np.median(rig.epipolar_distance(raw_a, raw_b + [0, 100], 0, 2)) > 30


def test_points_behind_cameras_are_rejected():
    rig = synthetic_rig()
    uv = np.array([rig.project([[.02, -.02, -.7]], c) for c in range(4)])
    points, mask, _ = rig.triangulate(uv, np.ones((4, 1)))
    assert np.isnan(points).all() and not mask.any()


def test_clarity_prefers_each_fingers_clear_camera_not_one_camera_for_whole_hand():
    keypoints = np.ones((4, 21, 3))
    keypoints[..., :2] = [500, 500]
    for finger in range(5):
        ids = np.arange(1 + finger * 4, 5 + finger * 4)
        keypoints[:, ids, 0] += np.arange(4) * 25
        keypoints[:, ids, 1] += finger * 30
    keypoints[0, 5:9, 2] = .95  # clear index in camera 0
    keypoints[1:, 5:9, 2] = .55
    keypoints[1, 13:17, 2] = .95  # clear ring in camera 1
    keypoints[[0, 2, 3], 13:17, 2] = .55
    weights, _ = visibility_weights(keypoints, np.ones(4), np.zeros(4), np.tile([1920, 1080], (4, 1)))
    assert weights[0, 8] > 5 * weights[1, 8]
    assert weights[1, 16] > 5 * weights[0, 16]
    # Insufficient confidence, out-of-image, and nonfinite points cannot contribute.
    keypoints[0, 8, 2] = .2
    keypoints[1, 16, 0] = 2000
    keypoints[2, 4, 1] = np.nan
    weights, _ = visibility_weights(keypoints, np.ones(4), np.zeros(4), np.tile([1920, 1080], (4, 1)))
    assert weights[0, 8] == weights[1, 16] == weights[2, 4] == 0
    assert np.isfinite(weights).all()


def test_confidence_filter_smooths_switches_but_drops_missing_and_resets_tracks_and_gaps():
    smoother = StableViewWeights()
    tracks = np.arange(4)
    a = np.ones((4, 21)) * .2
    smoother.update(a, 0., tracks)
    b = np.ones_like(a) * .9
    result = smoother.update(b, 1 / 30, tracks)
    assert np.all((result > .2) & (result < .9))
    b[0] = 0
    b[1] = .1
    b[2] = .01
    tracks[1] += 10
    result = smoother.update(b, 2 / 30, tracks)
    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_allclose(result[1], .1)
    assert np.all(result[2] <= .02), 'An obscured finger retained too much stale confidence'
    np.testing.assert_allclose(smoother.update(a, 1., tracks), a)


def test_hysteresis_never_retains_a_gross_geometric_outlier():
    rig = synthetic_rig()
    points = np.array([[.03, .02, .7]])
    uv = np.array([rig.project(points, c) for c in range(4)])
    uv[3] += [180, -110]
    recovered, mask, _ = rig.triangulate(uv, np.ones((4, 1)), threshold=20, previous_mask=np.ones((4, 1), bool))
    np.testing.assert_allclose(recovered, points, atol=1e-8)
    assert not mask[3, 0]


def test_uncertain_narrow_baseline_depth_receives_less_weight_than_clear_four_view_depth():
    rig = synthetic_rig()
    points = np.array([[0., 0., 1.5]])
    weights = np.ones((4, 1)) * .9
    errors = np.zeros((4, 1))
    all_views = np.ones((4, 1), bool)
    stereo_only = np.array([[True], [True], [False], [False]])
    supported = triangulation_reliability(rig, points, weights, all_views, errors, 20)
    narrow = triangulation_reliability(rig, points, weights, stereo_only, errors, 20)
    noisy = triangulation_reliability(rig, points, weights, all_views, errors + 18, 20)
    assert supported[0] > 3 * narrow[0]
    assert supported[0] > 2 * noisy[0]
    invalid = triangulation_reliability(rig, np.full((1, 3), np.nan), weights, all_views, errors, 20)
    assert invalid[0] == 0


def test_nonfinite_missing_camera_does_not_poison_other_views():
    rig = synthetic_rig()
    points = np.array([[.03, .02, .7]])
    uv = np.array([rig.project(points, c) for c in range(4)])
    uv[3] = np.nan
    recovered, mask, _ = rig.triangulate(uv, np.ones((4, 1)), previous_mask=np.ones((4, 1), bool))
    np.testing.assert_allclose(recovered, points, atol=1e-8)
    assert mask[:3].all() and not mask[3].any()


def test_preparation_keeps_fit_targets_finite_with_an_invalid_detector_coordinate():
    rig = synthetic_rig()
    points = np.tile([.03, .02, .7], (21, 1))
    uv = np.array([rig.project(points, c) for c in range(4)])
    keypoints = np.concatenate([uv, np.ones((4, 21, 1))], axis=-1)
    keypoints[3, 8, 0] = np.nan
    data = dict(frame_index=np.array([0]), timestamp_ns=np.array([123], np.int64),
                offsets=np.arange(5)[np.array([[0, 1], [1, 2], [2, 3], [3, 4]])][None],
                detector_keypoints=keypoints, joints_2d=uv, is_right=np.ones(4),
                confidence=np.ones(4), track_id=np.arange(4), sync=np.zeros((1, 4, 3)))
    _, pixels, weights, recovered, masks, quality = prepare_observations(data, rig, 1, return_quality=True)
    assert np.isfinite(pixels).all() and np.isfinite(weights).all()
    assert weights[0, 3, 8] == 0 and not masks[0, 3, 8]
    np.testing.assert_allclose(recovered[0], points, atol=1e-8)
    assert (quality['reliability'] > 0).all()
