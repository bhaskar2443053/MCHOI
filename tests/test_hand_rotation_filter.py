import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mchoi.hand.rotation_filter import OneEuroRotationFilter, filter_mano_rotations


def z_rotation(degrees):
    return Rotation.from_euler('z', degrees, degrees=True).as_matrix()


def test_rotation_filter_uses_short_path_across_angle_wrap():
    filt = OneEuroRotationFilter(beta=0)
    a = filt(z_rotation(179), 0)
    b = filt(z_rotation(-179), 1 / 30)
    step = Rotation.from_matrix(b @ a.T).as_rotvec()[2]
    assert 0 < np.rad2deg(step) < 1
    np.testing.assert_allclose(b @ b.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(b) - 1) < 1e-12


def test_jitter_reduction_and_adaptive_fast_motion_response():
    filt = OneEuroRotationFilter(min_cutoff=1, beta=.5)
    angles = 3 * np.sin(np.arange(180) / 30 * 2 * np.pi * 6)
    result = [Rotation.from_matrix(filt(z_rotation(a), n / 30)).as_euler('xyz', degrees=True)[2] for n, a in enumerate(angles)]
    assert np.std(result[30:]) < .3 * np.std(angles[30:])
    slow = OneEuroRotationFilter(beta=0)
    adaptive = OneEuroRotationFilter(beta=3)
    slow(np.eye(3), 0)
    adaptive(np.eye(3), 0)
    target = z_rotation(60)
    slow_error = Rotation.from_matrix(target @ slow(target, 1 / 30).T).magnitude()
    adaptive_error = Rotation.from_matrix(target @ adaptive(target, 1 / 30).T).magnitude()
    assert adaptive_error < .5 * slow_error


def test_each_joint_has_its_own_speed_and_filter_is_causal():
    filt = OneEuroRotationFilter(beta=3)
    filt(np.stack([np.eye(3), np.eye(3)]), 0)
    result = filt(np.stack([z_rotation(60), np.eye(3)]), 1 / 30)
    np.testing.assert_allclose(result[1], np.eye(3), atol=1e-12)
    assert filt.last_cutoff[0] > filt.last_cutoff[1] == 1
    # A sequence extension cannot change already emitted values.
    snapshot = result.copy()
    filt(np.stack([z_rotation(90), z_rotation(30)]), 2 / 30)
    np.testing.assert_array_equal(result, snapshot)


def test_gap_resets_and_invalid_rotations_are_rejected():
    filt = OneEuroRotationFilter()
    filt(np.eye(3), 0)
    np.testing.assert_allclose(filt(z_rotation(80), .5), z_rotation(80), atol=1e-12)
    with pytest.raises(ValueError):
        filt(np.eye(3), .5)
    with pytest.raises(ValueError):
        filt(np.eye(3) * 2, .6)
    with pytest.raises(ValueError):
        OneEuroRotationFilter(min_cutoff=0)


def test_mano_sequence_does_not_bridge_invalid_frames_or_change_wrist_by_default():
    raw = np.tile(np.eye(3), (5, 16, 1, 1))
    raw[1:, 1:] = z_rotation(60)
    raw[:, 0] = z_rotation(25)
    source = dict(mano_rotations=raw, fit_frame_rows=np.arange(5), frame_index=np.arange(5),
                  timestamp_ns=np.arange(5, dtype=np.int64) * 33_333_333,
                  valid=np.array([True, True, False, True, True]), is_right=np.ones(5))
    filtered, resets = filter_mano_rotations(source)
    assert resets == [0, 3]
    np.testing.assert_allclose(filtered[:, 0], raw[:, 0], atol=1e-7)
    np.testing.assert_allclose(filtered[3], raw[3], atol=1e-7)
    np.testing.assert_allclose(filtered[2], raw[2], atol=1e-7)
    assert Rotation.from_matrix(filtered[1, 1]).magnitude() < Rotation.from_matrix(raw[1, 1]).magnitude()
    prefix, _ = filter_mano_rotations({k: v[:2] for k, v in source.items()})
    np.testing.assert_array_equal(prefix, filtered[:2])
    source['is_right'][4] = 0
    source['mano_rotations'][4, 1:] = z_rotation(-30)
    filtered, resets = filter_mano_rotations(source)
    assert resets == [0, 3, 4]
    np.testing.assert_allclose(filtered[4], source['mano_rotations'][4], atol=1e-7)
