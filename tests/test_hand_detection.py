import numpy as np
import pytest

from mchoi.hand.detection import DetectionTracker, OneEuroFilter, expand_box


def detection(x=100, y=100, right=1):
    return [x, y, x + 80, y + 100, 0.9, right]


def test_expansion_is_ten_percent_not_twenty():
    np.testing.assert_allclose(expand_box([10, 20, 110, 220]), [5, 10, 115, 230])
    # Off-image margins must remain available for zero padding.
    np.testing.assert_allclose(expand_box([0, 0, 100, 100]), [-5, -5, 105, 105])


def test_one_euro_numeric_reference_and_elapsed_time():
    f = OneEuroFilter(min_cutoff=1, beta=0, d_cutoff=1)
    assert f([0], 0)[0] == 0
    alpha = (2 * np.pi * 0.1) / (1 + 2 * np.pi * 0.1)
    assert f([1], 0.1)[0] == pytest.approx(alpha)
    slow = OneEuroFilter(min_cutoff=1, beta=0)
    slow([0], 0)
    assert slow([1], 0.2)[0] > alpha


def test_filter_reduces_stationary_jitter_and_beta_reduces_motion_lag():
    rng = np.random.default_rng(19)
    raw = rng.normal(0, 0.01, 300)
    f = OneEuroFilter(min_cutoff=1, beta=0)
    filtered = [f([x], i / 30)[0] for i, x in enumerate(raw)]
    assert np.std(filtered[30:]) < np.std(raw[30:]) * 0.6
    stationary = OneEuroFilter(beta=0)
    adaptive = OneEuroFilter(beta=8)
    for i in range(30):
        stationary([0], i / 30)
        adaptive([0], i / 30)
    assert adaptive([0.5], 1)[0] > stationary([0.5], 1)[0]


def test_independent_tracks_survive_detection_order_changes():
    tracker = DetectionTracker()
    first = tracker.update([detection(100, right=0), detection(500, right=1)], 0, (1000, 1000))
    second = tracker.update([detection(505, right=1), detection(105, right=0)], 1 / 30, (1000, 1000))
    assert [x['track_id'] for x in second] == [first[1]['track_id'], first[0]['track_id']]
    assert 500 < second[0]['filtered_box'][0] < 505
    assert 100 < second[1]['filtered_box'][0] < 105


def test_same_handedness_hands_do_not_share_filter_state():
    tracker = DetectionTracker()
    first = tracker.update([detection(100), detection(500)], 0, (1000, 1000))
    second = tracker.update([detection(490), detection(110)], 1 / 30, (1000, 1000))
    assert [x['track_id'] for x in second] == [first[1]['track_id'], first[0]['track_id']]


def test_one_frame_handedness_flicker_keeps_track_and_side():
    tracker = DetectionTracker()
    first = tracker.update([detection(right=0)], 0, (1000, 1000))[0]
    next_hand = tracker.update([detection(right=1)], 1 / 30, (1000, 1000))[0]
    assert next_hand['track_id'] == first['track_id']
    assert next_hand['is_right'] == 0
    assert next_hand['detected_is_right'] == 1


def test_established_hand_keeps_side_through_short_ambiguous_sequence():
    tracker = DetectionTracker()
    for frame in range(60):
        hand = tracker.update([detection(right=1)], frame / 30, (1000, 1000))[0]
    identity = hand['track_id']
    for frame in range(60, 75):
        hand = tracker.update([detection(right=0)], frame / 30, (1000, 1000))[0]
        assert hand['track_id'] == identity
        assert hand['is_right'] == 1


def test_initial_wrong_handedness_can_be_corrected_by_later_evidence():
    tracker = DetectionTracker()
    tracker.update([detection(right=0)], 0, (1000, 1000))
    for frame in range(1, 8):
        hand = tracker.update([detection(right=1)], frame / 30, (1000, 1000))[0]
    assert hand['is_right'] == 1


def test_misses_emit_no_hands_and_gap_resets_filter():
    tracker = DetectionTracker()
    first = tracker.update([detection()], 0, (1000, 1000))[0]
    assert tracker.update([], 0.1, (1000, 1000)) == []
    again = tracker.update([detection(110)], 0.3, (1000, 1000))[0]
    assert again['track_id'] == first['track_id']
    assert again['filter_reset']
    np.testing.assert_allclose(again['filtered_box'], again['raw_box'])
    tracker.update([], 1, (1000, 1000))
    new = tracker.update([detection(110)], 1.1, (1000, 1000))[0]
    assert new['track_id'] != first['track_id']


def test_normalization_is_resolution_independent():
    a, b = DetectionTracker(), DetectionTracker()
    for i, x in enumerate([100, 105, 115, 115]):
        det = np.array([detection(x)])
        big = det.copy()
        big[:, :4] *= 2
        ra = a.update(det, i / 30, (1000, 500))[0]
        rb = b.update(big, i / 30, (2000, 1000))[0]
        np.testing.assert_allclose(ra['filtered_box'] * 2, rb['filtered_box'])
        np.testing.assert_allclose(ra['expanded_box'] * 2, rb['expanded_box'])


def test_invalid_detections_are_discarded():
    tracker = DetectionTracker()
    assert tracker.update([[100, 100, 90, 90, .8, 1], [0, 0, np.nan, 20, .8, 0]], 0, (1000, 1000)) == []


@pytest.mark.parametrize('timestamp', [0, -1, np.nan])
def test_nonmonotonic_or_invalid_timestamps_fail(timestamp):
    tracker = DetectionTracker()
    tracker.update([], 0, (1000, 1000))
    with pytest.raises(ValueError):
        tracker.update([], timestamp, (1000, 1000))
