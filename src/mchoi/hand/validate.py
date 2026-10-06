"""Validate fusion geometry, exact frame accounting, temporal metrics and rendered MP4s."""
import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np

from mchoi.multiview import Rig, CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.output / 'fit_summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('Fit is incomplete')
    cache = Path(summary['cache'])
    metadata = json.loads((cache / 'cache_metadata.json').read_text())
    calibration = json.loads((args.output / 'calibration.json').read_text())
    rig = Rig(calibration)
    data = dict(np.load(cache / 'observations.npz', allow_pickle=False))
    fused = dict(np.load(args.output / 'fused_hand.npz', allow_pickle=False))
    tri = dict(np.load(args.output / 'triangulation.npz', allow_pickle=False))
    count = len(data['frame_index'])
    assert count == summary['frames'] == metadata['frames']
    assert np.all(np.diff(data['frame_index']) == 1)
    assert np.all(np.diff(data['timestamp_ns']) > 0)
    assert np.array_equal(fused['timestamp_ns'], data['timestamp_ns'])
    assert np.array_equal(fused['frame_index'], data['frame_index'])
    valid = fused['valid']
    assert valid.sum() == summary['fitted_frames']
    assert np.isfinite(fused['vertices'][valid]).all()
    assert np.isfinite(fused['joints'][valid]).all()
    assert np.isnan(fused['vertices'][~valid]).all()
    assert fused['vertices'].shape == (count, 778, 3)
    assert fused['joints'].shape == (count, 21, 3)
    assert np.all(fused['used_views'][valid].sum(axis=1) >= 2)
    quality_report = None
    if summary.get('fusion_mode') == 'visibility':
        assert np.isfinite(tri['weights']).all() and (tri['weights'] >= 0).all()
        assert np.isfinite(tri['raw_weights']).all() and (tri['raw_weights'] >= 0).all()
        assert np.all(tri['weights'] <= 2 * tri['raw_weights'] + 1e-12)
        assert np.isfinite(tri['reliability']).all()
        assert np.all((tri['reliability'] >= 0) & (tri['reliability'] <= 1))
        supported = np.isfinite(tri['points']).all(axis=-1)
        assert np.all(tri['reliability'][~supported] == 0)
        assert np.all(tri['inliers'].sum(axis=1)[supported] >= 2)
        np.testing.assert_array_equal(fused['joint_reliability'], tri['reliability'])
        accepted_weights = tri['weights'] * tri['inliers']
        finger_weights = accepted_weights[:, :, 1:].reshape(count, 4, 5, 4).sum(axis=-1)
        total_weights = finger_weights.sum(axis=1)
        dominance = finger_weights.max(axis=1) / np.maximum(total_weights, 1e-12)
        winners = finger_weights.argmax(axis=1)
        finger_names = ('thumb', 'index', 'middle', 'ring', 'little')
        quality_report = dict(mode='visibility', stale_confidence_capped=True, unsupported_reliability_zero=True,
            median_largest_camera_weight_share=float(np.median(dominance[(total_weights > 0) & valid[:, None]])),
            highest_weight_camera_counts={finger: {camera: int(np.sum(valid & (total_weights[:, f] > 0) & (winners[:, f] == c)))
                for c, camera in enumerate(CAMERAS)} for f, finger in enumerate(finger_names)})
    rows = fused['fit_frame_rows']
    accepted_rows = valid[rows]
    np.testing.assert_allclose(fused['joints'][rows[accepted_rows], 0], fused['wrist_translation'][accepted_rows], atol=1e-5)
    rot = fused['mano_rotations']
    assert np.max(np.abs(np.linalg.det(rot) - 1)) < 1e-4
    assert np.max(np.abs(rot @ rot.swapaxes(-1, -2) - np.eye(3))) < 1e-4
    pose_filter_report = None
    if summary.get('one_euro_pose'):
        from mchoi.hand.rotation_filter import filter_mano_rotations
        source = dict(np.load(Path(summary['source_fit']) / 'fused_hand.npz', allow_pickle=False))
        config = summary['one_euro_pose']
        parameters = {k: config[k] for k in ('min_cutoff', 'beta', 'd_cutoff', 'wrist_min_cutoff', 'wrist_beta', 'reset_gap')}
        expected, resets = filter_mano_rotations(source, **parameters)
        np.testing.assert_allclose(rot, expected, atol=1e-7)
        np.testing.assert_array_equal(fused['one_euro_reset_rows'], resets)
        np.testing.assert_array_equal(fused['unfiltered_mano_rotations'], source['mano_rotations'])
        for key in ('frame_index', 'timestamp_ns', 'fit_frame_rows', 'wrist_translation', 'shape', 'selected'):
            np.testing.assert_array_equal(fused[key], source[key])
        assert np.all(~valid | source['valid']), 'Filter invented an unsupported pose'
        if not config['wrist_min_cutoff']:
            np.testing.assert_array_equal(rot[:, 0], source['mano_rotations'][:, 0])
        pose_filter_report = dict(recomputed_filter_matches=True, source_shape_and_wrist_positions_preserved=True,
                                  source_gaps_reset=True, no_invented_poses=True,
                                  additionally_rejected_frames=int((source['valid'] & ~valid).sum()))
    np.testing.assert_allclose(data['expanded_box'][:, 2:] - data['expanded_box'][:, :2],
                               1.10 * (data['filtered_box'][:, 2:] - data['filtered_box'][:, :2]), atol=.001)
    ledger = list(csv.DictReader((Path(metadata['take']) / 'frame_timestamps.csv').open(newline='')))
    timing = metadata['source']['timing']['cameras']
    verified = {}
    for cam, name in enumerate(CAMERAS):
        native_name = 'zed' if cam < 2 else 'realsense_' + calibration['cameras'][name]['device']['serial']
        stream = ('left', 'right')[cam] if cam < 2 else 'color'
        native = {int(row['frame_index']): int(row['pc_timestamp_ns']) for row in ledger
                  if row['camera'] == native_name and row['stream'] == stream}
        present = data['sync'][:, cam, 0] >= 0
        if not np.any(present):
            assert cam >= 2, 'ZED reference views must be present'
            assert np.all(data['sync'][:, cam] == -1)
            assert np.all(fused['selected'][:, cam] == -1)
            assert np.all(data['offsets'][:, cam, 0] == data['offsets'][:, cam, 1])
            assert not fused['used_views'][:, cam].any()
            verified[name] = 0
            continue
        if cam >= 2:
            assert timing[name].get('validated', False), 'Unvalidated timing contributed to fusion'
        indices = data['sync'][present, cam, 0]
        assert np.all(np.diff(indices) > 0), 'A source frame was reused'
        for i, timestamp in data['sync'][present, cam, :2]:
            assert native[int(i)] == int(timestamp)
        lag = 0 if cam < 2 else timing[name]['rs_minus_zed_lag_ns']
        delta = data['sync'][present, cam, 1] - lag - data['timestamp_ns'][present]
        assert np.array_equal(delta, data['sync'][present, cam, 2])
        assert np.max(np.abs(delta)) <= 20e6
        verified[name] = int(present.sum())
        selected = fused['selected'][:, cam]
        use = selected >= 0
        assert np.all(selected[use] >= data['offsets'][use, cam, 0])
        assert np.all(selected[use] < data['offsets'][use, cam, 1])
    projected = np.full((count, 21, 2), np.nan)
    mono = np.full_like(projected, np.nan)
    for row in range(count):
        if valid[row]:
            projected[row] = rig.project(fused['joints'][row], 0, distorted=True)
        obs = fused['selected'][row, 0]
        if obs >= 0:
            mono[row] = data['joints_2d'][obs]
    supported_in_left = valid & np.any(np.all((projected >= [0, 0]) & (projected < [1920, 1080]), axis=2), axis=1)
    # Measure high-frequency residual about timestamp-weighted linear motion.
    # This includes genuine acceleration and is explicitly a stability proxy, not pose accuracy.
    common = valid & (fused['selected'][:, 0] >= 0)
    times = (data['timestamp_ns'] - data['timestamp_ns'][0]) / 1e9
    center = np.flatnonzero(common[1:-1] & common[:-2] & common[2:]) + 1
    center = center[(times[center + 1] - times[center - 1]) < .12]
    if len(center):
        tracks = data['track_id'][fused['selected'][center, 0]]
        same_track = (tracks == data['track_id'][fused['selected'][center - 1, 0]]) & (tracks == data['track_id'][fused['selected'][center + 1, 0]])
        center = center[same_track]
    alpha = ((times[center] - times[center - 1]) / (times[center + 1] - times[center - 1]))[:, None, None]
    metrics = {}
    for name, points in [('monocular', mono), ('multiview', projected)]:
        interpolated = points[center - 1] * (1 - alpha) + points[center + 1] * alpha
        residual = points[center] - interpolated
        relative = points - points[:, :1]
        relative_residual = relative[center] - relative[center - 1] * (1 - alpha) - relative[center + 1] * alpha
        metrics[name] = dict(median_wrist_midpoint_residual_px=float(np.median(np.linalg.norm(residual[:, 0], axis=-1))) if len(center) else None,
                             median_finger_midpoint_residual_px=float(np.median(np.linalg.norm(relative_residual[:, 1:], axis=-1))) if len(center) else None)
    videos = {}
    video_specs = [('wilor_multiview_left.mp4', (1920, 1080)), ('comparison_left.mp4', (1920, 540))]
    if (args.output / 'comparison_hand.mp4').exists():
        video_specs.append(('comparison_hand.mp4', (1280, 720)))
    for name, size in video_specs:
        decoded = 0
        with av.open(str(args.output / name)) as container:
            for frame in container.decode(video=0):
                assert (frame.width, frame.height) == size
                assert decoded < count
                actual_ns = int(frame.pts * frame.time_base * 1_000_000_000)
                expected_ns = int(data['timestamp_ns'][decoded] - data['timestamp_ns'][0])
                assert abs(actual_ns - expected_ns) <= 1000, (name, decoded, actual_ns, expected_ns)
                decoded += 1
        assert decoded == count, (name, decoded, count)
        videos[name] = decoded
    report = dict(status='passed', frames=count, fitted_frames=int(valid.sum()),
                  fitted_frames_projecting_into_left=int(supported_in_left.sum()),
                  monocular_fallback_frames=int((~valid & (fused['selected'][:, 0] >= 0)).sum()),
                  source_timestamps_verified=verified, no_source_frame_reuse=True,
                  excluded_cameras=metadata['source'].get('excluded_cameras', {}),
                  camera_convention='native unrectified with calibrated distortion', valid_mano_rotations=True,
                  stability_comparison_frames=len(center), stability_proxy=metrics,
                  visibility_weight_checks=quality_report,
                  one_euro_pose_checks=pose_filter_report,
                  metric_limitation='Midpoint residual includes real acceleration; no ground-truth accuracy measurement.',
                  video_timestamps_verified_within_ns=1000,
                  video_frames_decoded=videos)
    (args.output / 'validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
