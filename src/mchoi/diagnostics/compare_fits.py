"""Compare two fitted runs on identical observations and common valid frames."""
import argparse
import json
from pathlib import Path

import numpy as np

from mchoi.multiview import Rig


def compare(reference, candidate):
    summaries = [json.loads((p / 'fit_summary.json').read_text()) for p in (reference, candidate)]
    fits = [dict(np.load(p / 'fused_hand.npz', allow_pickle=False)) for p in (reference, candidate)]
    triangulations = [dict(np.load(p / 'triangulation.npz', allow_pickle=False)) for p in (reference, candidate)]
    calibrations = [json.loads((p / 'calibration.json').read_text()) for p in (reference, candidate)]
    assert calibrations[0] == calibrations[1], 'Calibration differs'
    assert summaries[0]['cache'] == summaries[1]['cache'], 'Observation caches differ'
    for key in ('frame_index', 'timestamp_ns'):
        np.testing.assert_array_equal(fits[0][key], fits[1][key])
    rig = Rig(calibrations[0])
    common = fits[0]['valid'] & fits[1]['valid']
    # Stability in the requested left view uses frames where the target is
    # actually detected in that view. Geometry/coverage comparisons use all common fits.
    visible = common & (fits[0]['selected'][:, 0] >= 0) & (fits[1]['selected'][:, 0] >= 0)
    timestamps = fits[0]['timestamp_ns']
    times = (timestamps - timestamps[0]) / 1e9
    center = np.flatnonzero(visible[1:-1] & visible[:-2] & visible[2:]) + 1
    center = center[(times[center + 1] - times[center - 1]) < .12]
    with np.load(Path(summaries[0]['cache']) / 'observations.npz', allow_pickle=False) as cache:
        track_ids = cache['track_id']
    tracks = track_ids[np.maximum(fits[0]['selected'][:, 0], 0)]
    center = center[(tracks[center] == tracks[center - 1]) & (tracks[center] == tracks[center + 1])]
    alpha = ((times[center] - times[center - 1]) / (times[center + 1] - times[center - 1]))[:, None, None]
    # Fixed old observation set, so changing the new rejection mask cannot by
    # itself improve this comparison's reprojection error.
    old_mask = triangulations[0]['inliers'] & common[:, None, None]
    ids = np.flatnonzero(common)
    # Longer interval motion is a rough check against freezing articulation.
    start = np.flatnonzero(visible)
    end = np.searchsorted(times, times[start] + .20)
    eligible = end < len(common)
    start, end = start[eligible], end[eligible]
    eligible = visible[end] & ((times[end] - times[start]) < .25) & (tracks[start] == tracks[end])
    # Exclude any interval containing a reconstruction gap.
    missing = np.r_[0, np.cumsum(~visible)]
    eligible &= (missing[end + 1] - missing[start]) == 0
    start, end = start[eligible], end[eligible]
    result = dict(common_valid_frames=int(common.sum()), temporal_comparison_frames=len(center),
                  stability_scope='Common valid frames with continuous target detections in ZED left',
                  reference=str(reference.resolve()), candidate=str(candidate.resolve()),
                  limitation='Stability and motion proxies, not ground-truth accuracy; midpoint residual includes genuine acceleration.')
    for name, fit, tri in zip(('previous', 'refined'), fits, triangulations):
        projected = np.full((len(common), 21, 2), np.nan)
        all_pixels = np.full((len(common), 4, 21, 2), np.nan)
        for row in ids:
            projected[row] = rig.project(fit['joints'][row], 0, distorted=True)
            for cam in range(4):
                all_pixels[row, cam] = rig.project(fit['joints'][row], cam)
        relative = projected - projected[:, :1]
        residual = projected[center] - projected[center - 1] * (1 - alpha) - projected[center + 1] * alpha
        finger_residual = relative[center] - relative[center - 1] * (1 - alpha) - relative[center + 1] * alpha
        errors = np.linalg.norm(all_pixels - triangulations[0]['pixels'], axis=-1)[old_mask]
        motion = np.linalg.norm(relative[end, 1:] - relative[start, 1:], axis=-1)
        pairs = np.flatnonzero(common[:-1] & common[1:] & (np.diff(times) < .08))
        switches = np.any(tri['inliers'][pairs + 1] != tri['inliers'][pairs], axis=1)
        result[name] = dict(fitted_frames=int(fit['valid'].sum()),
            median_wrist_residual_px=float(np.median(np.linalg.norm(residual[:, 0], axis=-1))),
            median_finger_residual_px=float(np.median(np.linalg.norm(finger_residual[:, 1:], axis=-1))),
            median_tip_residual_px=float(np.median(np.linalg.norm(finger_residual[:, [4, 8, 12, 16, 20]], axis=-1))),
            p95_tip_residual_px=float(np.percentile(np.linalg.norm(finger_residual[:, [4, 8, 12, 16, 20]], axis=-1), 95)),
            median_error_on_previous_observations_px=float(np.median(errors)),
            p95_error_on_previous_observations_px=float(np.percentile(errors, 95)),
            median_200ms_finger_displacement_px=float(np.median(motion)),
            p95_200ms_finger_displacement_px=float(np.percentile(motion, 95)),
            fingertip_camera_membership_change_fraction=switches[:, [4, 8, 12, 16, 20]].mean(axis=0).tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    (args.candidate / 'refinement_comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
