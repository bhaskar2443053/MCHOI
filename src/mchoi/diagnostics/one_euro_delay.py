"""Estimate rotation-filter delay relative to its source fit, not ground truth."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_delay(source, candidate):
    a = dict(np.load(source / 'fused_hand.npz', allow_pickle=False))
    b = dict(np.load(candidate / 'fused_hand.npz', allow_pickle=False))
    np.testing.assert_array_equal(a['timestamp_ns'], b['timestamp_ns'])
    count = len(a['valid'])
    raw = np.full((count, 16, 3, 3), np.nan)
    filtered = raw.copy()
    raw[a['fit_frame_rows']] = a['mano_rotations']
    filtered[b['fit_frame_rows']] = b['mano_rotations']
    times = (a['timestamp_ns'] - a['timestamp_ns'][0]) / 1e9
    common = a['valid'] & b['valid'] & (a['selected'][:, 0] >= 0)
    segment_start = np.zeros(count, int)
    for i in range(1, count):
        segment_start[i] = (segment_start[i - 1] if common[i] and common[i - 1]
                            and times[i] - times[i - 1] <= .1 else i)
    rows = np.flatnonzero(common & ((times - times[segment_start]) >= .25))
    if not len(rows):
        return dict(status='insufficient_continuous_motion')

    def interpolate(query):
        hi = np.searchsorted(times, query, side='left')
        lo = hi - 1
        blend = (query - times[lo]) / (times[hi] - times[lo])
        delta = Rotation.from_matrix((raw[hi] @ raw[lo].swapaxes(-1, -2)).reshape(-1, 3, 3)).as_rotvec().reshape(-1, 16, 3)
        step = Rotation.from_rotvec((delta * blend[:, None, None]).reshape(-1, 3)).as_matrix().reshape(-1, 16, 3, 3)
        return step @ raw[lo]

    # Focus on moving joints; comparing stationary orientations cannot identify lag.
    old = interpolate(times[rows] - .2)
    speed = Rotation.from_matrix((raw[rows] @ old.swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude().reshape(-1, 16) / .2
    moving = speed > .3
    records = []
    for lag_ms in range(0, 161, 10):
        shifted = interpolate(times[rows] - lag_ms / 1000)
        error = np.rad2deg(Rotation.from_matrix((filtered[rows] @ shifted.swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude().reshape(-1, 16))
        records.append(dict(lag_ms=lag_ms,
            finger_median_error_deg=float(np.median(error[:, 1:][moving[:, 1:]])) if moving[:, 1:].any() else None,
            wrist_median_error_deg=float(np.median(error[:, 0][moving[:, 0]])) if moving[:, 0].any() else None))
    return dict(status='complete', continuous_frames=len(rows), moving_finger_samples=int(moving[:, 1:].sum()),
        estimated_finger_delay_ms=min(records, key=lambda r: r['finger_median_error_deg'])['lag_ms'] if moving[:, 1:].any() else None,
        estimated_wrist_orientation_delay_ms=min(records, key=lambda r: r['wrist_median_error_deg'])['lag_ms'] if moving[:, 0].any() else None,
        curve=records, limitation='Time shift minimizing angular disagreement with the unfiltered offline fit during motion, in 10 ms steps. Not latency relative to the real hand; no ground truth.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    report = rotation_delay(args.source, args.candidate)
    (args.candidate / 'one_euro_delay.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'curve'}, indent=2))
