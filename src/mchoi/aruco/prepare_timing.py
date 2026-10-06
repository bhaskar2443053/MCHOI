"""Reuse validated timing and try alternate calibrated marker views for missing offsets."""
import argparse
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import shutil

from mchoi.aruco.timing import estimate


def prepare(take):
    source = take / 'aruco_pose_all_cameras_new_calibration'
    destination = take / 'wilor_timing'
    final = destination / 'timing_alignment.json'
    if final.exists():
        return json.loads(final.read_text())
    destination.mkdir(exist_ok=True)
    existing = json.loads((source / 'timing_alignment.json').read_text())
    calibration = Path(existing['source_calibration'])
    if json.loads(calibration.read_text()) != json.loads((source / 'calibration_used.json').read_text()):
        raise ValueError(f'Calibration mismatch in {take.name}')
    result = copy.deepcopy(existing)
    result['original_timing_file'] = str((source / 'timing_alignment.json').resolve())
    result['policy'] = 'Use validated per-take offsets only; exclude unavailable timing. Alternate references use calibrated marker epipolar alignment with alternating-block holdout validation.'

    def trial(reference, reference_lag=0):
        output = destination / reference
        output.mkdir(exist_ok=True)
        if (source / 'marker_selection.json').exists():
            shutil.copyfile(source / 'marker_selection.json', output / 'marker_selection.json')
        else:
            # Older reviewed exports store their selected marker IDs in metadata.
            ids = json.loads((source / 'pose_metadata.json').read_text())['marker_ids']
            (output / 'marker_selection.json').write_text(json.dumps(dict(marker_ids=ids, source='pose_metadata.json')))
        with (output / 'estimation.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
            return estimate(take, calibration, output, reference, reference_lag)

    if any('validated' not in v for v in result['cameras'].values()):
        updated = trial('zed_left')
        for cam, entry in result['cameras'].items():
            if 'validated' not in entry:
                result['cameras'][cam] = updated['cameras'][cam]
    if any(not v['validated'] for v in result['cameras'].values()):
        updated = trial('zed_right')
        for cam, entry in result['cameras'].items():
            if not entry['validated'] and updated['cameras'][cam]['validated']:
                result['cameras'][cam] = updated['cameras'][cam]
    # A RealSense feed with a known absolute ZED offset can anchor the other
    # RealSense feed when they see a marker that is hidden from the ZED views.
    anchors = [(cam, entry['rs_minus_zed_lag_ns']) for cam, entry in result['cameras'].items() if entry['validated']]
    for anchor, lag in anchors:
        if all(v['validated'] for v in result['cameras'].values()):
            break
        updated = trial(anchor, lag)
        for cam, entry in updated['cameras'].items():
            if not result['cameras'][cam]['validated'] and entry['validated']:
                result['cameras'][cam] = entry
    result['active_cameras'] = ['zed_left', 'zed_right'] + [c for c, v in result['cameras'].items() if v['validated']]
    result['excluded_cameras'] = {c: v['status'] for c, v in result['cameras'].items() if not v['validated']}
    final.write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('takes', type=Path, nargs='+')
    args = parser.parse_args()
    for take in args.takes:
        result = prepare(take.resolve())
        print(json.dumps(dict(take=take.name, active=result['active_cameras'], excluded=result['excluded_cameras'],
                              offsets_ms={c: v['rs_minus_zed_lag_ns'] / 1e6 for c, v in result['cameras'].items()})), flush=True)
