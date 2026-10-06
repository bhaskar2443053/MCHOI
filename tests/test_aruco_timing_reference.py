"""Known-clock regression for timing recovery through another calibrated camera."""
import csv
import json
import numpy as np

from mchoi.aruco.timing import estimate


def test_indirect_timing_converts_relative_lag_to_absolute_zed_lag(tmp_path):
    take = tmp_path / 'take'
    detections = take / 'aruco_annotated'
    detections.mkdir(parents=True)
    output = tmp_path / 'timing'
    output.mkdir()
    (output / 'marker_selection.json').write_text(json.dumps(dict(marker_ids=[0])))
    cameras = {}
    for name, serial, center, lag in [('d435', '111', [-.2, 0, 0], .1), ('d455', '222', [.2, .05, 0], .075)]:
        T = np.eye(4)
        T[:3, 3] = -np.array(center)
        K = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]])
        cameras[name] = dict(device=dict(serial=serial), T_camera_from_reference=T.tolist(),
                             T_reference_from_camera=np.linalg.inv(T).tolist(), camera_matrix=K.tolist(),
                             distortion_coefficients=[0, 0, 0, 0, 0])
        fields = ['detected', 'marker_id', 'camera', 'timestamp_ns'] + [f'corner_{i}_{a}_px' for i in range(4) for a in ('x', 'y')]
        with (detections / f'realsense_{serial}_detections.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            corners = np.array([[-.012, -.012, 0], [.012, -.012, 0], [.012, .012, 0], [-.012, .012, 0]])
            for t in np.arange(0, 9, .025):
                position = [.05 * np.sin(t * 2), .05 * np.sin(t * 3), .8 + .06 * np.cos(t)]
                xyz = corners + position - center
                uv = xyz[:, :2] / xyz[:, 2:] * 800 + [960, 540]
                row = dict(detected=1, marker_id=0, camera=f'realsense_{serial}',
                           timestamp_ns=1_790_000_000_000_000_000 + int(round((t + lag) * 1e9)))
                row.update({f'corner_{i}_{a}_px': uv[i, j] for i in range(4) for j, a in enumerate(('x', 'y'))})
                writer.writerow(row)
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(dict(cameras=cameras)))
    result = estimate(take, calibration, output, reference_camera='d435', reference_lag_ns=100_000_000)
    recovered = result['cameras']['d455']
    assert recovered['validated']
    assert abs(recovered['rs_minus_zed_lag_ns'] - 75_000_000) <= 1_000_000
    assert abs(recovered['independent_holdout_best_ms'] - 75) <= 1
    assert abs(recovered['relative_training_best_ms'] + 25) <= 1
