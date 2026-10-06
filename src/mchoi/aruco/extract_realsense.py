"""Detect ArUco markers in every native RGB image of one RealSense recording (no re-encoding)."""
import argparse
from pathlib import Path

from mchoi.io.realsense import ColorReader


def extract(take, serial):
    import csv
    import json
    from collections import Counter
    import cv2
    output = take / 'aruco_annotated'
    output.mkdir(exist_ok=True)
    target = output / f'realsense_{serial}_detections.csv'
    if target.exists():
        raise FileExistsError(target)
    camera = f'realsense_{serial}'
    with (take / 'frame_timestamps.csv').open(newline='') as f:
        timestamps = [int(r['pc_timestamp_ns']) for r in csv.DictReader(f) if r['camera'] == camera and r['stream'] == 'color']
    reader = ColorReader(take / (camera + '.db3'))
    if reader.timestamps != timestamps:
        raise ValueError('Native timestamps differ from recording ledger')
    cv2.setNumThreads(4)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)
    fields = ['frame_index', 'timestamp_ns', 'camera', 'detected', 'marker_id'] + [f'corner_{i}_{a}_px' for i in range(4) for a in ('x', 'y')]
    counts, samples = Counter(), []
    sample_indices = {0, len(timestamps)//3, 2*len(timestamps)//3, len(timestamps)-1}
    try:
        with target.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index, timestamp in enumerate(timestamps):
                image = reader.read(index)
                if image.shape != (1080, 1920, 3):
                    raise ValueError('RGB resolution does not match calibration')
                corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                common = dict(frame_index=index, timestamp_ns=timestamp, camera=camera)
                if ids is None:
                    writer.writerow(dict(common, detected=0))
                    counts['no_detection'] += 1
                else:
                    for points, marker_id in zip(corners, ids.flatten()):
                        points = points.reshape(4, 2)
                        row = dict(common, detected=1, marker_id=int(marker_id))
                        row.update({f'corner_{i}_{a}_px': f'{points[i,j]:.3f}' for i in range(4) for j,a in enumerate(('x','y'))})
                        writer.writerow(row)
                        counts[str(int(marker_id))] += 1
                if index in sample_indices:
                    image = image.copy()
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(image, corners, ids)
                    cv2.putText(image, f'{camera} frame {index}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.imwrite(str(output / f'{camera}_sample_{index:04d}.jpg'), image)
                    samples.append(cv2.resize(image, (960, 540)))
                if (index+1) % 500 == 0:
                    print(f'{camera}: {index+1}/{len(timestamps)}', flush=True)
        cv2.imwrite(str(output / f'{camera}_contact_sheet.jpg'), cv2.vconcat(samples))
        report = dict(camera=camera, frames=len(timestamps), detections=dict(counts), native_timestamps_verified=True)
        (output / f'{camera}_summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        reader.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('take', type=Path)
    parser.add_argument('serial')
    args = parser.parse_args()
    extract(args.take.resolve(), args.serial)
