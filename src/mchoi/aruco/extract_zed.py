"""Extract native-resolution ArUco corners without encoding an intermediate video."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2

from mchoi.recorder.devices import load_zed


def extract(take, dictionary_name='DICT_4X4_50'):
    output = take / 'aruco_annotated'
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'detections.csv'
    if target.exists():
        raise FileExistsError(target)
    with (take / 'frame_timestamps.csv').open(newline='') as handle:
        timestamps = [int(row['pc_timestamp_ns']) for row in csv.DictReader(handle)
                      if row['camera'] == 'zed' and row['stream'] == 'left']
    cv2.setNumThreads(4)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name)), params)
    sl = load_zed()
    camera = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(take / 'zed.svo2'))
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.svo_real_time_mode = False
    init.sdk_verbose = 0
    status = camera.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(status)
    try:
        config = camera.get_camera_information().camera_configuration
        if (config.resolution.width, config.resolution.height) != (1920, 1080):
            raise ValueError('Expected native 1920 x 1080 ZED views')
        fields = ['frame_index', 'svo_position', 'timestamp_ns', 'elapsed_seconds', 'view', 'detected',
                  'dictionary', 'marker_id', 'center_x_px', 'center_y_px']
        fields += [f'corner_{i}_{axis}_px' for i in range(4) for axis in ('x', 'y')]
        counts = {view: Counter() for view in ('left', 'right')}
        samples = []
        sample_indices = {0, len(timestamps) // 3, 2 * len(timestamps) // 3, len(timestamps) - 1}
        mat = sl.Mat()
        index = 0
        with target.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while True:
                status = camera.grab()
                if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                if status > sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(status)
                timestamp = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
                if index >= len(timestamps) or timestamps[index] != timestamp:
                    raise ValueError('Source frame timestamp differs from native index')
                pair = []
                for view, enum in [('left', sl.VIEW.LEFT_UNRECTIFIED), ('right', sl.VIEW.RIGHT_UNRECTIFIED)]:
                    if camera.retrieve_image(mat, enum) > sl.ERROR_CODE.SUCCESS:
                        raise RuntimeError('Image retrieval failed')
                    image = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                    common = dict(frame_index=index, svo_position=camera.get_svo_position(), timestamp_ns=timestamp,
                                  elapsed_seconds=f'{(timestamp-timestamps[0])/1e9:.9f}', view=view, dictionary=dictionary_name)
                    if ids is None:
                        writer.writerow(dict(common, detected=0))
                        counts[view]['no_detection'] += 1
                    else:
                        for corner, marker_id in zip(corners, ids.flatten()):
                            points = corner.reshape(4, 2)
                            center = points.mean(axis=0)
                            row = dict(common, detected=1, marker_id=int(marker_id), center_x_px=f'{center[0]:.3f}', center_y_px=f'{center[1]:.3f}')
                            row.update({f'corner_{i}_{axis}_px': f'{points[i,j]:.3f}' for i in range(4) for j,axis in enumerate(('x','y'))})
                            writer.writerow(row)
                            counts[view][str(int(marker_id))] += 1
                    if index in sample_indices:
                        if ids is not None:
                            cv2.aruco.drawDetectedMarkers(image, corners, ids)
                        cv2.imwrite(str(output / f'sample_{index:04d}_{view}.jpg'), image)
                        small = cv2.resize(image, (960, 540))
                        cv2.putText(small, f'{view} | frame {index}', (14, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2)
                        pair.append(small)
                if pair:
                    samples.append(cv2.hconcat(pair))
                index += 1
                if index % 500 == 0:
                    print(f'ArUco detection: {index}/{len(timestamps)} frames', flush=True)
        if index != len(timestamps):
            raise ValueError('Not all source frames decoded')
        cv2.imwrite(str(output / 'sample_contact_sheet.jpg'), cv2.vconcat(samples))
        report = dict(frames=index, dictionary=dictionary_name, native_timestamps_verified=True,
                      per_view={view: dict(count) for view, count in counts.items()})
        (output / 'detection_summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        camera.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('take', type=Path)
    args = parser.parse_args()
    extract(args.take.resolve())
