"""Cache per-view YOLO-pose person keypoints (COCO-17) over time-aligned native frames of the four cameras."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np

from mchoi.paths import MODELS, calibration_path, sdk_python, timing_path
from mchoi.io.zed import read_exact

CAMERAS = ('zed_left', 'zed_right', 'd435', 'd455')
MAX_PEOPLE = 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='.npz path; must not exist')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--weights', type=Path, default=MODELS / 'yolo11x-pose.pt')
    parser.add_argument('--timing', type=Path, help='RealSense timing file (default: wilor_timing if present, else the ArUco estimate)')
    parser.add_argument('--sdk-python', type=Path, default=sdk_python())
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    calibration = calibration_path(args.take)
    timing = args.timing or timing_path(args.take)
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    process = subprocess.Popen([str(args.sdk_python), '-u', '-m', 'mchoi.io.stream_four_views', '--take', str(args.take),
                                '--calibration', str(calibration), '--timing', str(timing),
                                '--start-frame', str(args.start_frame), '--max-frames', str(args.max_frames), '--skip-unvalidated-timing'],
                               stdout=subprocess.PIPE, stderr=open(args.output.with_suffix('.decode.log'), 'wb'))
    pipe = process.stdout

    def message():
        length, = struct.unpack('<I', read_exact(pipe, 4))
        return json.loads(read_exact(pipe, length))

    if read_exact(pipe, 4) != b'W4V1':
        raise ValueError('Bad decoder header')
    info = message()
    count = info['count']
    keypoints = np.zeros((count, 4, MAX_PEOPLE, 17, 3), np.float32)
    present = np.zeros((count, 4), bool)
    frames, stamps = [], []
    start = time.perf_counter()
    for step in range(count):
        group = message()
        frames.append(group['frame_index'])
        stamps.append(group['timestamp_ns'])
        images, cams = [], []
        for cam in CAMERAS:
            if cam in group['matches']:
                images.append(np.frombuffer(read_exact(pipe, 1920 * 1080 * 3), np.uint8).reshape(1080, 1920, 3))
                cams.append(CAMERAS.index(cam))
        for cam, result in zip(cams, model.predict(images, imgsz=1280, conf=.25, verbose=False)):
            present[step, cam] = True
            if result.keypoints is None or not len(result.keypoints):
                continue
            xy, conf = result.keypoints.xy.cpu().numpy(), result.keypoints.conf.cpu().numpy()
            order = np.argsort(-result.boxes.conf.cpu().numpy())[:MAX_PEOPLE]
            for slot, person in enumerate(order):
                keypoints[step, cam, slot, :, :2] = xy[person]
                keypoints[step, cam, slot, :, 2] = conf[person]
        if step % 25 == 0:
            print(f'{step + 1}/{count}; {(step + 1) / (time.perf_counter() - start):.1f} groups/s', flush=True)
    process.wait(timeout=30)
    np.savez_compressed(args.output, frame_index=np.asarray(frames), timestamp_ns=np.asarray(stamps, np.int64),
                        keypoints=keypoints, view_present=present, cameras=np.asarray(CAMERAS))


if __name__ == '__main__':
    main()
