"""Run YOLO/One Euro/+10%/WiLoR across four time-aligned views, caching native predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

import cv2
import numpy as np

from mchoi.hand.detection import DetectionTracker
from mchoi.hand.wilor_backend import WilorBackend
from mchoi.io.zed import read_exact
from mchoi.common import json_default, save_json, draw_frame
from mchoi.paths import MANO_DIR, MODELS, WILOR_SRC, sdk_python

CAMERAS = ('zed_left', 'zed_right', 'd435', 'd455')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--sdk-python', type=Path, default=sdk_python())
    parser.add_argument('--timing', type=Path, help='Per-take validated timing override; original files are preserved')
    parser.add_argument('--skip-unvalidated-timing', action='store_true', help='Exclude RealSense views without validated timing')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    calibration = args.take / 'aruco_pose_all_cameras_new_calibration/calibration_used.json'
    timing = args.timing or args.take / 'aruco_pose_all_cameras_new_calibration/timing_alignment.json'
    save_json(output / 'calibration.json', json.loads(calibration.read_text()))
    save_json(output / 'timing_alignment.json', json.loads(timing.read_text()))
    info = dict(status='running', take=str(args.take.resolve()), start_frame=args.start_frame,
                crop_scale=1.10, filter=dict(min_cutoff=1.5, beta=8, d_cutoff=1),
                calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(), cameras=CAMERAS)
    save_json(output / 'cache_metadata.json', info)
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    start = time.perf_counter()
    backend = WilorBackend(WILOR_SRC, MODELS / 'wilor', MANO_DIR)
    trackers = {cam: DetectionTracker() for cam in CAMERAS}
    records, frame_indices, timestamps, offsets, sync = [], [], [], [], []
    process = None
    try:
        with (output / 'decode.log').open('wb') as log, (output / 'observations.jsonl').open('w', encoding='utf-8') as jsonl:
            decoder_args = [str(args.sdk_python), '-u', '-m', 'mchoi.io.stream_four_views',
                                        '--take', str(args.take), '--calibration', str(calibration), '--timing', str(timing),
                                        '--start-frame', str(args.start_frame), '--max-frames', str(args.max_frames)]
            if args.skip_unvalidated_timing:
                decoder_args.append('--skip-unvalidated-timing')
            process = subprocess.Popen(decoder_args,
                                       stdout=subprocess.PIPE, stderr=log)
            pipe = process.stdout
            if read_exact(pipe, 4) != b'W4V1':
                raise ValueError('Bad four-view decoder header')

            def message():
                length, = struct.unpack('<I', read_exact(pipe, 4))
                if length > 65536:
                    raise ValueError('Invalid decoder message size')
                return json.loads(read_exact(pipe, length))

            info['source'] = message()
            save_json(output / 'cache_metadata.json', info)
            samples = set(np.linspace(0, info['source']['count'] - 1, 8).astype(int))
            for step in range(info['source']['count']):
                group = message()
                index, timestamp = group['frame_index'], group['timestamp_ns']
                if step == 0:
                    origin = timestamp
                images, track_lists, active = [], [], []
                for cam in CAMERAS:
                    if cam not in group['matches']:
                        continue
                    image = np.frombuffer(read_exact(pipe, 1920 * 1080 * 3), np.uint8).reshape(1080, 1920, 3)
                    detections, keypoints = backend.detect_with_keypoints(image)
                    tracks = trackers[cam].update(detections, (group['matches'][cam]['timestamp_ns'] - origin) / 1e9, (1920, 1080))
                    for track in tracks:
                        j = int(np.argmin(np.linalg.norm(detections[:, :4] - track['raw_box'], axis=1)))
                        track['detector_keypoints'] = keypoints[j]
                    images.append(image)
                    track_lists.append(tracks)
                    active.append(cam)
                predictions = dict(zip(active, backend.reconstruct_many(images, track_lists)))
                row_offsets = []
                for cam in CAMERAS:
                    begin = len(records)
                    records.extend(predictions.get(cam, []))
                    row_offsets.append([begin, len(records)])
                row = dict(**group, hands={cam: [{k: v for k, v in h.items() if k not in ('vertices', 'vertices_2d')}
                                                for h in predictions.get(cam, [])] for cam in CAMERAS})
                jsonl.write(json.dumps(row, default=json_default, allow_nan=False) + '\n')
                frame_indices.append(index)
                timestamps.append(timestamp)
                offsets.append(row_offsets)
                sync.append([[group['matches'].get(cam, {}).get(k, -1) for k in ('frame_index', 'timestamp_ns', 'delta_ns')] for cam in CAMERAS])
                if step in samples:
                    panels = []
                    by_camera = dict(zip(active, images))
                    for cam in CAMERAS:
                        if cam not in by_camera:
                            panel = np.zeros((540, 960, 3), np.uint8)
                            cv2.putText(panel, cam + ': unavailable / excluded timing', (30, 270), cv2.FONT_HERSHEY_SIMPLEX, .7, (180, 180, 180), 2)
                            panels.append(panel)
                            continue
                        image = by_camera[cam]
                        panel = draw_frame(image, predictions[cam], backend.faces, index, (timestamp - origin) / 1e9)
                        cv2.putText(panel, cam + ' | unrectified', (1000, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                        panels.append(cv2.resize(panel, (960, 540)))
                    cv2.imwrite(str(output / f'views_{index:06d}.jpg'), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
                if step % 100 == 0:
                    jsonl.flush()
                    print(f'{step + 1}/{info["source"]["count"]} synchronized groups; {len(records)} hand observations; {(step + 1) / (time.perf_counter() - start):.1f} groups/s', flush=True)
            if process.wait(timeout=30) != 0:
                raise RuntimeError('Decoder exited with an error')
        keys = ['raw_box', 'filtered_box', 'expanded_box', 'joints_2d', 'joints_3d', 'vertices', 'camera_translation',
                'mano_global_orient', 'mano_hand_pose', 'mano_betas', 'detector_keypoints', 'is_right', 'confidence', 'track_id', 'focal_length']
        if not records:
            raise RuntimeError('No hands detected in any view')
        arrays = {key: np.asarray([record[key] for record in records]) for key in keys}
        np.savez_compressed(output / 'observations.npz', frame_index=np.asarray(frame_indices, np.int64),
                            timestamp_ns=np.asarray(timestamps, np.int64), offsets=np.asarray(offsets, np.int64),
                            sync=np.asarray(sync, np.int64), **arrays)
        np.save(output / 'mano_faces.npy', backend.faces)
        info.update(status='complete', frames=len(frame_indices), observations=len(records), elapsed_seconds=time.perf_counter() - start)
        save_json(output / 'cache_metadata.json', info)
        print(json.dumps(info, indent=2), flush=True)
    except BaseException as exc:
        info.update(status='failed', error=str(exc), frames=len(frame_indices))
        save_json(output / 'cache_metadata.json', info)
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            process.stdout.close()


if __name__ == '__main__':
    main()
