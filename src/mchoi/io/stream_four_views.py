"""Stream time-aligned native unrectified RGB frames using the existing SDK environment."""
import argparse
from bisect import bisect_left
import csv
import json
import os
from pathlib import Path
import struct
import sys

CAMERAS = ('zed_left', 'zed_right', 'd435', 'd455')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--timing', type=Path, required=True)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--tolerance-ms', type=float, default=20)
    parser.add_argument('--skip-unvalidated-timing', action='store_true')
    args = parser.parse_args()
    fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    if os.name == 'nt':
        import msvcrt
        msvcrt.setmode(fd, os.O_BINARY)
    pipe = os.fdopen(fd, 'wb')
    import cv2
    from mchoi.recorder.devices import load_zed
    from mchoi.io.realsense import ColorReader
    calibration = json.loads(args.calibration.read_text())
    timing = json.loads(args.timing.read_text())
    if calibration['reference_camera'] != 'zed_left' or calibration['status'] != 'calibrated':
        raise ValueError('Expected a calibrated rig with ZED left reference')
    # Timing file must have been fitted with this exact calibration, even if a copied file is used.
    original_calibration = json.loads(Path(timing['source_calibration']).read_text())
    if original_calibration != calibration:
        raise ValueError('Timing corrections and rig calibration do not match')
    rows = list(csv.DictReader((args.take / 'frame_timestamps.csv').open(newline='')))
    times = [int(r['pc_timestamp_ns']) for r in rows if r['camera'] == 'zed' and r['stream'] == 'left']
    if args.start_frame < 0 or args.start_frame >= len(times) or args.max_frames < 0:
        raise ValueError('Invalid frame interval')
    count = min(args.max_frames or len(times), len(times) - args.start_frame)
    readers, corrected, last, excluded = {}, {}, {}, {}
    sl = load_zed()
    camera = sl.Camera()
    try:
        for cam in CAMERAS[2:]:
            if not timing['cameras'][cam].get('validated', False):
                if not args.skip_unvalidated_timing:
                    raise ValueError(f'Unvalidated timing for {cam}')
                excluded[cam] = timing['cameras'][cam].get('status', 'timing_not_validated')
                continue
            serial = calibration['cameras'][cam]['device']['serial']
            readers[cam] = ColorReader(args.take / f'realsense_{serial}.db3')
            ledger = [int(r['pc_timestamp_ns']) for r in rows if r['camera'] == f'realsense_{serial}' and r['stream'] == 'color']
            if readers[cam].timestamps != ledger:
                raise ValueError(f'Native timestamp ledger mismatch for {cam}')
            lag = timing['cameras'][cam]['rs_minus_zed_lag_ns']
            corrected[cam] = [t - lag for t in ledger]
            last[cam] = -1
        init = sl.InitParameters()
        init.set_from_svo_file(str((args.take / 'zed.svo2').resolve()))
        init.svo_real_time_mode = False
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.sdk_verbose = 0
        if camera.open(init) > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Cannot open SVO')
        camera.set_svo_position(args.start_frame)
        mat = sl.Mat()
        meta = dict(cameras=CAMERAS, width=1920, height=1080, count=count, start_frame=args.start_frame,
                    active_cameras=[c for c in CAMERAS if c not in excluded], excluded_cameras=excluded,
                    image_convention='unrectified native BGR8', timing=timing, tolerance_ms=args.tolerance_ms)

        def message(value):
            encoded = json.dumps(value).encode('utf-8')
            pipe.write(struct.pack('<I', len(encoded)) + encoded)

        pipe.write(b'W4V1')
        message(meta)
        for index in range(args.start_frame, args.start_frame + count):
            if camera.grab() > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f'Cannot decode indexed SVO frame {index}')
            timestamp = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if camera.get_svo_position() != index or timestamp != times[index]:
                raise ValueError('SVO frame or timestamp mismatch')
            matches = {cam: dict(frame_index=index, timestamp_ns=timestamp, delta_ns=0) for cam in CAMERAS[:2]}
            for cam in readers:
                position = bisect_left(corrected[cam], timestamp)
                candidates = [i for i in (position - 1, position) if last[cam] < i < len(corrected[cam])]
                if candidates:
                    chosen = min(candidates, key=lambda i: abs(corrected[cam][i] - timestamp))
                    delta = corrected[cam][chosen] - timestamp
                    if abs(delta) <= args.tolerance_ms * 1e6:
                        matches[cam] = dict(frame_index=chosen, timestamp_ns=readers[cam].timestamps[chosen], delta_ns=delta)
                        last[cam] = chosen
            message(dict(frame_index=index, timestamp_ns=timestamp, matches=matches))
            for cam in CAMERAS:
                if cam not in matches:
                    continue
                if cam.startswith('zed'):
                    view = sl.VIEW.LEFT_UNRECTIFIED if cam == 'zed_left' else sl.VIEW.RIGHT_UNRECTIFIED
                    if camera.retrieve_image(mat, view) > sl.ERROR_CODE.SUCCESS:
                        raise RuntimeError('ZED view retrieval failed')
                    image = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
                else:
                    image = readers[cam].read(matches[cam]['frame_index'])
                if image.shape != (1080, 1920, 3):
                    raise ValueError(f'Unexpected {cam} resolution: {image.shape}')
                pipe.write(image.tobytes())
            pipe.flush()
    finally:
        camera.close()
        for reader in readers.values():
            reader.close()
        pipe.close()


if __name__ == '__main__':
    main()
