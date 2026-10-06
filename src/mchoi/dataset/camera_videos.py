"""Convert one recording's native camera files into standard RGB and depth videos with per-frame timestamps.

ZED Mini (zed.svo2): left and right unrectified RGB (the images the rig calibration describes), plus a depth map
computed by the ZED SDK (NEURAL mode). ZED depth is aligned to the *rectified* left image; its intrinsics are written
to the sidecar. RealSense (realsense_<serial>.db3): native BGR8 colour and native Z16 depth (848x480, depth-sensor
frame, not aligned to colour); raw Z16 values are kept and depth_units_m converts them to metres.

RGB -> H.264 MP4 (yuv420p, CRF 17). Depth -> FFV1 MKV, 16-bit grey, lossless, millimetres for ZED and raw Z16 units
for RealSense (both 0 = no measurement). Presentation times are source time since the stream's first frame; exact PC
clock times (Unix ns) are in <name>_frames.tsv. Every stream gets a .json sidecar.
"""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import sqlite3
import struct
import time

import av
import numpy as np

from mchoi.recorder.timestamps import parse_metadata, milliseconds_to_ns
from mchoi.paths import calibration_path, timing_path

TIME_BASE = Fraction(1, 1_000_000)


def decode(blob):
    """CDR sensor_msgs/Image -> (encoding, array)."""
    if blob[:4] != b'\x00\x01\x00\x00':
        raise ValueError('Expected little-endian CDR image')
    offset = 12

    def u32():
        nonlocal offset
        offset = (offset + 3) // 4 * 4
        value = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        return value

    def string():
        nonlocal offset
        length = u32()
        value = blob[offset:offset + length - 1].decode('utf-8')
        offset += length
        return value

    string()
    height, width = u32(), u32()
    encoding = string()
    offset += 1  # is_bigendian
    step, length = u32(), u32()
    if encoding == 'bgr8':
        return encoding, np.frombuffer(blob, np.uint8, count=length, offset=offset).reshape(height, step)[:, :width * 3].reshape(height, width, 3)
    if encoding in ('mono16', '16UC1'):
        return encoding, np.frombuffer(blob, np.uint16, count=length // 2, offset=offset).reshape(height, step // 2)[:, :width]
    raise ValueError(f'Unsupported encoding {encoding}')


class Writer:
    def __init__(self, path, width, height, depth):
        self.path, self.depth = path, depth
        self.container = av.open(str(path), 'w', options={} if depth else {'movflags': 'faststart'})
        self.stream = self.container.add_stream('ffv1' if depth else 'libx264', rate=30)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = 'gray16le' if depth else 'yuv420p'
        self.stream.options = {'level': '3', 'slices': '16', 'slicecrc': '1'} if depth else {'crf': '17', 'preset': 'medium'}
        self.stream.time_base = self.stream.codec_context.time_base = TIME_BASE
        self.first = None
        self.rows = []

    def add(self, image, timestamp_ns, frame_index, extra=None):
        self.first = timestamp_ns if self.first is None else self.first
        pts = (timestamp_ns - self.first) // 1000
        if self.rows and pts <= self.rows[-1]['pts_us']:
            raise ValueError(f'{self.path.name}: non-increasing timestamp at frame {frame_index}')
        frame = av.VideoFrame.from_ndarray(image, format='gray16le' if self.depth else 'bgr24')
        frame.pts, frame.time_base = pts, TIME_BASE
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.rows.append(dict(frame=len(self.rows), source_frame_index=frame_index, timestamp_ns=timestamp_ns, pts_us=pts, **(extra or {})))

    def close(self, sidecar):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        base = str(self.path)[:-len(self.path.suffix)]
        with open(base + '_frames.tsv', 'w', newline='') as f:
            w = csv.DictWriter(f, list(self.rows[0]), delimiter='\t')
            w.writeheader()
            w.writerows(self.rows)
        stamps = np.array([r['timestamp_ns'] for r in self.rows])
        sidecar.update(FrameCount=len(self.rows), Width=self.stream.width, Height=self.stream.height, NominalFrameRate=30,
                       MeasuredFrameRate=round(float((len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1e9)), 3),
                       DurationSeconds=round(float((stamps[-1] - stamps[0]) / 1e9), 3),
                       FirstTimestampUnixNs=int(stamps[0]), LastTimestampUnixNs=int(stamps[-1]),
                       Codec='FFV1 (lossless), Matroska, gray16le' if self.depth else 'H.264 (libx264, CRF 17, yuv420p), MP4',
                       PresentationTime='microseconds since this stream\'s first frame (source clock); exact times in _frames.tsv',
                       FramesFile=Path(base + '_frames.tsv').name)
        Path(base + '.json').write_text(json.dumps(sidecar, indent=2))


def export_zed(take, prefix, out, calibration, zed_label, max_frames):
    from mchoi.recorder.devices import load_zed
    sl = load_zed()
    ledger = {int(r['frame_index']): int(r['pc_timestamp_ns']) for r in csv.DictReader((take / 'frame_timestamps.csv').open(newline=''))
              if r['camera'] == 'zed' and r['stream'] == 'left'}
    camera = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str((take / 'zed.svo2').resolve()))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.sdk_verbose = 0
    if camera.open(init) > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError('Cannot open SVO')
    info = camera.get_camera_information()
    left_rect = info.camera_configuration.calibration_parameters.left_cam
    width, height = info.camera_configuration.resolution.width, info.camera_configuration.resolution.height
    writers = {view: Writer(out / f'{prefix}_cam-{zed_label}{view}_rgb.mp4', width, height, False) for view in ('left', 'right')}
    depth_writer = Writer(out / f'{prefix}_cam-{zed_label}left_depth.mkv', width, height, True)
    left, right, depth = sl.Mat(), sl.Mat(), sl.Mat()
    runtime = sl.RuntimeParameters()
    total = camera.get_svo_number_of_frames()
    count, start = 0, time.perf_counter()
    try:
        while True:
            status = camera.grab(runtime)
            if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if status > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f'grab: {status}')
            index = camera.get_svo_position()
            stamp = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if ledger.get(index) != stamp:
                raise ValueError(f'SVO frame {index} timestamp {stamp} differs from the recording ledger')
            camera.retrieve_image(left, sl.VIEW.LEFT_UNRECTIFIED)
            camera.retrieve_image(right, sl.VIEW.RIGHT_UNRECTIFIED)
            camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
            writers['left'].add(np.ascontiguousarray(left.get_data()[:, :, :3]), stamp, index)
            writers['right'].add(np.ascontiguousarray(right.get_data()[:, :, :3]), stamp, index)
            d = depth.get_data()
            d = np.where(np.isfinite(d) & (d > 0), np.minimum(np.rint(d), 65535), 0).astype(np.uint16)
            depth_writer.add(d, stamp, index)
            count += 1
            if count % 250 == 0:
                print(f'zed {count}/{total} {count / (time.perf_counter() - start):.1f} fps', flush=True)
            if max_frames and count >= max_frames:
                break
    finally:
        camera.close()
    cal = calibration['cameras']
    common = dict(CameraModel='Stereolabs ZED Mini', SerialNumber=str(info.serial_number), ClockDomain='PC Unix time (ns), ZED SDK image timestamp',
                  SourceFile='zed.svo2 (not distributed)')
    for view in ('left', 'right'):
        writers[view].close(dict(common, Stream='rgb', Camera=f'{zed_label}{view}', CalibrationName=f'zed_{view}',
                                 Intrinsics=dict(camera_matrix=cal[f'zed_{view}']['camera_matrix'], distortion_coefficients=cal[f'zed_{view}']['distortion_coefficients'],
                                                 model=cal[f'zed_{view}']['distortion_model']),
                                 IntrinsicsSource='rig calibration (unrectified native image)',
                                 Extrinsics=dict(T_camera_from_zedleft=cal[f'zed_{view}']['T_camera_from_reference'])))
    depth_writer.close(dict(common, Stream='depth', Camera=f'{zed_label}left', Units='millimetre', InvalidValue=0,
                            DepthMethod='ZED SDK NEURAL stereo depth', AlignedTo='rectified left image (ZED SDK convention), not the unrectified RGB video',
                            Intrinsics=dict(fx=left_rect.fx, fy=left_rect.fy, cx=left_rect.cx, cy=left_rect.cy, distortion='none (rectified)'),
                            IntrinsicsSource='ZED SDK factory calibration, rectified left'))
    return count


def export_realsense(take, prefix, out, calibration, serial, label, timing, max_frames):
    db = sqlite3.connect((take / f'realsense_{serial}.db3').resolve().as_uri() + '?mode=ro', uri=True)
    recording = json.loads((take / 'recording.json').read_text())['cameras'][f'realsense_{serial}']
    calib_name = next(k for k, v in calibration['cameras'].items() if str(v.get('device', {}).get('serial')) == serial)
    entry = timing['cameras'].get(calib_name, {})
    lag = int(entry['rs_minus_zed_lag_ns']) if entry.get('validated') else None
    ledger = {}
    for r in csv.DictReader((take / 'frame_timestamps.csv').open(newline='')):
        if r['camera'] == f'realsense_{serial}':
            ledger.setdefault(r['stream'], []).append(int(r['pc_timestamp_ns']))
    counts = {}
    for kind, stream in (('Color', 'color'), ('Depth', 'depth')):
        topic, name = db.execute("SELECT id,name FROM topics WHERE name LIKE ?", (f'%/{kind}_%/image/data',)).fetchone()
        meta_topic = db.execute('SELECT id FROM topics WHERE name=?', (name.replace('/data', '/metadata'),)).fetchone()[0]
        rows = db.execute('SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topic,)).fetchall()
        meta = db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (meta_topic,)).fetchall()
        stamps = [milliseconds_to_ns(parse_metadata(blob)['system_time']) for _, blob in meta]
        if [r[1] for r in rows] != [m[0] for m in meta] or stamps != ledger[stream]:
            raise ValueError(f'{serial} {stream}: native timestamps differ from the recording ledger')
        depth = stream == 'depth'
        size = recording['depth_size'] if depth else recording['rgb_size']
        writer = Writer(out / f'{prefix}_cam-{label}_{"depth.mkv" if depth else "rgb.mp4"}', size[0], size[1], depth)
        for i, ((message_id, _), stamp) in enumerate(zip(rows, stamps)):
            encoding, image = decode(db.execute('SELECT data FROM messages WHERE id=?', (message_id,)).fetchone()[0])
            writer.add(np.ascontiguousarray(image), stamp, i, None if lag is None else dict(zed_aligned_timestamp_ns=stamp - lag))
            if max_frames and i + 1 >= max_frames:
                break
        sidecar = dict(CameraModel=recording.get('name', 'RealSense D435'), SerialNumber=serial, Camera=label, CalibrationName=calib_name,
                       Stream=stream, ClockDomain='PC Unix time (ns), RealSense SDK system_time metadata',
                       SourceFile=f'realsense_{serial}.db3 (not distributed)',
                       TimingCorrectionNs=lag, TimingValidated=lag is not None,
                       TimingNote='zed_aligned_timestamp_ns = timestamp_ns - TimingCorrectionNs matches ZED image content'
                                  if lag is not None else 'Offset to the ZED clock could not be validated for this recording; '
                                                          'pair with other cameras only approximately',
                       Exposure=recording.get('controls', {}).get('depth' if depth else 'color'))
        if depth:
            sidecar.update(Units='RealSense Z16 units', DepthUnitsMetres=recording['depth_units_m'], InvalidValue=0,
                           AlignedTo='native depth sensor (not aligned to colour)', Intrinsics=recording.get('depth_intrinsics'),
                           IntrinsicsSource='RealSense factory calibration (recording.json)',
                           DepthToColorExtrinsics=recording.get('depth_to_color'))
        else:
            cal = calibration['cameras'][calib_name]
            sidecar.update(Intrinsics=dict(camera_matrix=cal['camera_matrix'], distortion_coefficients=cal['distortion_coefficients'], model=cal['distortion_model']),
                           IntrinsicsSource='rig calibration', FactoryIntrinsics=recording.get('color_intrinsics'),
                           Extrinsics=dict(T_camera_from_zedleft=cal['T_camera_from_reference']))
        writer.close(sidecar)
        counts[stream] = len(writer.rows)
        print(f'{label} {stream}: {counts[stream]} frames', flush=True)
    db.close()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('take', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prefix', required=True, help='File-name prefix, e.g. sub-01_ses-01_task-mustard_run-01')
    parser.add_argument('--cameras', nargs='+', default=('zed', 'realsense'), choices=('zed', 'realsense'))
    parser.add_argument('--labels', default='zedmini:261222078090=d435n1:261222076358=d435n2',
                        help='zed label, then serial=label for each RealSense')
    parser.add_argument('--max-frames', type=int, default=0, help='Testing only')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    zed_label, *rs = args.labels.split(':')
    rs_labels = dict(item.split('=') for item in rs)
    calibration = json.loads(calibration_path(args.take).read_text())
    timing = json.loads(timing_path(args.take).read_text())
    existing = list(args.output_dir.glob(f'{args.prefix}_cam-*'))
    if existing:
        parser.error(f'Outputs exist, e.g. {existing[0].name}')
    report = {}
    if 'zed' in args.cameras:
        report['zed'] = export_zed(args.take, args.prefix, args.output_dir, calibration, zed_label, args.max_frames)
    if 'realsense' in args.cameras:
        for serial, label in rs_labels.items():
            if (args.take / f'realsense_{serial}.db3').exists():
                report[label] = export_realsense(args.take, args.prefix, args.output_dir, calibration, serial, label, timing, args.max_frames)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
