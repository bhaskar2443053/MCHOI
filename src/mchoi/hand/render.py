"""Render calibrated fused MANO onto native ZED left, plus a monocular comparison."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
import numpy as np

from mchoi.multiview import Rig
from mchoi.recorder.devices import load_zed


class Writer:
    def __init__(self, path, width, height):
        if path.exists():
            raise FileExistsError(path)
        self.container = av.open(str(path), 'w', options={'movflags': '+faststart'})
        self.stream = self.container.add_stream('libx264', rate=30)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, 'yuv420p'
        self.time_base = Fraction(1, 1_000_000)
        self.stream.time_base = self.stream.codec_context.time_base = self.time_base
        self.stream.codec_context.thread_count = 3
        self.stream.options = {'crf': '19', 'preset': 'veryfast'}

    def add(self, image, timestamp_ns):
        frame = av.VideoFrame.from_ndarray(image, format='bgr24')
        frame.pts = int(timestamp_ns) // 1000
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def overlay(image, vertices, joints, faces, color, title, subtitle):
    canvas = image.copy()
    if vertices is not None:
        points = np.rint(np.clip(vertices, -10000, 10000)).astype(np.int32)
        wire = canvas.copy()
        cv2.polylines(wire, points[faces], True, color, 1, cv2.LINE_AA)
        cv2.addWeighted(wire, .45, canvas, .55, 0, canvas)
        points = np.rint(np.clip(joints, -10000, 10000)).astype(int)
        for base in (1, 5, 9, 13, 17):
            chain = [0] + list(range(base, base + 4))
            for a, b in zip(chain, chain[1:]):
                cv2.line(canvas, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, tuple(point), 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (1920, 80), (25, 25, 25), -1)
    cv2.putText(canvas, title, (20, 31), cv2.FONT_HERSHEY_SIMPLEX, .8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (20, 63), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compare-fit', type=Path, help='Compare against a previous multiview fit instead of monocular WiLoR')
    parser.add_argument('--closeup', action='store_true', help='Also export a shared-crop hand close-up comparison')
    parser.add_argument('--closeup-only', action='store_true', help='Export only the close-up, preserving existing full-view videos')
    args = parser.parse_args()
    if args.closeup_only:
        args.closeup = True
    summary = json.loads((args.output / 'fit_summary.json').read_text())
    cache = Path(summary['cache'])
    metadata = json.loads((cache / 'cache_metadata.json').read_text())
    data = dict(np.load(cache / 'observations.npz', allow_pickle=False))
    fused = dict(np.load(args.output / 'fused_hand.npz', allow_pickle=False))
    calibration = json.loads((args.output / 'calibration.json').read_text())
    rig = Rig(calibration)
    previous = None
    if args.compare_fit:
        previous = dict(np.load(args.compare_fit / 'fused_hand.npz', allow_pickle=False))
        if json.loads((args.compare_fit / 'calibration.json').read_text()) != calibration:
            raise ValueError('Comparison calibration differs')
        for key in ('frame_index', 'timestamp_ns'):
            np.testing.assert_array_equal(previous[key], fused[key])
    faces = np.load(args.output / 'mano_faces.npy')
    cv2.setNumThreads(4)
    sl = load_zed()
    camera = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(Path(metadata['take']) / 'zed.svo2'))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.sdk_verbose = 0
    if camera.open(init) > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError('Could not open SVO')
    writers = []
    count = 0
    samples = set(np.linspace(0, len(data['frame_index']) - 1, 8).astype(int))
    sample_paths = []
    try:
        main_video = None if args.closeup_only else Writer(args.output / 'wilor_multiview_left.mp4', 1920, 1080)
        if main_video is not None:
            writers.append(main_video)
        comparison = None if args.closeup_only else Writer(args.output / 'comparison_left.mp4', 1920, 540)
        if comparison is not None:
            writers.append(comparison)
        closeup = Writer(args.output / 'comparison_hand.mp4', 1280, 720) if args.closeup else None
        if closeup is not None:
            writers.append(closeup)
        camera.set_svo_position(int(data['frame_index'][0]))
        mat = sl.Mat()
        for row, index in enumerate(data['frame_index']):
            if camera.grab() > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f'Missing SVO frame {index}')
            timestamp = int(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if camera.get_svo_position() != index or timestamp != int(data['timestamp_ns'][row]):
                raise ValueError('SVO source frame does not match fitted data')
            if camera.retrieve_image(mat, sl.VIEW.LEFT_UNRECTIFIED) > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('Cannot retrieve unrectified left view')
            image = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2BGR)
            obs = int(fused['selected'][row, 0])
            mono_vertices, mono_joints = None, None
            if obs >= 0:
                xyz = data['vertices'][obs] + data['camera_translation'][obs]
                mono_vertices = data['focal_length'][obs] * xyz[:, :2] / xyz[:, 2:] + [960, 540]
                mono_joints = data['joints_2d'][obs]
            seconds = (timestamp - int(data['timestamp_ns'][0])) / 1e9
            baseline = overlay(image, mono_vertices, mono_joints, faces, (20, 190, 255),
                               f'ZED left only | WiLoR | frame {index} | {seconds:.2f}s', 'One Euro detection boxes | +10% crops')
            if previous is not None:
                old_vertices, old_joints = mono_vertices, mono_joints
                old_subtitle = 'Monocular fallback' if obs >= 0 else 'No supported target hand'
                if previous['valid'][row]:
                    old_vertices = rig.project(previous['vertices'][row], 0, distorted=True)
                    old_joints = rig.project(previous['joints'][row], 0, distorted=True)
                    old_subtitle = 'Source multi-view fit | before One Euro pose filter' if summary.get('one_euro_pose') else 'Previous multi-view result | original camera weights and temporal fit'
                baseline = overlay(image, old_vertices, old_joints, faces, (20, 190, 255),
                                   f'Previous multi-view | frame {index} | {seconds:.2f}s', old_subtitle)
            if fused['valid'][row]:
                vertices = rig.project(fused['vertices'][row], 0, distorted=True)
                joints = rig.project(fused['joints'][row], 0, distorted=True)
                views = int(fused['used_views'][row].sum())
                subtitle = (f'{views} views | clearer fingers weighted more | finger articulation smoothing'
                            if summary.get('fusion_mode') == 'visibility' else f'{views} supporting views | shared hand shape | temporal fit')
                if summary.get('one_euro_pose'):
                    cfg = summary['one_euro_pose']
                    subtitle = f'{views} views | One Euro fingers {cfg["min_cutoff"]:g} Hz, beta {cfg["beta"]:g} | wrist rotation {cfg["wrist_min_cutoff"]:g} Hz'
                result = overlay(image, vertices, joints, faces, (60, 230, 100),
                                 f'Multi-view MANO | ZED left | frame {index} | {seconds:.2f}s',
                                 subtitle)
            else:
                result = overlay(image, mono_vertices, mono_joints, faces, (20, 190, 255),
                                 f'Multi-view MANO | ZED left | frame {index} | {seconds:.2f}s',
                                 'Monocular fallback: insufficient multi-view support' if obs >= 0 else 'No supported target hand')
            pair = np.hstack([cv2.resize(baseline, (960, 540)), cv2.resize(result, (960, 540))])
            elapsed_ns = timestamp - int(data['timestamp_ns'][0])
            if main_video is not None:
                main_video.add(result, elapsed_ns)
            if comparison is not None:
                comparison.add(pair, elapsed_ns)
            if closeup is not None:
                # Both panels use the SAME rectangle, based on the detection
                # box, so different smoothing cannot change the crop comparison.
                if obs >= 0:
                    box = data['expanded_box'][obs]
                    center = (box[:2] + box[2:]) * .5
                    size = int(np.clip(np.max(box[2:] - box[:2]) * 1.30, 360, 1080))
                else:
                    center, size = np.array([960., 540.]), 1080
                center = np.clip(center, [size / 2, size / 2], [1920 - size / 2, 1080 - size / 2])
                x, y = np.rint(center - size / 2).astype(int)
                zoom = np.zeros((720, 1280, 3), np.uint8)
                for column, panel in enumerate((baseline, result)):
                    zoom[64:704, column * 640:(column + 1) * 640] = cv2.resize(panel[y:y + size, x:x + size], (640, 640))
                left_label = 'Single-view WiLoR' if obs >= 0 else 'No supported hand'
                if previous is not None:
                    left_label = 'Previous multi-view' if previous['valid'][row] else ('Previous: single-view fallback' if obs >= 0 else 'No supported hand')
                right_label = 'One Euro pose filtering' if summary.get('one_euro_pose') else 'Multi-view fit'
                if not fused['valid'][row]:
                    right_label = 'Single-view fallback: unfiltered' if obs >= 0 else 'No supported hand'
                cv2.putText(zoom, left_label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 190, 255), 2)
                cv2.putText(zoom, right_label, (655, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (60, 230, 100) if fused['valid'][row] else (20, 190, 255), 2)
                cv2.putText(zoom, f'frame {index} | {seconds:.2f}s | same crop in both panels', (15, 53), cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1)
                closeup.add(zoom, elapsed_ns)
                if row in samples:
                    cv2.imwrite(str(args.output / f'closeup_{int(index):06d}.jpg'), zoom)
            if row in samples:
                path = args.output / f'comparison_{int(index):06d}.jpg'
                cv2.imwrite(str(path), pair)
                cv2.imwrite(str(args.output / f'left_{int(index):06d}.jpg'), result)
                sample_paths.append(path)
            count += 1
            if count % 250 == 0:
                print(f'Rendered {count}/{len(data["frame_index"])} frames', flush=True)
    finally:
        camera.close()
        for writer in writers:
            writer.close()
    sheet = [cv2.resize(cv2.imread(str(path)), (1280, 360)) for path in sample_paths]
    if sheet:
        cv2.imwrite(str(args.output / 'comparison_contact_sheet.jpg'), np.vstack(sheet))
    (args.output / 'render_summary.json').write_text(json.dumps(dict(frames=count, timestamp_source='native SVO IMAGE',
        timestamp_precision='microsecond video PTS; exact nanoseconds in NPZ', codec='H264',
        output='wilor_multiview_left.mp4', comparison='comparison_left.mp4',
        closeup='comparison_hand.mp4' if args.closeup else None,
        comparison_reference=str(args.compare_fit.resolve()) if args.compare_fit else 'monocular WiLoR'), indent=2))
    print(f'Rendered {count} frames into {args.output}', flush=True)


if __name__ == '__main__':
    main()
