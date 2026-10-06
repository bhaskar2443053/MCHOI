"""Render the fitted right-arm joints over all four native views (2x2, 1920x1080 H.264)."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess
import sys

import av
import cv2
import numpy as np

from mchoi.multiview import Rig, CAMERAS
from mchoi.paths import calibration_path, timing_path
from mchoi.io.zed import read_exact

COLORS = ((255, 120, 0), (0, 220, 255), (60, 60, 255))  # BGR: shoulder, elbow, wrist
LABELS = ('shoulder', 'elbow', 'wrist')
METHOD = {1: 'triangulated', 2: 'single-view + bone length', 3: 'MANO anchor', 4: 'prior + IK fallback'}


def draw_raw(image, rig, c, joints):
    ok = np.isfinite(joints[:, 0]) & ((joints @ rig.T[c][:3, :3].T + rig.T[c][:3, 3])[:, 2] > .05)
    if not ok.any():
        return
    px = np.full((3, 2), np.nan)
    px[ok] = np.clip(rig.project(joints[ok], c, distorted=True), -1e4, 1e4)
    for a, b in ((0, 1), (1, 2)):
        if ok[a] and ok[b]:
            cv2.line(image, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), (0, 140, 255), 3, cv2.LINE_AA)
    for k in range(2):
        if ok[k]:
            cv2.circle(image, tuple(int(v) for v in px[k]), 8, (0, 140, 255), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm3d', type=Path)
    parser.add_argument('--take', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='.mp4 path; must not exist')
    parser.add_argument('--timing', type=Path, help='RealSense timing file (default: wilor_timing if present, else the ArUco estimate)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    arm = np.load(args.arm3d)
    calibration = calibration_path(args.take)
    timing = args.timing or timing_path(args.take)
    rig = Rig(json.loads(calibration.read_text()))
    process = subprocess.Popen([sys.executable, '-u', '-m', 'mchoi.io.stream_four_views', '--take', str(args.take),
                                '--calibration', str(calibration), '--timing', str(timing),
                                '--start-frame', str(int(arm['frame_index'][0])), '--max-frames', str(len(arm['frame_index'])), '--skip-unvalidated-timing'],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pipe = process.stdout

    def message():
        length, = struct.unpack('<I', read_exact(pipe, 4))
        return json.loads(read_exact(pipe, length))

    if read_exact(pipe, 4) != b'W4V1':
        raise ValueError('Bad decoder header')
    message()
    container = av.open(str(args.output), 'w', options={'movflags': 'faststart'})
    stream = container.add_stream('libx264', rate=30)
    stream.width, stream.height, stream.pix_fmt = 1920, 1080, 'yuv420p'
    stream.options = {'crf': '19', 'preset': 'veryfast'}
    stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
    origin = int(arm['timestamp_ns'][0])
    for t in range(len(arm['frame_index'])):
        group = message()
        panels = []
        joints, method = arm['joints'][t], arm['method'][t]
        raw = arm['raw_joints'][t] if 'raw_joints' in arm else None
        for c, cam in enumerate(CAMERAS):
            if cam not in group['matches']:
                panels.append(np.zeros((540, 960, 3), np.uint8))
                continue
            image = np.frombuffer(read_exact(pipe, 1920 * 1080 * 3), np.uint8).reshape(1080, 1920, 3).copy()
            if raw is not None:  # unfiltered fit, thin orange, beneath the One Euro result
                draw_raw(image, rig, c, raw)
            ok = np.isfinite(joints[:, 0])
            if ok.any():
                px = np.full((3, 2), np.nan)
                px[ok] = rig.project(joints[ok], c, distorted=True)
                px = np.clip(np.nan_to_num(px, nan=-1e4), -1e4, 1e4)
                behind = (joints @ rig.T[c][:3, :3].T + rig.T[c][:3, 3])[:, 2] < .05
                for a, b in ((0, 1), (1, 2)):
                    if ok[a] and ok[b] and not behind[a] and not behind[b]:
                        cv2.line(image, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), (80, 255, 80), 6, cv2.LINE_AA)
                for k in range(3):
                    if ok[k] and not behind[k]:
                        cv2.circle(image, tuple(int(v) for v in px[k]), 14, COLORS[k], -1, cv2.LINE_AA)
                        cv2.circle(image, tuple(int(v) for v in px[k]), 14, (255, 255, 255), 2, cv2.LINE_AA)
            panel = cv2.resize(image, (960, 540))
            cv2.putText(panel, cam, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
            panels.append(panel)
        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        seconds = (int(arm['timestamp_ns'][t]) - origin) / 1e9
        text = ('orange: raw | green: One Euro | ' if 'raw_joints' in arm else '') + f"right arm | frame {int(arm['frame_index'][t])} | {seconds:.2f}s | " + (
            'no valid hand fit' if not np.isfinite(joints[2, 0]) else
            ', '.join(f"{LABELS[k]}: {METHOD.get(int(method[k]), 'none')}" for k in (1, 2)))
        cv2.rectangle(grid, (0, 1040), (1920, 1080), (0, 0, 0), -1)
        cv2.putText(grid, text, (16, 1068), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        frame = av.VideoFrame.from_ndarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB), format='rgb24')
        frame.pts = (int(arm['timestamp_ns'][t]) - origin) // 1000  # source elapsed time, microseconds
        frame.time_base = stream.time_base
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    process.wait(timeout=30)


if __name__ == '__main__':
    main()
