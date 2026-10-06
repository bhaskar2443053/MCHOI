"""Render the exported hand/arm/object poses over all four native views (2x2, 1920x1080 H.264).

Reads the Mocap file and its sidecar .json (world frame), maps the arm bones and object pose back to each camera,
and draws: the MANO hand mesh from the fit (skin), the arm chain clavicle-shoulder-elbow-wrist (orange), and the
object mesh at the exported pose (yellow), each translucent with outlines. Because the arm and object are read back
from the Mocap file, a correct overlay also validates the export. Layers are painted object, arm, hand without
occlusion reasoning. Frame presentation times are the source ZED times. Frames absent from the Mocap file show the
camera image with a 'not exported' caption.
"""
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
from scipy.spatial.transform import Rotation
import trimesh

from mchoi.multiview import Rig, CAMERAS
from mchoi.paths import calibration_path, timing_path
from mchoi.io.zed import read_exact

ARM = ('clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r')
SKIN, ARM_COLOR, OBJECT_COLOR = (118, 154, 210), (40, 120, 235), (0, 200, 240)  # BGR


def parse_mocap(path):
    rows = {}
    for line in open(path):
        if not line[:1].isdigit():
            continue
        f = line.strip().split('|')
        i, bones = 11, {}
        for _ in range(int(f[10])):
            bones[f[i]] = np.array(f[i + 1:i + 4], float)
            i += 8
        obj = np.array(f[i + 2:i + 12], float) if int(f[i]) else None  # loc, quat, scale
        rows[int(f[1])] = (bones, obj)
    return rows


def decimate(mesh, cell):
    keys = np.floor(mesh.vertices / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    verts = np.zeros((inverse.max() + 1, 3))
    np.add.at(verts, inverse, mesh.vertices)
    verts /= np.bincount(inverse)[:, None]
    faces = inverse[mesh.faces]
    return verts, faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]


def project(rig, cam, points):
    """Distorted pixels and a mask of points at least 5 cm in front of the camera."""
    depth = points @ rig.T[cam][:3, :3].T + rig.T[cam][:3, 3]
    front = depth[:, 2] > .05
    px = np.full((len(points), 2), np.nan)
    if front.any():
        px[front] = rig.project(np.ascontiguousarray(points[front], float), cam, distorted=True)
    return np.clip(np.nan_to_num(px, nan=-1e4), -1e4, 1e4), front


def fill_mesh(layer, edges, px, front, faces, color):
    keep = front[faces].all(1)
    if not keep.any():
        return False
    tris = px[faces[keep]].astype(np.int32)
    cv2.fillPoly(layer, list(tris), color, cv2.LINE_AA)
    cv2.fillPoly(edges, list(tris), 255)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mocap', type=Path)
    parser.add_argument('--take', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True, help='MANO fit directory used for the export')
    parser.add_argument('--mesh', type=Path, help='Object STL (metres); omit for a take without an object')
    parser.add_argument('--sidecar', type=Path, help='export_mocap.py .json (default: next to the Mocap file)')
    parser.add_argument('--output', type=Path, required=True, help='.mp4 path; must not exist')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=.45)
    parser.add_argument('--timing', type=Path, help='RealSense timing file (default: wilor_timing if present, else the ArUco estimate)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    rig = Rig(json.loads(calibration_path(args.take).read_text()))
    world = json.loads((args.sidecar or args.mocap.with_suffix('.json')).read_text())['world']
    A = np.array([world['forward_in_zed_left'], world['right_in_zed_left'], world['up_in_zed_left']])
    origin = np.array(world['origin_in_zed_left_m'])
    to_cam = lambda w: np.asarray(w) / 100 @ A + origin  # A is orthonormal
    rows = parse_mocap(args.mocap)
    fit = np.load(args.fit / 'fused_hand.npz', allow_pickle=True)
    fit_rows = {int(f): r for r, f in enumerate(fit['frame_index'])}
    hand_faces = np.load(args.fit / 'mano_faces.npy')
    if args.mesh:
        obj_verts, obj_faces = decimate(trimesh.load(args.mesh), .004)
    else:  # an object pose without a mesh is drawn as a 4 cm cube proxy
        proxy = trimesh.creation.box(extents=(.04, .04, .04))
        obj_verts, obj_faces = proxy.vertices, proxy.faces
    obj_local = 100 * obj_verts * [1, -1, 1]  # mesh metres -> world-cm local axes, as written in the Mocap file

    process = subprocess.Popen([sys.executable, '-u', '-m', 'mchoi.io.stream_four_views', '--take', str(args.take),
                                '--calibration', str(calibration_path(args.take)), '--timing', str(args.timing or timing_path(args.take)),
                                '--start-frame', str(args.start_frame), '--max-frames', str(args.max_frames), '--skip-unvalidated-timing'],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    pipe = process.stdout

    def message():
        length, = struct.unpack('<I', read_exact(pipe, 4))
        return json.loads(read_exact(pipe, length))

    if read_exact(pipe, 4) != b'W4V1':
        raise ValueError('Bad decoder header')
    count = message()['count']
    container = av.open(str(args.output), 'w', options={'movflags': 'faststart'})
    stream = container.add_stream('libx264', rate=30)
    stream.width, stream.height, stream.pix_fmt = 1920, 1080, 'yuv420p'
    stream.options = {'crf': '19', 'preset': 'veryfast'}
    stream.time_base = stream.codec_context.time_base = Fraction(1, 1_000_000)
    origin_ns, drawn = None, 0
    for step in range(count):
        group = message()
        frame = group['frame_index']
        origin_ns = group['timestamp_ns'] if origin_ns is None else origin_ns
        row = rows.get(frame)
        hand = fit['vertices'][fit_rows[frame]] if row and frame in fit_rows else None
        obj = None
        if row and row[1] is not None:
            R = Rotation.from_quat(row[1][3:7]).as_matrix()
            obj = to_cam(row[1][:3] + (obj_local * row[1][7:10]) @ R.T)
        arm = to_cam(np.array([row[0][n] for n in ARM])) if row else None
        panels = []
        for c, cam in enumerate(CAMERAS):
            if cam not in group['matches']:
                panels.append(np.zeros((540, 960, 3), np.uint8))
                continue
            image = np.frombuffer(read_exact(pipe, 1920 * 1080 * 3), np.uint8).reshape(1080, 1920, 3).copy()
            if row:
                for points, faces, color in ((obj, obj_faces, OBJECT_COLOR), (hand, hand_faces, SKIN)):
                    if points is None:
                        continue
                    layer, edges = image.copy(), np.zeros(image.shape[:2], np.uint8)
                    px, front = project(rig, c, points)
                    if fill_mesh(layer, edges, px, front, faces, color):
                        image = cv2.addWeighted(layer, args.alpha, image, 1 - args.alpha, 0)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(image, contours, -1, color, 3, cv2.LINE_AA)
                px, front = project(rig, c, arm)
                for a, b in zip(range(3), range(1, 4)):
                    if front[a] and front[b]:
                        cv2.line(image, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), ARM_COLOR, 10, cv2.LINE_AA)
                for k in range(4):
                    if front[k]:
                        cv2.circle(image, tuple(int(v) for v in px[k]), 13, ARM_COLOR, -1, cv2.LINE_AA)
                        cv2.circle(image, tuple(int(v) for v in px[k]), 13, (255, 255, 255), 2, cv2.LINE_AA)
            panel = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.putText(panel, cam, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
            panels.append(panel)
        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        seconds = (group['timestamp_ns'] - origin_ns) / 1e9
        status = 'not exported (no hand fit or arm)' if not row else ('hand + arm + object' if obj is not None else 'hand + arm, no object pose')
        drawn += bool(row)
        cv2.rectangle(grid, (0, 1040), (1920, 1080), (0, 0, 0), -1)
        cv2.putText(grid, f'{args.mocap.name} | frame {frame} | {seconds:.2f}s | {status} | skin: MANO hand, orange: arm, yellow: object',
                    (16, 1068), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)
        video = av.VideoFrame.from_ndarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB), format='rgb24')
        video.pts, video.time_base = (group['timestamp_ns'] - origin_ns) // 1000, stream.time_base
        for packet in stream.encode(video):
            container.mux(packet)
        if step % 250 == 0:
            print(f'{step + 1}/{count} frames', flush=True)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    process.wait(timeout=60)
    print(json.dumps(dict(output=str(args.output), frames=count, frames_with_pose=drawn)))


if __name__ == '__main__':
    main()
