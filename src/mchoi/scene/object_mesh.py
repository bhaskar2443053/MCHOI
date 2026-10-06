"""Estimate the rigid transform from an object mesh (STL, metres) to its ArUco marker frame by silhouette fitting.

For sampled frames with a marker pose, the mesh is posed through the candidate marker<-mesh transform, projected into
every calibrated view (convex hull of its vertices) and compared with a colour mask of the object (HSV range). The
mean IoU inside a margin around the projection is maximised: a coarse grid (on a subset of views at quarter
resolution) over which side of the mesh carries the marker (6 principal directions), which way is marker-up (4),
height and sideways position on that side (marker placed on the outermost surface point there) and roll, then
Nelder-Mead over all six parameters on every view from the --refine best placements. A hue range with
low > high wraps around 0 (red objects, e.g. --hsv-low 170 100 60 --hsv-high 10 255 255). Output: T_marker_from_mesh (4x4, p_marker = R p_mesh + t).

With --gravity, a rest prior also penalises transforms that do not stand the mesh upright (mesh +Z along gravity up,
base between 1 cm below and 4 cm above the table plane, allowing for coasters) in --rest-frame, by default the first frame with a marker pose; YCB meshes are modelled in
that resting pose. Without it, near-symmetric objects seen from similar directions can fit equally well tipped over.

Frames: by default --sample-frames frames with a multi-view marker pose, evenly spread over the take, are decoded with
mchoi.io.stream_four_views (run this module with the ZED SDK environment). --frames-npz reuses saved images instead.
The default HSV range selects the yellow mustard bottle; set it for other objects.
"""
import argparse
import csv
import json
from pathlib import Path
import struct
import subprocess
import sys

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import trimesh

from mchoi.multiview import Rig
from mchoi.paths import ARUCO_DIR, timing_path
from mchoi.io.zed import read_exact

CAMERAS = ('zed_left', 'zed_right', 'd435', 'd455')


def grab_frames(take, calibration, timing, frames):
    """Time-aligned native four-view BGR images at the given ZED frame indexes."""
    images = np.zeros((len(frames), 4, 1080, 1920, 3), np.uint8)
    present = np.zeros((len(frames), 4), bool)
    for i, frame in enumerate(frames):
        process = subprocess.Popen([sys.executable, '-u', '-m', 'mchoi.io.stream_four_views', '--take', str(take),
                                    '--calibration', str(calibration), '--timing', str(timing),
                                    '--start-frame', str(frame), '--max-frames', '1', '--skip-unvalidated-timing'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        pipe = process.stdout
        if read_exact(pipe, 4) != b'W4V1':
            raise ValueError('Bad decoder header')

        def message():
            length, = struct.unpack('<I', read_exact(pipe, 4))
            return json.loads(read_exact(pipe, length))
        message()
        group = message()
        if group['frame_index'] != frame:
            raise ValueError(f'Decoder returned frame {group["frame_index"]} instead of {frame}')
        for c, cam in enumerate(CAMERAS):
            if cam in group['matches']:
                images[i, c] = np.frombuffer(read_exact(pipe, 1920 * 1080 * 3), np.uint8).reshape(1080, 1920, 3)
                present[i, c] = True
        process.wait(timeout=60)
        print(f'decoded frame {frame}: {int(present[i].sum())} views', flush=True)
    return dict(frames=np.asarray(frames), images=images, present=present)


def marker_poses(poses_csv, marker_id):
    out = {}
    for row in csv.DictReader(open(poses_csv, newline='')):
        if row['zed_left_frame_index'] == '' or row['pose_valid'] != '1' or int(row['marker_id']) != marker_id:
            continue
        R = np.array([[float(row[f'R{i}{j}']) for j in range(3)] for i in range(3)])
        out[int(row['zed_left_frame_index'])] = (R, np.array([float(row[k]) for k in ('tx_m', 'ty_m', 'tz_m')]), row['pose_status'])
    return out


def transform(params):
    """Affine marker<-mesh map: rotation vector, translation, then optional log-scales (none, uniform or per axis)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix() * np.exp(np.broadcast_to(params[6:], 3) if len(params) > 6 else np.zeros(3))
    T[:3, 3] = params[3:6]
    return T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--take', type=Path, required=True)
    parser.add_argument('--aruco', type=Path, help=f'ArUco output folder (default <take>/{ARUCO_DIR})')
    parser.add_argument('--sample-frames', type=int, default=12)
    parser.add_argument('--gravity', type=Path, help='estimate_zed_gravity.py output; enables the rest prior')
    parser.add_argument('--rest-frame', type=int, help='ZED frame where the object rests upright on the table (default: first marker pose)')
    parser.add_argument('--rest-weight', type=float, default=.3, help='IoU-equivalent penalty for a fully tilted or lifted rest pose')
    parser.add_argument('--fit-scale', choices=('none', 'uniform', 'axes'), default='none',
                        help='Also fit the mesh size: one factor, or one per mesh axis (e.g. a larger box of the same product)')
    parser.add_argument('--save-frames', type=Path, help='Save the decoded images (.npz) for reuse with --frames-npz')
    parser.add_argument('--refine', type=int, default=4, help='Number of best initial placements refined with Nelder-Mead')
    parser.add_argument('--timing', type=Path, help='RealSense timing file (default: wilor_timing if present, else the ArUco estimate)')
    parser.add_argument('--frames-npz', type=Path, help='Saved images instead of decoding: frames, images (N,4,H,W,3 BGR), present (N,4)')
    parser.add_argument('--marker', type=int, default=2)
    parser.add_argument('--hsv-low', type=int, nargs=3, default=(18, 110, 90))
    parser.add_argument('--hsv-high', type=int, nargs=3, default=(38, 255, 255))
    parser.add_argument('--output', type=Path, required=True, help='.json path; must not exist')
    parser.add_argument('--preview', type=Path, help='Optional .jpg contact sheet of the fitted silhouettes')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    aruco = args.aruco or args.take / ARUCO_DIR
    calibration, poses_csv = aruco / 'calibration_used.json', aruco / 'poses.csv'
    rig = Rig(json.loads(calibration.read_text()))
    mesh = trimesh.load(args.mesh)
    rng = np.random.default_rng(0)
    verts = mesh.vertices[rng.choice(len(mesh.vertices), min(3000, len(mesh.vertices)), replace=False)]
    poses = marker_poses(poses_csv, args.marker)
    if args.frames_npz:
        data = np.load(args.frames_npz)
    else:
        multiview = sorted(f for f, (_, _, status) in poses.items() if status == 'multiview')
        if not multiview:
            raise ValueError(f'Marker {args.marker} has no multi-view poses')
        picks = sorted({multiview[int(k)] for k in np.linspace(0, len(multiview) - 1, args.sample_frames)})
        data = grab_frames(args.take, calibration, args.timing or timing_path(args.take), picks)
        if args.save_frames:
            np.savez(args.save_frames, **data)
    samples = []  # (R_cm, t_cm, cam, mask) per usable view
    for i, frame in enumerate(data['frames']):
        if int(frame) not in poses:
            continue
        R, t, _ = poses[int(frame)]
        for cam in range(4):
            if data['present'][i, cam]:
                hsv = cv2.cvtColor(cv2.resize(data['images'][i, cam], (960, 540)), cv2.COLOR_BGR2HSV)
                low, high = np.array(args.hsv_low), np.array(args.hsv_high)
                if low[0] <= high[0]:
                    mask = cv2.inRange(hsv, low, high) > 0
                else:  # hue wraps around 0
                    mask = (cv2.inRange(hsv, low, np.r_[179, high[1:]]) > 0) | (cv2.inRange(hsv, np.r_[0, low[1:]], high) > 0)
                samples.append((R, t, cam, mask, i))
    if not samples:
        raise ValueError('No sampled frame has a marker pose')
    coarse = [(R, t, cam, cv2.resize(mask.astype(np.uint8), (480, 270), interpolation=cv2.INTER_NEAREST) > 0, i)
              for R, t, cam, mask, i in samples[::max(1, len(samples) // 16)]]

    def silhouettes(T, views=None):
        views = samples if views is None else views
        for R, t, cam, mask, i in views:
            pts = (verts @ T[:3, :3].T + T[:3, 3]) @ R.T + t
            depth = pts @ rig.T[cam][:3, :3].T + rig.T[cam][:3, 3]
            if (depth[:, 2] < .05).any():
                yield None, mask, i, cam
                continue
            px = rig.project(pts, cam, distorted=True) * (mask.shape[1] / 1920)
            if not np.isfinite(px).all() or np.abs(px).max() > 5000:
                yield None, mask, i, cam
                continue
            sil = np.zeros(mask.shape, np.uint8)
            cv2.fillConvexPoly(sil, cv2.convexHull(px.astype(np.int32)), 1)
            yield sil.astype(bool), mask, i, cam

    def score(T, detail=False, views=None):
        values = []
        for sil, mask, i, cam in silhouettes(T, views):
            if sil is None or not sil.any():
                values.append(0.)
                continue
            ys, xs = np.nonzero(sil)
            h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
            y0, y1 = max(ys.min() - h // 2, 0), ys.max() + h // 2
            x0, x1 = max(xs.min() - w // 2, 0), xs.max() + w // 2
            s, m = sil[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            union = (s | m).sum()
            values.append((s & m).sum() / union if union else 0.)
        if detail:
            return values
        return float(np.mean(values)) - rest_penalty(T)

    # Rest prior: in the rest frame the object stands on the table in its modelled (YCB: z-up, base at z=0) pose.
    # Silhouettes alone cannot tell e.g. an upright mug from one lying along the viewing direction when only the two
    # ZED views are available.
    rest = None
    if args.gravity:
        gravity = json.loads(args.gravity.read_text())
        gravity_up, plane = np.asarray(gravity['up_in_zed_left']), np.asarray(gravity['support_plane'])
        rest_frame = args.rest_frame if args.rest_frame is not None else int(min(poses))
        if rest_frame not in poses:
            raise ValueError(f'Marker {args.marker} has no pose in rest frame {rest_frame}')
        rest = poses[rest_frame][:2]

    def rest_penalty(T):
        if rest is None:
            return 0.
        R, t = rest
        R_cam_mesh = R @ T[:3, :3]
        tilt = np.degrees(np.arccos(np.clip(R_cam_mesh[:, 2] @ gravity_up / np.linalg.norm(R_cam_mesh[:, 2]), -1, 1)))
        lowest = ((mesh.vertices @ R_cam_mesh.T + R @ T[:3, 3] + t) @ plane[:3]).min() + plane[3]
        gap = max(0., -.01 - lowest, lowest - .04)  # objects may stand on a coaster or mat up to ~4 cm thick
        return args.rest_weight * (min(tilt / 30, 1) + min(gap / .03, 1)) / 2

    # Initial guesses: the marker lies on the outer surface along one of the six principal directions (its Z = that
    # outward normal), with marker +Y along one of the four perpendicular principal directions.
    centre_of_mass = mesh.vertices.mean(0)
    centred = mesh.vertices - centre_of_mass
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    directions = [sign * a for a in axes for sign in (1, -1)]
    n_scale = dict(none=0, uniform=1, axes=3)[args.fit_scale]
    candidates = []
    cell = .012
    for normal in directions:
        for up in directions:
            if abs(up @ normal) > .5:
                continue
            side = np.cross(up, normal)
            R_ms = np.column_stack([side, up, normal]).T  # rows: marker axes in mesh coordinates
            along_up, along_side, along_normal = centred @ up, centred @ side, centred @ normal
            for height in np.linspace(.85 * along_up.min(), .85 * along_up.max(), 7):
                for lateral in np.linspace(.85 * along_side.min(), .85 * along_side.max(), 5):
                    # Marker centre on the outermost surface point near this (height, lateral) position.
                    near = (np.abs(along_up - height) < cell) & (np.abs(along_side - lateral) < cell)
                    if not near.any():
                        continue
                    centre = centre_of_mass + up * height + side * lateral + normal * along_normal[near].max()
                    for roll in np.radians([-30, -15, 0, 15, 30]):
                        Rr = Rotation.from_rotvec([0, 0, roll]).as_matrix() @ R_ms
                        params = np.r_[Rotation.from_matrix(Rr).as_rotvec(), -Rr @ centre, np.zeros(n_scale)]
                        candidates.append((score(transform(params), views=coarse), params))
    candidates.sort(key=lambda c: -c[0])
    print(f'{len(candidates)} initial placements; best coarse IoU {candidates[0][0]:.3f}', flush=True)
    refined = []
    for coarse_iou, start in candidates[:args.refine]:
        result = minimize(lambda p: -score(transform(p)), start, method='Nelder-Mead',
                          options=dict(maxiter=1500, xatol=1e-4, fatol=1e-4, initial_simplex=start + np.vstack([np.zeros(6 + n_scale), np.diag([.08] * 3 + [.01] * 3 + [.1] * n_scale)])))
        refined.append((-result.fun, result.x, coarse_iou))
        print(f'  refined coarse {coarse_iou:.3f} -> IoU {-result.fun:.3f}', flush=True)
    best_iou, best_x, grid_iou = max(refined, key=lambda r: r[0])
    best = (grid_iou, best_x)
    T = transform(best_x)
    per_view = score(T, detail=True)
    scale = np.exp(np.broadcast_to(best_x[6:], 3)) if n_scale else np.ones(3)
    rigid = np.eye(4)
    rigid[:3, :3], rigid[:3, 3] = T[:3, :3] / scale, T[:3, 3]
    report = dict(mesh=str(args.mesh), marker=args.marker, T_marker_from_mesh=rigid.tolist(),
                  T_mesh_from_marker=np.linalg.inv(rigid).tolist(), scale=scale.tolist(), fit_scale=args.fit_scale,
                  marker_origin_in_mesh_m=(np.linalg.inv(rigid)[:3, 3] / scale).tolist(),
                  mean_iou=float(np.mean(per_view)), median_iou=float(np.median(per_view)), grid_iou=best[0], views=len(per_view),
                  frames=[int(data['frames'][s[4]]) for s in samples[::1]][::4],
                  rest_frame=None if rest is None else rest_frame, rest_penalty=rest_penalty(T),
                  rest_tilt_deg=None if rest is None else float(np.degrees(np.arccos(np.clip((rest[0] @ rigid[:3, :3])[:, 2] @ gravity_up, -1, 1)))),
                  convention='p_marker = R (scale * p_mesh) + t, R and t from T_marker_from_mesh, scale per mesh axis; '
                             'mesh in its own units (metres); marker X right, Y up, Z out of face')
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))
    if args.preview:
        tiles = []
        for (sil, mask, i, cam), v in zip(silhouettes(T), per_view):
            img = cv2.resize(data['images'][i, cam], (960, 540))
            if sil is not None:
                contours, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
            cv2.putText(img, f'frame {int(data["frames"][i])} cam {cam} IoU {v:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
            tiles.append(cv2.resize(img, (480, 270)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.preview), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))


if __name__ == '__main__':
    main()
