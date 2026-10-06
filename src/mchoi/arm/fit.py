"""Metric 3D right shoulder/elbow/wrist from cached YOLO-pose views, anchored on the multiview MANO wrist.

Per view, the person whose right wrist is closest to the projected MANO wrist is selected. Elbow and shoulder are
triangulated with the calibrated rig when two or more views agree; otherwise they are placed on the single
camera ray at the median bone length from the parent joint (method 2). Output is in the ZED-left frame, metres.

Fallback (method 4) for frames where no camera sees the shoulder or elbow, e.g. takes without validated D455 timing:
the shoulder is held at a static prior -- the median of this take's observed shoulders when there are at least
--min-observed-shoulders, else --shoulder-prior -- and the elbow is solved by two-bone IK between shoulder and wrist,
bending towards gravity-down and the subject's right. These joints are plausible placements, not measurements.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mchoi.multiview import Rig, CAMERAS

SHOULDER, ELBOW, WRIST = 6, 8, 10  # COCO-17 right side
NAMES = ('shoulder', 'elbow', 'wrist')
MIN_CONF = .35
GATE_PX = 150


def ray(rig, cam, pixel):
    n = np.append((rig.undistort(pixel[None], cam)[0] - rig.K[cam][:2, 2]) / np.diag(rig.K[cam])[:2], 1.)
    d = rig.T[cam][:3, :3].T @ n
    return rig.centers[cam], d / np.linalg.norm(d)


def on_sphere(origin, direction, center, radius, reference):
    """Point on the ray at distance radius from center, choosing the root nearest to reference."""
    offset = origin - center
    b = offset @ direction
    disc = b * b - (offset @ offset - radius ** 2)
    if disc < 0:
        t = -b
        point = origin + t * direction
        return center + radius * (point - center) / max(np.linalg.norm(point - center), 1e-9)
    roots = [origin + t * direction for t in (-b - np.sqrt(disc), -b + np.sqrt(disc)) if t > .05]
    if not roots:
        return None
    return min(roots, key=lambda p: np.linalg.norm(p - reference))


def candidates(rig, pixels, weights, parent, j, length, t):
    """Both ray/sphere intersections from the best single view at frame t."""
    cams = np.flatnonzero(weights[t, :, j] > 0)
    if not len(cams) or not np.isfinite(parent[t, 0]):
        return []
    cam = cams[np.argmax(weights[t, cams, j])]
    origin, direction = ray(rig, cam, pixels[t, cam, j])
    offset = origin - parent[t]
    b = offset @ direction
    disc = b * b - (offset @ offset - length ** 2)
    if disc < 0:
        point = origin - b * direction
        return [parent[t] + length * (point - parent[t]) / max(np.linalg.norm(point - parent[t]), 1e-9)]
    return [origin + s * direction for s in (-b - np.sqrt(disc), -b + np.sqrt(disc)) if s > .05]


def single_view_chain(rig, pixels, weights, parent, j, length, root='far'):
    """Viterbi over the two ray/sphere roots per frame, minimizing joint travel between frames."""
    count = len(parent)
    options = [candidates(rig, pixels, weights, parent, j, length, t) for t in range(count)]
    chosen = [None] * count
    if root != 'continuity':
        # Candidates are ordered near-to-far along the ray; the far root is the one away from the viewing camera.
        return [o[-1 if root == 'far' else 0] if o else None for o in options]
    t = 0
    while t < count:
        if not options[t]:
            t += 1
            continue
        end = t
        while end + 1 < count and options[end + 1]:
            end += 1
        cost = [np.zeros(len(options[t]))]
        back = []
        for u in range(t + 1, end + 1):
            step = np.array([[np.linalg.norm(a - b) for b in options[u - 1]] for a in options[u]])
            total = step + cost[-1][None]
            back.append(np.argmin(total, axis=1))
            cost.append(total.min(axis=1))
        k = int(np.argmin(cost[-1]))
        for u in range(end, t - 1, -1):
            chosen[u] = options[u][k]
            if u > t:
                k = int(back[u - t - 1][k])
        t = end + 1
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm2d', type=Path)
    parser.add_argument('fit', type=Path, help='MANO fit directory (fused_hand.npz, calibration.json)')
    parser.add_argument('--output', type=Path, required=True, help='.npz path; must not exist')
    parser.add_argument('--threshold', type=float, default=25)
    parser.add_argument('--root', choices=['far', 'near', 'continuity'], default='far', help='Which ray/sphere intersection to take for single-view joints')
    parser.add_argument('--forearm', type=float, default=.27, help='Elbow-wrist length (m) when multi-view triangulation cannot measure it')
    parser.add_argument('--upper-arm', type=float, default=.30, help='Shoulder-elbow length (m), same fallback')
    parser.add_argument('--gravity', type=Path, help='estimate_zed_gravity.py output; enables the method-4 fallback')
    parser.add_argument('--shoulder-prior', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Static shoulder (ZED-left metres) when this take observes too few shoulders')
    parser.add_argument('--min-observed-shoulders', type=int, default=100)
    parser.add_argument('--forward-camera', default='d455', help='Camera the subject faces (sets the elbow bend direction)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    rig = Rig(json.loads((args.fit / 'calibration.json').read_text()))
    arm = np.load(args.arm2d)
    fit = np.load(args.fit / 'fused_hand.npz', allow_pickle=True)
    rows = {int(f): i for i, f in enumerate(fit['frame_index'])}
    frames = arm['frame_index']
    count = len(frames)
    pixels = np.zeros((count, 4, 3, 2))
    weights = np.zeros((count, 4, 3))
    anchor = np.full((count, 3), np.nan)
    for t, frame in enumerate(frames):
        row = rows[int(frame)]
        if fit['valid'][row]:
            anchor[t] = fit['joints'][row, 0]
        for cam in range(4):
            people = arm['keypoints'][t, cam]
            if not arm['view_present'][t, cam] or not np.isfinite(anchor[t, 0]):
                continue
            target = rig.project(anchor[t][None], cam, distorted=True)[0]
            wrists = people[:, WRIST]
            dist = np.where(wrists[:, 2] > MIN_CONF, np.linalg.norm(wrists[:, :2] - target, axis=1), np.inf)
            person = int(np.argmin(dist))
            if dist[person] > GATE_PX:
                continue
            for j, joint in enumerate((SHOULDER, ELBOW, WRIST)):
                x, y, c = people[person, joint]
                if c >= MIN_CONF and x > 0 and y > 0:
                    pixels[t, cam, j] = rig.undistort(np.array([[x, y]]), cam)[0]
                    weights[t, cam, j] = c ** 2
    # Triangulate with the project's robust consensus (Rig.triangulate expects 4 views x joints).
    points = np.full((count, 3, 3), np.nan)
    masks = np.zeros((count, 4, 3), bool)
    for t in range(count):
        points[t], masks[t], _ = rig.triangulate(pixels[t], weights[t], args.threshold)
    wrist = np.where(np.isfinite(anchor[:, :1]), anchor, points[:, 2])
    result = np.full((count, 3, 3), np.nan)
    method = np.zeros((count, 3), np.int8)  # 0 none, 1 triangulated, 2 single-view on bone sphere, 3 MANO anchor
    result[:, 2], method[:, 2] = wrist, np.where(np.isfinite(wrist[:, 0]), 3, 0)
    lengths = {}
    for j, parent in ((1, 2), (0, 1)):
        multi = masks[:, :, j].sum(1) >= 2
        known = multi & np.isfinite(result[:, parent, 0]) & np.isfinite(points[:, j, 0])
        samples = np.linalg.norm(points[known, j] - result[known, parent], axis=1)
        if len(samples) >= 20:
            length = float(np.median(samples))
        else:
            length = args.forearm if j == 1 else args.upper_arm  # unobservable from one view: anthropometric prior
        lengths[j] = length
        if length is None:
            continue
        chosen = single_view_chain(rig, pixels, weights, result[:, parent], j, length, args.root)
        for t in range(count):
            tri = points[t, j]
            if multi[t] and np.isfinite(tri[0]) and np.isfinite(result[t, parent, 0])                     and abs(np.linalg.norm(tri - result[t, parent]) / length - 1) < .25:
                result[t, j], method[t, j] = tri, 1
            elif chosen[t] is not None:
                result[t, j], method[t, j] = chosen[t], 2
    prior, prior_source = None, None
    observed = method[:, 0] > 0
    if observed.sum() >= args.min_observed_shoulders:
        prior, prior_source = np.median(result[observed, 0], 0), f'median of {int(observed.sum())} observed shoulders'
    elif args.shoulder_prior:
        prior, prior_source = np.asarray(args.shoulder_prior), 'command line'
    if prior is not None and args.gravity:
        up = np.asarray(json.loads(args.gravity.read_text())['up_in_zed_left'])
        forward = -rig.T[CAMERAS.index(args.forward_camera)][2, :3]
        forward = (forward - (forward @ up) * up) / np.linalg.norm(forward - (forward @ up) * up)
        pole = -up + .5 * np.cross(forward, up)  # elbow bends down and to the subject's right
        upper, fore = lengths[0], lengths[1]
        for t in range(count):
            w = result[t, 2]
            if not np.isfinite(w[0]):
                continue
            if not np.isfinite(result[t, 0, 0]):
                e = result[t, 1]
                result[t, 0] = e + upper * (prior - e) / np.linalg.norm(prior - e) if np.isfinite(e[0]) else prior
                method[t, 0] = 4
            if not np.isfinite(result[t, 1, 0]):
                s = result[t, 0]
                d = np.linalg.norm(w - s)
                u = (w - s) / d
                if d >= upper + fore or d <= abs(upper - fore):
                    result[t, 1] = s + upper * u
                else:
                    a = (upper ** 2 - fore ** 2 + d ** 2) / (2 * d)
                    p = pole - (pole @ u) * u
                    result[t, 1] = s + a * u + np.sqrt(upper ** 2 - a ** 2) * p / np.linalg.norm(p)
                method[t, 1] = 4
    np.savez_compressed(args.output, frame_index=frames, timestamp_ns=arm['timestamp_ns'], joints=result, method=method,
                        views=masks.sum(1), joint_names=np.asarray(NAMES), forearm_length=np.nan if lengths[1] is None else lengths[1],
                        upper_arm_length=np.nan if lengths[0] is None else lengths[0],
                        shoulder_prior=np.full(3, np.nan) if prior is None else prior)
    report = dict(frames=count, forearm_m=lengths[1], upper_arm_m=lengths[0],
                  shoulder_prior=None if prior is None else prior.tolist(), shoulder_prior_source=prior_source, **{
        f'{n}_{label}': float((method[:, j] == m).mean()) for j, n in enumerate(NAMES)
        for label, m in (('triangulated', 1), ('single_view', 2), ('anchor', 3), ('fallback', 4))})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
