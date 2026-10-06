"""Export the right arm, MANO hand and ArUco object as a version-4 Mocap text file (the 001_Mug_Mocap.txt layout).

World frame (Unreal convention): centimetres, left-handed, X forward (direction the subject faces, i.e. towards the
D455 opposite them, made horizontal), Y right, Z up (ZED IMU gravity). Origin: the support plane (table top) directly
below the ZED-left camera, raised by --plane-height-cm. Bone axes follow the reference file: local -X points from each
bone to its child; upper arm and forearm Z is the elbow hinge axis; hand and finger Z points from the little-finger
side to the index side (thumb Z is the palm normal); wrist_inner/outer are the hand bone rotated +-90 deg about X.

The camera/head is the static ZED-left camera. Clavicle_r is placed --clavicle-cm medially of the fitted shoulder
(along world -Y); it is not observed. The object is posed from ArUco marker --object-markers[0] (default 2); the second marker
(default 1) alone is mapped through their median rigid offset. With --mesh-transform (fit_object_mesh_offset.py) the object pose is the mesh (STL) origin, and
a mesh vertex v (metres) lands at LOC + rotate(ROT, 100 * (SX v.x, -SY v.y, SZ v.z)) in world cm, with the fitted mesh
scale in SCALE_X/Y/Z (1 unless the fit estimated a size) (Y negated for the
left-handed world, as an FBX/STL importer does). Without it, the origin is the midpoint of the two markers, not a mesh origin.
The primary-marker pose is One Euro filtered before the mesh offset is applied (position per axis in m/s, rotation on SO(3)
in rad/s), resetting after source gaps over --object-reset-gap; it is not reset when the source switches between
the two markers, so the small offset between their estimates is smoothed instead of jumping. --object-min-cutoff 0
disables it.
Only frames with a valid hand fit and fitted shoulder/elbow are written; FRAME_NUMBER is the ZED-left frame index.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


import numpy as np
from scipy.spatial.transform import Rotation

from mchoi.hand.detection import OneEuroFilter
from mchoi.hand.rotation_filter import OneEuroRotationFilter

FINGERS = (('thumb', 1), ('index', 5), ('middle', 9), ('ring', 13), ('pinky', 17))
WRIST_OFFSET_CM = np.array([.717, 1.867, 0.])  # wrist_inner_r in hand_r local axes, median of the reference file
FLIP_Y = np.diag([1., -1., 1.])  # right-handed object axes -> left-handed world


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def frame_from(x_dir, z_hint):
    """Rotation with local X = x_dir, Z = z_hint made orthogonal to X, Y = Z x X (numeric, as in the reference)."""
    x = unit(x_dir)
    z = unit(z_hint - (z_hint @ x) * x)
    return np.column_stack([x, np.cross(z, x), z])


def quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()  # x, y, z, w


def fmt(values):
    return '|'.join(f'{v:.6f}' for v in values)


def second_difference(poses, frames, stamps, gap):
    """Median and p95 of the position (mm) and rotation (deg) second differences over consecutive frames."""
    pos, rot = [], []
    for k in range(1, len(frames) - 1):
        f = frames[k - 1:k + 2]
        if not all(x in poses for x in f) or stamps[k + 1] - stamps[k - 1] > 2 * gap:
            continue
        (R0, t0), (R1, t1), (R2, t2) = (poses[x] for x in f)
        pos.append(np.linalg.norm(t0 - 2 * t1 + t2) * 1000)
        rot.append(np.degrees(Rotation.from_matrix((R2 @ R1.T) @ (R1 @ R0.T).T).magnitude()))
    return dict(position_mm_median=float(np.median(pos)), position_mm_p95=float(np.percentile(pos, 95)),
                rotation_deg_median=float(np.median(rot)), rotation_deg_p95=float(np.percentile(rot, 95)))


def load_object(poses_csv, frames, stamps, local=None, filter_settings=None, markers=(2, 1)):
    """Object pose (R, t) in the ZED-left frame per ZED frame index, from markers[0] (preferred) and optional markers[1].

    Returns (filtered, raw, info); filtered is raw when filter_settings is None. stamps are seconds per frame. The
    secondary marker is ignored when it is seen together with the primary in fewer than 10 frames.
    """
    primary, secondary = markers[0], markers[1] if len(markers) > 1 else None
    marker = {primary: {}, secondary: {}}
    for row in csv.DictReader(open(poses_csv, newline='')):
        if row['zed_left_frame_index'] == '' or row['pose_valid'] != '1' or int(row['marker_id']) not in marker:
            continue
        R = np.array([[float(row[f'R{i}{j}']) for j in range(3)] for i in range(3)])
        t = np.array([float(row[k]) for k in ('tx_m', 'ty_m', 'tz_m')])
        marker[int(row['marker_id'])][int(row['zed_left_frame_index'])] = (R, t)
    both = [f for f in marker[primary] if f in marker[secondary]]
    if len(both) < 10:
        print(f'Secondary marker {secondary} seen with {primary} in only {len(both)} frames; using the primary marker only', flush=True)
        secondary = None
        marker[None] = {}
    # Pose of the secondary marker in primary-marker coordinates.
    rel_t = np.median([marker[primary][f][0].T @ (marker[secondary][f][1] - marker[primary][f][1]) for f in both], 0) if secondary is not None else np.zeros(3)
    rel_R = Rotation.from_matrix([marker[primary][f][0].T @ marker[secondary][f][0] for f in both]).mean().as_matrix() if secondary is not None else np.eye(3)
    if local is None:  # primary-marker axes at the marker midpoint
        local = np.eye(4)
        local[:3, 3] = rel_t / 2
    raw, sources = {}, {}
    for f in frames:
        if f in marker[primary]:
            raw[f], sources[f] = marker[primary][f], primary
        elif f in marker[secondary]:
            R1, t1 = marker[secondary][f]
            R = R1 @ rel_R.T
            raw[f], sources[f] = (R, t1 - R @ rel_t), secondary
    filtered, resets = dict(raw), 0
    fs = filter_settings
    if fs:
        rotation = OneEuroRotationFilter(fs['rot_min_cutoff'], fs['rot_beta'], fs['d_cutoff'], fs['reset_gap'])
        position, last = None, None
        for f, t in zip(frames, stamps):
            if f not in raw:
                continue
            if last is None or t - last > fs['reset_gap']:
                position = OneEuroFilter(fs['min_cutoff'], fs['beta'], fs['d_cutoff'])
                rotation.reset()
                resets += 1
            R, p = raw[f]
            filtered[f] = (rotation(R, t), position(p, t))
            last = t
    switches = sum(1 for a, b in zip(frames, frames[1:]) if a in sources and b in sources and sources[a] != sources[b])
    jitter = dict(raw=second_difference(raw, frames, stamps, fs['reset_gap'] if fs else .1))
    if fs:
        jitter['filtered'] = second_difference(filtered, frames, stamps, fs['reset_gap'])
    place = lambda poses: {f: (R @ local[:3, :3], t + R @ local[:3, 3]) for f, (R, t) in poses.items()}
    return place(filtered), place(raw), dict(
        markers=list(markers), marker_pairs=len(both), secondary_in_primary_m=rel_t.tolist(),
        secondary_in_primary_rotvec_deg=np.degrees(Rotation.from_matrix(rel_R).as_rotvec()).tolist(),
        frames_from_primary=sum(v == primary for v in sources.values()), frames_from_secondary=sum(v == secondary for v in sources.values()),
        marker_switches=switches, filter=fs, filter_resets=resets, primary_pose_jitter=jitter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path, help='MANO fit directory (fused_hand.npz, calibration.json)')
    parser.add_argument('arm', type=Path, help='Filtered arm .npz (filter_right_arm.py)')
    parser.add_argument('--gravity', type=Path, required=True, help='estimate_zed_gravity.py output')
    parser.add_argument('--poses', type=Path, help='ArUco poses.csv; omit for a take without an object')
    parser.add_argument('--output', type=Path, required=True, help='.txt path; must not exist')
    parser.add_argument('--mesh', default='mustard', help='Mesh name written in the header')
    parser.add_argument('--object-markers', type=int, nargs='+', default=(2, 1), metavar='ID',
                        help='ArUco IDs on the object: PRIMARY [SECONDARY]; the mesh transform is relative to PRIMARY')
    parser.add_argument('--mesh-transform', type=Path, help='fit_object_mesh_offset.py output: pose the mesh origin instead of the marker midpoint')
    parser.add_argument('--object-min-cutoff', type=float, default=1.5, help='Hz, object position; 0 disables object filtering')
    parser.add_argument('--object-beta', type=float, default=20., help='Per m/s of object speed')
    parser.add_argument('--object-rot-min-cutoff', type=float, default=1.5, help='Hz, object rotation')
    parser.add_argument('--object-rot-beta', type=float, default=8., help='Per rad/s of object angular speed')
    parser.add_argument('--object-d-cutoff', type=float, default=1.)
    parser.add_argument('--object-reset-gap', type=float, default=.1, help='Seconds')
    parser.add_argument('--clavicle-cm', type=float, default=15.58, help='Clavicle_r to upperarm_r length (reference file value)')
    parser.add_argument('--plane-height-cm', type=float, default=0., help='Height of the support plane above the world origin')
    parser.add_argument('--forward-camera', default='d455', help='Calibrated camera the subject faces')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    calibration = json.loads((args.fit / 'calibration.json').read_text())
    fit = np.load(args.fit / 'fused_hand.npz', allow_pickle=True)
    arm = np.load(args.arm)
    gravity = json.loads(args.gravity.read_text())
    if not np.array_equal(fit['frame_index'], arm['frame_index']) or not np.array_equal(fit['timestamp_ns'], arm['timestamp_ns']):
        raise ValueError('Fit and arm frames differ')

    # World frame: rows of A map ZED-left optical coordinates to (forward, right, up).
    up = np.asarray(gravity['up_in_zed_left'])
    T_face = np.asarray(calibration['cameras'][args.forward_camera]['T_camera_from_reference'])
    forward = -T_face[2, :3]  # minus that camera's optical axis, in the ZED-left frame
    forward = unit(forward - (forward @ up) * up)
    A = np.vstack([forward, np.cross(forward, up), up])
    plane = np.asarray(gravity['support_plane'])
    origin = -plane[3] * plane[:3] - args.plane_height_cm / 100 * up

    def world(p):
        return 100 * (np.asarray(p) - origin) @ A.T

    cam_R = A @ np.array([[0., 1, 0], [0, 0, -1], [1, 0, 0]])  # camera X = optical z, Y = optical x, Z = -optical y
    cam_loc, cam_q = world(np.zeros(3)), quat(cam_R)
    zed = calibration['cameras']['zed_left']
    K, (width, height) = np.asarray(zed['camera_matrix']), zed['image_size']
    fov_h, fov_v = (np.degrees(2 * np.arctan(width / (2 * K[0, 0]))), np.degrees(2 * np.arctan(height / (2 * K[1, 1]))))

    frames = [int(f) for f in fit['frame_index']]
    mesh_fit = json.loads(args.mesh_transform.read_text()) if args.mesh_transform else {}
    local = np.asarray(mesh_fit['T_marker_from_mesh']) if mesh_fit else None
    mesh_scale = np.asarray(mesh_fit.get('scale', [1., 1., 1.]))  # per mesh axis; written as SCALE_X/Y/Z
    stamps = fit['timestamp_ns'].astype(np.int64)
    settings = None if args.object_min_cutoff <= 0 else dict(
        min_cutoff=args.object_min_cutoff, beta=args.object_beta, rot_min_cutoff=args.object_rot_min_cutoff,
        rot_beta=args.object_rot_beta, d_cutoff=args.object_d_cutoff, reset_gap=args.object_reset_gap)
    objects, raw_objects, object_info = ({}, {}, dict(note='no object')) if args.poses is None else load_object(args.poses, frames, (stamps - stamps[0]) / 1e9, local, settings, tuple(args.object_markers))
    arm_joints = arm['joints']
    lines, written, hinge_prev, skipped = [], [], None, dict(no_hand=0, no_arm=0)
    straight = 0
    for t, frame in enumerate(frames):
        if not fit['valid'][t]:
            skipped['no_hand'] += 1
            continue
        if not np.isfinite(arm_joints[t, :2]).all():
            skipped['no_arm'] += 1
            continue
        hand = world(fit['joints'][t])
        shoulder, elbow = world(arm_joints[t, 0]), world(arm_joints[t, 1])
        wrist = hand[0]
        upper, fore = unit(elbow - shoulder), unit(wrist - elbow)
        lat = unit(hand[5] - hand[17])
        palm = unit(np.cross(unit(hand[9] - hand[0]), lat))
        hinge = np.cross(upper, fore)
        if np.linalg.norm(hinge) > np.sin(np.radians(10)):
            hinge_prev = unit(hinge)
        else:
            straight += 1
            hinge = hinge_prev if hinge_prev is not None else -lat
        clavicle = shoulder - np.array([0., args.clavicle_cm, 0.])
        bones = [('clavicle_r', clavicle, frame_from(clavicle - shoulder, np.array([0., 0., -1.]))),
                 ('upperarm_r', shoulder, frame_from(shoulder - elbow, hinge)),
                 ('lowerarm_r', elbow, frame_from(elbow - wrist, hinge))]
        hand_R = frame_from(hand[0] - hand[9], lat)
        bones.append(('hand_r', wrist, hand_R))
        for name, sign in (('wrist_inner_r', 1), ('wrist_outer_r', -1)):
            bones.append((name, wrist + hand_R @ (sign * WRIST_OFFSET_CM), hand_R @ Rotation.from_rotvec([sign * np.pi / 2, 0, 0]).as_matrix()))
        for finger, base in FINGERS:
            for k in range(3):
                j = base + k
                bones.append((f'{finger}_0{k + 1}_r', hand[j], frame_from(hand[j] - hand[j + 1], palm if finger == 'thumb' else lat)))
        seconds = (stamps[t] - stamps[0]) / 1e9
        fields = [f'{seconds:.6f}', str(frame), f'{seconds:.6f}', fmt(cam_loc), fmt(cam_q), str(len(bones))]
        fields += [f'{name}|{fmt(p)}|{fmt(quat(R))}' for name, p, R in bones]
        if frame in objects:
            R, p = objects[frame]
            fields += ['1', f'object_0|{fmt(world(p))}|{fmt(quat(A @ R @ FLIP_Y))}|{fmt(mesh_scale)}']
        else:
            fields += ['0']
        lines.append('|'.join(fields))
        written.append(frame)
    first = next((f for f in frames if f in objects), None)
    header_objects = []
    if first is not None:
        R0, p0 = objects[first]
        header_objects = [f'    {{mesh:{args.mesh},loc:[{",".join(f"{v:.6f}" for v in world(p0))}],rot:[{",".join(f"{v:.6f}" for v in quat(A @ R0 @ FLIP_Y))}]}}']
    created = datetime.fromtimestamp(stamps[0] / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    header = ['{', '  version:4,', f'  creation_time:{created},',
              f'  camera:{{loc:[{",".join(f"{v:.6f}" for v in cam_loc)}],rot:[{",".join(f"{v:.6f}" for v in cam_q)}],'
              f'fov:{fov_h:.6f},fov_h:{fov_h:.6f},fov_v:{fov_v:.6f}}},',
              '  objects:[', *header_objects, '  ]', '}',
              '# Format: TIMESTAMP|FRAME_NUMBER|FRAME_TIMESTAMP|HEAD_LOC_X|HEAD_LOC_Y|HEAD_LOC_Z|HEAD_ROT_X|HEAD_ROT_Y|HEAD_ROT_Z|HEAD_ROT_W|NUM_BONES|[BONE_DATA...]|NUM_OBJECTS|[OBJECT_DATA...]',
              '#         BONE_DATA: BONE_NAME|POS_X|POS_Y|POS_Z|ROT_X|ROT_Y|ROT_Z|ROT_W',
              '#         OBJECT_DATA: OBJECT_NAME|LOC_X|LOC_Y|LOC_Z|ROT_X|ROT_Y|ROT_Z|ROT_W|SCALE_X|SCALE_Y|SCALE_Z']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='\n') as f:
        f.write('\n'.join(header + lines) + '\n')
    report = dict(output=str(args.output), frames_total=len(frames), frames_written=len(written), skipped=skipped,
                  frames_with_object=sum(f in objects for f in written), straight_elbow_frames=straight,
                  first_frame=written[0], last_frame=written[-1],
                  world=dict(units='cm', axes='Unreal: X forward, Y right, Z up (left-handed)',
                             forward_in_zed_left=A[0].tolist(), right_in_zed_left=A[1].tolist(), up_in_zed_left=A[2].tolist(),
                             origin_in_zed_left_m=origin.tolist(), plane_height_cm=args.plane_height_cm,
                             camera_height_above_plane_cm=float(100 * plane[3])),
                  camera=dict(loc=cam_loc.tolist(), rot_xyzw=cam_q.tolist(), fov_h=fov_h, fov_v=fov_v),
                  object=dict(mesh=args.mesh, frame=('mesh origin and axes from ' + str(args.mesh_transform) + '; vertex v (m) -> LOC + ROT * 100*(v.x,-v.y,v.z)') if args.mesh_transform
                             else 'primary-marker axes (Y flipped for left-handed world), origin at the marker midpoint', **object_info),
                  clavicle_cm=args.clavicle_cm, sources=dict(fit=str(args.fit), arm=str(args.arm), gravity=str(args.gravity), poses=str(args.poses)))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('world', 'sources')}, indent=2))


if __name__ == '__main__':
    main()
