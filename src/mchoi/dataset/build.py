"""Assemble the publishable MCHOI dataset (BIDS-inspired layout) from the pipeline outputs.

Reads configs/mchoi_v1.0.json. Writes dataset-level metadata, the rig calibration with its ChArUco captures, the
object meshes, and per recording the derivatives: MANO hand pose, right-arm joints, object pose, ArUco marker poses,
mesh-to-marker transform, camera timing, Mocap file, 3D viewer and overlay video. Camera RGB/depth videos are written
separately by mchoi.dataset.videos into sub-XX/ses-YY/video/. Native SVO2/DB3 files are not distributed.
Existing files are overwritten, so the dataset can be rebuilt after pipeline changes. The dataset README is
docs/dataset_README.md of this repository.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from mchoi.paths import ARUCO_DIR, CONFIG_DIR, MODELS, REPO_ROOT, data_root, dataset_root, timing_path
from mchoi.export.mocap import load_object

PIPELINE = 'mchoi-pipeline'
JOINTS = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip', 'index_mcp', 'index_pip', 'index_dip', 'index_tip',
          'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip', 'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
          'little_mcp', 'little_pip', 'little_dip', 'little_tip']
ARM_METHOD = {0: 'none', 1: 'triangulated', 2: 'single-view at bone length', 3: 'MANO wrist', 4: 'prior + IK fallback'}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def write_tsv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fields, delimiter='\t', lineterminator='\n', restval='n/a')
        w.writeheader()
        for row in rows:
            w.writerow({k: 'n/a' if v is None or (isinstance(v, float) and not np.isfinite(v)) else v for k, v in row.items()})


def num(v, digits=6):
    return None if v is None or not np.isfinite(v) else round(float(v), digits)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def utc(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def export_hand(fit_dir, out, prefix):
    d = np.load(fit_dir / 'fused_hand.npz', allow_pickle=True)
    n = len(d['frame_index'])
    rot = np.full((n, 16, 3, 3), np.nan, np.float32)
    rot[d['fit_frame_rows']] = d['mano_rotations']
    aa = np.full((n, 16, 3), np.nan, np.float32)
    ok = np.isfinite(rot[:, 0, 0, 0]) & d['valid']
    aa[ok] = Rotation.from_matrix(rot[ok].reshape(-1, 3, 3)).as_rotvec().reshape(-1, 16, 3)
    valid = d['valid'].astype(bool)
    joints = np.where(valid[:, None, None], d['joints'], np.nan).astype(np.float32)
    vertices = np.where(valid[:, None, None], d['vertices'], np.nan).astype(np.float32)
    name = f'{prefix}_space-zedleft_desc-mano_handpose.npz'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / name, frame_index=d['frame_index'], timestamp_ns=d['timestamp_ns'], valid=valid,
                        joints=joints, vertices=vertices, faces=np.load(fit_dir / 'mano_faces.npy').astype(np.int32),
                        mano_rotmat=rot, mano_axis_angle=aa, betas=d['shape'].astype(np.float32),
                        wrist_translation=joints[:, 0], used_views=d['used_views'], joint_names=np.array(JOINTS))
    rows = []
    for i in range(n):
        row = dict(frame=int(d['frame_index'][i]), timestamp_ns=int(d['timestamp_ns'][i]), valid=int(valid[i]),
                   n_views=int(d['used_views'][i].sum()) if valid[i] else 0)
        for j, jn in enumerate(JOINTS):
            for a, ax in enumerate('xyz'):
                row[f'{jn}_{ax}'] = num(joints[i, j, a])
        rows.append(row)
    write_tsv(out / f'{prefix}_space-zedleft_desc-mano_handjoints.tsv', rows)
    summary = json.loads((fit_dir / 'fit_summary.json').read_text())
    write_json(out / f'{prefix}_space-zedleft_desc-mano_handjoints.json', dict(
        Description='Right-hand 3D joints from the multi-view MANO fit (one shared shape per recording, temporal fit, One Euro on rotations).',
        CoordinateSystem='space-zedleft: ZED Mini left optical frame (x right, y down, z forward), metres',
        frame='ZED-left frame index of the source recording', timestamp_ns='ZED-left image time, PC clock, Unix ns',
        valid='1 if a metric MANO fit passed the geometry/reprojection checks; joints are n/a otherwise',
        n_views='cameras contributing to the fit at this frame', JointOrder=JOINTS,
        Companion=f'{prefix}_space-zedleft_desc-mano_handpose.npz', FittedFrames=int(valid.sum()), Frames=n,
        MedianReprojectionErrorPx=summary.get('median_reprojection_after_px')))
    write_json(out / f'{prefix}_space-zedleft_desc-mano_handpose.json', dict(
        Description='MANO right-hand pose per frame.',
        Arrays=dict(frame_index='(N,) ZED-left frame index', timestamp_ns='(N,) PC clock, Unix ns', valid='(N,) bool',
                    joints='(N,21,3) metres, space-zedleft, JointOrder; NaN when invalid',
                    vertices='(N,778,3) metres, space-zedleft; NaN when invalid', faces='(1538,3) MANO triangle indices',
                    mano_rotmat='(N,16,3,3) global orientation then 15 joint rotations (MANO order: index, middle, little, ring, thumb; 3 each)',
                    mano_axis_angle='(N,16,3) the same rotations as axis-angle (radians)', betas='(10,) shared MANO shape',
                    wrist_translation='(N,3) wrist joint position, metres', used_views='(N,4) cameras used: zedminileft, zedminiright, d435n1, d435n2'),
        Reconstruction='vertices = MANO_right(global_orient=rotmat[0], hand_pose=rotmat[1:], betas, pose2rot=False).vertices '
                       '- MANO joint 0 + wrist_translation (smplx.MANOLayer via the WiLoR wrapper, use_pca=False; joints re-ordered '
                       'to the OpenPose 21-joint order). The stored joints and vertices are authoritative. The MANO model is not '
                       'distributed; obtain it from https://mano.is.tue.mpg.de under its licence.',
        JointOrder=JOINTS, CoordinateSystem='space-zedleft (metres)', Method=summary.get('processing', 'multi-view MANO fit'),
        Settings={k: summary[k] for k in ('iterations', 'temporal_weight', 'finger_temporal_weight', 'rotation_temporal_weight', 'fusion_mode') if k in summary}))
    return dict(hand_frames=int(valid.sum()), frames=n, first_ns=int(d['timestamp_ns'][0]), last_ns=int(d['timestamp_ns'][-1]))


def prior_m(raw_path):
    raw = np.load(raw_path)
    prior = raw['shoulder_prior'] if 'shoulder_prior' in raw else np.full(3, np.nan)
    return prior.tolist() if np.isfinite(prior).all() else None


def export_arm(arm_dir, out, prefix):
    d = np.load(arm_dir / 'arm3d_one_euro.npz')
    rows = []
    for i in range(len(d['frame_index'])):
        row = dict(frame=int(d['frame_index'][i]), timestamp_ns=int(d['timestamp_ns'][i]))
        for j, jn in enumerate(('shoulder', 'elbow', 'wrist')):
            for a, ax in enumerate('xyz'):
                row[f'{jn}_{ax}'] = num(d['joints'][i, j, a])
            row[f'{jn}_method'] = int(d['method'][i, j])
        for j, jn in enumerate(('shoulder', 'elbow')):
            for a, ax in enumerate('xyz'):
                row[f'{jn}_raw_{ax}'] = num(d['raw_joints'][i, j, a])
        rows.append(row)
    write_tsv(out / f'{prefix}_space-zedleft_desc-rightarm_joints.tsv', rows)
    report = json.loads((arm_dir / 'arm3d_one_euro.json').read_text())
    write_json(out / f'{prefix}_space-zedleft_desc-rightarm_joints.json', dict(
        Description='Right shoulder, elbow and wrist. Wrist = MANO wrist. Shoulder/elbow from YOLO-pose keypoints in the '
                    'validated views, placed at fixed bone lengths when only one view sees them, then One Euro filtered with bone lengths kept.',
        CoordinateSystem='space-zedleft (metres)', MethodCodes=ARM_METHOD,
        Columns=dict(x_y_z='filtered position', raw='before One Euro filtering', method='source of the raw estimate'),
        UpperArmLengthM=float(d['upper_arm_length']), ForearmLengthM=float(d['forearm_length']),
        ShoulderPriorM=prior_m(arm_dir / 'arm3d_raw.npz'),
        Filter=report['settings'], JitterMm=report['jitter_mm']))
    method = d['method']
    hand = method[:, 2] > 0
    return dict(shoulder_observed=float(np.isin(method[hand, 0], (1, 2)).mean()) if hand.any() else 0.)


def export_object(take, run, out, prefix, rec, mocap_meta):
    fit = np.load(run / 'hand_fit_one_euro/fused_hand.npz', allow_pickle=True)
    frames = [int(f) for f in fit['frame_index']]
    stamps = fit['timestamp_ns'].astype(np.int64)
    meshfit = next(iter(sorted((run / 'mocap').glob('*_mesh_to_marker*.json'))), None)
    local = scale = None
    if meshfit:
        mf = json.loads(meshfit.read_text())
        local, scale = np.asarray(mf['T_marker_from_mesh']), mf.get('scale', [1, 1, 1])
        copy(meshfit, out / f'{prefix}_desc-meshfit_transform.json')
        jpg = meshfit.with_suffix('.jpg')
        if jpg.exists():
            copy(jpg, out / f'{prefix}_desc-meshfit_contactsheet.jpg')
    settings = mocap_meta['object'].get('filter')
    poses, raw, info = load_object(take / ARUCO_DIR / 'poses.csv', frames, (stamps - stamps[0]) / 1e9, local, settings, tuple(rec['markers']))
    rows = []
    for f, t in zip(frames, stamps):
        row = dict(frame=f, timestamp_ns=int(t), valid=int(f in poses))
        for label, src in (('', poses), ('raw_', raw)):
            if f in src:
                R, p = src[f]
                q = Rotation.from_matrix(R).as_quat()
                row.update({f'{label}tx': num(p[0]), f'{label}ty': num(p[1]), f'{label}tz': num(p[2]),
                            f'{label}qx': num(q[0], 7), f'{label}qy': num(q[1], 7), f'{label}qz': num(q[2], 7), f'{label}qw': num(q[3], 7)})
            else:
                row.update({f'{label}{k}': None for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')})
        rows.append(row)
    write_tsv(out / f'{prefix}_space-zedleft_desc-object_pose.tsv', rows)
    write_json(out / f'{prefix}_space-zedleft_desc-object_pose.json', dict(
        Description=f'6-DoF pose of the {rec["object"]} from ArUco markers {rec["markers"]} (first = primary).',
        CoordinateSystem='space-zedleft (metres); quaternion (qx,qy,qz,qw) rotates object-frame vectors into space-zedleft',
        ObjectFrame=('mesh (STL) origin and axes; a mesh vertex v maps to R @ (scale * v) + t' if meshfit else
                     'no matching mesh: primary-marker axes at the midpoint of the two markers'),
        MeshScale=scale, Mesh=f'objects/obj-{rec["object"]}_mesh.stl' if meshfit else None,
        Columns=dict(tx_ty_tz_q='One Euro filtered pose', raw_='unfiltered marker-derived pose'),
        Filter=settings, Markers=info))
    rows = []
    for r in csv.DictReader(open(take / ARUCO_DIR / 'poses.csv', newline='')):
        if r['zed_left_frame_index'] == '':
            continue
        rows.append(dict(frame=int(r['zed_left_frame_index']), timestamp_ns=int(r['timestamp_ns']), marker_id=int(r['marker_id']),
                         valid=int(r['pose_valid']), status=r['pose_status'], cameras=r['source_cameras'] or None,
                         reprojection_rms_px=r['reprojection_rms_px'] or None,
                         **{k: r[k] or None for k in ('tx_m', 'ty_m', 'tz_m', 'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22')}))
    write_tsv(out / f'{prefix}_space-zedleft_desc-aruco_markerposes.tsv', rows)
    write_json(out / f'{prefix}_space-zedleft_desc-aruco_markerposes.json', dict(
        Description='Per-frame pose of every reviewed ArUco marker (DICT_4X4_50, 24 mm black square), no smoothing, no filled gaps.',
        CoordinateSystem='space-zedleft (metres); p_zedleft = R @ p_marker + t; marker X right, Y up, Z out of the face',
        status='multiview, single_view_tracked (orientation chosen by continuity), single_view_tracked_unanchored, or a rejection reason',
        MarkerRoles={str(m): 'object' for m in rec['markers']} | {'4': 'forearm', '5': 'forearm', '3': 'other'}))
    return dict(object_frames=len(poses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DIR / 'mchoi_v1.0.json')
    parser.add_argument('--checksums', action='store_true', help='Also write SHA256SUMS.txt (slow: reads every file)')
    parser.add_argument('--checksums-only', action='store_true', help='Only (re)write SHA256SUMS.txt of the finished dataset')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = dataset_root(config)
    if args.checksums_only:
        return write_checksums(root)
    sub, ses = f"sub-{config['subject']}", f"ses-{config['session']}"
    deriv = root / 'derivatives' / PIPELINE / sub / ses

    # Calibration: results and every ChArUco capture.
    session = data_root() / config['calibration_session']
    calib = session / config['calibration_run']
    for src, dst in (('calibration.json', 'calib-01_rig_calibration.json'), ('calibration.yml', 'calib-01_rig_calibration.yml'),
                     ('summary.txt', 'calib-01_rig_summary.txt')):
        copy(calib / src, root / 'calibration' / dst)
    cal = json.loads((calib / 'calibration.json').read_text())
    name_to_label = {c['calibration_name']: c['label'] for c in config['cameras']}
    poses = sorted(p for p in session.glob('pose_*') if p.is_dir())
    for p in poses:
        k = p.name.split('_')[1]
        copy(p / 'pose.json', root / 'calibration/captures' / f'calib-01_pose-{k}_charuco.json')
        for png in p.glob('*.png'):
            copy(png, root / 'calibration/captures' / f'calib-01_pose-{k}_cam-{name_to_label[png.stem]}_charuco.png')
    write_json(root / 'calibration/calib-01_rig_calibration_description.json', dict(
        Description='ChArUco calibration of the four RGB views, run after the recordings with the rig unmoved; used for every recording.',
        Board=cal.get('board', {}) and {k: cal['board'][k] for k in ('type', 'squares_x', 'squares_y', 'square_length_m', 'marker_length_m', 'dictionary') if k in cal['board']},
        Captures=len(poses), ReferenceFrame='zed_left optical frame (x right, y down, z forward), metres',
        Convention=cal.get('coordinate_convention'), CameraLabels=name_to_label,
        ReprojectionRmsPx={name_to_label[k]: v.get('rms_px') for k, v in cal['cameras'].items()},
        Files=dict(json='full intrinsics, distortion, transforms, errors, per-capture board poses', yml='OpenCV matrices',
                   captures='calib-01_pose-<n>_cam-<label>_charuco.png (lossless native images) and _charuco.json (detected corners, timestamps)')))

    # Objects.
    obj_rows = []
    for name, o in config['objects'].items():
        if o['mesh']:
            copy(MODELS / o['mesh'], root / 'objects' / f'obj-{name}_mesh.stl')
        obj_rows.append(dict(object=name, ycb_id=o['ycb_id'], mesh=f'obj-{name}_mesh.stl' if o['mesh'] else None, source=o['source'], note=o.get('note')))
    write_tsv(root / 'objects/objects.tsv', obj_rows)

    # Per recording.
    rec_rows, scans = [], []
    for rec in config['recordings']:
        take = data_root() / 'recordings' / rec['source']
        run = take / config['run_name']
        prefix = f"{sub}_{ses}_task-{rec['task']}_run-{rec['run']}"
        print(prefix, flush=True)
        mocap_json = next((run / 'mocap').glob('*_Mocap.json'))
        mocap_meta = json.loads(mocap_json.read_text())
        stats = export_hand(run / 'hand_fit_one_euro', deriv / 'hand', prefix)
        stats.update(export_arm(run / 'right_arm', deriv / 'arm', prefix))
        if rec['object']:
            stats.update(export_object(take, run, deriv / 'object', prefix, rec, mocap_meta))
        copy(mocap_json.with_suffix('.txt'), deriv / 'mocap' / f'{prefix}_space-world_mocap.txt')
        write_json(deriv / 'mocap' / f'{prefix}_space-world_mocap.json', {k: v for k, v in mocap_meta.items() if k not in ('sources', 'output')})
        copy(next((run / 'mocap').glob('*_mocap_viewer.html')), deriv / 'viz' / f'{prefix}_desc-3d_viewer.html')
        copy(next((run / 'mocap').glob('*_overlay.mp4')), deriv / 'viz' / f'{prefix}_desc-overlay_video.mp4')
        timing = json.loads(timing_path(take).read_text())
        write_json(deriv / 'sync' / f'{prefix}_desc-camera_timing.json', dict(
            Description='Content-based RealSense-to-ZED clock offsets (ArUco corner matching, hold-out validated). '
                        'zed_aligned_time = realsense_time - rs_minus_zed_lag_ns. Unvalidated cameras were excluded from multi-view processing.',
            Cameras={name_to_label[k]: v for k, v in timing['cameras'].items()},
            ActiveCameras=[name_to_label[c] for c in timing.get('active_cameras', name_to_label)],
            ExcludedCameras={name_to_label[k]: v for k, v in timing.get('excluded_cameras', {}).items()}))
        video_dir = root / sub / ses / 'video'
        for sidecar in sorted(video_dir.glob(f'{prefix}_cam-*.json')):
            s = json.loads(sidecar.read_text())
            media = next(video_dir.glob(sidecar.stem + '.m*'))
            scans.append(dict(filename=f'video/{media.name}', acq_time=utc(s['FirstTimestampUnixNs']), camera=s['Camera'], stream=s['Stream'],
                              frames=s['FrameCount'], duration_s=s['DurationSeconds'], source_recording=rec['source']))
        active = [name_to_label[c] for c in timing.get('active_cameras', name_to_label)]
        rec_rows.append(dict(recording=prefix, task=rec['task'], run=rec['run'], object=rec['object'], ycb_id=config['objects'].get(rec['object'] or '', {}).get('ycb_id'),
                             object_markers=','.join(map(str, rec['markers'])) or None, start_time_utc=utc(stats['first_ns']),
                             duration_s=round((stats['last_ns'] - stats['first_ns']) / 1e9, 2), zed_frames=stats['frames'],
                             validated_views=','.join(active), hand_pose_frames=stats['hand_frames'],
                             object_pose_frames=stats.get('object_frames'), mocap_frames=mocap_meta['frames_written'],
                             shoulder_observed_fraction=round(stats['shoulder_observed'], 3),
                             source_recording=rec['source']))
    write_tsv(root / 'recordings.tsv', rec_rows)
    if scans:
        write_tsv(root / sub / ses / f'{sub}_{ses}_scans.tsv', scans)

    # Dataset-level metadata.
    write_tsv(root / 'participants.tsv', [dict(participant_id=sub, handedness_recorded='right', note='single participant; demographics not collected in this release')])
    write_tsv(root / 'tasks.tsv', [dict(task=t, object=o, description=d) for t, o, d in (
        ('mustardbottle', 'mustardbottle', 'grasp, lift, move and replace the mustard bottle'),
        ('crackerbox', 'crackerbox', 'grasp, lift, move and replace the cracker box'),
        ('puddingbox', 'puddingbox', 'grasp, lift, move and replace the pudding box'),
        ('bowl', 'bowl', 'grasp, lift, move and replace the bowl'),
        ('mug', 'mug', 'grasp, lift, move and replace the mug (two runs)'),
        ('banana', 'banana', 'grasp, lift, move and replace the banana-shaped object'),
        ('gesture', None, 'free right-hand gestures without an object (two runs)'))])
    write_tsv(root / 'cameras.tsv', config['cameras'])
    write_json(root / 'dataset_description.json', dict(
        Name=config['title'], DatasetVersion=config['version'], DatasetType='raw+derivative (BIDS-inspired; not a validated BIDS modality)',
        Authors=['TO BE COMPLETED'], License='TO BE COMPLETED', HowToAcknowledge='TO BE COMPLETED',
        EthicsApprovals=['TO BE COMPLETED'], Keywords=['hand pose', 'MANO', 'hand-object interaction', 'multi-view', 'RGB-D', 'YCB', 'motion capture'],
        GeneratedBy=[dict(Name=PIPELINE, Description='mchoi.pipeline and mchoi.dataset.build of the mchoi repository')],
        AcquisitionDate=config['session_date']))
    write_json(root / 'derivatives' / PIPELINE / 'dataset_description.json', dict(
        Name=f"{config['name']} derivatives", DatasetType='derivative', GeneratedBy=[dict(Name=PIPELINE, Description='see README.md and docs/')],
        SourceDatasets=[dict(URL='../../', Version=config['version'])]))
    copy(REPO_ROOT / 'docs' / 'dataset_README.md', root / 'README.md')
    (root / 'CHANGES').write_text(f"{config['version']} {config['release_date']}\n  - Initial release: 9 recordings, calibration, RGB/depth videos, hand/arm/object poses, Mocap files, viewers.\n")
    if args.checksums:
        write_checksums(root)
    print(json.dumps(rec_rows, indent=1)[:3000])


def write_checksums(root):
    lines = []
    for path in sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt'):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 24), b''):
                h.update(block)
        lines.append(f'{h.hexdigest()}  {path.relative_to(root).as_posix()}')
    (root / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n')
    print(f'SHA256SUMS.txt: {len(lines)} files')


if __name__ == '__main__':
    main()
