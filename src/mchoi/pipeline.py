"""Run the full hand-object pipeline for one recording: calibration + raw recording -> poses, Mocap file, viewer, videos.

Each step is skipped when its output already exists, so an interrupted run can simply be restarted. Run with the SDK
environment (.venv); WiLoR/MANO steps are launched with .venv-wilor automatically. See README.md for inputs, outputs
and conventions. Steps, in order:

  aruco_detect   ArUco corners in ZED left/right and every RealSense recording
  aruco_review   marker_selection.json (from --marker-ids, or stop and print the decoded IDs for review)
  aruco_poses    RealSense timing offsets, marker poses, validation, annotated marker videos
  timing         per-take validated timing (wilor_timing/); views whose offset cannot be validated are left out
  hand_cache     YOLO + WiLoR predictions in all four views
  hand_fit       shared temporal MANO fit (visibility-weighted)
  hand_filter    One Euro on MANO rotations, meshes regenerated
  hand_videos    ZED-left overlay, before/after comparison, hand close-up; validation
  arm_2d         YOLO-pose body keypoints in all four views
  gravity        ZED IMU up direction and table plane
  arm_fit        metric right shoulder/elbow anchored on the MANO wrist (prior + IK where unobserved)
  arm_filter     One Euro on shoulder/elbow, bone lengths kept
  object_mesh    STL -> ArUco marker transform by silhouette fitting (skipped without --mesh)
  mocap          Mocap text file (001_Mug_Mocap.txt layout) + .json sidecar, One Euro on the object pose
  viewer         self-contained 3D HTML viewer
  overlay_video  hand mesh, arm and object mesh drawn over all four camera views
  arm_video      raw vs filtered arm over all four views
"""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys
from mchoi.paths import REPO_ROOT, sdk_python, subprocess_env, wilor_python

SDK = sdk_python()
WILOR = wilor_python()
ARUCO_DIR = 'aruco_pose_all_cameras_new_calibration'
STEPS = ('aruco_detect', 'aruco_review', 'aruco_poses', 'timing', 'hand_cache', 'hand_fit', 'hand_filter', 'hand_videos', 'arm_2d',
         'gravity', 'arm_fit', 'arm_filter', 'object_mesh', 'mocap', 'viewer', 'overlay_video', 'arm_video')


def decoded_ids(take):
    counts = Counter()
    for path in (take / 'aruco_annotated').glob('*detections.csv'):
        for row in csv.DictReader(path.open(newline='')):
            if row['detected'] == '1':
                counts[int(row['marker_id'])] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--take', type=Path, required=True, help='Recording folder (zed.svo2, realsense_*.db3, frame_timestamps.csv)')
    parser.add_argument('--calibration', type=Path, required=True, help='calibration.json from the calibration app')
    parser.add_argument('--mesh', type=Path, help='Object STL in metres; omit for a take without an object (hand and arm only)')
    parser.add_argument('--markers-only', action='store_true',
                        help='Object present but no matching mesh: export the marker-midpoint pose, drawn as a 4 cm cube proxy')
    parser.add_argument('--object-name', default='mustard', help='Mesh name written in the Mocap file')
    parser.add_argument('--object-markers', type=int, nargs='+', default=(2, 1), metavar='ID',
                        help='ArUco IDs on the object: PRIMARY [SECONDARY]; the mesh transform is fitted to PRIMARY')
    parser.add_argument('--marker-ids', type=int, nargs='+', help='All genuine ArUco IDs in the scene (writes marker_selection.json)')
    parser.add_argument('--object-hsv-low', type=int, nargs=3, default=(18, 110, 90), help='Object colour range for the mesh fit')
    parser.add_argument('--object-hsv-high', type=int, nargs=3, default=(38, 255, 255))
    parser.add_argument('--object-fit-scale', choices=('none', 'uniform', 'axes'), default='none',
                        help='Also fit the mesh size, when the real object differs from the model (e.g. a party-size box)')
    parser.add_argument('--plane-height-cm', type=float, default=0., help='Table height above the desired world origin')
    parser.add_argument('--run-name', default='pipeline', help='Output folder inside the take')
    parser.add_argument('--shoulder-prior', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Static shoulder in ZED-left metres for takes where no camera sees the shoulder (see README)')
    parser.add_argument('--from', dest='start', choices=STEPS, help='Start at this step')
    parser.add_argument('--only', nargs='+', choices=STEPS, help='Run only these steps')
    parser.add_argument('--dry-run', action='store_true', help='Print the commands without running them')
    args = parser.parse_args()

    take = args.take.resolve()
    if args.mesh and args.markers_only:
        parser.error('--markers-only is for objects without a mesh')
    for required in (take / 'zed.svo2', take / 'frame_timestamps.csv', args.calibration, *([args.mesh] if args.mesh else [])):
        if not required.exists():
            parser.error(f'Missing input: {required}')
    aruco = take / ARUCO_DIR
    run = take / args.run_name
    cache, source_fit, fit = run / 'hand_cache', run / 'hand_fit', run / 'hand_fit_one_euro'
    arm, mocap = run / 'right_arm', run / 'mocap'
    name = args.object_name
    mocap_txt = mocap / f'{name.capitalize()}_Mocap.txt'
    transform = mocap / f'{name}_mesh_to_marker{args.object_markers[0]}.json'
    timing = take / 'wilor_timing/timing_alignment.json'
    prior = ['--shoulder-prior', *args.shoulder_prior] if args.shoulder_prior else []
    mesh_args = (['--poses', aruco / 'poses.csv', '--mesh', name, '--object-markers', *args.object_markers, '--mesh-transform', transform]
                 if args.mesh else ['--poses', aruco / 'poses.csv', '--mesh', name, '--object-markers', *args.object_markers]
                 if args.markers_only else ['--mesh', 'none'])
    view_mesh = ['--mesh', args.mesh.resolve()] if args.mesh else []
    serials = sorted(p.stem.split('_', 1)[1] for p in take.glob('realsense_*.db3'))

    steps = {
        'aruco_detect': ([take / 'aruco_annotated/detection_summary.json'] + [take / f'aruco_annotated/realsense_{s}_summary.json' for s in serials],
                         [([SDK, '-m', 'mchoi.aruco.extract_zed', take], take / 'aruco_annotated/detection_summary.json')] +
                         [([SDK, '-m', 'mchoi.aruco.extract_realsense', take, s], take / f'aruco_annotated/realsense_{s}_summary.json') for s in serials]),
        'aruco_review': (aruco / 'marker_selection.json', 'review'),
        'aruco_poses': (aruco / 'video_validation.json', [
            [SDK, '-m', 'mchoi.aruco.poses', take, '--calibration', args.calibration.resolve(), '--output', aruco, '--data-only'],
            [SDK, '-m', 'mchoi.aruco.validate', aruco],
            [SDK, '-m', 'mchoi.aruco.poses', take, '--calibration', args.calibration.resolve(), '--output', aruco, '--render-only'],
            [SDK, '-m', 'mchoi.aruco.validate', aruco, '--video']]),
        'timing': (timing, [[SDK, '-m', 'mchoi.aruco.prepare_timing', take]]),
        'hand_cache': (cache / 'observations.npz', [[WILOR, '-m', 'mchoi.hand.cache', '--take', take, '--output', cache,
                                                    '--timing', timing, '--skip-unvalidated-timing']]),
        'hand_fit': (source_fit / 'fused_hand.npz', [[WILOR, '-m', 'mchoi.hand.fit', cache, '--output', source_fit, '--fusion-mode', 'visibility',
                                                     '--iterations', 300, '--temporal-weight', 4, '--finger-temporal-weight', 3,
                                                     '--rotation-temporal-weight', .25]]),
        'hand_filter': (fit / 'fused_hand.npz', [[WILOR, '-m', 'mchoi.hand.filter', source_fit, '--output', fit, '--min-cutoff', 1, '--beta', 1,
                                                 '--d-cutoff', 1, '--wrist-min-cutoff', 2.5, '--wrist-beta', 3, '--reset-gap', .1]]),
        'hand_videos': (fit / 'validation.json', [[SDK, '-m', 'mchoi.hand.render', fit, '--compare-fit', source_fit, '--closeup'],
                                                  [SDK, '-m', 'mchoi.hand.validate', fit]]),
        'arm_2d': (arm / 'arm2d.npz', [[WILOR, '-m', 'mchoi.arm.keypoints', '--take', take, '--output', arm / 'arm2d.npz']]),
        'arm_fit': (arm / 'arm3d_raw.npz', [[SDK, '-m', 'mchoi.arm.fit', arm / 'arm2d.npz', fit, '--output', arm / 'arm3d_raw.npz', '--root', 'far',
                                             '--gravity', arm / 'gravity.json', *prior]]),
        'arm_filter': (arm / 'arm3d_one_euro.npz', [[SDK, '-m', 'mchoi.arm.filter', arm / 'arm3d_raw.npz', '--output', arm / 'arm3d_one_euro.npz']]),
        'gravity': (arm / 'gravity.json', [[SDK, '-m', 'mchoi.scene.gravity', '--svo', take / 'zed.svo2', '--output', arm / 'gravity.json']]),
        'object_mesh': (transform, None if not args.mesh else [[SDK, '-m', 'mchoi.scene.object_mesh', args.mesh.resolve(), '--take', take, '--marker', args.object_markers[0],
                                     '--hsv-low', *args.object_hsv_low, '--hsv-high', *args.object_hsv_high, '--gravity', arm / 'gravity.json', '--fit-scale', args.object_fit_scale,
                                     '--output', transform, '--preview', transform.with_suffix('.jpg')]]),
        'mocap': (mocap_txt, [[SDK, '-m', 'mchoi.export.mocap', fit, arm / 'arm3d_one_euro.npz', '--gravity', arm / 'gravity.json',
                               *mesh_args, '--plane-height-cm', args.plane_height_cm, '--output', mocap_txt]]),
        'viewer': (mocap / f'{name}_mocap_viewer.html', [[SDK, '-m', 'mchoi.export.viewer', mocap_txt, *view_mesh, '--fit', fit,
                                                         '--title', f'{name} mocap viewer', '--output', mocap / f'{name}_mocap_viewer.html']]),
        'overlay_video': (mocap / f'{name}_overlay.mp4', [[SDK, '-m', 'mchoi.export.overlay', mocap_txt, '--take', take, '--fit', fit,
                                                          *view_mesh, '--output', mocap / f'{name}_overlay.mp4']]),
        'arm_video': (arm / 'right_arm_one_euro.mp4', [[SDK, '-m', 'mchoi.arm.render', arm / 'arm3d_one_euro.npz', '--take', take,
                                                       '--output', arm / 'right_arm_one_euro.mp4']]),
    }
    if args.only:
        selected = args.only
    elif args.start:
        selected = STEPS[STEPS.index(args.start):]
    else:
        selected = STEPS
    if not args.dry_run:
        for folder in (run, arm, mocap):  # tool-created folders (cache and fits) must not pre-exist
            folder.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        if step not in selected:
            continue
        done, commands = steps[step]
        if commands is None:
            print(f'[skip] {step}: no object mesh', flush=True)
            continue
        done = done if isinstance(done, list) else [done]
        if all(d.exists() for d in done):
            print(f'[skip] {step}: {done[0]} exists', flush=True)
            continue
        if commands == 'review':
            done = done[0]
            if args.marker_ids:
                print(f'[run ] {step}: marker IDs {args.marker_ids}', flush=True)
                if not args.dry_run:
                    aruco.mkdir(parents=True, exist_ok=True)
                    done.write_text(json.dumps(dict(marker_ids=sorted(args.marker_ids), review='Given on the mchoi.pipeline command line'), indent=2))
                continue
            counts = decoded_ids(take)
            print(f'[stop] {step}: decoded ArUco IDs (detections) = {dict(counts.most_common())}\n'
                  '       Rare IDs are usually misdecodes. Rerun with --marker-ids <genuine IDs> to continue.', flush=True)
            if args.dry_run:
                continue
            sys.exit(2)
        print(f'[run ] {step}', flush=True)
        for command in commands:
            if isinstance(command, tuple):  # (command, own output): skip parts that already finished
                command, output = command
                if output.exists():
                    continue
            command = [str(c) for c in command]
            print('       ' + ' '.join(shlex.quote(c) for c in command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO_ROOT, env=subprocess_env(), check=True)
    print(f'Outputs: {run}')


if __name__ == '__main__':
    main()
