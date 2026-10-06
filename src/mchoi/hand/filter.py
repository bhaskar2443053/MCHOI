"""Apply rotation-aware One Euro to a saved MANO fit and regenerate its meshes."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from mchoi.hand.rotation_filter import filter_mano_rotations
from mchoi.multiview import Rig
from mchoi.paths import MANO_DIR, WILOR_SRC


def regenerate(fused, rotations):
    import torch
    if os.name == 'nt':
        os.environ.setdefault('PYOPENGL_PLATFORM', 'win32')
    sys.path.insert(0, str(WILOR_SRC))
    from wilor.models.mano_wrapper import MANO
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layer = MANO(model_path=str(MANO_DIR), num_hand_joints=15, create_body_pose=False).to(device).eval()
    rows = fused['fit_frame_rows']
    vertices = np.full_like(fused['vertices'], np.nan)
    joints = np.full_like(fused['joints'], np.nan)
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            stop = min(start + 256, len(rows))
            r = torch.tensor(rotations[start:stop], dtype=torch.float32, device=device)
            betas = torch.tensor(fused['shape'], dtype=torch.float32, device=device).expand(stop - start, -1)
            pred = layer(global_orient=r[:, :1], hand_pose=r[:, 1:], betas=betas, pose2rot=False)
            mirror = torch.ones((stop - start, 1, 3), device=device)
            mirror[:, 0, 0] = torch.tensor(2 * fused['is_right'][rows[start:stop]] - 1, device=device)
            translation = torch.tensor(fused['wrist_translation'][start:stop], dtype=torch.float32, device=device)[:, None]
            vertices[rows[start:stop]] = ((pred.vertices - pred.joints[:, :1]) * mirror + translation).cpu().numpy()
            joints[rows[start:stop]] = ((pred.joints - pred.joints[:, :1]) * mirror + translation).cpu().numpy()
    vertices[~fused['valid']], joints[~fused['valid']] = np.nan, np.nan
    return vertices, joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-cutoff', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--d-cutoff', type=float, default=1.)
    parser.add_argument('--wrist-min-cutoff', type=float, default=0., help='Global wrist rotation cutoff; zero keeps its original rotation')
    parser.add_argument('--wrist-beta', type=float, default=3.)
    parser.add_argument('--reset-gap', type=float, default=.1)
    args = parser.parse_args()
    if not np.isfinite([args.wrist_min_cutoff, args.wrist_beta]).all() or min(args.wrist_min_cutoff, args.wrist_beta) < 0:
        parser.error('Wrist parameters must be finite and nonnegative')
    start = time.perf_counter()
    source_summary = json.loads((args.source / 'fit_summary.json').read_text())
    if source_summary['status'] != 'complete' or 'one_euro_pose' in source_summary:
        raise ValueError('Use a completed, unfiltered source fit to avoid accidental repeated filtering')
    fused = dict(np.load(args.source / 'fused_hand.npz', allow_pickle=False))
    config = dict(min_cutoff=args.min_cutoff, beta=args.beta, d_cutoff=args.d_cutoff,
                  wrist_min_cutoff=args.wrist_min_cutoff, wrist_beta=args.wrist_beta, reset_gap=args.reset_gap)
    rotations, resets = filter_mano_rotations(fused, **config)
    args.output.mkdir(parents=True, exist_ok=False)
    vertices, joints = regenerate(fused, rotations)
    rig = Rig(json.loads((args.source / 'calibration.json').read_text()))
    tri = dict(np.load(args.source / 'triangulation.npz', allow_pickle=False))
    active = (tri['weights'] * tri['inliers']) > 0
    errors = np.full(active.shape, np.nan)
    valid = fused['valid'].copy()
    for row in np.flatnonzero(valid):
        for cam in range(4):
            errors[row, cam] = np.linalg.norm(rig.project(joints[row], cam) - tri['pixels'][row, cam], axis=-1)
        depths = np.einsum('ci,ki->ck', rig.T[:, 2, :3], joints[row]) + rig.T[:, 2, 3, None]
        valid[row] = (np.all((depths > .05) | ~active[row]) and np.isfinite(vertices[row]).all()
                      and np.median(errors[row][active[row]]) < 35)
    vertices[~valid], joints[~valid] = np.nan, np.nan
    rejected = np.flatnonzero(fused['valid'] & ~valid).tolist()
    result = dict(fused, valid=valid, vertices=vertices, joints=joints, mano_rotations=rotations,
                  unfiltered_mano_rotations=fused['mano_rotations'], one_euro_reset_rows=np.asarray(resets, np.int64))
    np.savez_compressed(args.output / 'fused_hand.npz', **result)
    for name in ('calibration.json', 'mano_faces.npy', 'triangulation.npz'):
        shutil.copyfile(args.source / name, args.output / name)
    shutil.copyfile(args.source / 'fit_summary.json', args.output / 'source_fit_summary.json')
    summary = copy.deepcopy(source_summary)
    summary['source_fitting_history'] = summary.pop('fitting_history', [])
    summary['source_elapsed_seconds'] = summary.pop('elapsed_seconds', None)
    summary.update(status='complete', source_fit=str(args.source.resolve()), fitted_frames=int(valid.sum()),
        median_reprojection_after_px=float(np.median(errors[active & valid[:, None, None]])),
        p95_reprojection_after_px=float(np.percentile(errors[active & valid[:, None, None]], 95)),
        observed_views_per_fitted_frame={str(n): int(np.sum(fused['used_views'][valid].sum(axis=1) == n)) for n in range(2, 5)},
        one_euro_pose=dict(**config, representation='SO(3) shortest-path interpolation; angular speed rad/s',
                           scope='local finger rotations and global wrist rotation' if args.wrist_min_cutoff else 'local finger rotations only',
                           wrist_translation_filtered=False, filter_is_causal=True, source_fit_is_offline=True,
                           reset_rows=resets, additionally_rejected_rows=rejected),
        processing='One Euro post-filter of an existing offline temporal fit; regenerate MANO with fixed source shape and wrist positions',
        elapsed_seconds=time.perf_counter() - start)
    (args.output / 'fit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('status', 'fitted_frames', 'median_reprojection_after_px', 'p95_reprojection_after_px', 'one_euro_pose', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
