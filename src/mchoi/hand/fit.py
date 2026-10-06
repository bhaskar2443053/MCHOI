"""Fit one temporally consistent MANO hand to calibrated observations from four cameras."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from mchoi.multiview import Rig, prepare_observations
from mchoi.common import save_json
from mchoi.paths import MANO_DIR, WILOR_SRC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--temporal-weight', type=float, default=4.0)
    parser.add_argument('--finger-temporal-weight', type=float, default=3.0)
    parser.add_argument('--rotation-temporal-weight', type=float, default=.25)
    parser.add_argument('--fusion-mode', choices=('visibility', 'legacy'), default='visibility')
    parser.add_argument('--reprojection-threshold', type=float, default=20)
    args = parser.parse_args()
    if args.iterations < 1 or not np.isfinite([args.temporal_weight, args.finger_temporal_weight, args.rotation_temporal_weight, args.reprojection_threshold]).all():
        parser.error('Iterations must be positive and fit parameters finite')
    if min(args.temporal_weight, args.finger_temporal_weight, args.rotation_temporal_weight) < 0 or args.reprojection_threshold <= 0:
        parser.error('Smoothing weights must be nonnegative and reprojection threshold positive')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cache_meta = json.loads((args.cache / 'cache_metadata.json').read_text())
    if cache_meta['status'] != 'complete':
        raise ValueError('Observation cache is incomplete')
    calibration = json.loads((args.cache / 'calibration.json').read_text())
    rig = Rig(calibration)
    data = dict(np.load(args.cache / 'observations.npz', allow_pickle=False))
    # This trial reconstructs the hand visible in ZED left. Other resting hands
    # in the wide RealSense views are excluded by side + epipolar association.
    left_ids = [i for a, b in data['offsets'][:, 0] for i in range(a, b)]
    if not left_ids:
        raise ValueError('No target hand was observed in ZED left')
    side = int(np.average(data['is_right'][left_ids], weights=data['confidence'][left_ids]) >= .5)
    selected, pixels, weights, points, masks, quality = prepare_observations(data, rig, side, args.reprojection_threshold,
                                                                          mode=args.fusion_mode, return_quality=True)
    valid_joints = np.isfinite(points[:, :, 0])
    valid_frames = (valid_joints.sum(axis=1) >= 8) & (valid_joints[:, [0, 5, 9, 13, 17]].sum(axis=1) >= 3)
    ids = np.flatnonzero(valid_frames)
    if not len(ids):
        raise RuntimeError('No sufficiently consistent multi-view hand observations')
    print(f'Triangulation: {len(ids)}/{len(valid_frames)} frames supported; median {np.median(valid_joints[ids].sum(axis=1)):.0f}/21 joints', flush=True)
    np.savez_compressed(args.output / 'triangulation.npz', selected=selected, pixels=pixels, weights=weights,
                        points=points, inliers=masks, valid_frames=valid_frames, **quality)
    import torch
    import torch.nn.functional as F
    if os.name == 'nt':
        os.environ.setdefault('PYOPENGL_PLATFORM', 'win32')
    sys.path.insert(0, str(WILOR_SRC))
    from wilor.models.mano_wrapper import MANO
    torch.set_num_threads(4)
    device = torch.device('cuda')
    layer = MANO(model_path=str(MANO_DIR), num_hand_joints=15, create_body_pose=False).to(device).eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    # Fix one subject shape for the entire sequence instead of changing finger lengths every frame.
    shape_ids = [i for i in left_ids if data['confidence'][i] > .7 and int(data['is_right'][i]) == side]
    if not shape_ids:
        shape_ids = left_ids
    shape = np.median(data['mano_betas'][shape_ids], axis=0).astype(np.float32)
    betas = torch.tensor(shape, device=device).expand(len(ids), -1)
    initial_rotations = []
    initial_sources = []
    mirror = np.diag([2 * side - 1, 1., 1.])
    for frame in ids:
        available = np.flatnonzero(selected[frame] >= 0)
        consistent = [c for c in available if int(data['is_right'][selected[frame, c]]) == side]
        cam = int(consistent[0] if consistent else available[0])
        obs = selected[frame, cam]
        rotations = np.concatenate([data['mano_global_orient'][obs], data['mano_hand_pose'][obs]], axis=0).astype(float)
        camera_to_reference = np.linalg.inv(rig.T[cam])[:3, :3]
        rotations[0] = mirror @ camera_to_reference @ mirror @ rotations[0]
        initial_rotations.append(rotations)
        initial_sources.append(cam)
    initial_rotations = torch.tensor(np.asarray(initial_rotations), dtype=torch.float32, device=device)
    initial_6d = torch.cat([initial_rotations[:, :, :, 0], initial_rotations[:, :, :, 1]], dim=-1)
    rotations_6d = torch.nn.Parameter(initial_6d.clone())
    mirror_vector = torch.tensor([2 * side - 1, 1, 1], dtype=torch.float32, device=device)

    def matrices(parameters):
        a = F.normalize(parameters[..., :3], dim=-1)
        b = F.normalize(parameters[..., 3:] - (parameters[..., 3:] * a).sum(-1, keepdim=True) * a, dim=-1)
        c = torch.cross(a, b, dim=-1)
        return torch.stack([a, b, c], dim=-1)

    def mano(rotations):
        output = layer(global_orient=rotations[:, :1], hand_pose=rotations[:, 1:], betas=betas, pose2rot=False)
        vertices = (output.vertices - output.joints[:, :1]) * mirror_vector
        joints = (output.joints - output.joints[:, :1]) * mirror_vector
        return vertices, joints

    with torch.no_grad():
        initial_vertices, initial_joints = mano(matrices(initial_6d))
    local = initial_joints.cpu().numpy()
    translations = []
    for n, frame in enumerate(ids):
        joint_ids = np.flatnonzero(valid_joints[frame])
        translations.append(np.median(points[frame, joint_ids] - local[n, joint_ids], axis=0))
    translation = torch.nn.Parameter(torch.tensor(np.asarray(translations), dtype=torch.float32, device=device))
    baseline_joints = (initial_joints + translation[:, None]).detach().cpu().numpy()
    initial_vertices_world = (initial_vertices + translation[:, None]).detach().cpu().numpy()
    tensor = lambda value: torch.tensor(value, dtype=torch.float32, device=device)
    target = tensor(np.nan_to_num(points[ids]))
    valid = tensor(valid_joints[ids])
    reliability = tensor(quality['reliability'][ids])
    observation = tensor(pixels[ids])
    # Geometric outliers receive no fitting weight. Uncertain 3D targets are
    # downweighted independently, rather than overpowering clear 2D views.
    fitting_weights = tensor(weights[ids] * masks[ids])
    T, K = tensor(rig.T), tensor(rig.K)
    times = (data['timestamp_ns'][ids] - data['timestamp_ns'][ids[0]]) / 1e9
    dt = np.diff(times)
    continuity = (np.diff(ids) == 1) & (dt > 0) & (dt < .08)
    middle = np.flatnonzero(continuity[:-1] & continuity[1:]) + 1
    temporal_indices = torch.tensor(middle, device=device)
    interpolation = tensor(((times[middle] - times[middle - 1]) / (times[middle + 1] - times[middle - 1]))[:, None, None])
    finger_support = reliability[:, 1:].reshape(-1, 5, 4).mean(-1)
    # MANO's rotation order: index, middle, little, ring, thumb (3 joints each).
    rotation_support = finger_support[:, [1, 2, 4, 3, 0]].repeat_interleave(3, dim=1)

    def project(joints):
        pc = torch.einsum('cij,bkj->bcki', T[:, :3, :3], joints) + T[None, :, None, :3, 3]
        uv = pc[..., :2] / pc[..., 2:].clamp(min=.05)
        return uv * torch.diagonal(K, dim1=1, dim2=2)[None, :, None, :2] + K[None, :, None, :2, 2]

    optimizer = torch.optim.Adam([dict(params=[rotations_6d], lr=.02), dict(params=[translation], lr=.003)])
    history = []
    for iteration in range(args.iterations):
        # A constant final step size can itself cause parameter oscillation.
        decay = .03 + .97 * .5 * (1 + np.cos(np.pi * iteration / max(args.iterations - 1, 1)))
        optimizer.param_groups[0]['lr'] = .02 * decay
        optimizer.param_groups[1]['lr'] = .003 * decay
        optimizer.zero_grad()
        rotations = matrices(rotations_6d)
        vertices, relative_joints = mano(rotations)
        joints = relative_joints + translation[:, None]
        reprojection = F.smooth_l1_loss(project(joints) / 15, observation / 15, reduction='none').sum(-1)
        loss_2d = (reprojection * fitting_weights).sum() / fitting_weights.sum().clamp(min=1)
        loss_3d = (F.smooth_l1_loss(joints / .02, target / .02, reduction='none').sum(-1) * reliability).sum() / valid.sum().clamp(min=1)
        prior = ((rotations[:, 1:] - initial_rotations[:, 1:]) ** 2).mean()
        temporal = joints.sum() * 0
        finger_temporal = joints.sum() * 0
        rotation_temporal = joints.sum() * 0
        if len(middle):
            center = temporal_indices
            smooth = joints[center - 1] * (1 - interpolation) + joints[center + 1] * interpolation
            temporal = F.smooth_l1_loss(joints[center] / .006, smooth / .006)
            # Undo the global wrist orientation and handedness mirror, isolating
            # articulation from translation AND rigid rotation of the whole hand.
            local_joints = torch.matmul(relative_joints * mirror_vector, rotations[:, 0])
            local_smooth = local_joints[center - 1] * (1 - interpolation) + local_joints[center + 1] * interpolation
            finger_error = F.smooth_l1_loss(local_joints[center, 1:] / .004, local_smooth[:, 1:] / .004, reduction='none')
            finger_temporal = (finger_error * (1 + 1.5 * (1 - reliability[center, 1:, None]))).mean()
            # Chordal rotation acceleration avoids Euler angle wrapping. Only
            # the loss uses interpolated matrices; every emitted rotation is SO(3).
            blend = interpolation[:, None]
            rotation_smooth = rotations[center - 1, 1:] * (1 - blend) + rotations[center + 1, 1:] * blend
            rotation_error = F.smooth_l1_loss(rotations[center, 1:] / .06, rotation_smooth / .06, reduction='none')
            rotation_temporal = (rotation_error * (1 + 1.5 * (1 - rotation_support[center, :, None, None]))).mean()
        loss = (loss_2d + .35 * loss_3d + .15 * prior + args.temporal_weight * temporal
                + args.finger_temporal_weight * finger_temporal + args.rotation_temporal_weight * rotation_temporal)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite MANO fitting objective')
        loss.backward()
        optimizer.step()
        if iteration % 25 == 0 or iteration == args.iterations - 1:
            record = dict(iteration=iteration, total=float(loss.detach()), reprojection=float(loss_2d.detach()), triangulation=float(loss_3d.detach()),
                          temporal=float(temporal.detach()), finger_temporal=float(finger_temporal.detach()), rotation_temporal=float(rotation_temporal.detach()))
            history.append(record)
            print(json.dumps(record), flush=True)
    with torch.no_grad():
        rotations = matrices(rotations_6d)
        vertices, joints = mano(rotations)
        vertices = vertices + translation[:, None]
        joints = joints + translation[:, None]
        final_pixels = project(joints).cpu().numpy()
        baseline_pixels = project(tensor(baseline_joints)).cpu().numpy()
        fitted_vertices, fitted_joints = vertices.cpu().numpy(), joints.cpu().numpy()
    active = (weights[ids] * masks[ids]) > 0
    before = np.linalg.norm(baseline_pixels - pixels[ids], axis=-1)[active]
    after = np.linalg.norm(final_pixels - pixels[ids], axis=-1)[active]
    # Keep unsupported frames explicit. Rendering can show a clearly labeled monocular fallback.
    frames = len(valid_frames)
    full_vertices = np.full((frames, 778, 3), np.nan, np.float32)
    full_joints = np.full((frames, 21, 3), np.nan, np.float32)
    full_vertices[ids], full_joints[ids] = fitted_vertices, fitted_joints
    # nanmedian ignores excluded observations; require metric cheirality and a bounded consensus residual.
    accepted = np.nanmedian(np.where(active, np.linalg.norm(final_pixels - pixels[ids], axis=-1), np.nan), axis=(1, 2)) < 35
    camera_depths = np.einsum('ci,bki->bck', rig.T[:, 2, :3], fitted_joints) + rig.T[None, :, 2, 3, None]
    accepted &= np.all((camera_depths > .05) | ~active, axis=(1, 2))
    accepted &= np.isfinite(fitted_vertices).all(axis=(1, 2))
    valid_frames[ids[~accepted]] = False
    full_vertices[~valid_frames], full_joints[~valid_frames] = np.nan, np.nan
    np.savez_compressed(args.output / 'fused_hand.npz', frame_index=data['frame_index'], timestamp_ns=data['timestamp_ns'],
                        valid=valid_frames, vertices=full_vertices, joints=full_joints, selected=selected,
                        is_right=np.full(frames, side, np.int32), shape=shape, fit_frame_rows=ids,
                        mano_rotations=rotations.detach().cpu().numpy(), wrist_translation=translation.detach().cpu().numpy(),
                        initial_joints=baseline_joints, initial_vertices=initial_vertices_world,
                        used_views=np.any(masks, axis=2), support_per_joint=masks.sum(axis=1), joint_reliability=quality['reliability'])
    np.save(args.output / 'mano_faces.npy', layer.faces)
    save_json(args.output / 'calibration.json', calibration)
    summary = dict(status='complete', cache=str(args.cache.resolve()), frames=frames, fitted_frames=int(valid_frames.sum()),
                   target_hand='right' if side else 'left', subject_shape=shape,
                   observed_views_per_fitted_frame={str(n): int(np.sum(np.any(masks[valid_frames], axis=2).sum(axis=1) == n)) for n in range(2, 5)},
                   median_reprojection_before_px=float(np.median(before)), median_reprojection_after_px=float(np.median(after)),
                   p95_reprojection_after_px=float(np.percentile(after, 95)), iterations=args.iterations,
                   temporal_weight=args.temporal_weight, fitting_history=history,
                   fusion_mode=args.fusion_mode, finger_temporal_weight=args.finger_temporal_weight,
                   rotation_temporal_weight=args.rotation_temporal_weight, optimizer_schedule='cosine learning rate decay to 3 percent',
                   visibility_proxy='joint confidence^4, finger-chain confidence, bounded projected finger resolution; no explicit occlusion labels',
                   camera_weight_smoothing=dict(rise_seconds=.10, fall_seconds=.05, reset_gap_seconds=.08,
                                                max_current_weight_multiplier=2, exit_threshold_multiplier=1.25),
                   image_convention='unrectified native images; undistorted keypoints for fitting; distorted projection for display',
                   coordinates='metres in calibrated ZED left optical frame, x right, y down, z forward',
                   limitations='One target hand; estimates depend on calibration, timing and learned keypoints. Unsupported frames remain invalid. Reprojection error is consistency, not ground-truth accuracy.',
                   elapsed_seconds=time.perf_counter() - started)
    save_json(args.output / 'fit_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('fitting_history', 'subject_shape')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
