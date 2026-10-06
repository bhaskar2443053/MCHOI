"""Calibrated association and robust triangulation of hand observations."""
from itertools import combinations
import cv2
import numpy as np
from mchoi.fusion_quality import StableViewWeights, visibility_weights, triangulation_reliability

CAMERAS = ('zed_left', 'zed_right', 'd435', 'd455')


class Rig:
    def __init__(self, calibration):
        if calibration['reference_camera'] != 'zed_left' or calibration['units'] != 'metres':
            raise ValueError('Expected metric ZED-left-reference calibration')
        self.K, self.D, self.T, self.image_sizes = [], [], [], []
        for name in CAMERAS:
            camera = calibration['cameras'][name]
            self.K.append(np.asarray(camera['camera_matrix'], float))
            self.D.append(np.asarray(camera['distortion_coefficients'], float))
            self.T.append(np.asarray(camera['T_camera_from_reference'], float))
            self.image_sizes.append(camera.get('image_size', [1920, 1080]))
        self.K, self.T = np.asarray(self.K), np.asarray(self.T)
        self.centers = np.asarray([np.linalg.inv(t)[:3, 3] for t in self.T])

    def undistort(self, points, camera, pixels=True):
        return cv2.undistortPoints(np.asarray(points, float).reshape(-1, 1, 2), self.K[camera], self.D[camera],
                                   P=self.K[camera] if pixels else None).reshape(-1, 2)

    def project(self, points, camera, distorted=False):
        points = np.asarray(points, float)
        T = self.T[camera]
        if distorted:
            return cv2.projectPoints(points, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], self.K[camera], self.D[camera])[0].reshape(-1, 2)
        pc = points @ T[:3, :3].T + T[:3, 3]
        return (pc[:, :2] / pc[:, 2:]) * np.diag(self.K[camera])[:2] + self.K[camera][:2, 2]

    def epipolar_distance(self, a, b, ca, cb):
        relation = self.T[cb] @ np.linalg.inv(self.T[ca])
        t = relation[:3, 3]
        skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        F = np.linalg.inv(self.K[cb]).T @ skew @ relation[:3, :3] @ np.linalg.inv(self.K[ca])
        pa = np.c_[self.undistort(a, ca), np.ones(len(a))]
        pb = np.c_[self.undistort(b, cb), np.ones(len(b))]
        la, lb = pb @ F, pa @ F.T
        numerator = np.abs(np.sum(pb * lb, axis=1))
        return numerator * .5 * (1 / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-12) +
                                  1 / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-12))

    def triangulate(self, pixels, weights, threshold=20, previous_mask=None):
        """Pair hypotheses + reprojection consensus + weighted DLT refinement, per joint.

        pixels are undistorted pixel coordinates (C,J,2). Missing/weak observations
        have weight zero. Returns metric points, consensus mask, and pixel residuals.
        """
        count = pixels.shape[1]
        finite = np.isfinite(pixels).all(axis=-1) & np.isfinite(weights)
        weights = np.where(finite & (weights > 0), weights, 0)
        pixels = np.nan_to_num(pixels, nan=0, posinf=0, neginf=0)
        normalized = (pixels - self.K[:, None, :2, 2]) / np.diagonal(self.K, axis1=1, axis2=2)[:, None, :2]
        best_score = np.full(count, -np.inf)
        best = np.full((count, 3), np.nan)
        best_mask = np.zeros((4, count), bool)
        limits = np.full((4, count), threshold, dtype=float)
        if previous_mask is not None:
            # Previously consistent views may stay up to 25% beyond the entry
            # threshold. Gross outliers still leave immediately, even at high confidence.
            limits[np.asarray(previous_mask, bool)] *= 1.25
        for a, b in combinations(range(4), 2):
            X = cv2.triangulatePoints(self.T[a, :3], self.T[b, :3], normalized[a].T, normalized[b].T).T
            denom = X[:, 3:]
            xyz = np.divide(X[:, :3], denom, out=np.full((count, 3), 1e6), where=np.abs(denom) > 1e-10)
            rays_a, rays_b = xyz - self.centers[a], xyz - self.centers[b]
            cosine = np.sum(rays_a * rays_b, axis=1) / np.maximum(np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1), 1e-12)
            angle = np.arccos(np.clip(cosine, -1, 1))
            pc = np.einsum('cij,kj->cki', self.T[:, :3, :3], xyz) + self.T[:, None, :3, 3]
            projected = pc[:, :, :2] / np.maximum(pc[:, :, 2:], 1e-9)
            projected = projected * np.diagonal(self.K, axis1=1, axis2=2)[:, None, :2] + self.K[:, None, :2, 2]
            errors = np.linalg.norm(projected - pixels, axis=2)
            mask = (errors < limits) & (weights > 0) & (pc[:, :, 2] > .05)
            valid = (weights[a] > 0) & (weights[b] > 0) & mask[a] & mask[b] & (angle > np.deg2rad(1))
            valid &= (mask.sum(axis=0) >= 2) & (xyz[:, 2] > .05) & (np.linalg.norm(xyz, axis=1) < 10)
            score = (weights * mask * (1 - 0.2 * np.minimum(errors / threshold, 1))).sum(axis=0)
            if previous_mask is not None:
                score += .06 * (weights * mask * previous_mask).sum(axis=0)
            improve = valid & (score > best_score)
            best[improve], best_score[improve], best_mask[:, improve] = xyz[improve], score[improve], mask[:, improve]
        for joint in np.flatnonzero(np.isfinite(best[:, 0])):
            rows = []
            reference_errors = np.array([np.linalg.norm(self.project(best[joint:joint + 1], c)[0] - pixels[c, joint]) for c in range(4)])
            for cam in np.flatnonzero(best_mask[:, joint]):
                p = self.T[cam, :3]
                x, y = normalized[cam, joint]
                w = np.sqrt(weights[cam, joint])
                if previous_mask is not None:
                    w /= np.sqrt(1 + (reference_errors[cam] / (.65 * threshold)) ** 2)
                rows.extend([w * (x * p[2] - p[0]), w * (y * p[2] - p[1])])
            _, _, vh = np.linalg.svd(rows)
            if abs(vh[-1, 3]) < 1e-10:
                continue
            candidate = vh[-1, :3] / vh[-1, 3]
            errors = np.array([np.linalg.norm(self.project(candidate[None], cam)[0] - pixels[cam, joint]) for cam in range(4)])
            used = best_mask[:, joint]
            depth = np.einsum('ci,i->c', self.T[:, 2, :3], candidate) + self.T[:, 2, 3]
            bound = threshold * 1.5 if previous_mask is None else limits[used, joint]
            if np.isfinite(candidate).all() and candidate[2] > .05 and np.all(errors[used] <= bound) and np.all(depth[used] > .05):
                best[joint] = candidate
        errors = np.full((4, count), np.nan)
        valid = np.isfinite(best[:, 0])
        for cam in range(4):
            errors[cam, valid] = np.linalg.norm(self.project(best[valid], cam) - pixels[cam, valid], axis=1)
        return best, best_mask, errors


def select_observations(data, frame, rig, side):
    """Associate one target hand, anchored in ZED left when visible, using palm epipolar consistency."""
    candidates = []
    for begin, end in data['offsets'][frame]:
        ids = list(range(begin, end))
        candidates.append(ids)
    available = [c for c in range(4) if candidates[c]]
    selected = np.full(4, -1, np.int64)
    if not available:
        return selected
    anchors = [c for c in available if any(int(data['is_right'][i]) == side for i in candidates[c])]
    if not anchors:
        return selected
    anchor = anchors[0]
    seed = max((i for i in candidates[anchor] if int(data['is_right'][i]) == side), key=lambda i: data['confidence'][i])
    selected[anchor] = seed
    palm = [0, 1, 5, 9, 13, 17]
    for cam in available:
        if cam == anchor:
            continue
        scores = []
        for obs in candidates[cam]:
            distance = np.median(rig.epipolar_distance(data['joints_2d'][seed][palm], data['joints_2d'][obs][palm], anchor, cam))
            side_penalty = 15 * (int(data['is_right'][obs]) != side)
            scores.append((distance + side_penalty + 5 * (1 - data['confidence'][obs]), obs, distance))
        _, obs, distance = min(scores)
        if distance < 60:
            selected[cam] = obs
    return selected


def prepare_observations(data, rig, side, threshold=20, mode='visibility', return_quality=False):
    if mode not in ('visibility', 'legacy'):
        raise ValueError('Unknown camera weighting mode')
    frames = len(data['frame_index'])
    selected = np.full((frames, 4), -1, np.int64)
    pixels = np.zeros((frames, 4, 21, 2))
    weights = np.zeros((frames, 4, 21))
    points = np.full((frames, 21, 3), np.nan)
    masks = np.zeros((frames, 4, 21), bool)
    raw_weights = np.zeros_like(weights)
    clarity = np.zeros((frames, 4, 5))
    errors = np.full_like(weights, np.nan)
    reliability = np.zeros((frames, 21))
    smoother = StableViewWeights()
    previous_tracks = np.full(4, -1)
    for frame in range(frames):
        selected[frame] = select_observations(data, frame, rig, side)
        all_keypoints = np.zeros((4, 21, 3))
        hand_confidence = np.zeros(4)
        tracks = np.full(4, -1)
        for cam, obs in enumerate(selected[frame]):
            if obs < 0:
                continue
            keypoints = data['detector_keypoints'][obs]
            all_keypoints[cam], hand_confidence[cam] = keypoints, data['confidence'][obs]
            tracks[cam] = data['track_id'][obs]
            pixels[frame, cam] = rig.undistort(np.nan_to_num(keypoints[:, :2], nan=0, posinf=0, neginf=0), cam)
            confidence = np.nan_to_num(keypoints[:, 2], nan=0, posinf=0, neginf=0)
            valid = np.isfinite(keypoints).all(axis=-1) & (confidence >= .35) & (keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)
            time_weight = 1 / (1 + (abs(data['sync'][frame, cam, 2]) / 20e6) ** 2)
            weights[frame, cam] = valid * confidence ** 2 * data['confidence'][obs] * time_weight
        previous_mask = None
        if mode == 'visibility':
            raw_weights[frame], clarity[frame] = visibility_weights(all_keypoints, hand_confidence, data['sync'][frame, :, 2], rig.image_sizes)
            seconds = float(data['timestamp_ns'][frame] - data['timestamp_ns'][0]) / 1e9
            weights[frame] = smoother.update(raw_weights[frame], seconds, tracks)
            previous_mask = np.zeros((4, 21), bool)
            if frame and 0 < data['timestamp_ns'][frame] - data['timestamp_ns'][frame - 1] < 80e6:
                previous_mask[:] = masks[frame - 1]
                previous_mask[tracks != previous_tracks] = False
        else:
            raw_weights[frame] = weights[frame]
        points[frame], masks[frame], errors[frame] = rig.triangulate(pixels[frame], weights[frame], threshold, previous_mask)
        if mode == 'visibility':
            # Smooth robust falloff inside the consensus, hard zero outside it.
            taper = 1 / (1 + (np.nan_to_num(errors[frame], nan=1e6) / (.65 * threshold)) ** 2)
            weights[frame] *= taper
            reliability[frame] = triangulation_reliability(rig, points[frame], weights[frame], masks[frame], errors[frame], threshold)
        else:
            reliability[frame] = np.isfinite(points[frame, :, 0])
        previous_tracks = tracks
    result = (selected, pixels, weights, points, masks)
    if return_quality:
        return (*result, dict(raw_weights=raw_weights, finger_clarity=clarity, reprojection_errors=errors, reliability=reliability))
    return result
