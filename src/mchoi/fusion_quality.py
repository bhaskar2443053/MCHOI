"""Visibility proxies and temporal confidence for calibrated finger observations.

Detector confidence is not an occlusion label. Projected finger resolution is an
additional, deliberately bounded proxy; geometric consensus remains mandatory.
"""
import numpy as np

FINGERS = np.arange(1, 21).reshape(5, 4)


def visibility_weights(keypoints, hand_confidence, delta_ns, image_sizes):
    """Return per-joint weights and per-finger clarity, without moving keypoints."""
    keypoints = np.asarray(keypoints, float)
    xy = np.nan_to_num(keypoints[..., :2], nan=0, posinf=0, neginf=0)
    confidence = np.clip(np.nan_to_num(keypoints[..., 2]), 0, 1)
    valid = np.isfinite(keypoints).all(axis=-1) & (confidence >= .35)
    valid &= (xy > 0).all(axis=-1) & (xy < np.asarray(image_sizes)[:, None]).all(axis=-1)
    confidence = np.where(valid, confidence, 0)
    finger_confidence = confidence[:, FINGERS]
    # Combine the joint's own confidence with evidence from its entire finger.
    chain_confidence = np.mean(finger_confidence, axis=-1)
    segments = np.linalg.norm(np.diff(xy[:, FINGERS], axis=2), axis=-1)
    segment_valid = valid[:, FINGERS][:, :, 1:] & valid[:, FINGERS][:, :, :-1]
    length = (segments * segment_valid).sum(axis=-1)
    relative_resolution = np.sqrt(length / np.maximum(length.max(axis=0, keepdims=True), 1))
    resolution = np.clip(relative_resolution, .45, 1) * np.clip(length / 60, .5, 1)
    clarity = chain_confidence * resolution
    weights = confidence ** 2
    weights[:, 1:] = (finger_confidence ** 4 * clarity[..., None]).reshape(-1, 20)
    timing = 1 / (1 + (np.abs(np.asarray(delta_ns, float)) / 20e6) ** 2)
    weights *= np.clip(np.asarray(hand_confidence), 0, 1)[:, None] * timing[:, None]
    return weights, clarity


class StableViewWeights:
    """Smooth weight changes, reset on gaps/track changes, drop invalid input now.

    This filters confidence, not coordinates, so it introduces no unequal
    keypoint-position delay between cameras before triangulation.
    """
    def __init__(self, rise_seconds=.10, fall_seconds=.05, reset_gap=.08):
        if not np.isfinite([rise_seconds, fall_seconds, reset_gap]).all() or min(rise_seconds, fall_seconds, reset_gap) <= 0:
            raise ValueError('Weight time constants must be positive and finite')
        self.rise, self.fall, self.gap = rise_seconds, fall_seconds, reset_gap
        self.timestamp = self.weights = self.tracks = None

    def update(self, weights, timestamp, tracks):
        weights, tracks = np.asarray(weights, float), np.asarray(tracks)
        if not np.isfinite(weights).all() or (weights < 0).any() or not np.isfinite(timestamp):
            raise ValueError('Confidence input must be finite and nonnegative')
        if self.timestamp is None:
            result = weights.copy()
        else:
            dt = timestamp - self.timestamp
            if dt <= 0:
                raise ValueError('Confidence timestamps must strictly increase')
            tau = np.where(weights > self.weights, self.rise, self.fall)
            alpha = 1 - np.exp(-dt / tau)
            result = self.weights + alpha * (weights - self.weights)
            # A suddenly obscured finger must not inherit a large old weight.
            result = np.minimum(result, 2 * weights)
            reset = (tracks != self.tracks) | (dt > self.gap)
            result[reset] = weights[reset]
            # No retained confidence for a missing/invalid observation.
            result[weights == 0] = 0
        self.timestamp, self.weights, self.tracks = timestamp, result.copy(), tracks.copy()
        return result


def triangulation_reliability(rig, points, weights, masks, errors, threshold):
    """Bounded support score, not a calibrated statistical covariance.

    Penalize weak confidence, limited effective camera support, narrow ray angles,
    and residual disagreement. Unsupported joints always have zero reliability.
    """
    result = np.zeros(points.shape[0])
    for joint in np.flatnonzero(np.isfinite(points).all(axis=-1)):
        cameras = np.flatnonzero(masks[:, joint] & (weights[:, joint] > 0))
        if len(cameras) < 2:
            continue
        w = weights[cameras, joint]
        effective_views = w.sum() ** 2 / np.maximum(np.square(w).sum(), 1e-12)
        confidence = np.sum(w * w) / np.maximum(w.sum(), 1e-12)
        rays = points[joint] - rig.centers[cameras]
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        sine = np.sqrt(np.maximum(0, 1 - np.square(rays @ rays.T)))
        # Require meaningful confidence in BOTH members of a well-separated pair.
        pair_weight = np.minimum(w[:, None], w[None]) / max(w.max(), 1e-12)
        geometry = np.clip(np.max(sine * pair_weight) / np.sin(np.deg2rad(10)), .05, 1)
        residual = np.average(np.square(errors[cameras, joint]), weights=w)
        agreement = 1 / (1 + residual / (.6 * threshold) ** 2)
        result[joint] = np.clip(confidence * np.clip(effective_views / 3, .3, 1) * geometry * agreement, .02, 1)
    return result
