"""Timestamp-aware filtering of detections, before any image crop or pose inference."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


class OneEuroFilter:
    """Vector 1 Euro filter; derivative uses consecutive unfiltered observations."""

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        if not np.isfinite([min_cutoff, beta, d_cutoff]).all():
            raise ValueError("Filter parameters must be finite")
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("Cutoffs must be positive and beta nonnegative")
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.t = self.raw = self.value = self.derivative = None

    @staticmethod
    def alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (1 + r)

    def __call__(self, value, timestamp):
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all() or not np.isfinite(timestamp):
            raise ValueError("Filter input must be finite")
        if self.t is None:
            self.raw = self.value = value.copy()
            self.derivative = np.zeros_like(value)
        else:
            dt = timestamp - self.t
            if dt <= 0:
                raise ValueError("Filter timestamps must strictly increase")
            derivative = (value - self.raw) / dt
            a_d = self.alpha(self.d_cutoff, dt)
            self.derivative += a_d * (derivative - self.derivative)
            a = self.alpha(self.min_cutoff + self.beta * np.abs(self.derivative), dt)
            self.value = a * value + (1 - a) * self.value
            self.raw = value.copy()
        self.t = timestamp
        return self.value.copy()


def box_state(box):
    box = np.asarray(box, dtype=np.float64)
    return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]]


def state_box(state):
    return np.r_[state[:2] - state[2:] / 2, state[:2] + state[2:] / 2]


def expand_box(box, factor=1.10):
    """Grow width AND height by factor around the center; retain out-of-frame padding."""
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("Crop scale must be finite and at least 1")
    state = box_state(box)
    state[2:] *= factor
    return state_box(state)


def iou(a, b):
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = np.prod(overlap)
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return intersection / max(union, 1e-12)


@dataclass
class Track:
    identity: int
    raw_box: np.ndarray
    last_seen: float
    right_score: float
    score_weight: float
    side: int
    filter: OneEuroFilter


class DetectionTracker:
    """Spatial assignment, stable handedness and an independent box filter per track.

    Only current detections are returned. Misses never generate invented poses.
    Box center/size are normalized by image dimensions, so beta is resolution independent.
    """

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0,
                 max_age=0.5, reset_gap=0.25, crop_scale=1.10):
        OneEuroFilter(min_cutoff, beta, d_cutoff)
        expand_box([0, 0, 1, 1], crop_scale)
        if not np.isfinite([max_age, reset_gap]).all() or min(max_age, reset_gap) <= 0:
            raise ValueError("Track timeouts must be positive and finite")
        self.filter_kwargs = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.max_age, self.reset_gap, self.crop_scale = max_age, reset_gap, crop_scale
        self.tracks = {}
        self.next_id = 0
        self.last_timestamp = None

    def update(self, detections, timestamp, image_size):
        """detections: N x 6 [x1,y1,x2,y2,confidence,is_right]; size: (width,height)."""
        if not np.isfinite(timestamp):
            raise ValueError("Timestamp must be finite")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Frame timestamps must strictly increase")
        self.last_timestamp = timestamp
        width, height = image_size
        if min(width, height) <= 0:
            raise ValueError("Image size must be positive")
        normalizer = np.array([width, height, width, height], dtype=float)
        detections = np.asarray(detections, dtype=float).reshape(-1, 6)
        valid = np.isfinite(detections).all(axis=1)
        valid &= (detections[:, 2:4] > detections[:, :2]).all(axis=1)
        valid &= np.isin(detections[:, 5], [0, 1])
        detections = detections[valid]
        self.tracks = {k: t for k, t in self.tracks.items() if timestamp - t.last_seen <= self.max_age}
        existing = list(self.tracks.values())
        cost = np.full((len(existing), len(detections)), 1e6)
        for i, track in enumerate(existing):
            old = box_state(track.raw_box)
            for j, det in enumerate(detections):
                new = box_state(det[:4])
                distance = np.linalg.norm(old[:2] - new[:2]) / max(np.linalg.norm(old[2:]), np.linalg.norm(new[2:]), 1)
                overlap = iou(track.raw_box, det[:4])
                ratio = np.max(np.maximum(old[2:] / new[2:], new[2:] / old[2:]))
                if (overlap >= 0.05 or distance <= 0.8) and ratio < 3:
                    side_penalty = 0.35 * (track.side != int(det[5]))
                    cost[i, j] = 1 - overlap + 0.5 * distance + side_penalty
        assigned = {}
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            assigned = {j: existing[i] for i, j in zip(rows, cols) if cost[i, j] < 1e6}
        result = []
        for j, det in enumerate(detections):
            track = assigned.get(j)
            reset = track is None
            if track is None:
                track = Track(self.next_id, det[:4].copy(), timestamp, float(det[5]), float(det[4]), int(det[5]),
                              OneEuroFilter(**self.filter_kwargs))
                self.tracks[track.identity] = track
                self.next_id += 1
            else:
                gap = timestamp - track.last_seen
                reset = gap > self.reset_gap
                if reset:
                    track.filter = OneEuroFilter(**self.filter_kwargs)
                # Anatomical side is constant across a physical track. Accumulate
                # confidence-weighted evidence instead of flipping on brief ambiguity.
                track.score_weight += float(det[4])
                side_alpha = float(det[4]) / max(track.score_weight, 1e-12)
                track.right_score += side_alpha * (det[5] - track.right_score)
                if track.right_score > 0.6:
                    track.side = 1
                elif track.right_score < 0.4:
                    track.side = 0
            filtered = track.filter(box_state(det[:4]) / normalizer, timestamp) * normalizer
            smooth_box = state_box(filtered)
            track.raw_box = det[:4].copy()
            track.last_seen = timestamp
            result.append(dict(track_id=track.identity, confidence=float(det[4]),
                               detected_is_right=int(det[5]), is_right=track.side,
                               raw_box=det[:4].copy(), filtered_box=smooth_box,
                               expanded_box=expand_box(smooth_box, self.crop_scale), filter_reset=reset))
        return result
