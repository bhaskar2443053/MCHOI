"""Causal One Euro filtering on SO(3), with independent angular speed per joint."""
import numpy as np
from scipy.spatial.transform import Rotation


class OneEuroRotationFilter:
    """One Euro cutoff law with shortest-path rotation interpolation.

    Cutoffs are Hz; angular speed is radians/second. Beta therefore cannot be
    copied from a pixel-coordinate or normalized-box filter. The derivative
    uses consecutive RAW rotations, expressed in the rotation's parent frame.
    """
    def __init__(self, min_cutoff=1., beta=1., d_cutoff=1., reset_gap=.1):
        if not np.isfinite([min_cutoff, beta, d_cutoff, reset_gap]).all():
            raise ValueError('Filter parameters must be finite')
        if min(min_cutoff, d_cutoff, reset_gap) <= 0 or beta < 0:
            raise ValueError('Cutoffs/gap must be positive and beta nonnegative')
        self.min_cutoff, self.beta, self.d_cutoff, self.reset_gap = min_cutoff, beta, d_cutoff, reset_gap
        self.reset()

    def reset(self):
        self.timestamp = self.raw = self.filtered = self.velocity = None
        self.last_cutoff = None

    @staticmethod
    def alpha(cutoff, dt):
        r = 2 * np.pi * np.asarray(cutoff) * dt
        return r / (1 + r)

    def __call__(self, matrices, timestamp):
        matrices = np.asarray(matrices, float)
        if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all() or not np.isfinite(timestamp):
            raise ValueError('Expected finite rotation matrices and timestamp')
        if not np.allclose(matrices @ matrices.swapaxes(-1, -2), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(matrices), 1, atol=1e-4):
            raise ValueError('Input must contain proper rotations')
        shape = matrices.shape
        raw = matrices.reshape(-1, 3, 3)
        if self.timestamp is not None:
            dt = timestamp - self.timestamp
            if dt <= 0:
                raise ValueError('Filter timestamps must strictly increase')
            if dt > self.reset_gap:
                self.reset()
        if self.timestamp is None:
            self.filtered = raw.copy()
            self.velocity = np.zeros((len(raw), 3))
            self.last_cutoff = np.full(len(raw), self.min_cutoff)
        else:
            if raw.shape != self.raw.shape:
                raise ValueError('Joint count changed without a filter reset')
            delta = raw @ self.raw.swapaxes(-1, -2)
            velocity = Rotation.from_matrix(delta).as_rotvec() / dt
            self.velocity += self.alpha(self.d_cutoff, dt) * (velocity - self.velocity)
            self.last_cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.velocity, axis=-1)
            alpha = self.alpha(self.last_cutoff, dt)
            error = Rotation.from_matrix(raw @ self.filtered.swapaxes(-1, -2)).as_rotvec()
            step = Rotation.from_rotvec(alpha[:, None] * error).as_matrix()
            self.filtered = step @ self.filtered
        self.raw, self.timestamp = raw.copy(), float(timestamp)
        return self.filtered.reshape(shape).copy()


def filter_mano_rotations(fused, min_cutoff=1., beta=1., d_cutoff=1.,
                          wrist_min_cutoff=0., wrist_beta=3., reset_gap=.1):
    """Filter accepted MANO frames only; reset across missing frames and side changes.

    A wrist cutoff of zero disables global rotation filtering. Wrist translation
    is always retained from the source fit. Rejected parameter rows are preserved
    for provenance, but never become accepted poses or enter filter state.
    """
    fingers = OneEuroRotationFilter(min_cutoff, beta, d_cutoff, reset_gap)
    wrist = OneEuroRotationFilter(wrist_min_cutoff, wrist_beta, d_cutoff, reset_gap) if wrist_min_cutoff > 0 else None
    rotations = np.array(fused['mano_rotations'], dtype=float, copy=True)
    rows = fused['fit_frame_rows']
    previous = None
    resets = []
    for n, row in enumerate(rows):
        if not fused['valid'][row]:
            previous = None
            continue
        timestamp = float(fused['timestamp_ns'][row] - fused['timestamp_ns'][0]) / 1e9
        if (previous is None or row != previous + 1 or
                fused['frame_index'][row] != fused['frame_index'][previous] + 1 or
                fused['is_right'][row] != fused['is_right'][previous] or
                (fused['timestamp_ns'][row] - fused['timestamp_ns'][previous]) / 1e9 > reset_gap):
            fingers.reset()
            if wrist is not None:
                wrist.reset()
            resets.append(int(row))
        rotations[n, 1:] = fingers(rotations[n, 1:], timestamp)
        if wrist is not None:
            rotations[n, 0] = wrist(rotations[n, 0], timestamp)
        previous = row
    return rotations.astype(np.float32), resets
