"""One Euro filter for the fitted right shoulder and elbow, preserving the MANO wrist and the bone lengths.

The cutoff follows the filtered 3D joint speed (m/s): cutoff = min_cutoff + beta * speed. Filters reset when a joint
is missing or the source time gap exceeds --reset-gap. After filtering, the elbow is put back on the forearm-length
sphere around the (already filtered) MANO wrist and the shoulder on the upper-arm sphere around the filtered elbow.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SHOULDER, ELBOW, WRIST = 0, 1, 2


class OneEuro3D:
    """One Euro filter on a 3D point; the cutoff uses the norm of the filtered velocity."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = self.raw = self.value = self.velocity = None

    @staticmethod
    def alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (1 + r)

    def __call__(self, value, timestamp):
        if self.t is None:
            self.raw, self.value, self.velocity = value.copy(), value.copy(), np.zeros(3)
        else:
            dt = timestamp - self.t
            self.velocity += self.alpha(self.d_cutoff, dt) * ((value - self.raw) / dt - self.velocity)
            a = self.alpha(self.min_cutoff + self.beta * np.linalg.norm(self.velocity), dt)
            self.value = a * value + (1 - a) * self.value
            self.raw = value.copy()
        self.t = timestamp
        return self.value.copy()


def on_sphere(center, point, length):
    d = point - center
    return center + length * d / max(np.linalg.norm(d), 1e-9)


def jitter_mm(points, stamps):
    """Median magnitude of the second difference (acceleration proxy) over consecutive valid frames, in mm."""
    ok = np.isfinite(points[:, 0])
    good = ok[2:] & ok[1:-1] & ok[:-2] & (np.diff(stamps)[1:] < .1) & (np.diff(stamps)[:-1] < .1)
    second = points[2:] - 2 * points[1:-1] + points[:-2]
    return float(np.median(np.linalg.norm(second[good], axis=1)) * 1000) if good.any() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm3d', type=Path, help='Output of fit_right_arm.py')
    parser.add_argument('--output', type=Path, required=True, help='.npz path; must not exist')
    parser.add_argument('--elbow-min-cutoff', type=float, default=.7)
    parser.add_argument('--elbow-beta', type=float, default=4.)
    parser.add_argument('--shoulder-min-cutoff', type=float, default=.3)
    parser.add_argument('--shoulder-beta', type=float, default=2.)
    parser.add_argument('--d-cutoff', type=float, default=1.)
    parser.add_argument('--reset-gap', type=float, default=.1, help='Seconds')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    arm = np.load(args.arm3d)
    raw = arm['joints']
    stamps = (arm['timestamp_ns'] - arm['timestamp_ns'][0]) / 1e9
    lengths = {ELBOW: float(arm['forearm_length']), SHOULDER: float(arm['upper_arm_length'])}
    filters = {ELBOW: OneEuro3D(args.elbow_min_cutoff, args.elbow_beta, args.d_cutoff),
               SHOULDER: OneEuro3D(args.shoulder_min_cutoff, args.shoulder_beta, args.d_cutoff)}
    out = np.full_like(raw, np.nan)
    out[:, WRIST] = raw[:, WRIST]
    resets = {ELBOW: 0, SHOULDER: 0}
    for t in range(len(raw)):
        for j, parent in ((ELBOW, WRIST), (SHOULDER, ELBOW)):
            f = filters[j]
            if not np.isfinite(raw[t, j, 0]) or not np.isfinite(out[t, parent, 0]):
                f.reset()
                continue
            if f.t is not None and stamps[t] - f.t > args.reset_gap:
                f.reset()
            if f.t is None:
                resets[j] += 1
            out[t, j] = on_sphere(out[t, parent], f(raw[t, j], stamps[t]), lengths[j])
    report = dict(settings=vars(args) | dict(arm3d=str(args.arm3d), output=str(args.output)), resets={'elbow': resets[ELBOW], 'shoulder': resets[SHOULDER]},
                  jitter_mm={name: dict(raw=jitter_mm(raw[:, j], stamps), filtered=jitter_mm(out[:, j], stamps))
                             for j, name in ((SHOULDER, 'shoulder'), (ELBOW, 'elbow'), (WRIST, 'wrist'))},
                  max_bone_length_error_mm=float(max(np.nanmax(np.abs(np.linalg.norm(out[:, j] - out[:, p], axis=1) - lengths[j]))
                                                     for j, p in ((ELBOW, WRIST), (SHOULDER, ELBOW))) * 1000))
    np.savez_compressed(args.output, frame_index=arm['frame_index'], timestamp_ns=arm['timestamp_ns'], joints=out, raw_joints=raw,
                        method=arm['method'], views=arm['views'], joint_names=arm['joint_names'],
                        forearm_length=lengths[ELBOW], upper_arm_length=lengths[SHOULDER], settings=json.dumps(report['settings']))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
