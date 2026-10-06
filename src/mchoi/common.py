"""Small shared helpers: JSON output, WiLoR debug drawing, marker geometry and rotation angles."""
import json
from pathlib import Path

import cv2
import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value.resolve())
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False), encoding='utf-8')


def draw_frame(image, hands, faces, index, seconds):
    canvas = image.copy()
    colors = [(255, 180, 30), (70, 230, 100)]  # left / right (BGR)
    mesh = image.copy()
    for hand in hands:
        points = np.rint(np.clip(hand['vertices_2d'], -10000, 10000)).astype(np.int32)
        cv2.polylines(mesh, points[faces], True, colors[hand['is_right']], 1, cv2.LINE_AA)
    cv2.addWeighted(mesh, 0.45, canvas, 0.55, 0, canvas)
    for hand in hands:
        color = colors[hand['is_right']]
        for key, box_color, thickness in [('raw_box', (150, 150, 150), 1),
                                          ('expanded_box', color, 2)]:
            x1, y1, x2, y2 = np.rint(hand[key]).astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)
        points = np.rint(np.clip(hand['joints_2d'], -10000, 10000)).astype(int)
        for base in [1, 5, 9, 13, 17]:
            chain = [0] + list(range(base, base + 4))
            for a, b in zip(chain, chain[1:]):
                cv2.line(canvas, tuple(points[a]), tuple(points[b]), color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, tuple(point), 3, (255, 255, 255), -1, cv2.LINE_AA)
        x, y = np.rint(hand['expanded_box'][:2]).astype(int)
        label = f"{'R' if hand['is_right'] else 'L'} #{hand['track_id']}  {hand['confidence']:.2f}"
        cv2.putText(canvas, label, (max(0, x), max(100, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 77), (25, 25, 25), -1)
    cv2.putText(canvas, f'WiLoR | ZED left | frame {index} | {seconds:.2f}s | {len(hands)} hands',
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'YOLO -> One Euro detection boxes -> +10% crop -> WiLoR   |   grey: raw detection',
                (20, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def square_points(side_m):
    if not np.isfinite(side_m) or side_m <= 0:
        raise ValueError('Marker side must be a positive length in metres')
    half = side_m / 2
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], np.float64)


def roll_pitch_yaw(rotation):
    """Degrees, R = Rz(yaw) @ Ry(pitch) @ Rx(roll), fixed camera axes."""
    c = np.hypot(rotation[0, 0], rotation[1, 0])
    if c > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, np.arctan2(-rotation[2, 0], c), yaw])


def rotation_angle(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))
