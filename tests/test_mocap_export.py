"""Bone-frame conventions, arm filtering and mesh helpers used by the Mocap export."""
import numpy as np
import trimesh

from mchoi.export.viewer import decimate
from mchoi.export.mocap import frame_from, quat
from mchoi.arm.filter import OneEuro3D, on_sphere
from scipy.spatial.transform import Rotation


def test_frame_from_matches_reference_axes():
    R = frame_from(np.array([3., 0, 0]), np.array([.2, 0, 1]))
    assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R[:, 0], [1, 0, 0])  # local X along the given direction
    assert np.allclose(R[:, 2], [0, 0, 1])  # Z is the hint made orthogonal to X
    assert np.allclose(R[:, 1], np.cross(R[:, 2], R[:, 0]))
    assert np.allclose(Rotation.from_quat(quat(R)).as_matrix(), R)


def test_arm_filter_smooths_noise_and_bone_length_is_exact():
    rng = np.random.default_rng(0)
    f = OneEuro3D(.7, 4., 1.)
    wrist = np.zeros(3)
    raw, out = [], []
    for k in range(120):
        p = np.array([.27, 0, 0]) + rng.normal(0, .005, 3)
        raw.append(p)
        out.append(on_sphere(wrist, f(p, k / 30), .27))
    raw, out = np.array(raw[20:]), np.array(out[20:])
    assert np.allclose(np.linalg.norm(out - wrist, axis=1), .27)
    assert np.std(np.diff(out, 2, axis=0)) < .3 * np.std(np.diff(raw, 2, axis=0))


def test_decimate_keeps_shape_and_drops_degenerate_faces():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=.05)
    verts, faces = decimate(mesh, .01)
    assert len(faces) < len(mesh.faces)
    assert (faces[:, 0] != faces[:, 1]).all() and (faces[:, 1] != faces[:, 2]).all() and (faces[:, 0] != faces[:, 2]).all()
    assert np.allclose(np.linalg.norm(verts, axis=1), .05, atol=.006)


def test_mesh_fit_transform_scale():
    from mchoi.scene.object_mesh import transform
    rigid = transform(np.array([0, 0, np.pi / 2, .1, .2, .3]))
    assert np.allclose(rigid[:3, :3] @ [1, 0, 0], [0, 1, 0]) and np.allclose(rigid[:3, 3], [.1, .2, .3])
    scaled = transform(np.array([0, 0, 0, 0, 0, 0, np.log(2)]))
    assert np.allclose(scaled[:3, :3], 2 * np.eye(3))
    axes = transform(np.array([0, 0, 0, 0, 0, 0, 0, np.log(2), np.log(.5)]))
    assert np.allclose(np.diag(axes[:3, :3]), [1, 2, .5])


def test_load_object_primary_only_when_secondary_never_seen(tmp_path):
    import csv
    from mchoi.export.mocap import load_object
    path = tmp_path / 'poses.csv'
    fields = ['zed_left_frame_index', 'pose_valid', 'marker_id', 'tx_m', 'ty_m', 'tz_m'] + [f'R{i}{j}' for i in range(3) for j in range(3)]
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for k in range(5):
            w.writerow(dict(zed_left_frame_index=k, pose_valid=1, marker_id=2, tx_m=.01 * k, ty_m=0, tz_m=.5,
                            **{f'R{i}{j}': float(i == j) for i in range(3) for j in range(3)}))
    poses, raw, info = load_object(path, list(range(6)), np.arange(6) / 30, None, None, (2, 1))
    assert sorted(poses) == [0, 1, 2, 3, 4] and info['frames_from_secondary'] == 0
    assert np.allclose(poses[3][1], [.03, 0, .5])


def _cdr_image(encoding, height, width, step, data):
    import struct

    def pad(buf):
        return buf + b'\x00' * (-len(buf) % 4)
    blob = b'\x00\x01\x00\x00' + struct.pack('<II', 0, 0)
    for text in ('frame',):
        raw = text.encode() + b'\x00'
        blob = pad(blob) + struct.pack('<I', len(raw)) + raw
    blob = pad(blob) + struct.pack('<II', height, width)
    raw = encoding.encode() + b'\x00'
    blob = pad(blob) + struct.pack('<I', len(raw)) + raw + b'\x00'
    blob = pad(blob) + struct.pack('<II', step, len(data)) + data
    return blob


def test_decode_realsense_images():
    from mchoi.dataset.camera_videos import decode
    depth = (np.arange(12, dtype=np.uint16) * 1000).reshape(3, 4)
    enc, out = decode(_cdr_image('mono16', 3, 4, 8, depth.tobytes()))
    assert enc == 'mono16' and np.array_equal(out, depth)
    rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    enc, out = decode(_cdr_image('bgr8', 3, 4, 12, rgb.tobytes()))
    assert enc == 'bgr8' and np.array_equal(out, rgb)
