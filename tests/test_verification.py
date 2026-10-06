import copy
import json

import cv2
import numpy as np
import pytest

from mchoi.calibration.board import load_board
from mchoi.calibration.cameras import VIEW_ORDER
from mchoi.calibration.solver import transform
from mchoi.calibration.verification import (StereoDepth, VerificationError, board_point, check_frames,
                                     load_calibration, project_reference_point, realsense_point,
                                     validate_calibration)
from mchoi.paths import CONFIG_DIR




def rig_calibration(distorted=False):
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    transforms = {"zed_left": np.eye(4), "zed_right": transform(np.eye(3), [-.064, 0, 0]),
                  "d435": transform(cv2.Rodrigues(np.array([.05, -.12, .02]))[0], [.12, .02, 0]),
                  "d455": transform(cv2.Rodrigues(np.array([-.05, .12, -.02]))[0], [-.13, .03, .02])}
    d = [-.12, .02, .001, -.002, 0] if distorted else [0]*5
    cameras = {key: {"camera_matrix": k.tolist(), "distortion_coefficients": d,
                     "image_size": [640, 480], "T_camera_from_reference": t.tolist(),
                     "T_reference_from_camera": np.linalg.inv(t).tolist()}
               for key, t in transforms.items()}
    return {"reference_camera": "zed_left", "units": "metres", "cameras": cameras}


def test_board_pixel_transfer_uses_loaded_extrinsics_and_distortion():
    config, board = load_board(CONFIG_DIR/"board_config.json")
    calibration = rig_calibration(distorted=True)
    rb = cv2.Rodrigues(np.array([.2, -.17, .12]))[0]
    tb = np.array([-.08, -.10, .85])
    reference_from_board = transform(rb, tb)
    board_xyz = np.array([.075, .095, 0])
    expected = (reference_from_board @ np.append(board_xyz, 1))[:3]
    expected_pixels = project_reference_point(calibration, expected)
    for source in VIEW_ORDER:
        camera = calibration["cameras"][source]
        local_board = np.array(camera["T_camera_from_reference"]) @ reference_from_board
        observations, _ = cv2.projectPoints(board.getChessboardCorners(),
                                            cv2.Rodrigues(local_board[:3, :3])[0], local_board[:3, 3],
                                            np.array(camera["camera_matrix"]), np.array(camera["distortion_coefficients"], float))
        obs = {"ids": list(range(24)), "corners": observations.reshape(-1, 2).tolist()}
        actual, _ = board_point(calibration, board, source, expected_pixels[source]["pixel"], obs)
        assert np.linalg.norm(actual-expected) < 2e-6
        projections = project_reference_point(calibration, actual)
        for target in VIEW_ORDER:
            assert np.linalg.norm(np.array(projections[target]["pixel"])-expected_pixels[target]["pixel"]) < .002
    changed = copy.deepcopy(calibration)
    changed["cameras"]["d435"]["T_camera_from_reference"][0][3] += .05
    wrong = project_reference_point(changed, expected)
    assert np.linalg.norm(np.array(wrong["d435"]["pixel"])-expected_pixels["d435"]["pixel"]) > 10


def test_real_depth_pixel_transfer_and_missing_depth():
    calibration = rig_calibration()
    source = "d435"
    k = calibration["cameras"][source]["camera_matrix"]
    depth = {"image": np.full((480, 640), 1250, np.uint16), "units_m": .001,
             "intrinsics": {"camera_matrix": k, "distortion_coefficients": [0]*5,
                            "distortion_model": "distortion.none"},
             "T_color_from_depth": np.eye(4).tolist(), "device_timestamp_ms": 42.}
    packet = {"depth": depth, "device_timestamp_ms": 42.}
    pixel = (350, 250)
    point, _ = realsense_point(calibration, source, pixel, packet)
    expected_local = np.array([.06*1.25, .02*1.25, 1.25, 1.])
    expected = np.array(calibration["cameras"][source]["T_reference_from_camera"]) @ expected_local
    assert np.allclose(point, expected[:3], atol=1e-7)
    assert np.allclose(project_reference_point(calibration, point)[source]["pixel"], pixel)
    depth["image"][:] = 0
    with pytest.raises(VerificationError, match="No valid depth"):
        realsense_point(calibration, source, pixel, packet)


def test_stereo_depth_from_either_lens_recovers_known_point():
    calibration = rig_calibration()
    rng = np.random.default_rng(9)
    left = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), .5)
    right = np.zeros_like(left)
    right[:, :-32] = left[:, 32:]
    batch = {"zed_left": {"image": cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)},
             "zed_right": {"image": cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)}}
    stereo = StereoDepth(calibration, batch)
    for source, pixel in [("zed_left", (350., 240.)), ("zed_right", (318., 240.))]:
        point, _ = stereo.point(source, pixel)
        assert np.allclose(point, [.06, 0, 1.], atol=.005)


def test_invalid_calibration_and_frame_mismatch_are_rejected(tmp_path):
    calibration = rig_calibration()
    validate_calibration(calibration)
    bad = copy.deepcopy(calibration)
    bad["cameras"]["d435"]["T_reference_from_camera"][0][0] = 2
    with pytest.raises(VerificationError, match="rigid transform"):
        validate_calibration(bad)
    batch = {key: {"image": np.zeros((480, 640, 3), np.uint8), "received_at": 42.} for key in VIEW_ORDER}
    check_frames(calibration, batch)
    calibration["cameras"]["d435"]["device"] = {"serial": "original"}
    with pytest.raises(VerificationError, match="serial"):
        check_frames(calibration, batch, {"d435": {"serial": "replacement"}})
    batch["d435"]["image"] = np.zeros((1080, 1920, 3), np.uint8)
    with pytest.raises(VerificationError, match="resolution"):
        check_frames(calibration, batch)
    config, board = load_board(CONFIG_DIR/"board_config.json")
    path = tmp_path/"calibration.json"
    path.write_text(json.dumps(calibration))
    loaded = load_calibration(path, config, board.getChessboardCorners())
    assert loaded["board_object_points_m"] == board.getChessboardCorners().tolist()


def test_projection_reports_behind_camera_or_outside_image():
    calibration = rig_calibration()
    assert project_reference_point(calibration, [0, 0, -1])["zed_left"]["reason"] == "Behind camera"
    assert project_reference_point(calibration, [10, 0, 1])["zed_left"]["reason"] == "Outside image"
