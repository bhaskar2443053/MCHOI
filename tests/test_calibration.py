import json

import cv2
import numpy as np
import pytest

from mchoi.calibration.board import BoardDetector, load_board
from mchoi.calibration.cameras import VIEW_ORDER
from mchoi.calibration.solver import calibrate_session, connect_graph, export_result, transform
from mchoi.calibration.storage import Session
from mchoi.paths import CONFIG_DIR




def synthetic_session(tmp_path):
    config, board = load_board(CONFIG_DIR/"board_config.json")
    session = Session(tmp_path/"session", config)
    points = board.getChessboardCorners()
    rng = np.random.default_rng(71)
    sizes = {key: [1920, 1080] for key in VIEW_ORDER}
    sizes["d455"] = [1280, 800]
    matrices = {key: np.array([[size[0]*.8, 0, size[0]/2], [0, size[0]*.805, size[1]/2], [0, 0, 1]], float)
                for key, size in sizes.items()}
    rig = {"zed_left": np.eye(4), "zed_right": transform(np.eye(3), [-.063, 0, 0]),
           "d435": transform(cv2.Rodrigues(np.array([.03, -.13, .02]))[0], [.17, .025, .015]),
           "d455": transform(cv2.Rodrigues(np.array([-.10, .08, -.03]))[0], [-.14, .055, .04])}
    for index in range(22):
        angles = rng.uniform([-.48, -.48, -.4], [.48, .48, .4])
        t = rng.uniform([-.19, -.17, .48], [.1, .04, .95])
        reference_from_board = transform(cv2.Rodrigues(angles)[0], t)
        views = {}
        for key in VIEW_ORDER:
            camera_from_board = rig[key] @ reference_from_board
            projected, _ = cv2.projectPoints(points, cv2.Rodrigues(camera_from_board[:3, :3])[0],
                                            camera_from_board[:3, 3], matrices[key], np.zeros(5))
            ids = np.arange(24)
            if key == "d455" and index % 2:
                ids = ids[2:22]  # Partial observations and mismatched sensor resolutions.
            pixels = projected.reshape(-1, 2)[ids] + rng.normal(0, .04, (len(ids), 2))
            views[key] = {"image_size": sizes[key], "ids": ids.tolist(), "corners": pixels.tolist(),
                          "camera": {}, "signature": (pixels[[0, 3, -1, -4]]/sizes[key]).flatten().tolist()}
        pose_id = f"pose_{index+1:04d}"
        path = session.path/pose_id
        path.mkdir()
        (path/"pose.json").write_text(json.dumps({"id": pose_id, "views": views}))
    return session, board, matrices, rig


def test_full_calibration_recovers_known_intrinsics_and_rig(tmp_path):
    session, board, expected_k, expected_camera_from_reference = synthetic_session(tmp_path)
    result = calibrate_session(session, board)
    assert set(result["cameras"]) == set(VIEW_ORDER)
    assert len(result["pairs"]) == 6
    for key, camera in result["cameras"].items():
        k = np.array(camera["camera_matrix"])
        assert np.allclose(k[:2, :2], expected_k[key][:2, :2], rtol=.015, atol=.5)
        actual = np.array(camera["T_camera_from_reference"])
        expected = expected_camera_from_reference[key]
        assert np.linalg.norm(actual[:3, 3]-expected[:3, 3]) < .006
        assert np.linalg.norm(actual[:3, :3]-expected[:3, :3]) < .015
        assert camera["rms_px"] < .2
    destination = export_result(session, result)
    saved = json.loads((destination/"calibration.json").read_text())
    assert saved["units"] == "metres"
    assert (destination/"calibration.yml").exists()
    storage = cv2.FileStorage(str(destination/"calibration.yml"), cv2.FILE_STORAGE_READ)
    try:
        assert np.allclose(storage.getNode("d435_T_camera_from_reference").mat(),
                           result["cameras"]["d435"]["T_camera_from_reference"])
    finally:
        storage.release()


def test_calibration_rejects_insufficient_data(tmp_path):
    config, board = load_board(CONFIG_DIR/"board_config.json")
    session = Session(tmp_path, config)
    with pytest.raises(ValueError, match="need at least"):
        calibrate_session(session, board)


def test_graph_transform_direction_and_indirect_paths():
    b_from_a = transform(cv2.Rodrigues(np.array([0, .3, 0]))[0], [.2, 0, .1])
    c_from_b = transform(cv2.Rodrigues(np.array([.1, 0, 0]))[0], [0, .4, 0])
    pairs = [{"from_camera": "a", "to_camera": "b", "T_to_from_from": b_from_a.tolist(), "rms_px": .2},
             {"from_camera": "b", "to_camera": "c", "T_to_from_from": c_from_b.tolist(), "rms_px": .3}]
    matrices, _ = connect_graph(pairs, "a", ["a", "b", "c"])
    assert np.allclose(matrices["c"], np.linalg.inv(c_from_b@b_from_a))
    reverse, _ = connect_graph(pairs, "c", ["a", "b", "c"])
    assert np.allclose(reverse["a"], c_from_b@b_from_a)


def test_detector_ignores_unrelated_duplicate_marker_ids():
    _, board = load_board(CONFIG_DIR/"board_config.json")
    board_image = board.generateImage((600, 840), marginSize=30)
    canvas = np.full((1080, 1600), 255, np.uint8)
    canvas[100:940, 400:1000] = board_image
    for i, marker_id in enumerate([2, 5, 11]):
        marker = cv2.aruco.generateImageMarker(board.getDictionary(), marker_id, 90)
        canvas[100+i*250:190+i*250, 1200:1290] = marker
    detection = BoardDetector(board).detect(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    assert detection["ids"] == list(range(24))
    assert detection["marker_ids"] == list(range(17))
    assert all(400 < p[0] < 1000 for p in detection["corners"])


def test_session_native_pixels_duplicate_and_non_destructive_undo(tmp_path):
    config, _ = load_board(CONFIG_DIR/"board_config.json")
    session = Session(tmp_path, config)
    image = np.arange(120*160*3, dtype=np.uint8).reshape(120, 160, 3)
    observation = {"ids": list(range(8)), "corners": [[10+i, 20+i] for i in range(8)],
                   "image_size": [160, 120], "signature": [0, 0, 1, 0, 1, 1, 0, 1]}
    batch = {"d435": {"image": image, "received_at": 42., "device_timestamp_ms": 123.,
                      "depth": {"image": np.zeros((60, 80), np.uint16)}}}
    pose = session.save(batch, {"d435": observation}, {"d435": {"serial": "camera-a"}})
    assert "depth" not in pose["views"]["d435"]
    assert session.duplicate({"d435": observation}) == pose["id"]
    assert np.array_equal(cv2.imread(str(session.path/pose["id"]/"d435.png")), image)
    with pytest.raises(ValueError, match="resolution changed"):
        session.save(batch, {"d435": {**observation, "image_size": [80, 60]}}, {"d435": {"serial": "camera-a"}})
    assert session.exclude_last() == pose["id"]
    assert session.poses() == []
    assert (session.path/pose["id"]/"d435.png").exists()
