"""Pixel transfer through measured 3D geometry and loaded rig calibration."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from .cameras import VIEW_ORDER
from .coverage import attach_coverage


class VerificationError(ValueError):
    pass


def load_calibration(path, board_config, board_points):
    path = Path(path)
    if path.suffix.lower() == ".json":
        result = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in (".yml", ".yaml"):
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not storage.isOpened():
                raise VerificationError("Cannot open calibration file.")
            result = {"cameras": {}, "reference_camera": storage.getNode("reference_camera").string(),
                      "units": storage.getNode("units").string(), "status": storage.getNode("status").string()}
            for key in VIEW_ORDER:
                camera = {}
                for field in ("camera_matrix", "distortion_coefficients", "T_reference_from_camera",
                              "T_camera_from_reference", "T_reference_from_depth"):
                    node = storage.getNode(key+"_"+field)
                    if not node.empty():
                        camera[field] = node.mat().tolist()
                camera["image_size"] = [int(storage.getNode(key+"_image_width").real()),
                                        int(storage.getNode(key+"_image_height").real())]
                camera["device"] = {}
                for field in ("serial", "name"):
                    node = storage.getNode(key+"_device_"+field)
                    if not node.empty():
                        camera["device"][field] = node.string()
                polygons = storage.getNode(key+"_coverage_polygons_px")
                if not polygons.empty():
                    camera["calibration_coverage"] = {
                        "image_size": camera["image_size"], "pose_count": polygons.size(),
                        "polygons_px": [polygons.at(i).mat().tolist() for i in range(polygons.size())]}
                result["cameras"][key] = camera
        finally:
            storage.release()
    else:
        raise VerificationError("Choose this app's calibration.json or calibration.yml file.")
    result = validate_calibration(result)
    if "board" in result:
        for field in ("squares_x", "squares_y", "square_length_m", "marker_length_m", "dictionary", "marker_ids"):
            if result["board"].get(field) != board_config.get(field):
                raise VerificationError("The calibration board differs from this app's board configuration.")
    result.setdefault("board", board_config)
    result.setdefault("board_object_points_m", np.asarray(board_points).tolist())
    result["loaded_from"] = str(path.resolve())
    attach_coverage(result, path)
    return result


def validate_calibration(result):
    result = copy.deepcopy(result)
    if result.get("units") != "metres":
        raise VerificationError("Calibration must explicitly use metres.")
    if result.get("reference_camera") not in VIEW_ORDER:
        raise VerificationError("Unknown reference camera.")
    for key in VIEW_ORDER:
        try:
            camera = result["cameras"][key]
            k = np.asarray(camera["camera_matrix"], float)
            d = np.asarray(camera["distortion_coefficients"], float).flatten()
            size = camera["image_size"]
            t = np.asarray(camera["T_reference_from_camera"], float)
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(f"Missing or invalid calibration for {key}.") from exc
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or len(d) not in (4, 5, 8, 12, 14)
                or not np.isfinite(d).all() or len(size) != 2 or min(size) <= 0):
            raise VerificationError(f"Invalid camera intrinsics for {key}.")
        if (t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)):
            raise VerificationError(f"Invalid rigid transform for {key}.")
        inverse = np.linalg.inv(t)
        if "T_camera_from_reference" in camera and not np.allclose(camera["T_camera_from_reference"], inverse, atol=1e-5):
            raise VerificationError(f"Inconsistent forward/inverse transforms for {key}.")
        camera["T_camera_from_reference"] = inverse.tolist()
        camera["distortion_coefficients"] = d.tolist()
    reference = result["cameras"][result["reference_camera"]]["T_reference_from_camera"]
    if not np.allclose(reference, np.eye(4), atol=1e-5):
        raise VerificationError("Reference camera transform must be identity.")
    return result


def check_frames(calibration, batch, metadata=None):
    for key in VIEW_ORDER:
        if key not in batch:
            raise VerificationError(f"No frame from {key}.")
        expected = calibration["cameras"][key]
        image = batch[key]["image"]
        if [image.shape[1], image.shape[0]] != list(expected["image_size"]):
            raise VerificationError(f"{key}: image resolution differs from the loaded calibration.")
        if metadata:
            saved = expected.get("device", {})
            live = metadata.get(key, {})
            # Serial numbers identify physical RealSense devices; USB paths may change with ports.
            if saved.get("serial") and live.get("serial") and saved["serial"] != live["serial"]:
                raise VerificationError(f"{key}: connected camera serial does not match calibration.")
    stamps = [p["received_at"] for p in batch.values()]
    if max(stamps)-min(stamps) > .08:
        raise VerificationError("Frames are more than 80 ms apart. Refresh views with the scene stationary.")


def pixel_ray(camera, pixel):
    p = np.array(pixel, float).reshape(1, 1, 2)
    xy = cv2.undistortPoints(p, np.array(camera["camera_matrix"], float),
                             np.array(camera["distortion_coefficients"], float)).reshape(2)
    return np.array([xy[0], xy[1], 1.0])


def project_reference_point(calibration, point):
    result = {}
    p = np.append(np.asarray(point, float), 1.)
    for key, camera in calibration["cameras"].items():
        local = np.asarray(camera["T_camera_from_reference"]) @ p
        if local[2] <= 1e-6:
            result[key] = {"pixel": None, "reason": "Behind camera"}
            continue
        pixel, _ = cv2.projectPoints(local[:3].reshape(1, 3), np.zeros(3), np.zeros(3),
                                    np.array(camera["camera_matrix"]), np.array(camera["distortion_coefficients"]))
        u, v = pixel.reshape(2)
        width, height = camera["image_size"]
        if not np.isfinite(pixel).all() or not (0 <= u < width and 0 <= v < height):
            result[key] = {"pixel": None, "reason": "Outside image"}
        else:
            result[key] = {"pixel": [float(u), float(v)], "depth_m": float(local[2])}
    return result


def board_point(calibration, board, source, pixel, observation):
    if len(observation["ids"]) < 6:
        raise VerificationError("Board-plane mode needs at least six detected corners in the clicked view.")
    camera = calibration["cameras"][source]
    objects = board.getChessboardCorners()[np.array(observation["ids"], np.int32)]
    image_points = np.array(observation["corners"], np.float32)
    k, d = np.array(camera["camera_matrix"]), np.array(camera["distortion_coefficients"])
    ok, rvec, tvec = cv2.solvePnP(objects, image_points, k, d)
    if not ok or tvec[2, 0] <= 0:
        raise VerificationError("Could not estimate the board plane.")
    projected, _ = cv2.projectPoints(objects, rvec, tvec, k, d)
    rms = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-image_points)**2, axis=1)))
    if rms > 3:
        raise VerificationError("Board pose has high reprojection error; refresh with a clear board view.")
    rotation = cv2.Rodrigues(rvec)[0]
    origin = -rotation.T @ tvec.reshape(3)
    direction = rotation.T @ pixel_ray(camera, pixel)
    if abs(direction[2]) < 1e-8:
        raise VerificationError("Clicked ray is parallel to the board.")
    distance = -origin[2]/direction[2]
    local = origin+distance*direction
    sx, sy = board.getChessboardSize()
    length = board.getSquareLength()
    if distance <= 0 or not (0 <= local[0] <= sx*length and 0 <= local[1] <= sy*length):
        raise VerificationError("Click inside the printed grid in Board plane mode. Use Automatic depth for other objects.")
    source_point = rotation @ local+tvec.reshape(3)
    reference = np.array(camera["T_reference_from_camera"]) @ np.append(source_point, 1.)
    return reference[:3], f"Board plane · pose RMS {rms:.2f} px"


def realsense_point(calibration, source, pixel, packet):
    depth = packet.get("depth")
    if depth is None:
        raise VerificationError("No RealSense depth frame. Reconnect cameras to enable depth.")
    if abs(depth["device_timestamp_ms"]-packet["device_timestamp_ms"]) > 80:
        raise VerificationError("Color and depth timestamps are too far apart. Refresh views.")
    intr = depth["intrinsics"]
    d = np.array(intr["distortion_coefficients"], float)
    model = intr["distortion_model"]
    if model not in ("distortion.none", "distortion.brown_conrady", "distortion.modified_brown_conrady") and np.any(abs(d) > 1e-10):
        raise VerificationError(f"Unsupported depth distortion model: {model}")
    raw = depth["image"]
    ys, xs = np.nonzero(raw)
    if not len(xs):
        raise VerificationError("No valid depth in this view.")
    z = raw[ys, xs].astype(float)*depth["units_m"]
    pixels = np.column_stack((xs, ys)).astype(np.float64).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pixels, np.array(intr["camera_matrix"]), d).reshape(-1, 2)
    cloud = np.column_stack((xy*z[:, None], z))
    t = np.array(depth["T_color_from_depth"])
    cloud = cloud @ t[:3, :3].T+t[:3, 3]
    cloud = cloud[cloud[:, 2] > .05]
    if len(cloud) == 0:
        raise VerificationError("No valid depth in front of the color camera.")
    camera = calibration["cameras"][source]
    projected, _ = cv2.projectPoints(cloud, np.zeros(3), np.zeros(3),
                                    np.array(camera["camera_matrix"]), np.array(camera["distortion_coefficients"]))
    distance = np.linalg.norm(projected.reshape(-1, 2)-pixel, axis=1)
    nearby = cloud[distance <= 3.0, 2]
    if len(nearby) < 2:
        raise VerificationError("No reliable depth at this pixel. Try a nearby surface or Board plane mode.")
    z = float(np.median(nearby))
    if np.ptp(nearby) > max(.03, .04*z):
        raise VerificationError("Depth changes sharply at this edge. Click farther inside the surface.")
    local = pixel_ray(camera, pixel)*z
    reference = np.array(camera["T_reference_from_camera"]) @ np.append(local, 1.)
    return reference[:3], f"RealSense depth · {z:.3f} m along source optical axis"


class StereoDepth:
    """Stereo matching is computed once for a frozen pair, never on the UI thread."""
    def __init__(self, calibration, batch):
        self.calibration = calibration
        left, right = (calibration["cameras"][key] for key in ("zed_left", "zed_right"))
        size = tuple(left["image_size"])
        if size != tuple(right["image_size"]):
            raise VerificationError("ZED stereo image sizes differ.")
        relative = np.array(right["T_camera_from_reference"]) @ np.array(left["T_reference_from_camera"])
        kl, kr = np.array(left["camera_matrix"]), np.array(right["camera_matrix"])
        dl, dr = np.array(left["distortion_coefficients"]), np.array(right["distortion_coefficients"])
        self.size = (size[0]//2, size[1]//2)
        rl, rr, pl, pr, _, _, _ = cv2.stereoRectify(kl, dl, kr, dr, size, relative[:3, :3], relative[:3, 3],
                                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=self.size)
        if abs(pr[0, 3]) < abs(pr[1, 3]) or pr[0, 3] >= 0:
            raise VerificationError("ZED calibration does not describe the expected horizontal left/right stereo pair.")
        self.rotations, self.projections = (rl, rr), (pl, pr)
        rectified = []
        for key, k, d, r, p in zip(("zed_left", "zed_right"), (kl, kr), (dl, dr), (rl, rr), (pl, pr)):
            mx, my = cv2.initUndistortRectifyMap(k, d, r, p, self.size, cv2.CV_32FC1)
            gray = cv2.cvtColor(batch[key]["image"], cv2.COLOR_BGR2GRAY)
            rectified.append(cv2.remap(gray, mx, my, cv2.INTER_LINEAR))
        disparities = min(256, (self.size[0]//3//16)*16)
        settings = dict(numDisparities=disparities, blockSize=5, P1=8*25, P2=32*25,
                        disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=80,
                        speckleRange=2, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left = cv2.StereoSGBM_create(minDisparity=0, **settings).compute(*rectified).astype(float)/16
        self.right = cv2.StereoSGBM_create(minDisparity=-disparities, **settings).compute(*rectified[::-1]).astype(float)/16
        self.maximum = disparities

    def point(self, source, pixel):
        index = 0 if source == "zed_left" else 1
        camera = self.calibration["cameras"][source]
        uv = cv2.undistortPoints(np.array(pixel, float).reshape(1, 1, 2), np.array(camera["camera_matrix"]),
                                 np.array(camera["distortion_coefficients"]), R=self.rotations[index],
                                 P=self.projections[index]).reshape(2)
        x, y = np.rint(uv).astype(int)
        if not (2 <= x < self.size[0]-2 and 2 <= y < self.size[1]-2):
            raise VerificationError("Point is outside the usable rectified stereo overlap.")
        disparity_map = self.left if index == 0 else self.right
        patch = disparity_map[y-1:y+2, x-1:x+2]
        valid = patch[(patch > .5) & (patch < self.maximum-1)] if index == 0 else patch[(patch < -.5) & (patch > -self.maximum+1)]
        if len(valid) < 5 or np.ptp(valid) > 2:
            raise VerificationError("No reliable ZED stereo depth here. Click a textured surface away from edges.")
        disparity = float(np.median(valid))
        match_x = int(round(uv[0]-disparity))
        if not 0 <= match_x < self.size[0]:
            raise VerificationError("Point has no match in the other ZED lens.")
        opposite = self.right if index == 0 else self.left
        matched = opposite[y, match_x]
        opposite_valid = -self.maximum+1 < matched < -.5 if index == 0 else .5 < matched < self.maximum-1
        if not opposite_valid or abs(disparity+matched) > 1.0:
            raise VerificationError("ZED stereo match failed the left/right consistency check.")
        left_uv, right_uv = (uv, uv-[disparity, 0]) if index == 0 else (uv-[disparity, 0], uv)
        homogeneous = cv2.triangulatePoints(*self.projections, left_uv.reshape(2, 1), right_uv.reshape(2, 1)).reshape(4)
        if abs(homogeneous[3]) < 1e-10:
            raise VerificationError("Stereo point is too far away.")
        rectified = homogeneous[:3]/homogeneous[3]
        point_left = self.rotations[0].T @ rectified
        if not .1 < point_left[2] < 15:
            raise VerificationError("Stereo depth is outside the reliable 0.1–15 m range.")
        reference = np.array(self.calibration["cameras"]["zed_left"]["T_reference_from_camera"]) @ np.append(point_left, 1.)
        return reference[:3], f"ZED stereo depth · {point_left[2]:.3f} m along left optical axis"
