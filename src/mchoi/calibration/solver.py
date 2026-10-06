"""ChArUco intrinsics and a connected camera rig, all translations in metres."""
from datetime import datetime, timezone
from itertools import combinations

import cv2
import numpy as np

from .cameras import VIEW_ORDER
from .coverage import build_coverage
from .storage import write_json


MIN_INTRINSIC_POSES = 12
MIN_PAIR_POSES = 8
MIN_CORNERS = 8


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation).reshape(3)
    return matrix


def rotation_angle(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def _intrinsics(key, samples, board, progress):
    samples = [s for s in samples if len(s["view"]["ids"]) >= MIN_CORNERS]
    if len(samples) < MIN_INTRINSIC_POSES:
        raise ValueError(f"{key}: {len(samples)} usable poses; need at least {MIN_INTRINSIC_POSES}.")
    sizes = {tuple(s["view"]["image_size"]) for s in samples}
    if len(sizes) != 1:
        raise ValueError(f"{key}: mixed image resolutions in session.")
    size = sizes.pop()
    xyz = board.getChessboardCorners()

    def solve(selected):
        obj = [xyz[np.array(s["view"]["ids"], np.int32)].astype(np.float32) for s in selected]
        img = [np.array(s["view"]["corners"], np.float32).reshape(-1, 1, 2) for s in selected]
        return cv2.calibrateCameraExtended(obj, img, size, None, None, flags=cv2.CALIB_FIX_K3,
                                          criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 100, 1e-8))

    progress(f"Calibrating {key}: {len(samples)} board poses…")
    result = solve(samples)
    errors = result[-1].flatten()
    median = np.median(errors)
    mad = np.median(abs(errors-median))*1.4826
    limit = max(1.5, float(median+3*max(mad, .1)))
    keep = [s for s, e in zip(samples, errors) if e <= limit]
    rejected = [s["id"] for s, e in zip(samples, errors) if e > limit]
    if rejected and len(keep) >= MIN_INTRINSIC_POSES:
        samples = keep
        result = solve(samples)
    else:
        rejected = []
    rms, k, d, rvecs, tvecs, std_i, _, per_view = result
    if not np.isfinite(k).all() or not np.isfinite(d).all() or not np.isfinite(rms):
        raise ValueError(f"{key}: calibration diverged. Add varied, sharper board poses.")
    if not (.15*size[0] < k[0, 0] < 8*size[0] and .15*size[1] < k[1, 1] < 8*size[1]):
        raise ValueError(f"{key}: implausible focal length. Capture more tilted board poses.")
    if any(np.asarray(t).reshape(3)[2] <= 0 for t in tvecs):
        raise ValueError(f"{key}: board solved behind the camera. Check observations.")
    normals = np.array([cv2.Rodrigues(r)[0][:, 2] for r in rvecs])
    spread = max(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
                 for a, b in combinations(normals, 2))
    pixels = np.concatenate([np.array(s["view"]["corners"]) for s in samples])
    coverage = (np.ptp(pixels, axis=0)/size).tolist()
    warnings = []
    if rms > 1.0:
        warnings.append(f"{key}: reprojection RMS {rms:.2f} px; aim below 1 px.")
    if spread < 15:
        warnings.append(f"{key}: only {spread:.1f}° of board-normal variation; add tilted poses.")
    if min(coverage) < .4:
        warnings.append(f"{key}: board observations cover a small part of the image; capture nearer the edges.")
    if not (0 < k[0, 2] < size[0] and 0 < k[1, 2] < size[1]):
        warnings.append(f"{key}: principal point is outside the image; calibration needs review.")
    output = {"image_size": list(size), "camera_matrix": k.tolist(),
              "distortion_coefficients": d.flatten().tolist(),
              "distortion_model": "opencv_brown_conrady_k1_k2_p1_p2_k3; k3 fixed to zero",
              "rms_px": float(rms), "intrinsic_std_deviations": std_i.flatten().tolist(),
              "used_pose_ids": [s["id"] for s in samples], "rejected_pose_ids": rejected,
              "board_normal_spread_deg": spread, "observed_image_span_fraction_xy": coverage,
              "calibration_coverage": build_coverage([s["view"] for s in samples], size),
              "per_pose": [{"id": s["id"], "rms_px": float(e),
                            "T_camera_from_board": transform(cv2.Rodrigues(r)[0], t).tolist()}
                           for s, e, r, t in zip(samples, per_view.flatten(), rvecs, tvecs)],
              "device": samples[-1]["view"].get("camera", {})}
    return output, warnings


def _pair(a, b, poses, cameras, board, progress):
    accepted_a = set(cameras[a]["used_pose_ids"])
    accepted_b = set(cameras[b]["used_pose_ids"])
    rows = []
    xyz = board.getChessboardCorners()
    for pose in poses:
        if pose["id"] not in accepted_a & accepted_b:
            continue
        va, vb = pose["views"][a], pose["views"][b]
        common = sorted(set(va["ids"]) & set(vb["ids"]))
        if len(common) < 6:
            continue
        objects = xyz[np.array(common)]
        if np.linalg.matrix_rank(objects[:, :2]-objects[0, :2]) < 2:
            continue
        pa, pb = dict(zip(va["ids"], va["corners"])), dict(zip(vb["ids"], vb["corners"]))
        rows.append((pose["id"], objects.astype(np.float32),
                     np.array([pa[i] for i in common], np.float32).reshape(-1, 1, 2),
                     np.array([pb[i] for i in common], np.float32).reshape(-1, 1, 2)))
    if len(rows) < MIN_PAIR_POSES:
        return None
    progress(f"Aligning {a} ↔ {b}: {len(rows)} shared poses…")

    def solve(data):
        return cv2.stereoCalibrateExtended(
            [r[1] for r in data], [r[2] for r in data], [r[3] for r in data],
            np.array(cameras[a]["camera_matrix"]), np.array(cameras[a]["distortion_coefficients"]),
            np.array(cameras[b]["camera_matrix"]), np.array(cameras[b]["distortion_coefficients"]),
            tuple(cameras[a]["image_size"]), None, None, flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 100, 1e-8))

    result = solve(rows)
    per_view = np.sqrt(np.mean(np.square(result[-1]), axis=1))
    median = np.median(per_view)
    limit = max(2.0, float(median+3*max(.1, 1.4826*np.median(abs(per_view-median)))))
    keep = [r for r, e in zip(rows, per_view) if e <= limit]
    rejected = [r[0] for r, e in zip(rows, per_view) if e > limit]
    if rejected and len(keep) >= MIN_PAIR_POSES:
        rows = keep
        result = solve(rows)
    else:
        rejected = []
    rms, _, _, _, _, rotation, translation = result[:7]
    if not np.isfinite(rms) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return None
    return {"from_camera": a, "to_camera": b, "rms_px": float(rms),
            "T_to_from_from": transform(rotation, translation).tolist(),
            "used_pose_ids": [r[0] for r in rows], "rejected_pose_ids": rejected,
            "per_view_errors_px": result[-1].tolist()}


def connect_graph(pairs, reference, keys):
    transforms = {reference: np.eye(4)}
    paths = {reference: []}
    costs = {reference: 0.0}
    pending = set(keys)
    while pending:
        reachable = pending & costs.keys()
        if not reachable:
            break
        current = min(reachable, key=lambda k: costs[k])
        pending.remove(current)
        for pair in pairs:
            a, b = pair["from_camera"], pair["to_camera"]
            if pair["rms_px"] > 3 or current not in (a, b):
                continue
            neighbor = b if current == a else a
            if neighbor not in pending:
                continue
            edge = np.array(pair["T_to_from_from"])
            candidate = transforms[current] @ (np.linalg.inv(edge) if current == a else edge)
            cost = costs[current] + max(.01, pair["rms_px"])
            if cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = cost
                transforms[neighbor] = candidate
                paths[neighbor] = paths[current]+[f"{a}->{b}"]
    return transforms, paths


def calibrate_session(session, board, progress=lambda message: None):
    poses = session.poses()
    cameras, warnings = {}, []
    for key in VIEW_ORDER:
        samples = [{"id": p["id"], "view": p["views"][key]} for p in poses if key in p["views"]]
        cameras[key], messages = _intrinsics(key, samples, board, progress)
        warnings.extend(messages)
    pairs = []
    for a, b in combinations(VIEW_ORDER, 2):
        pair = _pair(a, b, poses, cameras, board, progress)
        if pair:
            pairs.append(pair)
            if pair["rms_px"] > 1.5:
                warnings.append(f"{a} ↔ {b}: stereo RMS {pair['rms_px']:.2f} px.")
    reference = "zed_left"
    transforms, paths = connect_graph(pairs, reference, VIEW_ORDER)
    missing = set(VIEW_ORDER)-transforms.keys()
    if missing:
        raise ValueError("Cannot connect all cameras: " + ", ".join(sorted(missing)) +
                         f". Capture at least {MIN_PAIR_POSES} shared, stationary poses with 6+ common corners per linked pair.")
    for key, camera in cameras.items():
        camera["T_reference_from_camera"] = transforms[key].tolist()
        camera["T_camera_from_reference"] = np.linalg.inv(transforms[key]).tolist()
        camera["extrinsic_path"] = paths[key]
        factory = camera["device"].get("factory_T_color_from_depth")
        if factory:
            camera["T_reference_from_depth"] = (transforms[key] @ np.array(factory)).tolist()
            camera["depth_transform_source"] = "Measured RGB rig transform composed with RealSense factory depth-to-color transform"
    for pair in pairs:
        a, b = pair["from_camera"], pair["to_camera"]
        predicted = np.linalg.inv(transforms[b]) @ transforms[a]
        residual = np.linalg.inv(np.array(pair["T_to_from_from"])) @ predicted
        pair["graph_residual_translation_m"] = float(np.linalg.norm(residual[:3, 3]))
        pair["graph_residual_rotation_deg"] = rotation_angle(residual[:3, :3])
        if pair["graph_residual_translation_m"] > .015 or pair["graph_residual_rotation_deg"] > 2:
            warnings.append(f"{a} ↔ {b}: independent pair disagrees with rig graph; check stationary capture and pose coverage.")
    return {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "review_required" if warnings else "calibrated",
            "reference_camera": reference, "units": "metres",
            "coordinate_convention": "OpenCV optical coordinates: x right, y down, z forward. Column vectors; T_target_from_source @ p_source = p_target.",
            "image_convention": "Unrectified native-resolution RGB images; ZED left and right calibrated separately.",
            "synchronization": "Host receipt matching, no cross-device hardware synchronization; board must remain still during each capture.",
            "board": session.manifest["board"], "board_object_points_m": board.getChessboardCorners().tolist(),
            "source_pose_ids": [pose["id"] for pose in poses],
            "cameras": cameras, "pairs": pairs, "warnings": warnings,
            "validation": "Training-image reprojection error, pose coverage, and pairwise graph consistency; no independent metrology validation."}


def export_result(session, result):
    directory = session.path / ("calibration_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir()
    write_json(directory/"calibration.json", result)
    storage = cv2.FileStorage(str(directory/"calibration.yml"), cv2.FILE_STORAGE_WRITE)
    try:
        storage.write("reference_camera", result["reference_camera"])
        storage.write("units", "metres")
        storage.write("status", result["status"])
        storage.write("coordinate_convention", result["coordinate_convention"])
        for key, camera in result["cameras"].items():
            for field in ("camera_matrix", "distortion_coefficients", "T_reference_from_camera",
                          "T_camera_from_reference", "T_reference_from_depth"):
                if field in camera:
                    storage.write(key+"_"+field, np.array(camera[field], dtype=float))
            storage.write(key+"_image_width", camera["image_size"][0])
            storage.write(key+"_image_height", camera["image_size"][1])
            storage.write(key+"_rms_px", camera["rms_px"])
            device = camera.get("device", {})
            for field in ("serial", "name"):
                if device.get(field):
                    storage.write(key+"_device_"+field, str(device[field]))
            coverage = camera.get("calibration_coverage")
            if coverage:
                storage.startWriteStruct(key+"_coverage_polygons_px", cv2.FileNode_SEQ)
                for polygon in coverage["polygons_px"]:
                    storage.write("", np.array(polygon, dtype=np.float32))
                storage.endWriteStruct()
    finally:
        storage.release()
    lines = ["Multi-camera calibration", "", f"Status: {result['status']}",
             "Reference: ZED left optical frame. Translations in metres.", ""]
    for key, camera in result["cameras"].items():
        lines.append(f"{key}: {camera['rms_px']:.3f} px RMS; {len(camera['used_pose_ids'])} poses")
    lines.extend(["", "Review notes:"] + (result["warnings"] or ["No automated quality flags."]))
    lines.extend(["", result["validation"]])
    (directory/"summary.txt").write_text("\n".join(lines), encoding="utf-8")
    write_json(session.path/"latest_calibration.json", {"path": str(directory.name)})
    return directory
