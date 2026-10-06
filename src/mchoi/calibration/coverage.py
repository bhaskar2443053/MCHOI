"""Image regions sampled by accepted calibration-board observations."""
import json
from pathlib import Path
import re

import cv2
import numpy as np


def build_coverage(observations, image_size):
    polygons = []
    for observation in observations:
        if list(observation.get("image_size", [])) != list(image_size):
            continue
        corners = np.asarray(observation.get("corners", []), dtype=np.float32)
        if corners.ndim != 2 or corners.shape[1] != 2 or len(corners) < 3 or not np.isfinite(corners).all():
            continue
        hull = cv2.convexHull(corners).reshape(-1, 2)
        if cv2.contourArea(hull) > 0:
            polygons.append(hull.tolist())
    return {"image_size": list(image_size), "polygons_px": polygons, "pose_count": len(polygons),
            "method": "Union of per-pose observed ChArUco corner hulls; sampled image coverage, not an accuracy boundary."}


def attach_coverage(calibration, calibration_path=None):
    """Recover coverage for older exports from their original accepted pose files."""
    if calibration_path is None:
        return
    path = Path(calibration_path)
    companion = None
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            candidate = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            matches = all(
                camera["image_size"] == candidate["cameras"][key]["image_size"]
                and np.allclose(camera["camera_matrix"], candidate["cameras"][key]["camera_matrix"])
                and np.allclose(np.asarray(camera["distortion_coefficients"]).flatten(),
                                np.asarray(candidate["cameras"][key]["distortion_coefficients"]).flatten())
                and np.allclose(camera["T_reference_from_camera"], candidate["cameras"][key]["T_reference_from_camera"])
                for key, camera in calibration["cameras"].items())
            if matches:
                companion = candidate
        except (OSError, ValueError, KeyError, TypeError):
            pass
    for key, camera in calibration["cameras"].items():
        if camera.get("calibration_coverage", {}).get("polygons_px"):
            continue
        source = companion["cameras"][key] if companion else camera
        if source.get("calibration_coverage", {}).get("polygons_px"):
            camera["calibration_coverage"] = source["calibration_coverage"]
            continue
        accepted = source.get("used_pose_ids", [])
        observations = []
        for pose_id in accepted:
            if not isinstance(pose_id, str) or not re.fullmatch(r"pose_\d+", pose_id):
                continue
            for directory in (path.parent.parent, path.parent):
                try:
                    pose = json.loads((directory/pose_id/"pose.json").read_text(encoding="utf-8"))
                    observations.append(pose["views"][key])
                    break
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        if observations:
            coverage = build_coverage(observations, camera["image_size"])
            coverage["expected_pose_count"] = len(accepted)
            camera["calibration_coverage"] = coverage


def coverage_mask(camera):
    coverage = camera.get("calibration_coverage", {})
    width, height = camera["image_size"]
    if coverage.get("image_size") != [width, height] or not coverage.get("polygons_px"):
        return None
    mask = np.zeros((height, width), np.uint8)
    for polygon in coverage["polygons_px"]:
        vertices = np.asarray(polygon, np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
            continue
        # Fill each hull separately: filling all polygons at once would create
        # even/odd holes where observations overlap.
        vertices = np.clip(vertices, [-width, -height], [2*width, 2*height])
        cv2.fillConvexPoly(mask, np.rint(vertices).astype(np.int32), 255)
    return mask if np.any(mask) else None


def overlay_coverage(image, mask):
    """Tint only the covered preview pixels; return a copy of the original image."""
    output = image.copy()
    resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    inside = resized != 0
    color = np.array([150, 220, 60], dtype=float)  # BGR teal/green, 22% opacity.
    output[inside] = np.rint(output[inside]*.78+color*.22).astype(np.uint8)
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (150, 235, 70), 1, cv2.LINE_AA)
    return output
