import json
from pathlib import Path

import cv2
import numpy as np


def load_board(path):
    config = json.loads(Path(path).read_text())
    board = cv2.aruco.CharucoBoard((config["squares_x"], config["squares_y"]),
                                 config["square_length_m"], config["marker_length_m"],
                                 cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["dictionary"])),
                                 np.array(config["marker_ids"], dtype=np.int32))
    return config, board


class BoardDetector:
    def __init__(self, board):
        self.board = board
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        self.markers = cv2.aruco.ArucoDetector(board.getDictionary(), params)
        self.charuco = cv2.aruco.CharucoDetector(board)
        self.object_markers = dict(zip(board.getIds().flatten().tolist(), board.getObjPoints()))

    def detect(self, image, roi=None):
        height, width = image.shape[:2]
        x0, y0, x1, y1 = (0, 0, width, height) if roi is None else roi
        x0, x1 = max(0, int(x0)), min(width, int(x1))
        y0, y1 = max(0, int(y0)), min(height, int(y1))
        gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        empty = {"ids": [], "corners": [], "marker_ids": [], "marker_corners": [],
                 "image_size": [width, height], "signature": None, "coverage": 0.0,
                 "sharpness": 0.0}
        if gray.size == 0:
            return empty
        corners, ids, _ = self.markers.detectMarkers(gray)
        if ids is None:
            return empty
        candidates = [(int(i), c) for i, c in zip(ids.flatten(), corners)
                      if int(i) in self.object_markers]
        if len(candidates) < 3:
            return empty
        # Printed tags elsewhere on the table share IDs. Fit the complete board geometry
        # robustly, then retain one geometrically consistent instance of each marker.
        objects = np.concatenate([self.object_markers[i][:, :2] for i, _ in candidates])
        pixels = np.concatenate([c.reshape(4, 2) for _, c in candidates])
        threshold = max(3.0, width / 450)
        homography, _ = cv2.findHomography(objects, pixels, cv2.RANSAC, threshold,
                                         maxIters=2000, confidence=.995)
        if homography is None:
            return empty
        accepted = {}
        for marker_id, c in candidates:
            predicted = cv2.perspectiveTransform(self.object_markers[marker_id][:, :2].reshape(1, 4, 2),
                                                homography).reshape(4, 2)
            error = float(np.mean(np.linalg.norm(predicted-c.reshape(4, 2), axis=1)))
            if error < threshold*2.5 and (marker_id not in accepted or error < accepted[marker_id][0]):
                accepted[marker_id] = (error, c)
        if len(accepted) < 3:
            return empty
        marker_ids = np.array(sorted(accepted), dtype=np.int32).reshape(-1, 1)
        marker_corners = [accepted[int(i)][1] for i in marker_ids.flatten()]
        cc, ci, _, _ = self.charuco.detectBoard(gray, markerCorners=marker_corners, markerIds=marker_ids)
        offset = np.array([x0, y0], dtype=np.float32)
        empty["marker_ids"] = marker_ids.flatten().tolist()
        empty["marker_corners"] = [(c.reshape(4, 2)+offset).tolist() for c in marker_corners]
        if ci is None:
            return empty
        points = cc.reshape(-1, 2)+offset
        order = np.argsort(ci.flatten())
        empty["ids"] = ci.flatten()[order].tolist()
        empty["corners"] = points[order].tolist()
        hull = cv2.convexHull(points)
        empty["coverage"] = float(cv2.contourArea(hull)/(width*height))
        bx, by, bw, bh = cv2.boundingRect(np.concatenate(marker_corners).reshape(-1, 2))
        board_pixels = gray[max(0, by):min(gray.shape[0], by+bh), max(0, bx):min(gray.shape[1], bx+bw)]
        empty["sharpness"] = float(cv2.Laplacian(board_pixels, cv2.CV_64F).var()) if board_pixels.size else 0
        if len(ci) >= 6:
            object_points = self.board.getChessboardCorners()[ci.flatten(), :2]
            h, _ = cv2.findHomography(object_points, points, 0)
            if h is not None:
                sx, sy = self.board.getChessboardSize()
                length = self.board.getSquareLength()
                bounds = np.array([[[0, 0], [sx*length, 0], [sx*length, sy*length], [0, sy*length]]], np.float32)
                projected = cv2.perspectiveTransform(bounds, h).reshape(4, 2)
                empty["signature"] = (projected / [width, height]).flatten().tolist()
        return empty


def movement(a, b):
    common = sorted(set(a["ids"]) & set(b["ids"]))
    if len(common) < 6:
        return float("inf")
    pa = dict(zip(a["ids"], a["corners"]))
    pb = dict(zip(b["ids"], b["corners"]))
    return float(np.median([np.linalg.norm(np.array(pa[i])-pb[i]) for i in common]))


def annotate(image, detection, camera=None):
    output = image.copy()
    ids = np.array(detection["ids"], np.int32).reshape(-1, 1)
    if len(ids):
        points = np.array(detection["corners"], np.float32).reshape(-1, 1, 2)
        cv2.aruco.drawDetectedCornersCharuco(output, points, ids, (100, 235, 80))
        if camera and len(ids) >= 6:
            k = np.array(camera["camera_matrix"], float)
            d = np.array(camera["distortion_coefficients"], float)
            objects = np.array(camera["board_points"], np.float32)[ids.flatten()]
            ok, r, t = cv2.solvePnP(objects, points, k, d)
            if ok and t[2, 0] > 0:
                cv2.drawFrameAxes(output, k, d, r, t, .06, 2)
    return output
