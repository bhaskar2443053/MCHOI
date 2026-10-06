from datetime import datetime, timezone
import json
import os
from pathlib import Path

import cv2
import numpy as np


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


class Session:
    def __init__(self, path, board_config=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / "session.json"
        if manifest.exists():
            self.manifest = json.loads(manifest.read_text(encoding="utf-8"))
            if board_config is not None and self.manifest["board"] != board_config:
                raise ValueError("This session uses a different board configuration.")
        else:
            self.manifest = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                             "board": board_config, "excluded_poses": []}
            write_json(manifest, self.manifest)

    def poses(self, include_excluded=False):
        return [json.loads(file.read_text(encoding="utf-8"))
                for file in sorted(self.path.glob("pose_*/pose.json"))
                if include_excluded or file.parent.name not in self.manifest["excluded_poses"]]

    def counts(self):
        counts = {}
        for pose in self.poses():
            for view in pose["views"]:
                counts[view] = counts.get(view, 0)+1
        return counts

    def duplicate(self, observations):
        for pose in self.poses():
            distances = []
            for key in set(observations) & set(pose["views"]):
                a, b = observations[key].get("signature"), pose["views"][key].get("signature")
                if a is not None and b is not None:
                    distances.append(np.mean(np.linalg.norm((np.array(a)-b).reshape(4, 2), axis=1)))
            if distances and max(distances) < .018 and set(observations) <= set(pose["views"]):
                return pose["id"]
        return None

    def save(self, batch, observations, camera_metadata):
        previous = self.poses(include_excluded=True)
        for pose in previous:
            for key in set(pose["views"]) & set(observations):
                old = pose["views"][key]
                if old["image_size"] != observations[key]["image_size"]:
                    raise ValueError(f"{key} resolution changed. Start a new session.")
                old_meta = old.get("camera", {})
                new_meta = camera_metadata.get(key, {})
                for field in ("serial", "device_path"):
                    if old_meta.get(field) != new_meta.get(field):
                        raise ValueError(f"{key} device changed. Start a new session.")
        number = max([int(p["id"].split("_")[1]) for p in previous] or [0])+1
        pose_id = f"pose_{number:04d}"
        temp = self.path / ("_writing_"+pose_id)
        temp.mkdir(exist_ok=True)
        stamps = [batch[key]["received_at"] for key in observations]
        data = {"id": pose_id, "captured_at": datetime.now(timezone.utc).isoformat(),
                "host_receipt_spread_ms": (max(stamps)-min(stamps))*1000,
                "synchronization": "Nearest host receipt times; stationary board required; no hardware sync",
                "views": {}}
        for key, observation in observations.items():
            image = batch[key]["image"]
            filename = f"{key}.png"
            if not cv2.imwrite(str(temp/filename), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not save {filename}")
            packet = {k: v for k, v in batch[key].items() if k not in ("image", "depth")}
            data["views"][key] = {**observation, **packet, "image_file": filename,
                                   "camera": camera_metadata.get(key, {})}
        write_json(temp/"pose.json", data)
        os.replace(temp, self.path/pose_id)
        return data

    def exclude_last(self):
        poses = self.poses()
        if not poses:
            return None
        pose_id = poses[-1]["id"]
        self.manifest["excluded_poses"].append(pose_id)
        write_json(self.path/"session.json", self.manifest)
        return pose_id
