"""Persistent, validated manual exposure controls in hardware-native units."""
from copy import deepcopy
import json
from pathlib import Path


DEFAULTS = {
    "schema_version": 1,
    "zed": {"exposure_percent": 30, "gain": 60},
    "realsense": {
        "261222078090": {"color_exposure": 156, "color_gain": 128, "depth_exposure_us": 8500, "depth_gain": 16},
        "261222076358": {"color_exposure": 156, "color_gain": 128, "depth_exposure_us": 8500, "depth_gain": 16},
    },
}


def validate_settings(value):
    result = deepcopy(value)
    limits = {"color_exposure": (1, 333), "color_gain": (0, 128),
              "depth_exposure_us": (1, 33000), "depth_gain": (16, 248)}
    def integer(v, low, high, label):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not low <= v <= high:
            raise ValueError(f"{label} must be an integer from {low} to {high}")
        return int(v)
    for key, bounds in {"exposure_percent": (1, 100), "gain": (0, 100)}.items():
        result["zed"][key] = integer(result["zed"][key], *bounds, "ZED "+key)
    if set(result["realsense"]) != set(DEFAULTS["realsense"]):
        raise ValueError("Exposure settings must identify both configured RealSense serials")
    for serial, options in result["realsense"].items():
        for key, bounds in limits.items():
            options[key] = integer(options[key], *bounds, serial+" "+key)
    return result


def load_settings(workspace):
    path = Path(workspace)/"recording_settings.json"
    return validate_settings(json.loads(path.read_text()) if path.exists() else deepcopy(DEFAULTS))


def save_settings(workspace, value):
    value = validate_settings(value)
    path = Path(workspace)/"recording_settings.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)
