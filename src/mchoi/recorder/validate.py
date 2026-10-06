"""Read native recordings back and check stream sizes, types, counts and timing."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
import pyrealsense2 as rs

from mchoi.recorder.devices import load_zed


def timing_summary(timestamps_ns, fps=30):
    differences_ms = np.diff(np.asarray(timestamps_ns, dtype=np.int64))/1e6
    indices = np.flatnonzero(differences_ms > 1500/fps)
    return {"timestamp_gap_events": len(indices),
            "estimated_missing_frames": sum(max(1, round(float(differences_ms[i])*fps/1000)-1) for i in indices),
            "first_gap_events": [{"after_frame_index": int(i), "gap_ms": float(differences_ms[i])}
                                 for i in indices[:100]]}


def validate_realsense(path):
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True) as db:
        streams = {}
        topics = db.execute("SELECT id,name FROM topics WHERE type='sensor_msgs/msg/Image'").fetchall()
        for topic, name in topics:
            times = np.array([row[0] for row in db.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (topic,))], dtype=np.int64)
            assert len(times) >= 2, (name, len(times))
            key = "color" if "/Color_" in name else "depth"
            streams[key] = {"frames": len(times), "fps": float((len(times)-1)*1e9/(times[-1]-times[0])),
                            "duration_s": float((times[-1]-times[0])/1e9),
                            "largest_gap_ms": float(np.diff(times).max()/1e6), **timing_summary(times)}
        assert set(streams) == {"color", "depth"}, streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(path), repeat_playback=False)
    active = pipeline.start(config)
    try:
        active.get_device().as_playback().set_real_time(False)
        frames = pipeline.wait_for_frames(5000)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        rgb = np.asanyarray(color.get_data())
        z16 = np.asanyarray(depth.get_data())
        assert rgb.shape == (1080, 1920, 3) and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
        assert z16.shape == (480, 848) and z16.dtype == np.uint16, (z16.shape, z16.dtype)
        assert np.any(z16 > 0), "Depth contains no valid samples"
        for key, stream in [("color", rs.stream.color), ("depth", rs.stream.depth)]:
            profile = active.get_stream(stream)
            assert profile.fps() == 30
            streams[key].update(shape=list(rgb.shape if key == "color" else z16.shape), format=str(profile.format()))
        serial = active.get_device().get_info(rs.camera_info.serial_number)
    finally:
        pipeline.stop()
    return {"serial": serial, "streams": streams}


def validate_zed(path):
    sl = load_zed()
    camera = sl.Camera()
    params = sl.InitParameters()
    params.set_from_svo_file(str(path))
    params.svo_real_time_mode = False
    params.depth_mode = sl.DEPTH_MODE.NONE
    params.sdk_verbose = 0
    result = camera.open(params)
    assert result <= sl.ERROR_CODE.SUCCESS, str(result)
    try:
        config = camera.get_camera_information().camera_configuration
        assert (config.resolution.width, config.resolution.height) == (1920, 1080)
        assert round(config.fps) == 30
        timestamps = []
        images = {}
        while True:
            result = camera.grab()
            if result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            assert result <= sl.ERROR_CODE.SUCCESS, str(result)
            timestamps.append(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if len(timestamps) == 1:
                for name, view in [("left", sl.VIEW.LEFT_UNRECTIFIED), ("right", sl.VIEW.RIGHT_UNRECTIFIED)]:
                    mat = sl.Mat()
                    camera.retrieve_image(mat, view)
                    images[name] = list(mat.get_data().shape)
                    assert images[name][:2] == [1080, 1920], images[name]
        assert len(timestamps) >= 2
        declared = camera.get_svo_number_of_frames()
        # SDK 5.3.1 reports one more indexed frame than it decodes in our live
        # probes. Preserve that discrepancy in the report, and measure real frames.
        assert 0 <= declared-len(timestamps) <= 1, (declared, len(timestamps))
        return {"frames": len(timestamps), "sdk_reported_frames": declared,
                "sdk_frame_count_difference": declared-len(timestamps),
                "fps": (len(timestamps)-1)*1e9/(timestamps[-1]-timestamps[0]),
                "duration_s": (timestamps[-1]-timestamps[0])/1e9, "views": images,
                "largest_gap_ms": float(np.diff(timestamps).max()/1e6), **timing_summary(timestamps)}
    finally:
        camera.close()


def validate(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"recording.json").read_text())
    assert manifest["status"] == "complete", manifest["status"]
    result = {"directory": str(directory), "zed": validate_zed(directory/"zed.svo2"), "realsense": {}}
    for name, metadata in manifest["cameras"].items():
        if name.startswith("realsense_"):
            info = validate_realsense(directory/metadata["file"])
            assert info["serial"] == metadata["serial"]
            result["realsense"][name] = info
    assert len(result["realsense"]) == 2
    rates = [result["zed"]["fps"]]+[stream["fps"] for camera in result["realsense"].values()
                                     for stream in camera["streams"].values()]
    result["playback_ok"] = True
    result["rate_ok"] = all(29 <= rate <= 31 for rate in rates)
    result["continuous"] = result["zed"]["timestamp_gap_events"] == 0 and all(
        stream["timestamp_gap_events"] == 0 for camera in result["realsense"].values() for stream in camera["streams"].values())
    result["passed"] = result["playback_ok"] and result["rate_ok"] and result["continuous"]
    (directory/"playback_validation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["passed"] else 1)
