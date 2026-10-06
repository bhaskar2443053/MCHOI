"""One acquisition thread per physical camera; SDKs write full-resolution data."""
from collections import deque
from concurrent.futures import Future
import csv
import json
import os
from pathlib import Path
import queue
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone

import cv2
import numpy as np
import pyrealsense2 as rs

from .settings import DEFAULTS

RGB_SIZE = (1920, 1080)
DEPTH_SIZE = (848, 480)
FPS = 30
PREVIEW_SERIAL = "261222076358"


def load_slots(workspace):
    slots = json.loads((Path(workspace)/"camera_config.json").read_text())["realsense_slots"]
    ordered = [slots[key] for key in ("d435", "d455")]
    serials = [str(slot["serial"]) for slot in ordered]
    if len(set(serials)) != 2 or PREVIEW_SERIAL not in serials:
        raise ValueError("Configure two different RealSense serials, including preview camera 261222076358.")
    return ordered


def load_zed():
    # Python 3.8+ needs explicit DLL search directories on Windows.
    handles = []
    if os.name == "nt":
        roots = [os.environ.get("ZED_SDK_ROOT_DIR", "C:/Program Files (x86)/ZED SDK")]
        cuda = os.environ.get("CUDA_PATH")
        if cuda:
            roots.append(cuda)
        for root in roots:
            directory = Path(root)/"bin"
            if directory.is_dir():
                handles.append(os.add_dll_directory(str(directory)))
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise RuntimeError("ZED Python API is unavailable. Install the matching API for the installed ZED SDK; see README.md.") from exc
    # Retain directory handles for the lifetime of the loaded module.
    sl._recorder_dll_handles = handles
    return sl


def intrinsics(profile):
    i = profile.as_video_stream_profile().get_intrinsics()
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
            "ppx": i.ppx, "ppy": i.ppy, "coeffs": list(i.coeffs), "model": str(i.model)}


class CameraWorker(threading.Thread):
    def __init__(self, key):
        super().__init__(name="record-"+key, daemon=True)
        self.key = key
        self.commands = queue.Queue()
        self.ready = Future()
        self.lock = threading.Lock()
        self.shutdown = threading.Event()
        self.error = None
        self.metadata = {}
        self.arrivals = deque(maxlen=90)
        self.preview = None
        self.preview_sequence = 0
        self.recording = False
        self.prepared = False
        self.csv_file = None
        self.csv_writer = None
        self.stats = {}
        self.last_received = time.monotonic()
        self.last_stream_timestamp = {}

    def command(self, action, *args):
        future = Future()
        with self.lock:
            if self.error or not self.is_alive():
                future.set_exception(RuntimeError(self.error or self.key+" is not running"))
            else:
                self.commands.put((action, args, future))
        return future

    def snapshot(self):
        with self.lock:
            fps = ((len(self.arrivals)-1)/(self.arrivals[-1]-self.arrivals[0])
                   if len(self.arrivals) > 1 and self.arrivals[-1] > self.arrivals[0] else 0)
            return {"key": self.key, "error": self.error, "fps": fps,
                    "age_s": time.monotonic()-self.last_received,
                    "recording": self.recording, "metadata": dict(self.metadata),
                    "stats": deepcopy(self.stats)}

    def get_preview(self):
        with self.lock:
            return self.preview_sequence, self.preview

    def observed(self, stream, number, device_ms, domain, host_ns, mono_ns, pc_timestamp_ns=None):
        if not self.recording:
            return
        pc_timestamp_ns = host_ns if pc_timestamp_ns is None else pc_timestamp_ns
        self.csv_writer.writerow([stream, number, f"{device_ms:.6f}", domain, host_ns, mono_ns,
                                  pc_timestamp_ns, datetime.fromtimestamp(pc_timestamp_ns/1e9, timezone.utc).astimezone().isoformat()])
        with self.lock:
            item = self.stats.setdefault(stream, {"observed_frames": 0, "observed_number_gaps": 0,
                                                 "first_timestamp_ms": device_ms, "first_host_unix_ns": host_ns})
            previous = item.get("last_frame_number")
            if previous is not None and number > previous:
                item["observed_number_gaps"] += max(0, number-previous-1)
            item.update(last_frame_number=number, last_timestamp_ms=device_ms, last_host_unix_ns=host_ns)
            item["observed_frames"] += 1
            previous_ms = self.last_stream_timestamp.get(stream)
            if previous_ms is not None:
                gap_ms = device_ms-previous_ms
                item["max_timestamp_gap_ms"] = max(item.get("max_timestamp_gap_ms", 0), gap_ms)
                if gap_ms > 1500/FPS:
                    item["timestamp_gap_events"] = item.get("timestamp_gap_events", 0)+1
                    item["estimated_missing_frames"] = item.get("estimated_missing_frames", 0)+max(1, round(gap_ms*FPS/1000)-1)
                    # Bounded details: totals keep increasing without growing RAM indefinitely.
                    events = item.setdefault("first_gap_events", [])
                    if len(events) < 100:
                        events.append({"after_observed_index": item["observed_frames"]-2, "gap_ms": gap_ms})
            self.last_stream_timestamp[stream] = device_ms

    def received(self):
        now = time.monotonic()
        with self.lock:
            self.arrivals.append(now)
            self.last_received = now

    def prepare(self, directory):
        if self.prepared:
            raise RuntimeError("Already prepared for recording")
        with self.lock:
            self.stats = {}
        self.last_stream_timestamp = {}
        self.csv_file = (directory/(self.key+"_timestamps.csv")).open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["stream", "frame_number", "device_timestamp_ms", "timestamp_domain",
                                  "host_unix_ns", "host_monotonic_ns", "pc_timestamp_ns", "pc_local_time"])
        # Set before opening so a partial prepare always gets cleaned up.
        self.prepared = True
        self.prepare_native(directory)

    def begin(self):
        if not self.prepared:
            raise RuntimeError("Recorder is not prepared")
        self.begin_native()
        self.recording = True
        with self.lock:
            self.stats["start_host_unix_ns"] = time.time_ns()

    def settle(self):
        # Pipelines were reopened during prepare. Keep files paused while exposure,
        # timestamps and synchronized RGB/depth frames settle.
        until = time.monotonic()+.8
        while time.monotonic() < until:
            if self.shutdown.is_set():
                raise RuntimeError("Recording preparation cancelled")
            self.read_native()

    def finish(self):
        if not self.prepared:
            return self.stats
        self.recording = False
        try:
            self.finish_native()
        finally:
            self.prepared = False
            with self.lock:
                self.stats["stop_host_unix_ns"] = time.time_ns()
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
        return self.stats

    def run(self):
        try:
            self.open_native()
            self.last_received = time.monotonic()
            # Require an actual frame before enabling the Record button.
            while not self.read_native():
                if self.shutdown.is_set():
                    raise RuntimeError("Camera connection cancelled")
            self.ready.set_result(True)
            while not self.shutdown.is_set():
                try:
                    action, args, future = self.commands.get_nowait()
                except queue.Empty:
                    self.read_native()
                    continue
                try:
                    result = getattr(self, action)(*args)
                    future.set_result(result)
                except Exception as exc:
                    future.set_exception(exc)
                    raise
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            if not self.ready.done():
                self.ready.set_exception(exc)
        finally:
            try:
                self.finish()
            except Exception as exc:
                self.error = self.error or str(exc)
            try:
                self.close_native()
            except Exception as exc:
                self.error = self.error or str(exc)
            if not self.ready.done():
                self.ready.set_exception(RuntimeError("Camera closed"))
            while True:
                try:
                    _, _, future = self.commands.get_nowait()
                    future.set_exception(RuntimeError(self.error or "Camera closed"))
                except queue.Empty:
                    break


class RealSenseWorker(CameraWorker):
    def __init__(self, slot, settings=None):
        super().__init__("realsense_"+str(slot["serial"]))
        self.serial = str(slot["serial"])
        self.label = slot.get("label", "RealSense")
        self.pipeline = None
        self.active = None
        self.native_recorder = None
        self.context = None
        self.last_preview = 0
        self.last_numbers = {}
        self.settings = deepcopy(settings if settings is not None else DEFAULTS["realsense"][self.serial])

    def apply_settings(self, settings):
        if self.prepared:
            raise RuntimeError("Stop recording before changing exposure")
        self.settings = deepcopy(settings)
        device = self.active.get_device()
        self.set_sensor_options(device)
        self.metadata["controls"] = self.read_sensor_options(device)
        return self.metadata["controls"]

    def set_sensor_options(self, device):
        for sensor, exposure, gain in [
            (device.first_color_sensor(), self.settings["color_exposure"], self.settings["color_gain"]),
            (device.first_depth_sensor(), self.settings["depth_exposure_us"], self.settings["depth_gain"])]:
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, exposure)
            sensor.set_option(rs.option.gain, gain)
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 0)

    def read_sensor_options(self, device):
        controls = {"focus": "Fixed lens; no software focus control", "requested": dict(self.settings)}
        for name, sensor in [("color", device.first_color_sensor()), ("depth", device.first_depth_sensor())]:
            controls[name] = {"auto_exposure": bool(sensor.get_option(rs.option.enable_auto_exposure)),
                              "exposure": sensor.get_option(rs.option.exposure),
                              "gain": sensor.get_option(rs.option.gain),
                              "exposure_unit": "0.1 ms (Windows quantizes values)" if name == "color" else "microseconds"}
        return controls

    def open_native(self, filename=None):
        self.context = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d for d in self.context.query_devices()}
        if self.serial not in devices:
            raise RuntimeError("RealSense "+self.serial+" is not connected")
        device = devices[self.serial]
        for stream, size, fmt in [(rs.stream.color, RGB_SIZE, rs.format.bgr8),
                                  (rs.stream.depth, DEPTH_SIZE, rs.format.z16)]:
            available = [p.as_video_stream_profile() for s in device.query_sensors()
                         for p in s.get_stream_profiles() if p.stream_type() == stream and p.is_video_stream_profile()]
            if not any((p.width(), p.height()) == size and p.fps() == FPS and p.format() == fmt for p in available):
                raise RuntimeError(f"{self.serial}: {stream} {size[0]}x{size[1]} at 30 fps is unavailable. Check the USB 3 connection.")
        self.set_sensor_options(device)
        self.pipeline = rs.pipeline(self.context)
        if filename:
            # Pause before streaming so no warm-up frames enter the take.
            # Uncompressed native recording preserves BGR8 and metric Z16 samples.
            self.native_recorder = rs.recorder(str(filename), device, False)
            self.native_recorder.pause()
            self.pipeline.set_device(self.native_recorder)
        else:
            self.pipeline.set_device(device)
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, *RGB_SIZE, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, *DEPTH_SIZE, rs.format.z16, FPS)
        self.active = self.pipeline.start(cfg)
        if self.active.get_device().get_info(rs.camera_info.serial_number) != self.serial:
            raise RuntimeError("RealSense serial binding failed")
        color = self.active.get_stream(rs.stream.color)
        depth = self.active.get_stream(rs.stream.depth)
        ext = depth.get_extrinsics_to(color)
        self.metadata = {"serial": self.serial, "name": device.get_info(rs.camera_info.name),
                         "label": self.label, "rgb_size": list(RGB_SIZE), "depth_size": list(DEPTH_SIZE),
                         "fps": FPS, "rgb_format": "BGR8", "depth_format": "Z16",
                         "usb": device.get_info(rs.camera_info.usb_type_descriptor),
                         "controls": self.read_sensor_options(device),
                         "depth_units_m": device.first_depth_sensor().get_depth_scale(),
                         "color_intrinsics": intrinsics(color), "depth_intrinsics": intrinsics(depth),
                         "depth_to_color": {"rotation_column_major": list(ext.rotation),
                                            "translation_m": list(ext.translation)},
                         "file": Path(filename).name if filename else None}
        self.last_numbers = {}
        self.last_received = time.monotonic()

    def read_native(self):
        ok, frames = self.pipeline.try_wait_for_frames(250)
        if not ok:
            if time.monotonic()-self.last_received > 4:
                raise RuntimeError(self.serial+": no camera frames for four seconds")
            return False
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return False
        self.received()
        host_ns, mono_ns = time.time_ns(), time.monotonic_ns()
        for name, frame in [("color", color), ("depth", depth)]:
            number = frame.get_frame_number()
            if self.last_numbers.get(name) != number:
                self.observed(name, number, frame.get_timestamp(), str(frame.get_frame_timestamp_domain()), host_ns, mono_ns)
                self.last_numbers[name] = number
        if self.serial == PREVIEW_SERIAL and time.monotonic()-self.last_preview > .08:
            thumbnail = cv2.resize(np.asanyarray(color.get_data()), (960, 540), interpolation=cv2.INTER_AREA)
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.preview = thumbnail
                self.preview_sequence += 1
            self.last_preview = time.monotonic()
        return True

    def prepare_native(self, directory):
        self.close_native()
        self.open_native(directory/(self.key+".db3"))

    def begin_native(self):
        self.last_numbers = {}
        self.native_recorder.resume()

    def finish_native(self):
        if self.native_recorder:
            self.native_recorder.pause()
            # Release every recorder/profile reference to flush the native file.
            self.close_native()
            if not self.shutdown.is_set() and not self.error:
                self.open_native()

    def close_native(self):
        try:
            if self.active is not None:
                self.pipeline.stop()
        finally:
            self.active = None
            self.pipeline = None
            self.native_recorder = None
            self.context = None


class ZedWorker(CameraWorker):
    def __init__(self, settings=None):
        super().__init__("zed")
        self.camera = None
        self.sl = None
        self.native_recording = False
        self.frame_number = 0
        self.settings = deepcopy(settings if settings is not None else DEFAULTS["zed"])

    def apply_settings(self, settings):
        if self.prepared:
            raise RuntimeError("Stop recording before changing exposure")
        self.settings = deepcopy(settings)
        sl = self.sl
        for setting, value in [(sl.VIDEO_SETTINGS.AEC_AGC, 0),
                               (sl.VIDEO_SETTINGS.EXPOSURE, self.settings["exposure_percent"]),
                               (sl.VIDEO_SETTINGS.GAIN, self.settings["gain"])]:
            self.require_success(self.camera.set_camera_settings(setting, value), "Set ZED exposure/gain")
        controls = {"focus": "Fixed lens; no software focus control", "requested": dict(settings)}
        for key, setting in [("auto_exposure", sl.VIDEO_SETTINGS.AEC_AGC),
                              ("exposure_percent", sl.VIDEO_SETTINGS.EXPOSURE), ("gain", sl.VIDEO_SETTINGS.GAIN)]:
            result, value = self.camera.get_camera_settings(setting)
            self.require_success(result, "Read ZED controls")
            controls[key] = value
        self.metadata["controls"] = controls
        return controls

    def require_success(self, result, action):
        if result > self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(action+": "+str(result))

    def open_native(self):
        self.sl = load_zed()
        self.camera = self.sl.Camera()
        params = self.sl.InitParameters()
        params.camera_resolution = self.sl.RESOLUTION.HD1080
        params.camera_fps = FPS
        params.depth_mode = self.sl.DEPTH_MODE.NONE
        params.async_image_retrieval = False
        params.sdk_verbose = 0
        self.require_success(self.camera.open(params), "Open ZED Mini (close other camera apps)")
        info = self.camera.get_camera_information()
        config = info.camera_configuration
        if (config.resolution.width, config.resolution.height) != RGB_SIZE or round(config.fps) != FPS:
            raise RuntimeError("ZED did not accept 1920x1080 per lens at 30 fps")
        self.metadata = {"name": str(info.camera_model), "serial": int(info.serial_number),
                         "per_lens_size": list(RGB_SIZE), "fps": config.fps,
                         "views": ["left", "right"], "sdk_version": self.camera.get_sdk_version(),
                         "compression": "H264", "container": "SVO2", "file": None}
        self.apply_settings(self.settings)

    def read_native(self):
        result = self.camera.grab()
        if result > self.sl.ERROR_CODE.SUCCESS:
            if time.monotonic()-self.last_received > 4:
                raise RuntimeError("ZED grab failed: "+str(result))
            time.sleep(.01)
            return False
        self.received()
        self.frame_number += 1
        if self.recording:
            status = self.camera.get_recording_status()
            if not status.status:
                # NVENC may report no completed write for the first frame after resume.
                if time.monotonic()-self.last_recorded_at > 2:
                    raise RuntimeError("ZED has not written an SVO frame for two seconds")
            else:
                self.last_recorded_at = time.monotonic()
            # Include encoder warm-up grabs in the ledger. Native file indexing
            # verifies which endpoint frames actually reached the finalized SVO.
            image_timestamp_ns = self.camera.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            timestamp_ms = image_timestamp_ns/1e6
            self.observed("stereo", self.frame_number, timestamp_ms, "zed_image_timestamp",
                          time.time_ns(), time.monotonic_ns(), image_timestamp_ns)
            with self.lock:
                self.stats["sdk_dropped_frames"] = self.camera.get_frame_dropped_count()-self.drop_count_start
        return True

    def prepare_native(self, directory):
        filename = directory/"zed.svo2"
        params = self.sl.RecordingParameters(str(filename), self.sl.SVO_COMPRESSION_MODE.H264)
        self.require_success(self.camera.enable_recording(params), "Start SVO recording")
        self.native_recording = True
        self.camera.pause_recording(True)
        self.drop_count_start = self.camera.get_frame_dropped_count()
        self.metadata["file"] = filename.name

    def begin_native(self):
        self.drop_count_start = self.camera.get_frame_dropped_count()
        self.camera.pause_recording(False)
        self.last_recorded_at = time.monotonic()

    def finish_native(self):
        if self.native_recording:
            self.camera.disable_recording()
            self.native_recording = False

    def close_native(self):
        if self.camera is not None:
            self.camera.close()
            self.camera = None
