"""Camera acquisition runs independently of detection, disk writes, and the UI."""
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from cv2_enumerate_cameras import enumerate_cameras
from mchoi.paths import CONFIG_DIR


VIEW_NAMES = {"zed_left": "ZED Mini · Left", "zed_right": "ZED Mini · Right",
              "d435": "RealSense #1 · RGB", "d455": "RealSense #2 · RGB"}
VIEW_ORDER = tuple(VIEW_NAMES)


def bind_realsense_devices(devices, slots):
    """Bind physical cameras by serial, independently of model or enumeration order."""
    required = ("d435", "d455")
    if set(slots) != set(required):
        raise ValueError("camera_config.json must define the two RealSense slots: d435 and d455.")
    serials = [str(slots[key].get("serial", "")) for key in required]
    if any(not serial for serial in serials) or len(set(serials)) != 2:
        raise ValueError("Configure two different RealSense serial numbers in camera_config.json.")
    available = {device.get_info(rs.camera_info.serial_number): device for device in devices}
    missing = [serial for serial in serials if serial not in available]
    if missing:
        raise RuntimeError("Configured RealSense camera missing: "+", ".join(missing)+
                           ". Connected serials: "+(", ".join(sorted(available)) or "none")+
                           ". Update camera_config.json after replacing a camera.")
    return [(key, available[serial]) for key, serial in zip(required, serials)]


def view_label(key, info):
    if key.startswith("zed_"):
        return VIEW_NAMES[key]
    name = info.get("name", "RealSense").replace("Intel ", "")
    number = 1 if key == "d435" else 2
    serial = info.get("serial", "")
    return f"{name} #{number}"+(f" · {serial[-6:]}" if serial else "")


@dataclass(frozen=True)
class Frame:
    images: dict
    received_at: float
    device_timestamp_ms: float | None
    timestamp_domain: str
    sequence: int
    depth: dict | None = None


def intrinsics_dict(profile):
    i = profile.as_video_stream_profile().get_intrinsics()
    return {"image_size": [i.width, i.height], "camera_matrix":
            [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]],
            "distortion_coefficients": list(i.coeffs), "distortion_model": str(i.model)}


class CameraWorker(threading.Thread):
    def __init__(self, spec, stop_event):
        super().__init__(daemon=True, name=f"capture-{spec['key']}")
        self.spec = spec
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.frames = deque(maxlen=5)
        self.arrivals = deque(maxlen=90)
        self.metadata = dict(spec)
        self.status = "Opening…"
        self.error = None

    def push(self, frame):
        with self.lock:
            self.frames.append(frame)
            self.arrivals.append(frame.received_at)
            self.status = "Streaming"

    def snapshot(self):
        with self.lock:
            return list(self.frames)

    def info(self):
        with self.lock:
            fps = ((len(self.arrivals)-1) / (self.arrivals[-1]-self.arrivals[0])
                   if len(self.arrivals) > 2 and self.arrivals[-1] > self.arrivals[0] else 0)
            return {**self.metadata, "measured_fps": fps, "status": self.status,
                    "error": self.error,
                    "age_s": time.monotonic()-self.arrivals[-1] if self.arrivals else None}

    def run(self):
        try:
            if self.spec["backend"] == "realsense":
                self.run_realsense()
            else:
                self.run_zed()
        except Exception as exc:
            self.error = str(exc)
            self.status = "Disconnected / error"

    def run_realsense(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.spec["serial"])
        w, h = self.spec["image_size"]
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
        depth_size = self.spec.get("depth_size")
        if depth_size:
            config.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, 30)
        started = False
        try:
            active = pipeline.start(config)
            started = True
            color_profile = active.get_stream(rs.stream.color)
            self.metadata["factory_intrinsics"] = intrinsics_dict(color_profile)
            depth_description = None
            if depth_size:
                depth_profile = active.get_stream(rs.stream.depth)
                extrinsics = depth_profile.get_extrinsics_to(color_profile)
                depth_transform = np.eye(4)
                depth_transform[:3, :3] = np.array(extrinsics.rotation).reshape(3, 3, order="F")
                depth_transform[:3, 3] = extrinsics.translation
                depth_description = {"intrinsics": intrinsics_dict(depth_profile),
                                     "units_m": active.get_device().first_depth_sensor().get_depth_scale(),
                                     "T_color_from_depth": depth_transform.tolist()}
            sensor = active.get_device().first_color_sensor()
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
            if sensor.supports(rs.option.auto_exposure_priority):
                sensor.set_option(rs.option.auto_exposure_priority, 0)
            for depth_sensor in active.get_device().query_sensors():
                profiles = [p for p in depth_sensor.get_stream_profiles()
                            if p.stream_type() == rs.stream.depth and p.is_video_stream_profile()]
                if profiles:
                    depth = max(profiles, key=lambda p: p.as_video_stream_profile().width()*
                                p.as_video_stream_profile().height())
                    ext = depth.get_extrinsics_to(color_profile)
                    transform = np.eye(4)
                    transform[:3, :3] = np.array(ext.rotation).reshape(3, 3, order="F")
                    transform[:3, 3] = ext.translation
                    self.metadata["factory_T_color_from_depth"] = transform.tolist()
                    break
            sequence = 0
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1500)
                color = frames.get_color_frame()
                if not color:
                    continue
                received = time.monotonic()
                image = np.asanyarray(color.get_data()).copy()
                depth_data = None
                depth_frame = frames.get_depth_frame()
                if depth_description is not None and depth_frame:
                    depth_data = {**depth_description, "image": np.asanyarray(depth_frame.get_data()).copy(),
                                  "device_timestamp_ms": depth_frame.get_timestamp()}
                sequence += 1
                self.push(Frame({self.spec["key"]: image}, received,
                                color.get_timestamp(), str(color.get_frame_timestamp_domain()), sequence, depth_data))
        finally:
            if started:
                pipeline.stop()

    def run_zed(self):
        cap = cv2.VideoCapture(self.spec["index"], cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open ZED Mini; close other camera applications.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            self.metadata["reported_fps"] = cap.get(cv2.CAP_PROP_FPS)
            sequence = 0
            while not self.stop_event.is_set():
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError("ZED stopped delivering frames.")
                if image.shape[:2] != (1080, 3840):
                    raise RuntimeError(f"ZED did not accept 1080p30: received {image.shape[1]}×{image.shape[0]}.")
                sequence += 1
                self.push(Frame({"zed_left": image[:, :1920], "zed_right": image[:, 1920:]},
                                time.monotonic(), None, "host_receive_only", sequence))
        finally:
            cap.release()


class CameraRig:
    def __init__(self, config_path=None):
        self.stop_event = threading.Event()
        self.workers = []
        self.config_path = Path(config_path) if config_path else CONFIG_DIR/"camera_config.json"

    def start(self):
        specs = []
        try:
            slots = json.loads(self.config_path.read_text(encoding="utf-8"))["realsense_slots"]
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError("Cannot read camera_config.json; configure both RealSense serial numbers.") from exc
        context = rs.context()
        for key, device in bind_realsense_devices(context.query_devices(), slots):
            name = device.get_info(rs.camera_info.name)
            profiles = [p.as_video_stream_profile() for s in device.query_sensors()
                        for p in s.get_stream_profiles() if p.stream_type() == rs.stream.color
                        and p.fps() == 30 and p.format() == rs.format.bgr8]
            if not profiles:
                raise RuntimeError(f"{name} has no available RGB 30 fps profile.")
            best = max(profiles, key=lambda p: p.width()*p.height())
            depth_profiles = [p.as_video_stream_profile() for s in device.query_sensors()
                              for p in s.get_stream_profiles() if p.stream_type() == rs.stream.depth
                              and p.fps() == 30 and p.format() == rs.format.z16]
            economical = [p for p in depth_profiles if p.width() <= 848 and p.height() <= 480]
            depth = max(economical or depth_profiles, key=lambda p: p.width()*p.height()) if depth_profiles else None
            specs.append({"key": key, "backend": "realsense", "name": name,
                          "label": slots[key].get("label", name),
                          "serial": device.get_info(rs.camera_info.serial_number),
                          "usb": device.get_info(rs.camera_info.usb_type_descriptor),
                          "image_size": [best.width(), best.height()], "requested_fps": 30,
                          "depth_size": [depth.width(), depth.height()] if depth else None,
                          "available_30fps_sizes": sorted({(p.width(), p.height()) for p in profiles},
                                                          key=lambda s: s[0]*s[1], reverse=True)})
        zeds = [c for c in enumerate_cameras(cv2.CAP_DSHOW) if "ZED-M" in c.name]
        if zeds:
            camera = zeds[0]
            specs.append({"key": "zed", "backend": "directshow", "name": camera.name,
                          "index": camera.index, "device_path": camera.path,
                          "image_size": [1920, 1080], "requested_fps": 30,
                          "image_type": "unrectified stereo; 3840x1080 side-by-side"})
        missing = {"d435", "d455", "zed"} - {s["key"] for s in specs}
        if missing:
            raise RuntimeError("Missing cameras: " + ", ".join(sorted(missing)))
        self.workers = [CameraWorker(spec, self.stop_event) for spec in specs]
        for worker in self.workers:
            worker.start()

    def batch(self):
        """Select recent frames nearest a common host receipt time, without claiming hardware sync."""
        rings = [worker.snapshot() for worker in self.workers]
        if not rings or any(not ring for ring in rings):
            return {}
        anchor = min(ring[-1].received_at for ring in rings)
        batch = {}
        for ring in rings:
            packet = min(ring, key=lambda frame: abs(frame.received_at-anchor))
            for key, image in packet.images.items():
                batch[key] = {"image": image, "received_at": packet.received_at,
                              "device_timestamp_ms": packet.device_timestamp_ms,
                              "timestamp_domain": packet.timestamp_domain,
                              "sequence": packet.sequence}
                if packet.depth is not None:
                    batch[key]["depth"] = packet.depth
        return batch

    def metadata(self):
        result = {}
        for worker in self.workers:
            info = worker.info()
            keys = ("zed_left", "zed_right") if info["key"] == "zed" else (info["key"],)
            for key in keys:
                result[key] = {**info, "view": key}
        return result

    def stop(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=3)
