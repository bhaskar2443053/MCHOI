from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .board import BoardDetector, annotate, load_board, movement
from .cameras import CameraRig, VIEW_NAMES, VIEW_ORDER, view_label
from .coverage import attach_coverage, coverage_mask, overlay_coverage
from .solver import MIN_CORNERS, MIN_INTRINSIC_POSES, calibrate_session, export_result
from .storage import Session
from .verification import (VerificationError, StereoDepth, board_point, check_frames,
                           load_calibration, project_reference_point, realsense_point)
from mchoi.paths import CONFIG_DIR


BACKGROUND = "#11171e"
PANEL = "#1b2530"
TEXT = "#e6edf5"
MUTED = "#99adbd"
GREEN = "#66dbb0"
AMBER = "#f3c477"


class CalibrationApp:
    def __init__(self, root, workspace, offline=False):
        self.root = root
        self.workspace = Path(workspace)
        self.config, self.board = load_board(CONFIG_DIR/"board_config.json")
        self.session = self.new_session_object()
        self.events = queue.Queue()
        self.rig = None
        self.stop_detection = threading.Event()
        self.detection_thread = None
        self.state_lock = threading.Lock()
        self.latest = None
        self.history = {key: deque(maxlen=10) for key in VIEW_ORDER}
        self.rois = {}
        self.calibration = None
        self.busy = False
        self.connecting = False
        self.closing = False
        self.last_drawn = None
        self.last_auto = 0
        self.images = {}
        self.image_geometry = {}
        self.drag = None
        self.ready = {}
        self.counts = {}
        self.view_names = dict(VIEW_NAMES)
        self.coverage_masks = {}
        self.coverage_visible = False
        self.verification_frames = None
        self.verification_waiting = False
        self.verification_result = None
        self.verification_token = 0
        self.verification_working = False
        self.stereo_depth = None
        self.loaded_calibration_path = None
        self.verification_message = "Load a calibration file to check pixel alignment."
        self.build_ui()
        self.refresh_session()
        self.log("Keep cameras fixed. Move and tilt the board between captures; hold it still to capture.")
        self.root.after(80, self.tick)
        if not offline:
            self.root.after(250, self.connect)

    def new_session_object(self):
        return Session(self.workspace/"sessions"/datetime.now().strftime("%Y%m%d_%H%M%S_%f"), self.config)

    def build_ui(self):
        root = self.root
        root.title("Multi-camera calibration")
        window_width = min(1440, root.winfo_screenwidth()-60)
        window_height = min(960, root.winfo_screenheight()-100)
        root.geometry(f"{window_width}x{window_height}+20+20")
        root.minsize(1000, 700)
        root.configure(bg=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8), background="#2b3e4c", foreground=TEXT)
        style.map("TButton", background=[("active", "#3c5667"), ("disabled", "#26313c")],
                  foreground=[("disabled", "#6f8392")])
        style.configure("Accent.TButton", background="#277a65", foreground="white")
        style.map("Accent.TButton", background=[("active", "#32977c"), ("disabled", "#264b43")])
        style.configure("TCheckbutton", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", BACKGROUND)], foreground=[("active", TEXT)])
        style.configure("TProgressbar", background=GREEN, troughcolor=PANEL)

        header = ttk.Frame(root, padding=(18, 14, 18, 7))
        header.pack(fill="x")
        ttk.Label(header, text="Multi-camera calibration", font=("Segoe UI Semibold", 21)).pack(side="left")
        ttk.Label(header, text="5 × 7 ChArUco  ·  30 mm squares / 22 mm markers  ·  4×4_50",
                  foreground=MUTED).pack(side="right")
        toolbar = ttk.Frame(root, padding=(18, 5))
        toolbar.pack(fill="x")
        self.connect_button = ttk.Button(toolbar, text="Connect cameras", command=self.toggle_connection)
        self.connect_button.pack(side="left", padx=(0, 8))
        self.capture_button = ttk.Button(toolbar, text="Capture pose  [Space]", style="Accent.TButton", command=self.capture)
        self.capture_button.pack(side="left", padx=(0, 12))
        self.auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Auto capture", variable=self.auto_var).pack(side="left", padx=(0, 12))
        self.all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Require all views", variable=self.all_var).pack(side="left")
        self.calibrate_button = ttk.Button(toolbar, text="Calibrate all", command=self.calibrate)
        self.calibrate_button.pack(side="right")
        self.progress = ttk.Progressbar(toolbar, length=120, maximum=20)
        self.progress.pack(side="right", padx=12)
        self.pose_label = ttk.Label(toolbar, text="0 poses")
        self.pose_label.pack(side="right")

        verification = ttk.Frame(root, padding=(18, 3))
        verification.pack(fill="x")
        self.load_calibration_button = ttk.Button(verification, text="Load calibration…", command=self.load_calibration_file)
        self.load_calibration_button.pack(side="left", padx=(0, 10))
        self.verify_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(verification, text="Verify clicks", variable=self.verify_var,
                        command=self.toggle_verification).pack(side="left", padx=(0, 12))
        self.depth_mode = tk.StringVar(value="Automatic depth")
        depth_choice = ttk.Combobox(verification, textvariable=self.depth_mode, width=18,
                                   values=("Automatic depth", "Board plane"), state="readonly")
        depth_choice.pack(side="left", padx=(0, 8))
        depth_choice.bind("<<ComboboxSelected>>", lambda event: self.clear_verification())
        ttk.Button(verification, text="Refresh views", command=self.refresh_verification).pack(side="left", padx=(0, 6))
        ttk.Button(verification, text="Clear dots", command=self.clear_verification).pack(side="left")
        self.coverage_button = ttk.Button(verification, text="Show calibration ROI", command=self.toggle_coverage)
        self.coverage_button.pack(side="left", padx=(8, 0))
        self.calibration_label = ttk.Label(verification, text="No calibration loaded", foreground=MUTED)
        self.calibration_label.pack(side="right")

        helpbar = ttk.Frame(root, padding=(18, 4, 18, 8))
        helpbar.pack(fill="x")
        ttk.Label(helpbar, text="Aim for 20–30 poses: vary position, distance, and tilt. Hold still for 0.6 s. Drag on a view to restrict detection; right-click to clear.",
                  foreground=MUTED).pack(anchor="w")
        self.status = tk.StringVar(value="Opening cameras…")
        ttk.Label(helpbar, textvariable=self.status, foreground=GREEN).pack(anchor="w", pady=(4, 0))

        grid = ttk.Frame(root, padding=(14, 0))
        grid.pack(fill="both", expand=True)
        for i in range(2):
            grid.columnconfigure(i, weight=1, uniform="camera")
            grid.rowconfigure(i, weight=1, uniform="camera")
        self.canvases, self.camera_labels, self.camera_details, self.camera_headings = {}, {}, {}, {}
        for index, key in enumerate(VIEW_ORDER):
            card = tk.Frame(grid, bg=PANEL, padx=8, pady=7)
            card.grid(row=index//2, column=index%2, sticky="nsew", padx=4, pady=4)
            heading = tk.Frame(card, bg=PANEL)
            heading.pack(fill="x")
            self.camera_headings[key] = tk.Label(heading, text=self.view_names[key], bg=PANEL, fg=TEXT,
                                                  font=("Segoe UI Semibold", 12))
            self.camera_headings[key].pack(side="left")
            self.camera_labels[key] = tk.Label(heading, text="Waiting", bg=PANEL, fg=MUTED, font=("Segoe UI", 10))
            self.camera_labels[key].pack(side="right")
            canvas = tk.Canvas(card, bg="#080d12", highlightthickness=0, width=450, height=175)
            canvas.pack(fill="both", expand=True, pady=5)
            canvas.bind("<ButtonPress-1>", lambda e, k=key: self.start_roi(k, e))
            canvas.bind("<B1-Motion>", lambda e, k=key: self.move_roi(k, e))
            canvas.bind("<ButtonRelease-1>", lambda e, k=key: self.end_roi(k, e))
            canvas.bind("<Button-3>", lambda e, k=key: self.clear_roi(k))
            canvas.bind("<Configure>", lambda event: setattr(self, "last_drawn", None))
            self.canvases[key] = canvas
            self.camera_details[key] = tk.Label(card, text="Native-resolution frames are saved without overlays.",
                                               anchor="w", bg=PANEL, fg=MUTED, font=("Segoe UI", 9))
            self.camera_details[key].pack(fill="x")

        bottom = ttk.Frame(root, padding=(18, 6))
        bottom.pack(fill="x")
        self.session_label = ttk.Label(bottom, text="", foreground=MUTED)
        self.session_label.pack(side="left")
        self.session_buttons = []
        for text, command in [("Open folder", self.open_folder), ("Resume session", self.resume_session),
                              ("New session", self.new_session), ("Undo last pose", self.undo)]:
            button = ttk.Button(bottom, text=text, command=command)
            button.pack(side="right", padx=(5, 0))
            self.session_buttons.append(button)
        self.logbox = tk.Text(root, height=4, bg="#0c1218", fg=MUTED, relief="flat",
                              font=("Consolas", 9), wrap="word", padx=12, pady=6, state="disabled")
        self.logbox.pack(fill="x", padx=18, pady=(0, 10))
        root.bind("<space>", self.on_space)

    def on_space(self, event):
        if self.root.focus_get() and self.root.focus_get().winfo_toplevel() == self.root:
            self.capture()
        return "break"

    def log(self, message):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", datetime.now().strftime("%H:%M:%S")+"  "+message+"\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def toggle_connection(self):
        if self.rig:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        if self.connecting or self.closing:
            return
        self.connecting = True
        self.connect_button.configure(state="disabled")
        self.status.set("Opening cameras at their highest available 30 fps modes…")

        def work():
            rig = CameraRig(CONFIG_DIR/"camera_config.json")
            try:
                rig.start()
                self.events.put(("connected", rig))
            except Exception as exc:
                rig.stop()
                self.events.put(("connect_error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def disconnect(self):
        if self.connecting or self.busy:
            return
        self.auto_var.set(False)
        self.connecting = True
        self.connect_button.configure(state="disabled")
        self.stop_detection.set()
        rig = self.rig

        def work():
            if self.detection_thread:
                self.detection_thread.join(timeout=3)
            if rig:
                rig.stop()
            self.events.put(("disconnected", None))
        threading.Thread(target=work, daemon=True).start()

    def detection_loop(self, rig, stop_event):
        detector = BoardDetector(self.board)
        previous_sequences = None
        while not stop_event.is_set():
            started = time.monotonic()
            try:
                batch = rig.batch()
                sequences = tuple((key, value["sequence"]) for key, value in sorted(batch.items()))
                if not batch or sequences == previous_sequences:
                    stop_event.wait(.03)
                    continue
                previous_sequences = sequences
                with self.state_lock:
                    rois = dict(self.rois)
                observations = {key: detector.detect(packet["image"], rois.get(key))
                                for key, packet in batch.items()}
                with self.state_lock:
                    for key, observation in observations.items():
                        stamp = batch[key]["received_at"]
                        history = self.history[key]
                        if not history or history[-1][0] != stamp:
                            history.append((stamp, observation))
                    self.latest = {"batch": batch, "observations": observations, "generation": time.monotonic()}
            except Exception as exc:
                self.events.put(("log", "Detection error: "+str(exc)))
                stop_event.wait(.5)
            stop_event.wait(max(.005, .12-(time.monotonic()-started)))

    def stable(self, key, observation, stamp):
        history = list(self.history[key])
        old = [(t, obs) for t, obs in history if .55 <= stamp-t <= 1.2]
        if not old:
            return False
        threshold = max(1.4, observation["image_size"][0]*.0012)
        # Check the entire interval, so motion away and back is not mistaken for a stable board.
        oldest = old[-1][0]
        return all(movement(obs, observation) < threshold for t, obs in history if t >= oldest)

    def capture(self, automatic=False):
        if self.busy or self.connecting or self.rig is None or self.verify_var.get():
            return
        with self.state_lock:
            latest = self.latest
            if latest is None:
                return
            batch, observations = latest["batch"], latest["observations"]
            now = time.monotonic()
            selected = {key: obs for key, obs in observations.items()
                        if len(obs["ids"]) >= MIN_CORNERS and now-batch[key]["received_at"] < .7
                        and self.stable(key, obs, batch[key]["received_at"])}
        if self.all_var.get() and len(selected) != len(VIEW_ORDER):
            if not automatic:
                missing = [self.view_names[k] for k in VIEW_ORDER if k not in selected]
                self.log("Not saved: hold the board still with 8+ corners visible in "+", ".join(missing)+".")
            return
        if not selected:
            if not automatic:
                self.log("Not saved: no stationary board with at least 8 corners.")
            return
        stamps = [batch[key]["received_at"] for key in selected]
        if max(stamps)-min(stamps) > .08:
            if not automatic:
                self.log("Not saved: camera frames are more than 80 ms apart. Wait for fresh frames.")
            return
        duplicate = self.session.duplicate(selected)
        if duplicate:
            if not automatic:
                self.log(f"Not saved: this resembles {duplicate}. Move, tilt, or change the board distance.")
            return
        self.busy = True
        self.status.set("Saving native-resolution pose images…")
        session = self.session
        metadata = self.rig.metadata()

        def work():
            try:
                result = session.save(batch, selected, metadata)
                self.events.put(("captured", result))
            except Exception as exc:
                self.events.put(("operation_error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def refresh_session(self):
        self.counts = self.session.counts()
        count = len(self.session.poses())
        self.pose_label.configure(text=f"{count} pose" if count == 1 else f"{count} poses")
        minimum = min((self.counts.get(key, 0) for key in VIEW_ORDER), default=0)
        self.progress["value"] = minimum
        self.session_label.configure(text="Session: "+self.session.path.name)
        self.calibrate_button.configure(text=f"Calibrate all  ({minimum}/{MIN_INTRINSIC_POSES} min)")

    def calibrate(self):
        if self.busy:
            return
        counts = self.session.counts()
        missing = [f"{self.view_names[k]}: {counts.get(k, 0)}/{MIN_INTRINSIC_POSES}"
                   for k in VIEW_ORDER if counts.get(k, 0) < MIN_INTRINSIC_POSES]
        if missing:
            self.log("Need more poses — "+"; ".join(missing)+". Aim for 20–30 diverse poses.")
            return
        self.busy = True
        self.auto_var.set(False)
        session = self.session
        self.status.set("Calibrating camera intrinsics and rig geometry…")
        self.log("Calibration started. Live acquisition continues; this may take a minute.")

        def work():
            try:
                result = calibrate_session(session, self.board, lambda msg: self.events.put(("progress", msg)))
                directory = export_result(session, result)
                self.events.put(("calibrated", (result, directory)))
            except Exception as exc:
                self.events.put(("operation_error", str(exc)))
                (session.path/"calibration_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        threading.Thread(target=work, daemon=True).start()

    def show_result(self, result, directory):
        window = tk.Toplevel(self.root)
        window.title("Calibration results")
        window.geometry("790x570")
        window.configure(bg=BACKGROUND)
        ttk.Label(window, text="Calibration saved" if result["status"] == "calibrated" else "Calibration saved · review needed",
                  font=("Segoe UI Semibold", 17), padding=16).pack(anchor="w")
        text = tk.Text(window, bg=PANEL, fg=TEXT, relief="flat", font=("Consolas", 11), wrap="word", padx=16, pady=12)
        text.pack(fill="both", expand=True, padx=16)
        text.insert("end", "Reference: ZED left optical frame. Translations in metres.\n\n")
        for key, camera in result["cameras"].items():
            label = view_label(key, camera.get("device", {}))
            text.insert("end", f"{label}\n  RMS {camera['rms_px']:.3f} px · {len(camera['used_pose_ids'])} poses\n")
        text.insert("end", "\n"+"\n".join(result["warnings"] or ["No automated quality flags."]))
        text.insert("end", "\n\nSaved JSON, OpenCV YAML, per-pose board transforms, and quality report.\n"+str(directory))
        text.configure(state="disabled")
        ttk.Button(window, text="Open results folder", command=lambda: os.startfile(directory)).pack(pady=12)

    def open_folder(self):
        os.startfile(self.session.path)

    def set_calibration(self, calibration, path=None):
        attach_coverage(calibration, path)
        self.calibration = calibration
        self.coverage_masks = {key: mask for key, camera in calibration["cameras"].items()
                               if (mask := coverage_mask(camera)) is not None}
        self.loaded_calibration_path = Path(path) if path else None
        self.clear_verification()
        label = self.loaded_calibration_path.parent.name if path else "Current calibration"
        if calibration.get("status") == "review_required":
            label += " · review needed"
        self.calibration_label.configure(text=label)
        self.verification_frames = None
        self.verification_waiting = True
        self.last_drawn = None

    def toggle_coverage(self):
        if self.calibration is None:
            self.log("Load a calibration file first to display its sampled area.")
            return
        if not self.coverage_visible and not self.coverage_masks:
            self.log("Coverage is unavailable: keep the original pose folders alongside this older calibration, or load an export containing coverage.")
            return
        self.coverage_visible = not self.coverage_visible
        self.coverage_button.configure(text="Hide calibration ROI" if self.coverage_visible else "Show calibration ROI",
                                       style="Accent.TButton" if self.coverage_visible else "TButton")
        self.last_drawn = None
        self.log("Calibration ROI shown: transparent green marks the sampled board-corner area." if self.coverage_visible
                 else "Calibration ROI hidden.")

    def load_calibration_file(self, path=None):
        if self.busy:
            return
        if path is None:
            latest_files = sorted((self.workspace/"sessions").glob("*/calibration_*/calibration.json"))
            initial = latest_files[-1].parent if latest_files else self.workspace/"sessions"
            path = filedialog.askopenfilename(title="Load rig calibration", initialdir=initial,
                                             filetypes=[("Calibration", "*.json *.yml *.yaml")])
        if not path:
            return
        try:
            calibration = load_calibration(path, self.config, self.board.getChessboardCorners())
            self.set_calibration(calibration, path)
            self.verify_var.set(True)
            self.toggle_verification()
            self.log("Loaded calibration: "+str(path))
        except Exception as exc:
            messagebox.showerror("Cannot load calibration", str(exc))

    def toggle_verification(self):
        self.auto_var.set(False)
        self.clear_verification()
        self.verification_frames = None
        if self.verify_var.get():
            if self.calibration is None:
                self.verify_var.set(False)
                self.log("Load a calibration JSON or YAML file first.")
                return
            self.refresh_verification()
        else:
            self.last_drawn = None

    def clear_verification(self):
        self.verification_token += 1
        self.verification_result = None
        self.verification_working = False
        self.stereo_depth = None
        self.verification_message = "Frozen views · click a stationary object. Depth is required; occlusion is not checked."
        if hasattr(self, "depth_mode") and self.depth_mode.get() == "Board plane":
            self.verification_message = "Frozen views · click inside the printed board grid."
        for canvas in self.canvases.values():
            canvas.delete("verification")
        self.last_drawn = None

    def refresh_verification(self):
        if not self.verify_var.get() or self.calibration is None:
            self.log("Load calibration and enable Verify clicks first.")
            return
        with self.state_lock:
            latest = self.latest
        if latest is None:
            self.verification_waiting = True
            self.verification_message = "Waiting for camera frames…"
            return
        self.verification_waiting = False
        try:
            metadata = self.rig.metadata() if self.rig else {}
            check_frames(self.calibration, latest["batch"], metadata)
            if max(time.monotonic()-p["received_at"] for p in latest["batch"].values()) > .7:
                raise VerificationError("Frames are stale. Reconnect cameras and refresh.")
            self.clear_verification()
            self.verification_frames = latest
            self.draw(latest, metadata)
        except VerificationError as exc:
            self.verification_message = str(exc)
            self.log(str(exc))

    def verify_pixel(self, key, pixel):
        if self.verification_working:
            return
        pixel = tuple(float(value) for value in pixel)
        snapshot = self.verification_frames
        if snapshot is None or self.calibration is None:
            self.log("Refresh the views before selecting a point.")
            return
        self.verification_token += 1
        token = self.verification_token
        self.verification_result = None
        for canvas in self.canvases.values():
            canvas.delete("verification")
        self.verification_working = True
        self.verification_message = "Computing depth and projected pixels…"
        calibration, mode, cached_stereo = self.calibration, self.depth_mode.get(), self.stereo_depth

        def work():
            try:
                stereo = cached_stereo
                if mode == "Board plane":
                    point, description = board_point(calibration, self.board, key, pixel, snapshot["observations"][key])
                elif key in ("d435", "d455"):
                    point, description = realsense_point(calibration, key, pixel, snapshot["batch"][key])
                else:
                    if stereo is None:
                        stereo = StereoDepth(calibration, snapshot["batch"])
                    point, description = stereo.point(key, pixel)
                projections = project_reference_point(calibration, point)
                self.events.put(("verified", {"token": token, "source": key, "clicked_pixel": pixel,
                                              "projections": projections, "reference_point_m": point.tolist(),
                                              "description": description, "stereo": stereo}))
            except Exception as exc:
                self.events.put(("verify_error", (token, str(exc))))
        threading.Thread(target=work, daemon=True).start()

    def draw_verification(self):
        for key, canvas in self.canvases.items():
            canvas.delete("verification")
            if not self.verification_result or key not in self.image_geometry:
                continue
            projection = self.verification_result["projections"].get(key, {})
            pixel = projection.get("pixel")
            if pixel is None:
                canvas.create_text(12, 18, text=projection.get("reason", "No projection"), anchor="w",
                                   fill=AMBER, font=("Segoe UI", 10), tags="verification")
                continue
            ox, oy, scale, _, _ = self.image_geometry[key]
            x, y = ox+pixel[0]*scale, oy+pixel[1]*scale
            # Screen-sized dot stays small and visible at every preview zoom level.
            canvas.create_oval(x-4, y-4, x+4, y+4, fill="#ff3030", outline="white", width=1, tags="verification")
            canvas.create_text(12, 18, text=f"({pixel[0]:.1f}, {pixel[1]:.1f}) px", anchor="w",
                               fill="#ff7777", font=("Segoe UI Semibold", 10), tags="verification")

    def new_session(self):
        if self.busy:
            return
        self.auto_var.set(False)
        self.session = self.new_session_object()
        self.calibration = None
        self.verify_var.set(False)
        self.verification_frames = None
        self.clear_verification()
        self.calibration_label.configure(text="No calibration loaded")
        self.refresh_session()
        self.log("Started a new session. Previous captures remain saved.")

    def resume_session(self, path=None):
        if self.busy:
            return
        self.auto_var.set(False)
        if path is None:
            path = filedialog.askdirectory(title="Select a calibration session", initialdir=self.workspace/"sessions")
        if not path:
            return
        try:
            if not (Path(path)/"session.json").exists():
                raise ValueError("Select a folder containing session.json.")
            self.session = Session(path, self.config)
            self.calibration = None
            self.verify_var.set(False)
            self.verification_frames = None
            self.clear_verification()
            self.calibration_label.configure(text="No calibration loaded")
            latest = self.session.path/"latest_calibration.json"
            if latest.exists():
                result = self.session.path/json.loads(latest.read_text())["path"]/"calibration.json"
                candidate = json.loads(result.read_text())
                if candidate.get("source_pose_ids") == [p["id"] for p in self.session.poses()]:
                    self.set_calibration(candidate, result)
            self.refresh_session()
            self.log("Resumed session "+self.session.path.name)
        except Exception as exc:
            messagebox.showerror("Cannot resume session", str(exc))

    def undo(self):
        if self.busy:
            return
        self.auto_var.set(False)
        pose = self.session.exclude_last()
        if pose:
            self.calibration = None
            self.refresh_session()
            self.log(f"Excluded {pose} from mchoi.calibration. Its original images remain on disk.")

    def start_roi(self, key, event):
        self.drag = (key, event.x, event.y)

    def move_roi(self, key, event):
        if self.verify_var.get():
            return
        if self.drag and self.drag[0] == key:
            canvas = self.canvases[key]
            canvas.delete("roi_drag")
            canvas.create_rectangle(self.drag[1], self.drag[2], event.x, event.y,
                                    outline=AMBER, width=2, tags="roi_drag")

    def end_roi(self, key, event):
        if not self.drag or self.drag[0] != key or key not in self.image_geometry:
            self.drag = None
            return
        _, x, y = self.drag
        self.drag = None
        self.canvases[key].delete("roi_drag")
        if self.verify_var.get():
            if abs(event.x-x) < 8 and abs(event.y-y) < 8:
                ox, oy, scale, width, height = self.image_geometry[key]
                pixel = ((event.x-ox)/scale, (event.y-oy)/scale)
                if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                    self.verify_pixel(key, pixel)
            return
        if abs(event.x-x) < 15 or abs(event.y-y) < 15:
            return
        ox, oy, scale, width, height = self.image_geometry[key]
        a, b = sorted((int((x-ox)/scale), int((event.x-ox)/scale)))
        c, d = sorted((int((y-oy)/scale), int((event.y-oy)/scale)))
        roi = (max(0, a), max(0, c), min(width, b), min(height, d))
        if roi[2]-roi[0] < 30 or roi[3]-roi[1] < 30:
            return
        with self.state_lock:
            self.rois[key] = roi
            self.history[key].clear()
        self.log(f"{self.view_names[key]} detection region set. Right-click the view to clear it.")

    def clear_roi(self, key):
        if self.verify_var.get():
            self.clear_verification()
            return
        with self.state_lock:
            self.rois.pop(key, None)
            self.history[key].clear()
        self.log(f"{self.view_names[key]}: searching the full image.")

    def draw(self, latest, metadata):
        for key in VIEW_ORDER:
            if key not in latest["batch"]:
                continue
            packet = latest["batch"][key]
            obs = latest["observations"][key]
            camera = None
            if self.calibration and key in self.calibration["cameras"]:
                candidate = self.calibration["cameras"][key]
                if candidate["image_size"] == obs["image_size"]:
                    camera = {**candidate, "board_points": self.calibration["board_object_points_m"]}
            annotated = annotate(packet["image"], obs, camera)
            roi = self.rois.get(key)
            if roi:
                cv2.rectangle(annotated, roi[:2], roi[2:], (80, 195, 240), 2)
            height, width = annotated.shape[:2]
            canvas = self.canvases[key]
            cw, ch = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 80)
            scale = min(cw/width, ch/height)
            dw, dh = max(1, int(width*scale)), max(1, int(height*scale))
            thumbnail = cv2.resize(annotated, (dw, dh), interpolation=cv2.INTER_AREA)
            mask = self.coverage_masks.get(key) if self.calibration else None
            coverage_text = None
            if self.coverage_visible:
                if mask is not None and mask.shape == (height, width):
                    thumbnail = overlay_coverage(thumbnail, mask)
                    fraction = np.count_nonzero(mask)/mask.size
                    coverage = self.calibration["cameras"][key]["calibration_coverage"]
                    count = coverage["pose_count"]
                    expected = coverage.get("expected_pose_count", count)
                    partial = f" · {count}/{expected} poses available" if count < expected else ""
                    coverage_text = f"Sampled calibration area: {fraction:.0%}{partial}"
                else:
                    coverage_text = "Calibration coverage unavailable for this view"
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(thumbnail))
            self.images[key] = photo
            ox, oy = (cw-dw)//2, (ch-dh)//2
            canvas.delete("frame")
            canvas.create_image(ox, oy, image=photo, anchor="nw", tags="frame")
            canvas.tag_lower("frame")
            canvas.delete("coverage_label")
            if coverage_text:
                label = canvas.create_text(ox+8, oy+dh-12, text=coverage_text, anchor="sw",
                                           fill="#9befb8", font=("Segoe UI Semibold", 9), tags="coverage_label")
                box = canvas.bbox(label)
                backdrop = canvas.create_rectangle(box[0]-4, box[1]-2, box[2]+4, box[3]+2,
                                                    fill="#14251c", outline="", tags="coverage_label")
                canvas.tag_lower(backdrop, label)
            self.image_geometry[key] = (ox, oy, scale, width, height)
            info = metadata.get(key, {})
            fps = info.get("measured_fps", 0)
            ready = self.ready.get(key, False)
            text = f"{len(obs['marker_ids'])}/17 markers · {len(obs['ids'])}/24 corners"
            if self.verify_var.get() and self.depth_mode.get() == "Automatic depth":
                text = "Depth verification"
            self.camera_labels[key].configure(text=text, fg=GREEN if ready else AMBER)
            usb = f" · USB {info['usb']}" if info.get("usb") else ""
            state = "Frozen for verification" if self.verify_var.get() else "Ready" if ready else "Hold still" if len(obs["ids"]) >= MIN_CORNERS else "Show more of the board"
            count = self.counts.get(key, 0)
            count_text = f"{count} pose" if count == 1 else f"{count} poses"
            self.camera_details[key].configure(
                text=(f"{width}×{height} · frozen snapshot  |  {self.depth_mode.get()}" if self.verify_var.get()
                      else f"{width}×{height} · {fps:.1f} fps{usb}  |  {count_text}  |  {state}"),
                fg=AMBER if not self.verify_var.get() and fps and fps < 27 else MUTED)
        self.draw_verification()

    def tick(self):
        if self.closing:
            return
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "connected":
                self.rig = value
                self.connecting = False
                self.connect_button.configure(text="Disconnect", state="normal")
                self.stop_detection = threading.Event()
                self.detection_thread = threading.Thread(target=self.detection_loop,
                                                        args=(self.rig, self.stop_detection), daemon=True)
                self.detection_thread.start()
                for key, info in self.rig.metadata().items():
                    self.view_names[key] = view_label(key, info)
                    self.camera_headings[key].configure(text=self.view_names[key])
                    self.log(f"{self.view_names[key]}: {info['image_size'][0]}×{info['image_size'][1]} at 30 fps requested"+
                             (f" · USB {info['usb']}" if info.get("usb") else ""))
            elif event in ("disconnected", "connect_error"):
                self.rig = None
                self.connecting = False
                self.latest = None
                self.history = {key: deque(maxlen=10) for key in VIEW_ORDER}
                self.ready = {}
                self.verification_frames = None
                self.clear_verification()
                self.connect_button.configure(text="Connect cameras", state="normal")
                self.status.set("Disconnected" if event == "disconnected" else "Connection failed")
                if value:
                    self.log(value)
            elif event in ("log", "progress"):
                self.log(value)
                if event == "progress":
                    self.status.set(value)
            elif event == "captured":
                self.busy = False
                self.calibration = None
                self.calibration_label.configure(text="No calibration loaded")
                self.refresh_session()
                self.log(f"Saved {value['id']} · {len(value['views'])} views · {value['host_receipt_spread_ms']:.0f} ms host receipt spread")
            elif event == "operation_error":
                self.busy = False
                self.log(value)
                self.status.set(value)
            elif event == "calibrated":
                self.busy = False
                calibration, directory = value
                self.set_calibration(calibration, directory/"calibration.json")
                self.log("Calibration exported to "+str(directory))
                self.show_result(self.calibration, directory)
            elif event == "verified":
                if value["token"] == self.verification_token:
                    self.verification_working = False
                    self.verification_result = value
                    self.stereo_depth = value["stereo"]
                    self.verification_message = value["description"]+" · red dots are predicted pixels; occlusion is not checked."
                    self.draw_verification()
                    self.log(f"Clicked {self.view_names[value['source']]} ({value['clicked_pixel'][0]:.1f}, {value['clicked_pixel'][1]:.1f}) · "+value["description"])
            elif event == "verify_error":
                token, message = value
                if token == self.verification_token:
                    self.verification_working = False
                    self.verification_message = message
                    self.log(message)
        metadata = self.rig.metadata() if self.rig else {}
        now = time.monotonic()
        with self.state_lock:
            latest = self.latest
            if latest:
                self.ready = {key: len(obs["ids"]) >= MIN_CORNERS
                              and now-latest["batch"][key]["received_at"] < .7
                              and self.stable(key, obs, latest["batch"][key]["received_at"])
                              for key, obs in latest["observations"].items()}
        if self.verify_var.get() and self.calibration and self.verification_waiting and latest:
            self.refresh_verification()
        displayed = self.verification_frames if self.verify_var.get() else latest
        if displayed and displayed["generation"] != self.last_drawn:
            self.draw(displayed, metadata)
            self.last_drawn = displayed["generation"]
        if self.verify_var.get() and not self.busy:
            self.status.set(self.verification_message)
        elif self.rig and not self.busy:
            errors = [f"{self.view_names[key]}: {info['error']}" for key, info in metadata.items() if info.get("error")]
            if errors:
                self.status.set(" | ".join(errors))
            elif latest:
                stamps = [p["received_at"] for p in latest["batch"].values()]
                spread = (max(stamps)-min(stamps))*1000
                n = sum(self.ready.values())
                self.status.set(f"{n}/4 views ready · {spread:.0f} ms host receipt spread · "+
                                ("Capture a pose, then move and tilt the board." if n == 4 else "Show the board and hold still; 8+ corners per selected view."))
        self.capture_button.configure(state="normal" if self.rig and not self.busy and not self.connecting and not self.verify_var.get() else "disabled")
        self.load_calibration_button.configure(state="disabled" if self.busy else "normal")
        self.coverage_button.configure(state="normal" if self.calibration is not None else "disabled")
        self.calibrate_button.configure(state="disabled" if self.busy else "normal")
        for button in self.session_buttons:
            button.configure(state="disabled" if self.busy else "normal")
        if self.auto_var.get() and not self.busy and not self.verify_var.get() and now-self.last_auto > 3:
            self.last_auto = now
            self.capture(automatic=True)
        self.root.after(80, self.tick)

    def close(self):
        if self.busy:
            self.log("Wait for the current save or calibration to finish before closing.")
            return
        self.closing = True
        self.stop_detection.set()
        self.status.set("Releasing cameras…")
        done = threading.Event()

        def work():
            if self.rig:
                self.rig.stop()
            # A connection may have completed while the window was closing.
            if self.connecting:
                deadline = time.monotonic()+15
                while time.monotonic() < deadline:
                    try:
                        event, value = self.events.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if event == "connected":
                        value.stop()
                        break
                    if event in ("connect_error", "disconnected"):
                        break
            done.set()
        threading.Thread(target=work, daemon=True).start()

        def finish():
            if done.is_set():
                self.root.destroy()
            else:
                self.root.after(100, finish)
        finish()
