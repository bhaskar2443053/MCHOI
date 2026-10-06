"""A single Record/Stop control and a live RGB preview from RealSense 076358."""
import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from copy import deepcopy

from PIL import Image, ImageTk

from .controller import Recorder

BG = "#101820"
PANEL = "#1b2834"
TEXT = "#eaf1f6"
MUTED = "#9fb2c3"
GREEN = "#52d9a4"
RED = "#ee626b"


class RecordingApp:
    def __init__(self, root, workspace, output=None, offline=False):
        self.root = root
        self.workspace = Path(workspace)
        self.output = Path(output) if output else self.workspace/"recordings"
        self.recorder = None
        self.events = queue.Queue()
        self.busy = False
        self.closing = False
        self.photo = None
        self.last_preview = -1
        self.last_size = None
        self.last_health = 0
        self.last_state = None
        self.settings_window = None
        self.settings_entries = {}
        self.status = tk.StringVar(value="Connecting cameras…" if not offline else "Offline preview")
        root.title("Multi-camera recorder")
        width = min(1000, root.winfo_screenwidth()-80)
        height = min(760, root.winfo_screenheight()-100)
        root.geometry(f"{width}x{height}")
        root.minsize(760, 600)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        if not offline:
            self.launch(self.connect)
        root.after(100, self.tick)

    def _build(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=24, pady=(20, 14))
        tk.Label(top, text="Multi-camera recorder", bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 22)).pack(anchor="w")
        tk.Label(top, text="ZED stereo · 2 RealSense cameras   |   RGB 1920 × 1080 at 30 fps",
                 bg=BG, fg=MUTED, font=("Segoe UI", 11)).pack(anchor="w", pady=(5, 0))
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=24, pady=(0, 12))
        self.record_button = tk.Button(controls, text="Record", command=self.toggle, bg="#237b66",
                                       activebackground="#2d997f", fg="white", activeforeground="white",
                                       disabledforeground="#aac2ba", font=("Segoe UI Semibold", 15),
                                       relief="flat", padx=35, pady=10, state="disabled", cursor="hand2")
        self.record_button.pack(side="left")
        self.timer = tk.Label(controls, text="00:00:00", bg=BG, fg=TEXT, font=("Consolas", 25))
        self.timer.pack(side="left", padx=24)
        self.settings_button = tk.Button(controls, text="Exposure settings", command=self.show_settings,
                                        bg=PANEL, fg=TEXT, relief="flat", padx=14, pady=10,
                                        font=("Segoe UI", 10), state="disabled")
        self.settings_button.pack(side="left", padx=8)
        self.state_label = tk.Label(controls, text="CONNECTING", bg=BG, fg=MUTED, font=("Segoe UI Semibold", 11))
        self.state_label.pack(side="right")
        frame = tk.Frame(self.root, bg=PANEL, padx=10, pady=10)
        frame.pack(fill="both", expand=True, padx=24)
        tk.Label(frame, text="RealSense camera 2 · 076358", bg=PANEL, fg=TEXT,
                 font=("Segoe UI Semibold", 12)).pack(anchor="w", pady=(0, 8))
        self.canvas = tk.Canvas(frame, bg="#080e13", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(400, 190, text="Waiting for camera…", fill=MUTED, font=("Segoe UI", 14))
        self.camera_status = tk.Label(self.root, text="", bg=BG, fg=MUTED, anchor="w", font=("Segoe UI", 10))
        self.camera_status.pack(fill="x", padx=24, pady=(10, 3))
        self.timing_label = tk.Label(self.root, text="Exposure and gain: manual · Focus: fixed lenses",
                                     bg=BG, fg=MUTED, anchor="w", font=("Segoe UI", 9))
        self.timing_label.pack(fill="x", padx=24, pady=(0, 3))
        tk.Label(self.root, text="Depth 848 × 480 / 30 fps · PC timestamp matching within ±20 ms · No hardware synchronization",
                 bg=BG, fg=MUTED, anchor="w", font=("Segoe UI", 9)).pack(fill="x", padx=24)
        tk.Label(self.root, textvariable=self.status, wraplength=1000, justify="left", anchor="w",
                 bg=BG, fg=TEXT, font=("Segoe UI", 10)).pack(fill="x", padx=24, pady=(10, 4))
        self.folder = tk.Label(self.root, text="Save folder: "+str(self.output), bg=BG, fg="#83baff",
                               anchor="w", cursor="hand2", font=("Segoe UI", 9))
        self.folder.pack(fill="x", padx=24, pady=(0, 16))
        self.folder.bind("<Button-1>", lambda _: self.open_folder())

    def launch(self, action):
        if self.busy:
            return
        self.busy = True
        self.record_button.configure(state="disabled")
        def run():
            try:
                action()
                self.events.put(("done", None))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=run, daemon=True, name="recorder-operation").start()

    def connect(self):
        if self.recorder is not None:
            self.recorder.close()
        self.recorder = Recorder(self.workspace, self.output)
        self.recorder.connect()
        self.events.put(("message", "Ready. Record starts all cameras; Stop finalizes the take."))

    def toggle(self):
        if self.busy or self.closing:
            return
        if self.recorder is None or self.recorder.state in ("error", "closed", "disconnected"):
            self.status.set("Connecting cameras…")
            self.launch(self.connect)
        elif self.recorder.state == "ready":
            self.status.set("Preparing all cameras…")
            self.launch(self.start)
        elif self.recorder.state == "recording":
            self.status.set("Saving and finalizing files…")
            self.launch(self.stop)

    def start(self):
        directory = self.recorder.start_recording()
        self.events.put(("message", "Recording all cameras to "+directory.name))

    def show_settings(self):
        if self.busy or not self.recorder or self.recorder.state != "ready":
            return
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        win = self.settings_window = tk.Toplevel(self.root)
        win.title("Exposure settings · fixed focus")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.transient(self.root)
        body = tk.Frame(win, bg=BG, padx=22, pady=20)
        body.pack()
        tk.Label(body, text="Manual exposure and gain", bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 16)).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        tk.Label(body, text="Focus is fixed on these cameras; there is no software focus adjustment.\n"
                           "Shorter exposure reduces motion blur. Higher gain brightens the image but adds noise.",
                 bg=BG, fg=MUTED, justify="left", font=("Segoe UI", 10)).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 16))
        for col, text in enumerate(("Camera / stream", "Exposure", "Units", "Gain")):
            tk.Label(body, text=text, bg=BG, fg=TEXT, font=("Segoe UI Semibold", 10)).grid(row=2, column=col, sticky="w", padx=(0, 20), pady=(0, 8))
        self.settings_entries = {}
        rows = [("ZED · both lenses", "zed", "exposure_percent", "% of frame period", 1),
                ("078090 · RGB", "261222078090", "color_exposure", "ms", .1),
                ("078090 · depth", "261222078090", "depth_exposure_us", "ms", .001),
                ("076358 · RGB", "261222076358", "color_exposure", "ms", .1),
                ("076358 · depth", "261222076358", "depth_exposure_us", "ms", .001)]
        for row, (label, serial, key, unit, scale) in enumerate(rows, 3):
            settings = self.recorder.settings["zed"] if serial == "zed" else self.recorder.settings["realsense"][serial]
            gain_key = "gain" if serial == "zed" else key.split("_")[0]+"_gain"
            tk.Label(body, text=label, bg=BG, fg=TEXT).grid(row=row, column=0, sticky="w", padx=(0, 20), pady=6)
            exposure, gain = tk.StringVar(value=f"{settings[key]*scale:g}"), tk.StringVar(value=str(settings[gain_key]))
            ttk.Entry(body, textvariable=exposure, width=12).grid(row=row, column=1, sticky="w", padx=(0, 20))
            tk.Label(body, text=unit, bg=BG, fg=MUTED).grid(row=row, column=2, sticky="w", padx=(0, 20))
            ttk.Entry(body, textvariable=gain, width=10).grid(row=row, column=3, sticky="w")
            self.settings_entries[(serial, key)] = (exposure, gain, gain_key, scale)
        tk.Label(body, text="RGB exposure may be rounded by the Windows driver; applied values appear below.\n"
                           "Settings persist between takes. Controls are locked while recording.",
                 bg=BG, fg=MUTED, justify="left").grid(row=8, column=0, columnspan=4, sticky="w", pady=(16, 10))
        self.settings_feedback = tk.Label(body, text="", wraplength=630, bg=BG, fg=GREEN, justify="left")
        self.settings_feedback.grid(row=9, column=0, columnspan=4, sticky="w", pady=(0, 10))
        self.settings_apply = tk.Button(body, text="Apply and save", command=self.apply_exposure,
                                       bg="#237b66", fg="white", padx=20, pady=8, relief="flat")
        self.settings_apply.grid(row=10, column=0, columnspan=4, sticky="e")
        self.show_applied_settings()

    def show_applied_settings(self):
        if not self.settings_window or not self.settings_window.winfo_exists():
            return
        summaries = []
        for worker in self.recorder.workers:
            c = worker.snapshot()["metadata"].get("controls", {})
            if worker.key == "zed" and c:
                summaries.append(f"ZED: exposure {c['exposure_percent']}%, gain {c['gain']}")
            elif c:
                summaries.append(f"{worker.serial[-6:]}: RGB {c['color']['exposure']*.1:g} ms / gain {c['color']['gain']:g}; "
                                 f"depth {c['depth']['exposure']*.001:g} ms / gain {c['depth']['gain']:g}")
        self.settings_feedback.configure(text="Applied · auto exposure OFF\n"+"\n".join(summaries), fg=GREEN)

    def apply_exposure(self):
        if self.busy or self.recorder.state != "ready":
            return
        settings = deepcopy(self.recorder.settings)
        try:
            for (serial, key), (exposure, gain, gain_key, scale) in self.settings_entries.items():
                options = settings["zed"] if serial == "zed" else settings["realsense"][serial]
                raw = float(exposure.get())/scale
                if abs(raw-round(raw)) > .00001:
                    raise ValueError("Exposure is finer than the camera's supported units")
                options[key] = round(raw)
                options[gain_key] = float(gain.get())
            from .settings import validate_settings
            settings = validate_settings(settings)
        except (ValueError, OverflowError) as exc:
            self.settings_feedback.configure(text=str(exc), fg=RED)
            return
        self.settings_feedback.configure(text="Applying camera controls…", fg=MUTED)
        def apply():
            self.recorder.apply_settings(settings)
            self.events.put(("settings_applied", None))
            self.events.put(("message", "Manual exposure/gain saved. Auto exposure is off on all cameras."))
        self.launch(apply)

    def stop(self, reason=None):
        directory = self.recorder.stop_recording(reason)
        if self.recorder.error:
            self.events.put(("error", "Take interrupted; files retained. "+self.recorder.error))
        else:
            warnings = self.recorder.manifest.get("timing_warnings", [])
            self.events.put(("message", "Saved "+directory.name+(" · Timing gaps detected; see recording.json." if warnings
                                     else " · ZED SVO2 + two RealSense RGB/depth DB3 files.")))

    def open_folder(self):
        directory = self.recorder.directory if self.recorder and self.recorder.directory else self.output
        directory.mkdir(parents=True, exist_ok=True)
        os.startfile(directory)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.status.set("Closing safely. Finalizing any active recording…")
        self.record_button.configure(state="disabled")
        self.wait_to_close()

    def wait_to_close(self):
        if self.busy:
            self.root.after(100, self.wait_to_close)
            return
        def close():
            if self.recorder:
                self.recorder.close()
            self.events.put(("closed", None))
        self.launch(close)

    def tick(self):
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "closed":
                self.root.destroy()
                return
            if event in ("done", "error"):
                self.busy = False
            if event in ("message", "error"):
                self.status.set(value)
                if event == "error" and self.settings_window and self.settings_window.winfo_exists():
                    self.settings_feedback.configure(text=value, fg=RED)
            if event == "settings_applied":
                self.show_applied_settings()
            if event == "error" and self.closing:
                self.closing = False
                self.status.set("Could not finish closing: "+value+". Close again to retry.")
        state = self.recorder.state if self.recorder else "disconnected"
        if state == "indexing":
            self.status.set("Files saved. Building per-frame PC timestamps and alignment…")
        controls_enabled = not self.busy and not self.closing and state == "ready"
        self.settings_button.configure(state="normal" if controls_enabled else "disabled")
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_apply.configure(state="normal" if controls_enabled else "disabled")
        if not self.closing:
            self.state_label.configure(text=state.upper(), fg=RED if state == "recording" else GREEN if state == "ready" else MUTED)
            self.record_button.configure(text="Stop recording" if state == "recording" else "Retry cameras" if state == "error" else "Record",
                                         bg="#a93242" if state == "recording" else "#237b66",
                                         state="normal" if not self.busy and state in ("ready", "recording", "error") else "disabled")
        if self.recorder:
            elapsed = int(time.monotonic()-self.recorder.started_monotonic) if self.recorder.started_monotonic else 0
            self.timer.configure(text=f"{elapsed//3600:02}:{elapsed//60%60:02}:{elapsed%60:02}")
            summaries = []
            for worker in self.recorder.workers:
                info = worker.snapshot()
                label = "ZED stereo" if worker.key == "zed" else info["metadata"].get("serial", worker.key)[-6:]
                summaries.append(label+(" · error" if info["error"] else f" · {info['fps']:.1f} fps"))
            self.camera_status.configure(text="     |     ".join(summaries))
            zed = next((w.snapshot() for w in self.recorder.workers if w.key == "zed"), {})
            stereo = zed.get("stats", {}).get("stereo", {})
            gaps = stereo.get("timestamp_gap_events", 0)
            timing_text = f"ZED timing: {gaps} gaps"+(f" · about {stereo.get('estimated_missing_frames', 0)} missing frame intervals" if gaps else "")
            self.timing_label.configure(text="Exposure/gain: manual · Focus: fixed lenses   |   "+timing_text,
                                        fg="#f0b86e" if gaps else MUTED)
            if self.recorder.directory:
                self.folder.configure(text="Save folder: "+str(self.recorder.directory))
            sequence, pixels = self.recorder.preview()
            size = (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
            if pixels is not None and (sequence != self.last_preview or size != self.last_size):
                im = Image.fromarray(pixels)
                im.thumbnail(size, Image.Resampling.BILINEAR)
                self.photo = ImageTk.PhotoImage(im)
                self.canvas.delete("all")
                self.canvas.create_image(size[0]//2, size[1]//2, image=self.photo)
                self.last_preview, self.last_size = sequence, size
            if not self.busy and not self.closing and time.monotonic()-self.last_health > 1:
                self.last_health = time.monotonic()
                error = self.recorder.health_error()
                if error and state == "recording":
                    self.status.set("Stopping all recordings: "+error)
                    self.launch(lambda message=error: self.stop(message))
                elif error and state == "ready":
                    self.recorder.state = "error"
                    self.status.set(error+". Close other camera apps and retry.")
        self.root.after(100, self.tick)
