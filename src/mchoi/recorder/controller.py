"""Coordinate native recorders and persist honest take status and timing."""
from concurrent.futures import wait
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time

from .devices import RealSenseWorker, ZedWorker, load_slots, PREVIEW_SERIAL
from .settings import load_settings, save_settings, validate_settings
from .timestamps import export_timestamps
from mchoi.paths import CONFIG_DIR


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class Recorder:
    def __init__(self, workspace, output=None, workers=None, timestamp_exporter=export_timestamps):
        self.workspace = Path(workspace)
        self.output = Path(output) if output else self.workspace/"recordings"
        self.settings = load_settings(CONFIG_DIR)
        self.timestamp_exporter = timestamp_exporter
        self.workers = workers if workers is not None else [
            *[RealSenseWorker(slot, self.settings["realsense"][slot["serial"]]) for slot in load_slots(CONFIG_DIR)],
            ZedWorker(self.settings["zed"])]
        self.operation = threading.Lock()
        self.state = "disconnected"
        self.error = None
        self.directory = None
        self.manifest = None
        self.started_monotonic = None

    def apply_settings(self, settings):
        settings = validate_settings(settings)
        with self.operation:
            if self.state != "ready":
                raise RuntimeError("Stop recording before changing exposure")
            previous = self.settings
            try:
                # Validate the whole request before modifying any camera. Drivers
                # may quantize values, so report the actual readback to the UI.
                result = {}
                for worker in self.workers:
                    options = settings["zed"] if worker.key == "zed" else settings["realsense"][worker.serial]
                    result[worker.key] = worker.command("apply_settings", options).result(10)
                save_settings(CONFIG_DIR, settings)
                self.settings = settings
                return result
            except Exception:
                for worker in self.workers:
                    if not worker.error and worker.is_alive():
                        options = previous["zed"] if worker.key == "zed" else previous["realsense"][worker.serial]
                        try:
                            worker.command("apply_settings", options).result(10)
                        except Exception:
                            pass
                raise

    def connect(self):
        with self.operation:
            self.state = "connecting"
            try:
                for worker in self.workers:
                    worker.start()
                for worker in self.workers:
                    worker.ready.result(timeout=40)
                self.state = "ready"
            except Exception as exc:
                self.error = str(exc)
                self.state = "error"
                self._shutdown()
                raise

    def _all(self, action, *args):
        requests = [(worker, worker.command(action, *args)) for worker in self.workers]
        # Every camera gets the request before waiting; independent SDKs can flush together.
        _, pending = wait([f for _, f in requests], timeout=60)
        errors = []
        for worker, future in requests:
            if future in pending:
                errors.append(worker.key+": timed out during "+action)
            else:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(worker.key+": "+str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    def start_recording(self):
        with self.operation:
            if self.state != "ready":
                raise RuntimeError("Cameras are not ready")
            self.error = None
            self.output.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.output).free < 2*1024**3:
                raise RuntimeError("Less than 2 GB free in the recording folder")
            self.directory = self.output/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.directory.mkdir()
            self.state = "starting"
            self.manifest = {
                "schema_version": 1, "status": "starting", "created_utc": utc_now(),
                "format": {"zed": "SVO2 H264 (both stereo views)",
                           "realsense": "native RealSense DB3; uncompressed BGR8 + Z16"},
                "synchronization": "Software coordinated start/stop; cameras are NOT hardware synchronized.",
                "timestamps": "Native files preserve device timestamps. CSVs map observed frames to host UTC and monotonic clocks; observers may see fewer frames than native files.",
                "pc_clock": {"unix_ns_at_start": time.time_ns(), "monotonic_ns_at_start": time.monotonic_ns(),
                             "local_time_at_start": datetime.now().astimezone().isoformat(),
                             "timezone": datetime.now().astimezone().tzname()},
                "preview_serial": PREVIEW_SERIAL,
                "exposure_settings": self.settings,
                "depth": "Native, unaligned depth; metres = Z16 value * depth_units_m. Factory depth-to-color calibration is included.",
            }
            try:
                self._save()
                self._all("prepare", self.directory)
                # Wait until restarted RealSense streams settle before releasing all recorders.
                self._all("settle")
                self.manifest["cameras"] = {w.key: w.snapshot()["metadata"] for w in self.workers}
                self.manifest["start_requested_host_unix_ns"] = time.time_ns()
                self._all("begin")
                self.started_monotonic = time.monotonic()
                self.manifest.update(status="recording", started_utc=utc_now())
                self.state = "recording"
                self._save()
            except Exception as exc:
                self._finish(str(exc))
                raise
        return self.directory

    def stop_recording(self, reason=None):
        with self.operation:
            if self.state not in ("recording", "starting"):
                return self.directory
            self._finish(reason)
        return self.directory

    def _finish(self, error=None):
        self.state = "stopping"
        self.manifest["stop_requested_host_unix_ns"] = time.time_ns()
        self.manifest["status"] = "stopping"
        try:
            self._save()
        except OSError as exc:
            error = (error+"; " if error else "")+"Could not update manifest: "+str(exc)
        try:
            self._all("finish")
        except Exception as exc:
            error = (error+"; " if error else "")+str(exc)
        # A failed worker also finalizes its own files in its finally block.
        for worker in self.workers:
            if worker.error:
                worker.join(timeout=30)
                if worker.is_alive():
                    error = (error+"; " if error else "")+worker.key+": still finalizing"
        snapshots = {w.key: w.snapshot() for w in self.workers}
        self.manifest.update(stopped_utc=utc_now(), observations=snapshots)
        files = {p.name: p.stat().st_size for p in self.directory.iterdir()
                 if p.suffix in (".svo2", ".db3", ".csv")}
        self.manifest["files_bytes"] = files
        expected = ["zed.svo2"]+[w.key+".db3" for w in self.workers if w.key.startswith("realsense_")]
        if not error and any(files.get(name, 0) < 1024 for name in expected):
            error = "A required native recording is missing or empty"
        for key, snapshot in snapshots.items():
            streams = ("stereo",) if key == "zed" else ("color", "depth")
            if not error and any(snapshot["stats"].get(s, {}).get("observed_frames", 0) == 0 for s in streams):
                error = key+": no recorded frames were observed"
        self.manifest["timing_warnings"] = [
            f"{key} {stream}: {stats['timestamp_gap_events']} timestamp gaps; approximately "
            f"{stats.get('estimated_missing_frames', 0)} missing frame intervals in the observed stream"
            for key, snapshot in snapshots.items() for stream, stats in snapshot["stats"].items()
            if isinstance(stats, dict) and stats.get("timestamp_gap_events", 0)]
        if not error:
            self.state = "indexing"
            try:
                self.manifest["timestamp_alignment"] = self.timestamp_exporter(self.directory)
            except Exception as exc:
                error = "Native files saved, but frame timestamp alignment failed: "+str(exc)
                self.manifest["timestamp_alignment"] = {"status": "failed", "error": str(exc)}
        self.manifest.update(status="interrupted" if error else "complete", error=error)
        self.started_monotonic = None
        self.error = error
        self.state = "error" if any(w.error or w.is_alive() is False for w in self.workers) else "ready"
        try:
            self._save()
        except OSError as exc:
            self.error = (self.error+"; " if self.error else "")+"Files finalized, but manifest could not be saved: "+str(exc)

    def _save(self):
        write_json(self.directory/"recording.json", self.manifest)

    def health_error(self):
        for worker in self.workers:
            info = worker.snapshot()
            if info["error"]:
                return worker.key+": "+info["error"]
            if self.state == "recording" and info["age_s"] > 6:
                return worker.key+": camera stopped delivering frames"
        if self.state == "recording" and shutil.disk_usage(self.output).free < 1024**3:
            return "Recording stopped because less than 1 GB remains on the output drive"
        return None

    def _shutdown(self):
        for worker in self.workers:
            worker.shutdown.set()
        for worker in self.workers:
            if worker.ident is not None:
                worker.join(timeout=35)
        alive = [w.key for w in self.workers if w.is_alive()]
        if alive:
            raise RuntimeError("Cameras still finalizing: "+", ".join(alive))

    def close(self):
        with self.operation:
            try:
                if self.state in ("recording", "starting", "stopping"):
                    self._finish()
            finally:
                self._shutdown()
                self.state = "closed"

    def preview(self):
        worker = next((w for w in self.workers if getattr(w, "serial", None) == PREVIEW_SERIAL), None)
        return worker.get_preview() if worker else (0, None)
