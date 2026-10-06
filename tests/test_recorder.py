"""Exercise failure cleanup and take boundaries without camera hardware."""
import json
import time
from copy import deepcopy
import io
import csv

import pytest

from mchoi.recorder.controller import Recorder
from mchoi.recorder.devices import CameraWorker
from mchoi.recorder.settings import DEFAULTS, validate_settings
from mchoi.recorder.validate import timing_summary


class FakeCamera(CameraWorker):
    def __init__(self, key, fail_prepare=False):
        super().__init__(key)
        self.fail_prepare = fail_prepare
        self.native_file = None
        self.closed = False
        self.sequence = 0
        self.did_finish = False
        self.group = []

    def open_native(self):
        self.metadata = {"name": self.key}

    def read_native(self):
        time.sleep(.002)
        self.received()
        self.sequence += 1
        if self.recording:
            for stream in (("stereo",) if self.key == "zed" else ("color", "depth")):
                self.observed(stream, self.sequence, self.sequence*33.333, "fake", time.time_ns(), time.monotonic_ns())
        return True

    def prepare_native(self, directory):
        name = "zed.svo2" if self.key == "zed" else self.key+".db3"
        self.native_file = (directory/name).open("wb")
        self.native_file.write(b"x"*2048)
        if self.fail_prepare:
            raise RuntimeError("Simulated camera start failure")

    def settle(self):
        self.read_native()

    def begin_native(self):
        assert all(camera.prepared for camera in self.group)

    def finish_native(self):
        if self.native_file:
            self.native_file.close()
        self.did_finish = True

    def close_native(self):
        self.closed = True


def make_recorder(tmp_path, fail=False):
    cameras = [FakeCamera("zed"), FakeCamera("realsense_a"), FakeCamera("realsense_b", fail)]
    for camera in cameras:
        camera.group = cameras
    recorder = Recorder(tmp_path, tmp_path/"takes", cameras, timestamp_exporter=lambda directory: {"test_export": True})
    recorder.connect()
    return recorder, cameras


def test_repeated_takes_are_distinct_and_close_finalizes_active_take(tmp_path):
    recorder, cameras = make_recorder(tmp_path)
    try:
        first = recorder.start_recording()
        time.sleep(.025)
        recorder.stop_recording()
        assert recorder.state == "ready"
        first_manifest = (first/"recording.json").read_text()
        assert json.loads(first_manifest)["status"] == "complete"
        second = recorder.start_recording()
        time.sleep(.025)
        recorder.close()
        assert first != second
        assert (first/"recording.json").read_text() == first_manifest
        assert json.loads((second/"recording.json").read_text())["status"] == "complete"
        assert all(c.closed and c.csv_file is None and c.native_file.closed for c in cameras)
    finally:
        recorder.close()


def test_partial_prepare_failure_retains_files_and_stops_every_recorder(tmp_path):
    recorder, cameras = make_recorder(tmp_path, fail=True)
    try:
        with pytest.raises(RuntimeError, match="Simulated camera start failure"):
            recorder.start_recording()
        manifest = json.loads((recorder.directory/"recording.json").read_text())
        assert manifest["status"] == "interrupted"
        assert all(c.did_finish and c.csv_file is None and c.native_file.closed for c in cameras)
        assert not any(c.recording for c in cameras)
        assert list(recorder.directory.glob("*.db3"))
    finally:
        recorder.close()


def test_manifest_write_failure_still_flushes_native_files(tmp_path, monkeypatch):
    recorder, cameras = make_recorder(tmp_path)
    try:
        recorder.start_recording()
        time.sleep(.025)
        def disk_error():
            raise OSError("Simulated disk-full manifest write")
        monkeypatch.setattr(recorder, "_save", disk_error)
        recorder.stop_recording()
        assert all(c.did_finish and c.native_file.closed and c.csv_file is None for c in cameras)
        assert "manifest" in recorder.error
        assert recorder.manifest["status"] == "interrupted"
    finally:
        recorder.close()


def test_disconnect_marks_take_interrupted_and_flushes_other_cameras(tmp_path):
    recorder, cameras = make_recorder(tmp_path)
    try:
        recorder.start_recording()
        time.sleep(.025)
        # A failed acquisition loop runs CameraWorker's finalizer before it exits.
        cameras[0].error = "Simulated USB disconnect"
        cameras[0].shutdown.set()
        cameras[0].join(timeout=2)
        reason = recorder.health_error()
        assert "USB disconnect" in reason
        recorder.stop_recording(reason)
        assert recorder.state == "error"
        assert json.loads((recorder.directory/"recording.json").read_text())["status"] == "interrupted"
        assert all(c.native_file.closed for c in cameras)
    finally:
        recorder.close()


def test_timing_gap_is_detected_even_when_frame_numbers_are_sequential():
    camera = FakeCamera("zed")
    camera.recording = True
    camera.csv_file = io.StringIO()
    camera.csv_writer = csv.writer(camera.csv_file)
    for i, timestamp in enumerate((0.0, 33.4, 66.8, 133.6, 167.0)):
        camera.observed("stereo", i, timestamp, "test", 0, 0)
    stats = camera.snapshot()["stats"]["stereo"]
    assert stats["observed_number_gaps"] == 0
    assert stats["timestamp_gap_events"] == 1
    assert stats["estimated_missing_frames"] == 1
    assert stats["first_gap_events"][0]["after_observed_index"] == 2
    audit = timing_summary([0, 33_400_000, 66_800_000, 133_600_000, 167_000_000])
    assert audit["timestamp_gap_events"] == 1
    assert audit["estimated_missing_frames"] == 1
    assert audit["first_gap_events"][0]["after_frame_index"] == 2


def test_manual_settings_reject_exposures_that_cannot_fit_at_30fps():
    invalid = deepcopy(DEFAULTS)
    invalid["realsense"]["261222076358"]["depth_exposure_us"] = 165000
    with pytest.raises(ValueError, match="depth_exposure_us"):
        validate_settings(invalid)
    assert validate_settings(DEFAULTS) == DEFAULTS
