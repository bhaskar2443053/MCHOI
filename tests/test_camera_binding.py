import pytest
import pyrealsense2 as rs

from mchoi.calibration.cameras import bind_realsense_devices, view_label


class Device:
    def __init__(self, serial):
        self.serial = serial

    def get_info(self, field):
        if field == rs.camera_info.serial_number:
            return self.serial
        return "RealSense D435"


def test_identical_models_keep_their_slots_when_device_order_changes():
    first, second = Device("original"), Device("replacement")
    slots = {"d435": {"serial": "original"}, "d455": {"serial": "replacement"}}
    for devices in ([first, second], [second, first]):
        bound = dict(bind_realsense_devices(devices, slots))
        assert bound["d435"] is first
        assert bound["d455"] is second
    assert "D435 #2" in view_label("d455", {"name": "RealSense D435", "serial": "123456"})


def test_replacement_is_not_silently_assigned_to_a_missing_camera():
    slots = {"d435": {"serial": "original"}, "d455": {"serial": "old-device"}}
    with pytest.raises(RuntimeError, match="old-device"):
        bind_realsense_devices([Device("original"), Device("replacement")], slots)
    slots["d455"]["serial"] = "original"
    with pytest.raises(ValueError, match="different"):
        bind_realsense_devices([Device("original")], slots)
