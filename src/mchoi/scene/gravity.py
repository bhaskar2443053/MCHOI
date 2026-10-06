"""Estimate gravity 'up' and the dominant support plane in the ZED-left optical frame (x right, y down, z forward, metres).

Up is the mean IMU specific force (reaction to gravity) rotated into the left camera frame. The ZED floor-plane search
is also sampled; with a table-top rig it finds the table, which is reported as the support plane.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mchoi.recorder.devices import load_zed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--svo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='.json path; must not exist')
    parser.add_argument('--frames', type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    sl = load_zed()
    camera = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo.resolve()))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init.sdk_verbose = 0
    if camera.open(init) > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError('Opening SVO failed')
    try:
        camera.enable_positional_tracking(sl.PositionalTrackingParameters())
        imu_pose = camera.get_camera_information().sensors_configuration.camera_imu_transform
        R_cam_imu = np.asarray(imu_pose.get_rotation_matrix().r, float)
        sensors, plane, reset = sl.SensorsData(), sl.Plane(), sl.Transform()
        accel, planes = [], []
        for i in range(args.frames):
            if camera.grab() > sl.ERROR_CODE.SUCCESS:
                break
            camera.get_sensors_data(sensors, sl.TIME_REFERENCE.IMAGE)
            accel.append(np.asarray(sensors.get_imu_data().get_linear_acceleration(), float))
            if i % 10 == 0 and camera.find_floor_plane(plane, reset) == sl.ERROR_CODE.SUCCESS:
                planes.append(np.asarray(plane.get_plane_equation(), float))
    finally:
        camera.close()
    accel = np.asarray(accel)
    up = R_cam_imu @ accel.mean(0)
    up /= np.linalg.norm(up)
    planes = np.asarray(planes)
    planes[planes[:, :3] @ up < 0] *= -1
    plane = np.median(planes, 0)
    plane /= np.linalg.norm(plane[:3])
    result = dict(svo=str(args.svo), frames=len(accel), up_in_zed_left=up.tolist(), gravity_m_s2=float(np.linalg.norm(accel.mean(0))),
                  accel_std_m_s2=accel.std(0).tolist(), support_plane=plane.tolist(), support_plane_samples=len(planes),
                  camera_height_above_plane_m=float(plane[3]),
                  plane_vs_imu_angle_deg=float(np.degrees(np.arccos(np.clip(plane[:3] @ up, -1, 1)))),
                  convention='plane n.p + d = 0 with n pointing up (ZED-left optical frame); up is the IMU specific-force direction')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
