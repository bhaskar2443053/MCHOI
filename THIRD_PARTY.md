# Third-party software, models and data

This repository's code is MIT-licensed. It uses, but does not contain, the components below; each keeps its own terms.
Several are **non-commercial** or **copyleft**: check them before any use outside academic research.

| Component | Used for | Obtained by | Licence |
| --- | --- | --- | --- |
| [WiLoR](https://github.com/rolpotamias/WiLoR) code (commit `fcb9113`) and weights | hand detection and MANO reconstruction per view | `scripts/setup_wilor_env.ps1`, `scripts/download_models.py` | CC BY-NC-ND 4.0 |
| [MANO](https://mano.is.tue.mpg.de) `MANO_RIGHT.pkl` | hand model | manual download after registration | MANO licence (non-commercial research) |
| [smplx](https://github.com/vchoutas/smplx) | MANO layer | pip | SMPL-X licence (non-commercial research) |
| [Ultralytics](https://github.com/ultralytics/ultralytics) and YOLO11x-pose weights | body keypoints (shoulder, elbow); WiLoR's hand detector | pip, `scripts/download_models.py` | AGPL-3.0 |
| [YCB object models](https://www.ycbbenchmarks.com) (google_16k) | bowl, mug, pudding box meshes | `scripts/download_models.py` | CC BY 4.0 |
| [GRASPA-benchmark](https://github.com/hsp-iit/GRASPA-benchmark) YCB copies | mustard bottle, cracker box meshes | `scripts/download_models.py` | MIT (models from YCB) |
| [ZED SDK](https://www.stereolabs.com/developers) and `pyzed` | reading/recording ZED SVO2 | Stereolabs installer and wheel | Stereolabs licence |
| [librealsense](https://github.com/IntelRealSense/librealsense) `pyrealsense2` | reading/recording RealSense DB3 | pip | Apache-2.0 |
| OpenCV, NumPy, SciPy, PyAV, trimesh, Pillow, PyTorch | general | pip | Apache-2.0 / BSD / MIT-style |
| [pptxgenjs](https://github.com/gitbrent/PptxGenJS) | pipeline slides | `npm ci` in `deck/` | MIT |
| three.js (loaded from a CDN by the generated viewer) | 3D viewer | at view time | MIT |
