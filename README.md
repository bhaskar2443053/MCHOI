# mchoi: multi-camera hand–object interaction capture

Code behind the **MCHOI v1.0** dataset: a four-view RGB-D rig (ZED Mini left and right lenses, two RealSense D435)
records a person manipulating YCB objects. From each recording and the rig calibration it produces:

- the right hand as a MANO mesh, fitted jointly to all views (WiLoR initialization, temporal fit, One Euro filtering),
- the right arm (shoulder, elbow, wrist),
- the object's 6-DoF pose from ArUco markers, with its mesh registered to the markers,
- a Mocap text file (`001_Mug_Mocap.txt` layout), an interactive 3D viewer and annotated videos,
- and, for a release, the BIDS-style dataset with RGB/depth videos, calibration, metadata, figures and slides.

It also contains the two desktop apps used to capture the data: ChArUco rig calibration and synchronized recording.

## Repository layout

```text
configs/            release config (mchoi_v1.0.json: recordings and per-recording settings), board, cameras, recorder exposure,
                    paths.example.json -> copy to paths.local.json (git-ignored)
src/mchoi/
  calibration/      ChArUco rig calibration app             python -m mchoi.calibration
  recorder/         ZED SVO2 + RealSense DB3 recorder app   python -m mchoi.recorder
  io/               native SVO2/DB3 readers, synchronized four-view frame stream
  aruco/            marker detection, RealSense-to-ZED timing, multi-view marker poses
  hand/             WiLoR detections, multi-view MANO fit, SO(3) One Euro filter, videos
  arm/              2D keypoints, 3D shoulder/elbow fit, filter, video
  scene/            gravity/table frame, mesh-to-marker registration
  export/           Mocap file, 3D viewer, overlay video
  dataset/          camera videos, dataset layout, documentation media, summaries
  diagnostics/      fit comparison, One Euro delay
  pipeline.py       all steps for one recording
  reproduce.py      the whole release from configs/mchoi_v1.0.json
  paths.py          where data, models and the two Python environments are
deck/               pipeline slides (pptxgenjs + native PowerPoint equations)
environment/        pinned requirements of both environments
scripts/            environment setup, model download, app launchers
tests/              unit tests
docs/               pipeline, methods, capture apps, dataset README
```

## Installation (Windows 11)

The work is split over two Python environments because the ZED SDK and WiLoR need different Python versions. `mchoi.pipeline`
runs each step in the right one.

| Environment | Python | Contents | Used for |
| --- | --- | --- | --- |
| `.venv-sdk` | 3.14 | ZED SDK 5.3 (`pyzed`), pyrealsense2, OpenCV, PyAV, trimesh | capture apps, ArUco, arm/object fit, export, dataset |
| `.venv-wilor` | 3.10 | PyTorch 2.8 (CUDA 12.8), WiLoR, smplx, Ultralytics | hand detection/fit, arm keypoints |

Prerequisites: [ZED SDK 5.3](https://www.stereolabs.com/developers/release), Python 3.14 and 3.10 (`py` launcher), git, an NVIDIA
GPU with a CUDA 12.8 driver, and, for the slides only, [Node.js](https://nodejs.org) 24.

```powershell
git clone <this repository> mchoi; cd mchoi
powershell -ExecutionPolicy Bypass -File scripts\setup_sdk_env.ps1     # .venv-sdk
powershell -ExecutionPolicy Bypass -File scripts\setup_wilor_env.ps1   # .venv-wilor, third_party\WiLoR, models\
```

`setup_wilor_env.ps1` ends with `scripts\download_models.py`. That script downloads the WiLoR weights, YOLO11x-pose and the YCB
meshes into `models\`, and checks each against the SHA-256 of the file used for the release. **MANO** needs a registration: download
it from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) and copy `MANO_RIGHT.pkl` to `models\mano\`.

For the slides: `cd deck; npm ci`.

### Where things are

Data, models and the environments default to folders inside the repository (`data\`, `models\`, `.venv-sdk\`, `.venv-wilor\`).
To use other locations, copy `configs\paths.example.json` to `configs\paths.local.json` and edit it, or set environment variables
(`MCHOI_DATA`, `MCHOI_MODELS`, `MCHOI_MANO`, `MCHOI_WILOR_SRC`, `MCHOI_SDK_PYTHON`, `MCHOI_WILOR_PYTHON`; see
[src/mchoi/paths.py](src/mchoi/paths.py)). The data root holds:

```text
<data>/
  sessions/<session>/pose_*/ and calibration_*/calibration.json    calibration app
  recordings/<take>/zed.svo2, realsense_<serial>.db3, frame_timestamps.csv, recording.json    recorder app
  recordings/<take>/pipeline/ ...                                   processing outputs
  publication/MCHOI_v1.0/                                           the dataset
  logs/
```

## Use

**Capture.** Double-click `scripts\start_calibration.cmd` or `scripts\start_recorder.cmd` (or run `python -m mchoi.calibration` /
`python -m mchoi.recorder` in `.venv-sdk`). See [docs/calibration_and_recording.md](docs/calibration_and_recording.md).

**Process one recording.**

```powershell
.\.venv-sdk\Scripts\python.exe -m mchoi.pipeline --take <data>\recordings\<take> `
    --calibration <data>\sessions\<session>\calibration_<time>\calibration.json `
    --mesh models\objects\mustard_bottle\nontextured.stl --object-name mustard --object-markers 2 1 --marker-ids 1 2 3 4 5
```

Every step is skipped when its output exists, so a stopped run resumes. `--dry-run` prints the commands. See
[docs/pipeline.md](docs/pipeline.md) for the steps, options, outputs and file conventions.

**Reproduce MCHOI v1.0.** With the native recordings and the calibration session in the data root:

```powershell
.\.venv-sdk\Scripts\python.exe -m mchoi.reproduce              # pipeline, summary, videos, dataset, media, deck, checksums
.\.venv-sdk\Scripts\python.exe -m mchoi.reproduce --dry-run    # print every command
.\.venv-sdk\Scripts\python.exe -m mchoi.reproduce --stages dataset media deck checksums   # rebuild the dataset only
```

The per-recording settings (marker IDs, object mesh, colour range, scale fit, shoulder prior) are in
[configs/mchoi_v1.0.json](configs/mchoi_v1.0.json). Set `MCHOI_DATASET` to write the dataset somewhere else, for example to
compare it with a published copy. The native SVO2/DB3 recordings (about 25 GB per minute) are not part of the published dataset,
which contains RGB and lossless depth videos instead. The dataset itself is described in [docs/dataset_README.md](docs/dataset_README.md).

**Tests.** `.\.venv-sdk\Scripts\python.exe -m pytest -q`. On some PCs, importing PyAV 18 makes pytest print a
"Windows fatal exception: access violation" trace. That exception is raised and handled inside the FFmpeg DLLs, and the tests
still pass.

## Reproducibility check

The code was checked against the MCHOI v1.0 release on the original PC (2026-10-06):

| Check | Result |
| --- | --- |
| Unit tests | 58 passed |
| Dataset rebuild from the pipeline outputs (`mchoi.dataset.build`) | 281 of 283 files byte-identical; README and `dataset_description.json` differ only in the provenance text |
| Documentation media (`mchoi.dataset.media`) | 15 of 16 byte-identical; the 3D-viewer GIF depends on the browser's screenshot size |
| Slides (`deck/build.py`) | same 25 slides and 32 equations; theme identical |
| Model downloads (`scripts/download_models.py`) | all meshes match the release SHA-256 |
| gesture1 from the native recording (`mchoi.reproduce --only gesture1_…`, all steps) | Mocap file identical; 57 of 67 outputs byte-identical (all arrays and videos, the viewer); the other 10 are JSON metadata that differ only in file paths, run times and three added timing fields. ArUco detections, marker poses and camera timing offsets identical |

## Tested environment

Windows 11 Pro; Python 3.14.4 with ZED SDK 5.3.1 and librealsense 2.58.4; Python 3.10.20 with PyTorch 2.8.0+cu128 on an RTX 5090
(driver 591.86); Node.js 24.19; Microsoft Edge (viewer screenshots for the media). Exact package versions:
[environment/](environment/).

## Documentation

- [docs/pipeline.md](docs/pipeline.md): steps, options, outputs, STL-to-marker registration, Mocap conventions, smoothing, processed recordings.
- [docs/hand_object_pose.md](docs/hand_object_pose.md): how hand and object poses are estimated.
- [docs/calibration_and_recording.md](docs/calibration_and_recording.md): the capture apps.
- [docs/dataset_README.md](docs/dataset_README.md): the dataset (copied into the release as `README.md`).
- The slides `docs/MCHOI_v1.0_pipeline.pptx` in the dataset: workflow with snapshots and the maths of every stage.

## Licence and citation

The code is MIT-licensed ([LICENSE](LICENSE)). WiLoR, MANO, smplx, Ultralytics, YCB and the ZED SDK keep their own terms, and
several are non-commercial or copyleft: see [THIRD_PARTY.md](THIRD_PARTY.md). Citation metadata: [CITATION.cff](CITATION.cff).
