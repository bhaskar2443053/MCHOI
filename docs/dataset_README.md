# MCHOI v1.0: Multi-Camera Hand–Object Interaction dataset

A person's right hand and arm manipulating YCB objects. The scene was recorded by four calibrated RGB views, three of which also record depth. Each recording comes with:

- RGB and depth videos with exact per-frame timestamps,
- a multi-view MANO right-hand pose,
- right-arm joints,
- the 6-DoF object pose, registered to the object's mesh,
- a motion-capture text file (Unreal Engine skeleton layout),
- an interactive 3D viewer and an annotated overlay video.

The rig calibration is included with all its raw ChArUco captures. The processing workflow is shown step by step in [`docs/MCHOI_v1.0_pipeline.pptx`](docs/MCHOI_v1.0_pipeline.pptx).

> **Before public release, complete:** authors, licence, citation (`dataset_description.json`), and ethics/consent statement. The participant's torso and arms are visible in one camera view.

## At a glance

| | |
| --- | --- |
| Participants | 1 (right hand) |
| Recordings | 9: six object tasks (7 runs) and 2 gesture runs |
| Objects | mustard bottle, cracker box (party size), pudding box, bowl, mug: YCB meshes included; banana-shaped object: pose only |
| Total duration | 13.3 min (795.8 s) |
| Synchronized frames (ZED, ~30 fps) | 22,887 |
| Frames with a valid hand pose | 16,263 (71%) |
| Frames with an object pose | 18,838 (of 18,891 object-task frames) |
| Cameras | ZED Mini stereo (left, right; SDK depth), 2 × RealSense D435 (RGB + depth) |
| Resolution | RGB 1920 × 1080; ZED depth 1920 × 1080; RealSense depth 848 × 480 |
| Calibration | 25 ChArUco board poses; reprojection RMS 0.22–0.35 px per camera |
| Size | 18 GB, 490 files (`SHA256SUMS.txt` lists a checksum for every file) |

| Recording | Object | Duration (s) | Frames | Hand-pose frames | Object-pose frames | Views with validated timing |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| task-mug_run-01 | mug (YCB 025) | 83.5 | 2,386 | 1,360 | 2,386 | ZED left, right |
| task-mug_run-02 | mug (YCB 025) | 104.0 | 3,015 | 1,956 | 3,012 | ZED left, right |
| task-mustardbottle_run-01 | mustard bottle (YCB 006) | 107.5 | 3,109 | 2,686 | 3,103 | all 4 |
| task-banana_run-01 | banana-shaped object | 79.8 | 2,311 | 1,802 | 2,309 | all 4 |
| task-bowl_run-01 | bowl (YCB 024) | 92.5 | 2,585 | 1,476 | 2,543 | all 4 |
| task-puddingbox_run-01 | pudding box (YCB 008) | 102.1 | 2,954 | 2,199 | 2,954 | all 4 |
| task-crackerbox_run-01 | cracker box (YCB 003, party size) | 88.3 | 2,531 | 1,695 | 2,531 | all 4 |
| task-gesture_run-01 | none | 36.6 | 1,061 | 732 | — | ZED left, right, d435n1 |
| task-gesture_run-02 | none | 101.4 | 2,935 | 2,357 | — | ZED left, right, d435n1 |

The machine-readable version is `recordings.tsv`. Frames without a hand pose are frames where the hand was out of view or not supported by enough views; they are marked `valid = 0`, never interpolated.

## Folder structure and naming

The layout follows BIDS conventions: `key-value` entities joined by `_`, a suffix naming the data type, raw data under `sub-*/ses-*/`, processed data under `derivatives/`, and a JSON sidecar describing each data file.

```text
MCHOI_v1.0/
├── README.md  CHANGES  dataset_description.json
├── participants.tsv  tasks.tsv  cameras.tsv  recordings.tsv
├── calibration/
│   ├── calib-01_rig_calibration.json / .yml        intrinsics, distortion, camera-to-reference transforms
│   ├── calib-01_rig_summary.txt  calib-01_rig_calibration_description.json
│   └── captures/calib-01_pose-<NNNN>_cam-<camera>_charuco.png   + calib-01_pose-<NNNN>_charuco.json
├── objects/            obj-<object>_mesh.stl (metres, YCB frame)  objects.tsv
├── sub-01/ses-01/
│   ├── sub-01_ses-01_scans.tsv                      every video file with acquisition time
│   └── video/
│       ├── <rec>_cam-<camera>_rgb.mp4    + .json  + _rgb_frames.tsv
│       └── <rec>_cam-<camera>_depth.mkv  + .json  + _depth_frames.tsv
├── derivatives/mchoi-pipeline/sub-01/ses-01/
│   ├── hand/    <rec>_space-zedleft_desc-mano_handpose.npz/.json, _desc-mano_handjoints.tsv/.json
│   ├── arm/     <rec>_space-zedleft_desc-rightarm_joints.tsv/.json
│   ├── object/  <rec>_space-zedleft_desc-object_pose.tsv/.json, _desc-aruco_markerposes.tsv/.json,
│   │            _desc-meshfit_transform.json, _desc-meshfit_contactsheet.jpg
│   ├── mocap/   <rec>_space-world_mocap.txt/.json
│   ├── sync/    <rec>_desc-camera_timing.json
│   └── viz/     <rec>_desc-3d_viewer.html, _desc-overlay_video.mp4
└── docs/        MCHOI_v1.0_pipeline.pptx, media/ (figures, GIFs)
```

| Entity | Values | Meaning |
| --- | --- | --- |
| `sub` | `01` | participant |
| `ses` | `01` | session (2026-09-23) |
| `task` | `mustardbottle`, `crackerbox`, `puddingbox`, `bowl`, `mug`, `banana`, `gesture` | object manipulated |
| `run` | `01`, `02` | repetition of a task |
| `cam` | `zedminileft`, `zedminiright`, `d435n1`, `d435n2` | camera (see `cameras.tsv`) |
| `space` | `zedleft` | ZED Mini left optical frame: x right, y down, z forward, metres |
| `space` | `world` | gravity-aligned Unreal frame: X forward, Y right, Z up, centimetres |
| `desc` | e.g. `mano`, `rightarm`, `object`, `aruco` | which derivative |
| `<rec>` | e.g. `sub-01_ses-01_task-mug_run-02` | one recording |

## Data files

### Camera videos (`sub-01/ses-01/video/`)

| File | Content |
| --- | --- |
| `_cam-zedminileft_rgb.mp4`, `_cam-zedminiright_rgb.mp4` | unrectified native ZED images (the images the calibration describes); H.264, CRF 17 |
| `_cam-zedminileft_depth.mkv` | ZED SDK NEURAL depth, uint16 millimetres, 0 = invalid; aligned to the **rectified** left image (intrinsics in the sidecar); FFV1 lossless |
| `_cam-d435n1_rgb.mp4`, `_cam-d435n2_rgb.mp4` | RealSense colour, native 1920 × 1080; H.264 |
| `_cam-d435n1_depth.mkv`, `_cam-d435n2_depth.mkv` | native RealSense Z16 depth, 848 × 480, in the depth sensor's frame (not aligned to colour); multiply by `DepthUnitsMetres` from the sidecar; FFV1 lossless (bit-exact) |
| `*_frames.tsv` | `frame`, `source_frame_index`, `timestamp_ns` (PC clock, Unix ns), `pts_us`; RealSense rows with validated timing also have `zed_aligned_timestamp_ns` |
| `*.json` | camera, serial number, calibration name, intrinsics/extrinsics, codec, frame count, exposure, timing correction |

Read depth losslessly with PyAV (`frame.to_ndarray()` returns uint16) or ffmpeg (`-pix_fmt gray16le`). The native ZED SVO2 and RealSense DB3 recordings are not distributed, for size reasons (about 320 GB).

**Synchronization.** Cameras were not hardware-synchronized. The RealSense clocks were aligned to the ZED by matching ArUco marker content, with hold-out validation (`derivatives/.../sync/`). Where an offset could not be validated, that camera is listed as not validated and was not used for multi-view processing: both RealSense cameras for the mug runs, and d435n2 for the gesture runs. Its video is still provided.

### Hand (`derivatives/.../hand/`)

- `_desc-mano_handjoints.tsv`: one row per ZED frame, with `frame`, `timestamp_ns`, `valid`, `n_views`, then x/y/z of the 21 joints in OpenPose order (`wrist`, `thumb_cmc` … `little_tip`), in metres, `space-zedleft`.
- `_desc-mano_handpose.npz`: `joints` (N,21,3), `vertices` (N,778,3), `faces`, `mano_rotmat` (N,16,3,3), `mano_axis_angle`, `betas` (one shape per recording), `wrist_translation`, `valid`, `used_views`. The sidecar gives the reconstruction rule. The MANO model itself must be obtained from <https://mano.is.tue.mpg.de>.

Method: hands are detected by YOLO and box-smoothed with One Euro; WiLoR predicts MANO in every view; keypoints are triangulated with visibility weights. One MANO hand is fitted jointly to all views with temporal terms, then the rotations are One Euro filtered. The median reprojection error is about 11 px at 1920 × 1080.

### Right arm (`derivatives/.../arm/`)

The `_desc-rightarm_joints.tsv` file has shoulder, elbow and wrist positions (filtered and raw) plus a per-joint `method` code:

| Code | Meaning |
| --- | --- |
| 1 | triangulated |
| 2 | single view at bone length |
| 3 | MANO wrist |
| 4 | prior + inverse kinematics |

Only RealSense d435n2 sees the shoulder. Where it is not validated (mug and gesture runs), shoulder and elbow are a static shoulder prior plus inverse kinematics: plausible placements, not measurements. Upper arm and forearm lengths are fixed at 30 cm and 27 cm.

### Object (`derivatives/.../object/`)

- **`_desc-object_pose.tsv`**: per-frame object pose, translation `tx ty tz` (m) and quaternion `qx qy qz qw`, `space-zedleft`. Columns without a prefix are One Euro filtered; `raw_` columns are unfiltered.
  - The pose is of the **mesh origin**: a mesh vertex `v` (metres) is at `R · (scale ∘ v) + t`, with `MeshScale` in the sidecar.
  - The banana-shaped object has no matching mesh, so its pose is the midpoint of its two markers.
- **`_desc-aruco_markerposes.tsv`**: every reviewed marker (24 mm, DICT_4X4_50) per frame, with status (multi-view or tracked single-view), source cameras and reprojection RMS. Markers 4 and 5 are on the forearm.
- **`_desc-meshfit_transform.json`** and **`_desc-meshfit_contactsheet.jpg`**: the mesh-to-marker transform and the evidence for it.
  - It was estimated by fitting the projected mesh outline to the object's colour in 12 multi-view frames, with an "upright on the table at the start" prior.
  - Check the contact sheet for each object.

### Motion-capture file (`derivatives/.../mocap/`)

`_space-world_mocap.txt` uses the `version:4` text layout:

```text
TIMESTAMP|FRAME_NUMBER|FRAME_TIMESTAMP|HEAD loc+quat|21|<bone|pos|quat>...|NUM_OBJECTS|<object|loc|quat|scale>
```

- **Bones:** 21 right-arm and hand bones (`clavicle_r`, `upperarm_r`, `lowerarm_r`, `hand_r`, `wrist_inner_r`, `wrist_outer_r`, `thumb/index/middle/ring/pinky_01..03_r`) and the object.
- **World frame:** Unreal axes (X forward, Y right, Z up, left-handed), in centimetres.
  - Up comes from the ZED IMU.
  - Forward points towards d435n2.
  - The origin is the table plane below the ZED camera.
- **Head/camera:** the static ZED-left camera.
- **Mesh placement:** an object mesh vertex `v` (m) is at `LOC + rotate(ROT, 100 · (SX·v.x, −SY·v.y, SZ·v.z))`.
- **Frames:** only frames with a valid hand pose are written, and `FRAME_NUMBER` is the ZED frame index.
- **Sidecar JSON:** gives the exact world axes and origin in `space-zedleft`, the filter settings, and jitter statistics.

### Visualization (`derivatives/.../viz/`)

- `_desc-3d_viewer.html`: a self-contained 3D player for the hand mesh, arm, object mesh and camera. Open it in a browser; it needs internet access for three.js. Append `#i=<row>` to open at a row.
- `_desc-overlay_video.mp4`: all four camera views with the hand mesh, arm and object mesh projected from the exported data.

## Loading examples (Python)

```python
import numpy as np, pandas as pd, av
rec = 'sub-01_ses-01_task-mustardbottle_run-01'
d = 'derivatives/mchoi-pipeline/sub-01/ses-01'
hand = np.load(f'{d}/hand/{rec}_space-zedleft_desc-mano_handpose.npz')        # hand['joints'][hand['valid']]
obj = pd.read_csv(f'{d}/object/{rec}_space-zedleft_desc-object_pose.tsv', sep='\t', na_values='n/a')
frames = pd.read_csv(f'sub-01/ses-01/video/{rec}_cam-d435n1_depth_frames.tsv', sep='\t')
with av.open(f'sub-01/ses-01/video/{rec}_cam-d435n1_depth.mkv') as c:
    depth_m = next(c.decode(video=0)).to_ndarray() * 0.001                   # DepthUnitsMetres from the sidecar
```

To project 3D results into a camera, use `calibration/calib-01_rig_calibration.json` (`T_camera_from_reference`, `camera_matrix`, `distortion_coefficients`). It applies to the unrectified RGB videos.

## Known limitations

- Post-hoc, content-based synchronization; not hardware-triggered.
- The calibration was captured after the recordings with the rig unmoved. The mug runs were recorded about 1.5 h earlier, and their RealSense timing could not be validated.
- Shoulder and elbow are estimates: single-view with fixed bone lengths, or a prior plus inverse kinematics.
- Hand–object contact is not enforced; fingers can slightly penetrate meshes.
- Object facing is ambiguous for nearly front/back-symmetric objects (mustard bottle). The cracker box uses a fitted scale (party-size box).
- The banana-shaped object has no mesh.
- Scores reported are consistency measures (reprojection, jitter, silhouette IoU), not ground-truth accuracy.

## Provenance

Generated with the `mchoi` code repository (`python -m mchoi.reproduce` runs every step):
- `mchoi.pipeline` (processing of each recording),
- `mchoi.dataset.videos` (camera RGB/depth videos),
- `mchoi.dataset.build` (this layout),
- `mchoi.dataset.media` and `deck/build.py` (figures and slides).

Source recording names are listed in `recordings.tsv` (`source_recording`).
