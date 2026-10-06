# Hand–object pipeline

From one recording of the rig (ZED Mini left and right, two RealSense D435) and the rig calibration, `mchoi.pipeline` produces the right hand as a MANO mesh, the right arm (shoulder, elbow, wrist), the object's 6-DoF pose with its STL mesh, a Mocap text file, a 3D viewer and annotated videos. Recording and calibration are described in [calibration_and_recording.md](calibration_and_recording.md); the algorithms in [hand_object_pose.md](hand_object_pose.md) and the slides in the dataset (`docs/MCHOI_v1.0_pipeline.pptx`).

## Inputs

1. **Calibration**: the `calibration.json` written by the calibration app, for example `<data>\sessions\<session>\calibration_<time>\calibration.json`. It must come from the same, unmoved rig as the recording.
2. **Recording folder** from the recorder app, containing:
   - `zed.svo2`
   - `realsense_<serial>.db3` (one per RealSense)
   - `frame_timestamps.csv`
   - `recording.json`
3. **Object mesh** (optional): an STL in metres. The YCB models of the objects used so far are downloaded to `models/objects/` by `scripts/download_models.py`.
4. **ArUco markers** (DICT_4X4_50, 24 mm squares) on the object. Use two markers on opposite faces when possible: a *primary* marker that the mesh is fitted to, and a *secondary* marker used when the primary is hidden.

Only the **right** hand and arm are reconstructed.

## Run

From the project root in PowerShell:

```powershell
$data = 'D:\mchoi_data'   # the data root (configs\paths.local.json or MCHOI_DATA)
$take = "$data\recordings\mustard_20260923_154513_076645"
$cal  = "$data\sessions\20260923_163029_830758\calibration_20260923_163607_631740\calibration.json"

.\.venv-sdk\Scripts\python.exe -m mchoi.pipeline --take $take --calibration $cal --mesh models\objects\mustard_bottle\nontextured.stl `
    --object-name mustard --object-markers 2 1 --marker-ids 1 2 3 4 5
```

- **Marker IDs:** `--marker-ids` lists the genuine marker IDs in the scene. If you leave it out, the run stops after ArUco detection and prints every decoded ID with its count. Rare IDs are usually misdecodes; rerun with the real ones.
- **Restarting:** every step is skipped if its output already exists, so a stopped or failed run just needs restarting.
- **Step control:** `--from <step>` or `--only <step> ...` reruns specific steps. Delete or rename that step's output first, because the tools refuse to overwrite.
- **Preview:** `--dry-run` prints every command without running it.
- **Run folder:** outputs go into `<take>\<run-name>`, default `pipeline`. Use another `--run-name` to keep several runs side by side.

Object options:

| Option | Use |
| --- | --- |
| `--object-hsv-low/high H S V` | Object colour for the mesh fit. The default is yellow. If low hue > high hue, the range wraps around 0, for red: `170 90 50` / `10 255 255`. |
| `--object-fit-scale uniform\|axes` | Also fit the mesh size, when the real object differs from the model, for example a party-size box. |
| `--markers-only` | The object has markers but no matching mesh: export the marker-midpoint pose, drawn as a 4 cm cube. |
| no `--mesh` | No object (gesture takes): hand and arm only. |
| `--shoulder-prior X Y Z` | Static shoulder (ZED-left metres) for takes where no camera sees the shoulder; see [the arm section](#right-arm). |
| `--plane-height-cm` | Table height, to put the world origin on the floor. |

`configs/mchoi_v1.0.json` holds the exact settings used for every September recording; `python -m mchoi.reproduce --dry-run` prints the resulting commands. See [Processed recordings](#processed-recordings).

On the mustard take (3,109 frames, about 104 s of video), a full run took about 45 minutes on this PC from the hand steps onwards. The ArUco steps come on top of that. The longest steps are the hand cache (about 10 min), the hand videos and the arm keypoints (about 6 min each), and the overlay video (about 6 min).

## What each step does

| Step | Module (`mchoi.`) | Env | Output (in `<take>`) |
| --- | --- | --- | --- |
| `aruco_detect` | `aruco.extract_zed`, `aruco.extract_realsense` | SDK | `aruco_annotated\` corner detections |
| `aruco_review` | (writes `marker_selection.json` from `--marker-ids`) | — | `aruco_pose_all_cameras_new_calibration\marker_selection.json` |
| `aruco_poses` | `aruco.poses` (with `aruco.timing`, `aruco.track`), `aruco.validate` | SDK | same folder: `calibration_used.json`, `timing_alignment.json`, `poses.csv`, marker videos |
| `timing` | `aruco.prepare_timing` | SDK | `wilor_timing\timing_alignment.json`: validated offsets; unvalidated views are dropped |
| `hand_cache` | `hand.cache` | WiLoR | `<run>\hand_cache\` |
| `hand_fit` | `hand.fit` | WiLoR | `<run>\hand_fit\fused_hand.npz` |
| `hand_filter` | `hand.filter` | WiLoR | `<run>\hand_fit_one_euro\fused_hand.npz` |
| `hand_videos` | `hand.render`, `hand.validate` | SDK | `<run>\hand_fit_one_euro\*.mp4`, `validation.json` |
| `arm_2d` | `arm.keypoints` | WiLoR | `<run>\right_arm\arm2d.npz` |
| `gravity` | `scene.gravity` | SDK | `<run>\right_arm\gravity.json` |
| `arm_fit` | `arm.fit` | SDK | `<run>\right_arm\arm3d_raw.npz` |
| `arm_filter` | `arm.filter` | SDK | `<run>\right_arm\arm3d_one_euro.npz` + `.json` |
| `object_mesh` | `scene.object_mesh` | SDK | `<run>\mocap\<object>_mesh_to_marker<id>.json` + `.jpg` |
| `mocap` | `export.mocap` | SDK | `<run>\mocap\<Object>_Mocap.txt` + `.json` |
| `viewer` | `export.viewer` | SDK | `<run>\mocap\<object>_mocap_viewer.html` |
| `overlay_video` | `export.overlay` | SDK | `<run>\mocap\<object>_overlay.mp4` |
| `arm_video` | `arm.render` | SDK | `<run>\right_arm\right_arm_one_euro.mp4` |

"SDK" is `.venv-sdk` (Python 3.14 with the ZED SDK, pyrealsense2, OpenCV, PyAV and trimesh). "WiLoR" is `.venv-wilor` (Python 3.10 with CUDA PyTorch, WiLoR, MANO and Ultralytics). The driver picks the right one for each step. The WiLoR steps read synchronized native frames from `mchoi.io.stream_four_views`, which runs in the SDK environment and pipes raw images to them.

The ArUco step must come first: besides the object poses, it measures each RealSense camera's clock offset from the image content (`timing_alignment.json`). The `timing` step keeps only offsets that pass hold-out validation. Every multi-view step pairs frames with them and leaves out views whose offset could not be validated: for example both RealSense cameras for mug1 and mug2, and the D455 for the gesture takes. The ArUco folder name `aruco_pose_all_cameras_new_calibration` is fixed, because the tools look for it there.

## Outputs to look at

| File | What it shows |
| --- | --- |
| `<run>\mocap\<Object>_Mocap.txt` | Final pose data: 21 right-arm/hand bones and the object per frame, in the `001_Mug_Mocap.txt` layout. |
| `<run>\mocap\<Object>_Mocap.json` | World-frame definition, object transform, filter settings, jitter numbers and skipped-frame counts. |
| `<run>\mocap\<object>_mocap_viewer.html` | Interactive 3D player: MANO hand mesh, arm mesh, object mesh, camera. Add `#i=<row>` to the URL to open at a row. It needs internet access for three.js. |
| `<run>\mocap\<object>_overlay.mp4` | Four camera views with the MANO hand, arm chain and object mesh drawn from the exported data. |
| `<run>\hand_fit_one_euro\wilor_multiview_left.mp4`, `comparison_left.mp4`, `comparison_hand.mp4` | Hand overlay on ZED left, plus before/after-filter comparisons. |
| `<run>\right_arm\right_arm_one_euro.mp4` | Raw (orange) vs filtered (green) arm in four views. |
| `aruco_pose_all_cameras_new_calibration\all_cameras_annotated.mp4` | Detected markers and their axes in all views. |

## How the object mesh is placed (STL → marker)

ArUco gives the pose of the **marker**, but the STL has its own origin (for the YCB mustard bottle, the bottom of the bottle). `mchoi.scene.object_mesh` estimates the fixed transform `T_marker_from_mesh` between the two:

1. It picks `--sample-frames` frames (default 12) where the primary marker has a multi-view pose, spread over the take, and decodes all four camera images for them.
2. In each image it masks the object by colour (HSV range; the defaults select yellow).
3. A candidate transform places the mesh through the marker pose. The mesh's projected outline (convex hull) is compared with the mask by IoU, inside a margin around the projection.
4. **Search.** About 4,000 starting placements are scored on a subset of views at quarter resolution:
   - the marker on the outer surface along each of the mesh's 6 principal directions,
   - marker-up along each of the 4 perpendicular directions,
   - 7 heights × 5 sideways positions on that side,
   - 5 rolls.

   The best 4 are refined with Nelder–Mead on all views, over 6 pose parameters plus optional scale (`--fit-scale`).
5. **Rest prior** (`--gravity`): in the first frame with a marker pose, the object should stand in its modelled pose: YCB meshes are +Z up with the base at z = 0. The prior penalizes:
   - tilt between mesh +Z and gravity,
   - a base more than 1 cm below or 4 cm above the table plane (objects often stand on a coaster).

   Without it, a near-symmetric object seen only from the two ZED views can fit equally well tipped over. This happened with the mug.
6. It saves:
   - `T_marker_from_mesh`, its inverse, and `scale` (`p_marker = R (scale · p_mesh) + t`),
   - the marker position in mesh coordinates,
   - the IoU and the rest tilt,
   - a contact sheet of the fitted outlines.

**Check the `.jpg` contact sheet:** the outline should hug the object in every view. IoU is only a rough guide: it is lowered by holes in the colour mask (white print, shadows, the hand) and by views under different lighting. Two limits:
- Objects that look the same front and back, such as the mustard bottle, can be fitted with the marker on either face. That turns the mesh 180° about its vertical axis without changing its outline.
- If the real object differs from the model, scale fitting helps only when the difference is mostly size.

`mchoi.export.mocap --mesh-transform` then writes the **STL origin** pose: object pose = marker pose ∘ `T_marker_from_mesh`. To place the mesh from a Mocap row:

```text
world_cm(v) = LOC + rotate(ROT, 100 * (SX v.x, -SY v.y, SZ v.z))      v = STL vertex in metres, S = SCALE_X/Y/Z
```

The Y flip converts the right-handed STL into the left-handed Unreal world, as Unreal's mesh importer does. The viewer and overlay video use exactly this rule, so they also check the file.

## Mocap file conventions

The layout follows the reference file `001_Mug_Mocap.txt`: a `version:4` header with camera and object, then one line per frame:

```text
TIMESTAMP|FRAME_NUMBER|FRAME_TIMESTAMP|HEAD loc+quat|21|<bone|pos|quat>...|NUM_OBJECTS|<object|loc|quat|scale>
```

- **World**: centimetres, Unreal axes (X forward, Y right, Z up, left-handed).
  - Up is gravity from the ZED IMU.
  - Forward is the horizontal direction towards the D455 opposite the subject.
  - The origin is the table plane directly below the ZED-left camera; `--plane-height-cm` shifts it.
  - The exact axes and origin in ZED-left coordinates are in the `.json` sidecar.
- **Head/camera**: the static ZED-left camera. The subject's head is not observed.
- **Bones**: `clavicle_r, upperarm_r, lowerarm_r, hand_r, wrist_inner_r, wrist_outer_r` and `thumb/index/middle/ring/pinky_01..03_r`. Axes match the reference file:
  - local −X points from a bone to its child,
  - upper-arm and forearm Z is the elbow hinge axis,
  - hand and finger Z points from the little-finger side to the index side; thumb Z is the palm normal,
  - `wrist_inner/outer` are `hand_r` rotated ±90° about X.
- **Not observed**: `clavicle_r` is placed 15.58 cm medially of the shoulder. Upper arm and forearm lengths are fixed at 30 and 27 cm, because only one camera sees the shoulder.

### Right arm

Only the D455 (opposite the subject) sees the shoulder; the ZED views see the forearm and sometimes the elbow. Each frame is labelled with a `method` code in `arm3d_raw.npz`:

| Code | Meaning |
| --- | --- |
| 1 | Triangulated from two or more views. |
| 2 | Single view, placed at bone length from the parent joint. |
| 3 | The wrist, taken from MANO. |
| 4 | Fallback, used when no camera sees the joint: the shoulder is held at a static prior and the elbow is solved by two-bone IK, bending down and to the subject's right. |

The prior is the take's median observed shoulder when at least 100 frames observe it. Otherwise it comes from `--shoulder-prior`. For mug1, mug2 and the gesture takes, the D455 timing is not validated, so their prior is the median of the four takes that do see the shoulder: (0.103, 0.255, −0.031) m in ZED-left. Those takes' medians differ by up to 9 cm, so on these takes the shoulder and elbow are plausible placements, not measurements.
- **Frames**: only frames with a valid hand fit and a fitted shoulder/elbow are written. `FRAME_NUMBER` is the ZED-left frame index, so skipped frames leave gaps. Times are seconds from the first ZED frame.

## Smoothing

| What | Filter | Settings |
| --- | --- | --- |
| Hand detection boxes | One Euro | 1.5 Hz, beta 8 |
| Hand articulation | temporal MANO fit, then One Euro on SO(3) | fingers 1 Hz / beta 1; wrist rotation 2.5 Hz / beta 3 |
| Shoulder, elbow | 3D One Euro, bone lengths re-imposed | elbow 0.7 Hz / beta 4; shoulder 0.3 Hz / beta 2 (per m/s) |
| Object pose | One Euro on the primary-marker pose (position + SO(3)) | 1.5 Hz; beta 20 per m/s, 8 per rad/s; no reset at marker switches |

All filters reset after gaps of more than 100 ms. On the mustard take:
- Shoulder/elbow jitter (median second difference) falls from 7.2/5.4 mm to 0.7/1.0 mm, with about 33 ms of delay.
- Object spikes (95th percentile) fall from 11.1 to 3.6 mm and from 4.3° to 2.1°. During fast moves the filtered mesh origin trails the raw pose by under 9.8 mm and 1.7° in 95% of frames.
- The object's beta is high on purpose: with beta 0.5 the jitter was lower, but the mesh lagged up to 23 mm (95th percentile) behind the real bottle during fast moves, which was visible in the overlay video.

`mchoi.diagnostics.one_euro_delay` measures the filter delay and jitter numbers above; `mchoi.diagnostics.compare_fits` compares two hand fits frame by frame.

## Processed recordings

All nine September 23 recordings were processed on 2026-10-01 with the same calibration (`sessions\20260923_163029_830758\calibration_20260923_163607_631740`). Outputs are in each take's `pipeline\` folder. The settings of every take are in `configs/mchoi_v1.0.json` (`recordings[].pipeline`); `python -m mchoi.reproduce` re-runs all of them, and its `summary` stage regenerates the table below (`mchoi.dataset.summarize`).

| Take | Views | Frames written / total | Shoulder seen | Object | Mesh fit (IoU, rest tilt, scale) | Object p95 jump raw→filtered |
| --- | --- | --- | --- | --- | --- | --- |
| banana_20260923_154931_267883 | ZED left, ZED right, d435, d455 | 1802 / 2311 | 94% | banana | — | 5.8 → 1.9 mm |
| bowl_20260923_155225_577637 | ZED left, ZED right, d435, d455 | 1476 / 2585 | 96% | bowl | 0.68, 0.9°, ×1.0 | 4.4 → 1.3 mm |
| CrackerBox_20260923_155853_433482 | ZED left, ZED right, d435, d455 | 1695 / 2531 | 84% | cracker_box | 0.52, 0.0°, ×1.28/1.2/1.2 | 10.4 → 3.2 mm |
| gesture1_20260923_160157_967044 | ZED left, ZED right, d435 | 732 / 1061 | 0% | — | — | — |
| gesture2_20260923_160512_971417 | ZED left, ZED right, d435 | 2357 / 2935 | 0% | — | — | — |
| jello_20260923_155604_473408 | ZED left, ZED right, d435, d455 | 2199 / 2954 | 97% | pudding_box | 0.24, 0.0°, ×0.97 | 6.1 → 2.4 mm |
| mug1_20260923_143947_317314 | ZED left, ZED right | 1360 / 2386 | 0% | mug | 0.6, 0.0°, ×1.0 | 7.2 → 2.3 mm |
| mug2_20260923_145820_415750 | ZED left, ZED right | 1956 / 3015 | 0% | mug | 0.65, 2.7°, ×1.0 | 12.3 → 3.6 mm |
| mustard_20260923_154513_076645 | ZED left, ZED right, d435, d455 | 2686 / 3109 | 96% | mustard | 0.56, 0.0°, ×1.0 | 11.1 → 3.6 mm |

"Shoulder seen" is the share of hand frames where a camera sees the shoulder; elsewhere the shoulder prior and IK fill in. Frames are skipped when the hand fit is invalid (not visible or not supported by enough views).

| Take | Object model and settings | Notes |
| --- | --- | --- |
| mustard | YCB mustard bottle, markers 2+1 | The bottle looks the same front and back; the fit put the marker on the opposite face from the first run (180° about vertical, same outline). |
| banana | **markers only** (4 cm cube proxy), markers 2+1 | The banana in the take is a smaller object with the marker on a block; the YCB banana does not fit even with scale. Supply a matching mesh to place it. |
| bowl | YCB bowl, markers 2+1, red | Marker inside on the bottom. |
| CrackerBox | YCB cracker box, per-axis scale ×1.28/1.20/1.20, markers 2+1, red | The real box is a party-size box. |
| jello | YCB **pudding box** (Jell-O chocolate pudding), uniform scale ×0.97, markers 2+1, purple | IoU is low only because the RealSense views fail the colour mask; the outline fits in all views. |
| mug1, mug2 | YCB mug, markers 0+1, red | Only the ZED views have validated timing; the rest prior keeps the mug upright. Shoulder from the prior. |
| gesture1, gesture2 | none | Hand and arm only; D455 timing not validated, shoulder from the prior. gesture1 is a short, interrupted take. |

## Limitations

- Camera timing is aligned after recording, from image content, and is not hardware-synchronized.
- The shoulder is seen by one camera only. Arm depth therefore relies on fixed bone lengths and the far ray/sphere intersection, so it is plausible rather than measured.
- Hand–object contact is not enforced; fingers can slightly penetrate the mesh.
- The object colour mask and marker placement are object-specific; check every mesh fit's contact sheet.
- Where RealSense timing cannot be validated (mug1, mug2, gesture takes), only the remaining views are used, and the shoulder comes from a prior.
- Overlay drawing paints the object, then the arm, then the hand, without occlusion.
