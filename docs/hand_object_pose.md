# How hand and object poses become video annotations

Hand reconstruction and ArUco marker tracking are separate pipelines using the calibrated RGB cameras. Both express metric results in the **ZED-left optical coordinate system**, allowing their results to be compared when frame times and calibration agree.

The hand videos of the `hand_videos` step ([pipeline](pipeline.md)) contain hand annotations. The object/marker annotations are exported separately by [aruco/poses.py](../src/mchoi/aruco/poses.py). The current hand renderer does not read object poses or draw object axes, and neither estimator uses hand–object contact constraints.

## Shared inputs and coordinates

Both pipelines use ZED left/right and the available RealSense RGB views. They use saved calibration intrinsics, lens distortion, and rigid camera transforms. They do not use RealSense depth or ZED depth to reconstruct these poses. Calibration comes from a separate ChArUco session; it is not re-estimated from the manipulation video.

The reference axes are X right, Y down, Z forward, with distances in metres. This reference is fixed to the ZED camera, not to the table or gravity. For any camera `C`, calibration supplies:

```text
p_C = T_camera_from_reference[C] @ p_Z
pixel_C = project_with_distortion(K_C, D_C, p_C)
```

Here positions are homogeneous column vectors. Raw ZED images are unrectified, so displayed geometry must use the corresponding distortion model. The hand fitter uses undistorted observations internally; the renderer restores distortion when projecting onto the raw image.

ZED stereo shares a frame timestamp. RealSense frames are aligned using per-recording timing corrections and nearest unused frames within 20 ms of the ZED timestamp. For the mustard recording, validated corrections subtract 93 ms and 80 ms from the two RealSense saved times. These are content-based alignment estimates, not hardware synchronization. Other recordings have different corrections and support.

## MANO hand pose

### 1. Detect the hand and predict an initial pose

[hand/cache.py](../src/mchoi/hand/cache.py) decodes aligned images and calls [hand/wilor_backend.py](../src/mchoi/hand/wilor_backend.py) for each camera. Hand YOLO supplies detection boxes, handedness, confidence, and 21 confidence-scored keypoints. [hand/detection.py](../src/mchoi/hand/detection.py) associates detections into tracks and applies One Euro to each track's normalized box center and size.

The filtered box width and height are multiplied by **1.10** around the center before WiLoR inference. WiLoR's required aspect-ratio padding still applies. WiLoR predicts MANO shape coefficients, global wrist orientation, 15 local finger-joint rotations, mesh geometry, and a monocular camera translation. That translation is an initial monocular estimate, not measured metric depth from the calibrated rig.

An important division of responsibility: **YOLO keypoints provide the confidence-scored multiview measurements; WiLoR provides the MANO initialization and shape estimate.** The reconstruction does not simply average independent WiLoR meshes from all cameras.

### 2. Associate the same hand and triangulate joints

[multiview.py](../src/mchoi/multiview.py) selects the target handedness from ZED-left observations and associates other views using handedness and epipolar consistency. This trial reconstructs one target hand.

Each joint receives a view weight based on its detection confidence, confidence along its finger, projected finger size, hand confidence, and timing agreement. Thus different cameras can contribute more strongly to different fingers. Clear-view weighting is a visibility proxy; it does not explicitly measure occlusion.

The solver undistorts keypoints, triangulates candidate camera pairs, and retains reprojection-consistent observations. Confidence below 0.35 and invalid geometry are excluded. The visibility mode uses 20 px to enter the camera consensus and 25 px to retain an existing member. Weight smoothing reduces abrupt camera changes while immediately removing missing observations. Accepted views contribute with unequal weights rather than a fixed equal average or one winning camera per finger.

The fitter initially requires at least eight triangulated joints, including at least three from wrist and the four non-thumb finger bases (joint indexes 0, 5, 9, 13, 17). It records confidence and triangulation reliability so poorly supported 3D points exert less influence.

### 3. Fit one articulated MANO hand

[hand/fit.py](../src/mchoi/hand/fit.py) optimizes the wrist translation, global orientation, and 15 local rotations for supported frames. It fixes one shape for the sequence: the median WiLoR shape from high-confidence target-hand detections in ZED left, with a fallback to all left-view detections if none meet the selection rule. This avoids changing finger lengths every frame.

The per-frame rotation initialization comes from the first available handedness-consistent selected camera in the stored camera order, then is transformed into the reference convention. Translation starts from the median difference between triangulated joints and initialized MANO joints. Subsequent optimization uses all accepted weighted measurements.

The selected fitting objective is:

```text
L = L_2D + 0.35 L_3D + 0.15 L_pose_prior
    + 4 L_world_temporal + 3 L_finger_temporal + 0.25 L_rotation_temporal
```

| Term | What it constrains |
| --- | --- |
| `L_2D` | Robust reprojection agreement with accepted camera keypoints, weighted by per-joint view confidence. |
| `L_3D` | Robust agreement with triangulated joints, scaled by their reliability. |
| `L_pose_prior` | Local rotations remain near the WiLoR initialization. |
| `L_world_temporal` | World-coordinate joint positions remain near timestamp-weighted interpolation between neighboring frames. |
| `L_finger_temporal` | Finger articulation is smooth after removing wrist translation, global rotation, and handedness mirroring. Weakly supported fingers receive stronger regularization. |
| `L_rotation_temporal` | Local finger rotations change smoothly, using matrix differences rather than Euler angles. |

The robust losses use normalized scales (15 px for reprojection, 20 mm for triangulated targets, 6 mm for world temporal motion, 4 mm for local articulation, and 0.06 for rotation-matrix entries). The weights above therefore apply to these normalized terms. The optimizer uses a continuous 6D rotation parameterization converted to proper rotation matrices, Adam, 300 iterations, and cosine learning-rate decay.

Temporal terms use actual timestamps and only consecutive supported intervals shorter than 80 ms. They use both preceding and following frames, so this is an **offline fit**. They are not One Euro filters.

MANO produces 778 vertices and 21 output joints. The implementation subtracts the MANO wrist, mirrors X for a left hand, and adds the fitted wrist translation. It uses the supplied right-hand MANO model for both sides following the WiLoR convention. A fit is accepted only with finite geometry, supporting-camera joint depths above 5 cm, and median accepted-observation reprojection residual below 35 px. Rejected metric geometry is stored as NaN with `valid=False`.

### 4. Apply the optional pose One Euro filter

[hand/filter.py](../src/mchoi/hand/filter.py) and [hand/rotation_filter.py](../src/mchoi/hand/rotation_filter.py) filter the fitted rotations on SO(3). The selected trial uses finger minimum cutoff 1 Hz/beta 1 and wrist-orientation minimum cutoff 2.5 Hz/beta 3, with derivative cutoff 1 Hz. Each joint has independent angular-velocity state and uses shortest-path rotation interpolation.

The filter preserves the original subject shape and wrist translation, regenerates MANO vertices/joints, and repeats the acceptance checks. It resets across source-invalid frames, frame discontinuities, handedness changes, or gaps over 100 ms. It does not fill missing poses. The filter itself is causal; the complete reconstruction remains offline because of its source fit.

### 5. Draw and save the hand annotation

[hand/render.py](../src/mchoi/hand/render.py) projects accepted vertices and joints through the saved camera calibration onto each native ZED-left image. It draws mesh edges, finger chains, joint dots, and captions. Comparison videos select an earlier fit or the cached monocular prediction as the baseline. Invalid metric fits can show a labeled monocular fallback; that displayed fallback is not a valid multiview pose.

`fused_hand.npz` is the numerical annotation: `frame_index`, `timestamp_ns`, `valid`, metric `vertices`/`joints`, camera observation selections, support, and shape. `mano_rotations` and `wrist_translation` are indexed by `fit_frame_rows`, not directly by every video frame. Consult `valid` even for rows with saved parameters, since some optimized fits are subsequently rejected. Filtered exports also preserve unfiltered rotations and filter-reset rows.

## Object pose from ArUco markers

### 1. What the current object pose represents

The current implementation estimates **a separate six-degree-of-freedom pose for each decoded marker ID**. For the mustard recording, the reviewed IDs are 1, 2, 3, 4, and 5. IDs alone do not identify an object center or establish which markers belong to one rigid object.

The tracking marker model is a **24 mm black square**, excluding its white margin, from `DICT_4X4_50`. Its origin is the black-square center; X points decoded right, Y decoded up, and Z out of the printed face. The four object points, in detector order, are:

```text
(-0.012, +0.012, 0), (+0.012, +0.012, 0),
(+0.012, -0.012, 0), (-0.012, -0.012, 0)  metres
```

These dimensions differ from the ChArUco board used to calibrate the cameras. A marker attached rigidly to an object provides a marker-frame proxy for that object's movement. The current code does not recover an object mesh, infer its center, or merge different marker IDs into a rigid body.

### 2. Detect corners and build time groups

[aruco/extract_zed.py](../src/mchoi/aruco/extract_zed.py) and [aruco/extract_realsense.py](../src/mchoi/aruco/extract_realsense.py) detect marker IDs and subpixel-refined corners in native images. They save detections under `<take>/aruco_annotated/`, including explicit empty frames. Reviewed ID selections are saved in `marker_selection.json`; ambiguous duplicate instances of the same ID in one camera/frame are removed from pose observations.

[aruco/timing.py](../src/mchoi/aruco/timing.py) estimates constant RealSense time corrections from calibrated epipolar agreement, with held-out temporal blocks for validation. `dataset()` in [aruco/poses.py](../src/mchoi/aruco/poses.py) groups corrected frames within 20 ms without reuse. Unmatched RealSense images become additional single-camera groups. Consequently, marker `group_index` is not interchangeable with ZED video `frame_index`.

The historical object exporter retains geometrically acceptable cameras even when timing is unverified, applying zero correction and recording that uncertainty. The hand pipeline can instead exclude unvalidated RealSense views. Always consult each export's timing metadata; identical recordings do not guarantee identical camera inclusion across the two pipelines.

### 3. Estimate rigid marker pose

For each group and marker ID, `Geometry` in [aruco/poses.py](../src/mchoi/aruco/poses.py) builds monocular candidates using `solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` and refines them with `solvePnPRefineLM`. A flat square can have competing pose branches. Each candidate is converted from camera coordinates to ZED-left coordinates:

```text
T_Z_from_marker = inverse(T_C_from_Z) @ T_C_from_marker
```

For multiview estimation, the solver tests camera subsets from largest to smallest. Each pair must have symmetric epipolar RMS at most 3 px. It jointly optimizes one rigid-square pose against the measured corner pixels in the subset, initialized from monocular candidates and triangulated-corner rigid alignment:

```text
minimize over R_ZM, t_ZM:
    sum over cameras C and corners j
    || project(K_C, D_C, T_CZ @ (R_ZM @ corner_M[j] + t_ZM))
       - measured_pixel[C,j] ||^2
```

This uses equally weighted pixel residuals and SciPy Levenberg–Marquardt. Robustness comes from camera-subset selection and rejection gates, not from the hand pipeline's per-finger weighting. Accepted poses must have positive corner depth, a front-facing printed square, RMS at most 2 px, and maximum corner residual at most 5 px. Competing solutions differing by more than 5 degrees but less than 0.25 px in RMS cause the subset to be rejected as ambiguous. The lowest-RMS accepted fit from the largest passing camera-count subsets is selected.

### 4. Resolve temporal branch ambiguity

[aruco/track.py](../src/mchoi/aruco/track.py) performs offline dynamic programming over measured candidates for each ID. Accepted multiview fits are fixed candidate states. Monocular branches are chosen using reprojection error plus rotation/translation continuity costs. Sequences split after more than 250 ms without a candidate.

This **selects candidate poses; it does not One Euro filter or average object poses**. Missing measurements stay missing. A forward gate rejects marker-normal changes exceeding 45 degrees within 100 ms. Monocular sequences without a multiview anchor are explicitly labeled uncertain because temporal continuity cannot establish the correct absolute planar orientation.

### 5. Draw and save the object annotation

The object exporter's `overlay()` draws detected marker outlines and projects marker-frame axes into each camera. X is red, Y green, and Z blue. Axes are drawn only for valid poses in contributing views where that marker is detected. A table shows all selected IDs' shared ZED-frame positions in millimetres, roll/pitch/yaw in degrees, reprojection error, method, and contributing cameras. Table positions are not coordinates in each panel's own camera frame.

The exporter creates `zed_left_annotated.mp4`, `zed_right_annotated.mp4`, `d435_annotated.mp4`, `d455_annotated.mp4`, and `all_cameras_annotated.mp4`. The montage uses ZED-left time groups, with blank tiles for unavailable aligned frames. Individual exports retain their native stream timing.

| Data file | Annotation meaning |
| --- | --- |
| `poses.csv` | One final pose or explicit invalid status per time group and selected ID. |
| `per_camera_poses.csv` | Maps native camera/frame indexes to the shared group pose, with detection/use flags and timing differences. |
| `monocular_candidates.csv` | Retained alternative branches in ZED-left coordinates. |
| `independent_estimates.csv` | Provisional estimates saved before final offline branch tracking. |
| `aligned_frames.csv` | Source frame membership and corrected-time offsets for each group. |
| `pose_metadata.json` | Marker size, coordinate conventions, algorithms, source paths, and limitations. |

CSV translation `tx_m,ty_m,tz_m` is in metres. `R00` through `R22` form the marker-to-reference rotation. `rx_rad,ry_rad,rz_rad` are a Rodrigues rotation vector, **not Euler angles**. Displayed Euler angles follow `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Use matrices for transform composition.

The detailed [ArUco processing reference](pose_estimation_flow/processing_details.md) and [diagram viewer](pose_estimation_flow/index.html) describe the existing object export. Their batch results and path names are historical; the mustard take has since been renamed. Object tracking/validation also reads `pose_metadata.source_recording` and `source_calibration`, so those embedded paths must resolve when reusing moved exports.

## Relating the annotations

For a ZED-left video frame, match the hand's `frame_index` to `per_camera_poses.csv` rows with `camera=zed_left` and the same native frame index. Select the intended marker ID, check both validity flags, and verify source timestamps and identical calibration. Do not join by CSV row number or marker group number. If the hand cache starts partway through a take, its array row zero is not source frame zero.

For a valid marker pose, a hand joint expressed relative to that marker is:

```text
p_marker = R_ZM.T @ (p_hand_Z - t_ZM)
```

To obtain a pose at a chosen object origin, an additional measured rigid transform is required. If `T_M_from_O` maps object coordinates into marker coordinates, then:

```text
T_Z_from_O = T_Z_from_M @ T_M_from_O
```

That marker-to-object calibration and object-ID mapping are not currently implemented. Neither are a combined hand/object renderer, rigid multi-marker object fitting, contact detection, or collision-constrained hand fitting. These formulas describe how the saved annotations can be related, not extra processing already performed by the videos.

Validation checks geometry, source-frame accounting, and video timing through [hand/validate.py](../src/mchoi/hand/validate.py) for hands and [aruco/validate.py](../src/mchoi/aruco/validate.py) for markers. Low reprojection error and smooth movement demonstrate consistency with the input observations, not ground-truth pose accuracy, object contact, or correct hidden-finger reconstruction.
