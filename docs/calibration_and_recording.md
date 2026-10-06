# Calibration and recording apps

For the hand–object pipeline see [pipeline.md](pipeline.md); for installation the [README](../README.md).

A small Windows desktop app for two RealSense cameras and a ZED Mini. This PC now uses two D435 cameras; the second replaced the D455. The ZED's two lenses are calibrated separately, giving four optical views. It captures ChArUco observations, estimates each view's intrinsics and lens distortion, and calculates camera transforms in the ZED left coordinate frame.

Double-click **scripts\start_calibration.cmd** (or run `.\.venv-sdk\Scripts\python.exe -m mchoi.calibration`). Cameras connect automatically; close other apps using them first. Configuration (board, camera serials, recorder exposure) is read from `configs/`; sessions and recordings are written to the data root (`<data>`, see the README).

## Capture and calibrate

1. Keep all cameras fixed throughout a session. Put the printed board on a flat, rigid backing. The confirmed board settings are in `configs/board_config.json`: 5 × 7 squares, 30 mm squares, 22 mm markers, DICT_4X4_50, IDs 0–16.
2. Move the board through different positions, distances, and tilts. Hold it still for at least 0.6 seconds until the views show **Ready**.
3. Press **Space** or **Capture pose**. **Auto capture** tries every three seconds and skips repeated poses. Aim for 20–30 diverse captures; calibration requires at least 12 usable poses per view.
4. **Require all views** is enabled by default. Turn it off to collect intrinsics in individual views when the board cannot be seen by every camera. For rig alignment, also capture at least eight shared poses for every camera pair needed to connect the rig, with six or more common corners per pair. Not every camera pair has to overlap directly.
5. Press **Calibrate all**. Results include per-camera and stereo reprojection errors, coverage warnings, transforms, and board poses. Review the quality notes. A low reprojection error alone does not establish physical accuracy.

Drag a rectangle on a preview to restrict board detection if other printed boards interfere. Right-click to clear the rectangle. The detector also filters unrelated markers using the expected board geometry. **Undo last pose** excludes it without deleting its original files. **Resume session** loads saved captures; **New session** preserves previous sessions.

## Verify a saved calibration by clicking

1. Click **Load calibration…** and select this app's `calibration.json` or `calibration.yml`. Loading enables **Verify clicks** and freezes a matched set of camera views. The app checks resolution and, when available, camera serial numbers against the file.
2. Select **Automatic depth**, then click an object in any of the four views. A small red dot marks the predicted pixel in every camera where the 3D point projects inside the image. Displayed coordinates are native-image pixels. The black margins around a preview are not clickable image pixels.
3. Click **Refresh views** after moving an object, then select another point. **Clear dots** removes the selection. Turn off **Verify clicks** to return to live pose capture.

**Show calibration ROI / Hide calibration ROI** toggles a transparent green overlay in every view. It shows the union of the board-corner regions from the poses actually accepted for that camera's calibration, with the percentage of the image covered. Overlapping poses do not make the shading darker. This is sampled image coverage, not a guaranteed accuracy boundary or a 3D working volume. Click verification and red dots remain usable through the overlay. Original captures are not modified.

New JSON and YAML exports embed this coverage. Existing calibrations recover it from their original accepted `pose_*/pose.json` files; keep those folders with the session. Older YAML files also need their matching companion JSON. If coverage data is unavailable, the app reports it instead of inventing a region.

Automatic depth supports arbitrary objects without the board. RealSense depth samples are transformed into the calibrated RGB view; the app now streams depth at up to 848 × 480 / 30 fps alongside native RGB. ZED depth comes from the calibrated left/right pair using stereo rectification, SGBM matching, and a left/right consistency check. Stereo is calculated on half-resolution images in a background thread, then points are projected back into the native-resolution feeds with lens distortion included.

Depth may be unavailable on textureless, reflective, transparent, occluded, or boundary pixels. The app reports that condition instead of guessing a dot. Points behind a camera or outside its image are labeled. **Target-view occlusion is not checked:** a projected point on the front of a bottle can be hidden from a camera looking at the back. Use surface features visible in the compared views to judge alignment. Depth uncertainty contributes to projection error; this visual check is not an independent metrology test.

**Board plane** mode provides a second check: click inside the detected printed grid. The clicked view alone estimates the board plane and the loaded rig transforms project the point into the other views; target dots are not snapped to detections. This mode requires at least six board corners in the clicked view and does not assume other objects are on that plane.

Verification freezes the displayed images so that a click and its dots refer to the same image set. Acquisition continues in the background. Keep the scene stationary when refreshing because physical cameras are matched by host receipt times, not hardware synchronized.

## Camera modes

| Camera | Highest resolution at 30 fps |
|---|---|
| D435 #1 RGB · serial 261222078090 | 1920 × 1080 on USB 3 |
| D435 #2 RGB · serial 261222076358 | 1920 × 1080 on USB 3 |
| ZED Mini | 1920 × 1080 per lens; 3840 × 1080 combined |

The RealSense SDK enumerates actual 30 fps profiles and chooses the largest available RGB mode. On USB 2 the D435 may be limited to 640 × 480 at 30 fps. After changing a cable or port, disconnect/reconnect in the app, and start a new session if the resolution changes. The ZED requests its highest 30 fps mode; 2K is limited to 15 fps. Each panel displays measured acquisition fps. Live detection and previews update independently at a lower rate to keep acquisition responsive; saved images retain native resolution.

`configs/camera_config.json` assigns the two RealSense cameras by serial number, so identical models cannot swap views when Windows changes device order. Update this file after a physical replacement. Headings show the detected model, camera number, and final six serial digits. The stable file keys `d435` and `d455` are retained for older sessions; the `d455` slot now contains D435 #2. The model and serial in each capture and calibration identify the actual hardware. New YAML exports also include serials. Replacing a camera requires a new calibration; start a new session and retain the old files for the original hardware.

Acquisition uses native RealSense RGB streams and unrectified ZED UVC stereo images. Camera hardware is not rewritten. This calibrates RGB camera geometry and the rig. It does not perform RealSense depth self-calibration or overwrite ZED SDK calibration. For each RealSense, the export also composes the measured RGB transform with the device's factory depth-to-color transform to provide `T_reference_from_depth`; no depth images are needed for that composition.

## Timing and data

All physical cameras acquire concurrently at 30 fps. The ZED left and right images come from one stereo frame. Across physical cameras, the app selects frames nearest a common host receipt time and rejects pose captures with a spread over 80 ms, stale frames, fewer than eight corners, or observed board motion. **This is not hardware synchronization. Hold the board still during each capture.** Device timestamps and host receipt times are saved for inspection.

Each `<data>/sessions/<date>/pose_0001/` contains lossless native-resolution PNGs and `pose.json` with exact observed corners, IDs, device identity, resolution, and timestamps. The app rejects mixing resolutions or camera identities within a session.

Each calibration run writes a new results folder containing:

- `calibration.json`: complete intrinsics, distortion, metric transforms, errors, rejected observations, and each captured board's estimated pose.
- `calibration.yml`: OpenCV matrices, image sizes, status, units, and transform convention.
- `summary.txt`: readable quality summary.

Transforms use OpenCV optical coordinates (x right, y down, z forward) and column vectors. `T_reference_from_camera @ p_camera = p_reference`; the reference is the ZED left optical frame. Distortion uses OpenCV's five-coefficient Brown model with k3 fixed to zero. Intrinsics are fit independently, then common ChArUco corners solve pairwise stereo geometry with fixed intrinsics. A connected graph combines pairwise transforms and reports disagreement among alternate paths. Calibration can flag limited tilt/coverage, excessive residual error, or inconsistent geometry as **review required**.

## Diagnostics and tests

The SDK environment (`scripts\setup_sdk_env.ps1`, see the README) runs both apps. A finite camera probe and the tests:

```powershell
.\.venv-sdk\Scripts\python.exe -m mchoi.calibration --probe
.\.venv-sdk\Scripts\python.exe -m pytest -q
```

The probe saves `<data>/camera_diagnostic.json` and releases the cameras when done. Logs from the launchers are saved under `logs/`.

## Record video from all cameras

Double-click **scripts\start_recorder.cmd** (or run `.\.venv-sdk\Scripts\python.exe -m mchoi.recorder`) to open the separate recording app. Close the calibration window or other camera applications first. The app connects the ZED Mini and both configured RealSense cameras; its single live preview is **RealSense camera 2, serial 261222076358**.

Press **Record** once to prepare and start all three physical cameras (four optical views). The button becomes **Stop recording**. Press it again to finish the take and flush all native files. Closing the window also finalizes an active take. Preparation and finalization can take a few seconds; the preview briefly pauses when its pipeline is reopened. Click the save-folder path to open the take in Explorer.

**Exposure settings** opens manual exposure and gain controls for ZED stereo, each RealSense RGB camera, and each RealSense depth sensor. Automatic exposure is disabled for all of them. Settings are saved in `configs/recording_settings.json`, reapplied when native recorders reopen, and included with actual device readbacks in each take. Controls are disabled during recording. The dialog shows the values actually accepted by the hardware: on Windows, D435 RGB exposure is quantized (for example, a requested 10 ms can read back as 7.8 ms).

Default manual settings are D435 RGB 15.6 ms / gain 128, D435 depth 8.5 ms / gain 16, and ZED exposure 30% of the frame period / gain 60. Adjust these for the lighting and motion in your scene. A shorter exposure reduces motion blur; increasing gain brightens the image at the cost of noise. Focus on these devices is fixed and they expose no software focus control. The UI therefore describes fixed focus instead of offering an ineffective focus slider. These controls apply to the recorder; the calibration app retains its own exposure behavior.

Every take gets a new folder under `<data>/recordings/<date_time>/`:

- `zed.svo2`: native ZED SVO version 2, containing **both lenses at 1920 × 1080 / 30 fps**, timestamps and available sensor data. Hardware H.264 encoding is used. Depth can be computed when playing the SVO through the ZED SDK.
- `realsense_261222078090.db3` and `realsense_261222076358.db3`: one native RealSense recording per device, containing **BGR8 color at 1920 × 1080 / 30 fps** and **Z16 depth at 848 × 480 / 30 fps**, timestamps and factory calibration. The installed RealSense SDK 2.58.4 records native DB3 files instead of legacy BAG files. These are RGB/depth sequences, not ordinary video-player files; read them with `pyrealsense2` playback.
- `recording.json`: requested modes, device serials, per-camera factory intrinsics/extrinsics, depth scale, start/stop times, observed frame statistics, file sizes and completion status.
- `*_timestamps.csv`: observed device frame numbers and timestamps with host UTC/monotonic timestamps. The native SDK writes independently, so the CSV observer can see fewer frames than are present in the recording.
- `frame_timestamps.csv`: the complete saved-frame index on the PC clock, covering both ZED views and both RealSense RGB/depth streams. Columns include native file frame index, device frame number when available, integer Unix nanoseconds, local ISO time with timezone offset, timestamp source and the original native timestamp.
- `frame_alignment.csv`: one row per ZED stereo frame, with the closest unused RGB and depth frame from each RealSense camera, each frame's PC timestamp, signed time difference, and total time span of the matched group. Default tolerance is ±20 ms from the ZED timestamp. Blank entries explicitly mark unmatched frames; a dropped frame never shifts all subsequent matches or causes another frame to be reused.
- `timestamp_alignment.json`: clock source, matching tolerance, indexed/matched frame counts and synchronization limitations.

**All frame indexes use the PC wall clock.** The human-readable column uses this PC's local timezone (currently UTC+04:00); the integer timestamp is Unix nanoseconds, independent of timezone. RealSense times come from the SDK `system_time` metadata recorded with every native image, including images not observed by the UI. ZED times come from the SDK's image timestamps on the PC system clock. The CSVs preserve available precision; nanosecond notation does not imply nanosecond synchronization accuracy.

After Stop, the app finalizes the native files and builds the timestamp indexes and matching table. For new SVO recordings it checks the live SDK timestamp ledger against native frame count and first/intermediate/last frames to avoid decoding the entire video; if those checks disagree, it reads timestamps directly from every saved SVO frame. Older recordings use the native-file path. This extra indexing step runs in the background while the UI remains responsive.

PC timestamps describe SDK frame arrival/image timing. USB latency, sensor readout and exposure timing can differ between cameras. Timestamp matching provides **post-recording alignment, not hardware-synchronized exposure**. Retain the original timestamp columns and review the signed offsets when working with moving scenes.

Validation on this PC: all 407 indexed ZED timestamps in a test take matched a full native SVO decode exactly. A subsequent live take indexed every saved RGB/depth image and matched all 121 ZED frames within tolerance. The original user take was also indexed: 415 of 416 ZED frames had matches in all RealSense streams. Twenty-two automated tests cover cleanup, camera settings, missing-frame detection, timestamp precision, clock regressions, and matching without frame reuse.

Generate or rebuild the indexes for an existing recording with:

```powershell
.\.venv-sdk\Scripts\python.exe -m mchoi.recorder.align <data>\recordings\<take_folder> --tolerance-ms 20
```

RealSense depth is **native and unaligned**. Its size remains the same as in calibration verification; it is not upscaled to 1080p. Convert depth to metres by multiplying Z16 samples by `depth_units_m`, and use the saved factory depth-to-color transform when aligning it with RGB. Zero depth denotes a missing measurement. RealSense frames are saved uncompressed to preserve their original values, using approximately **25 GB per minute for the two cameras together**; output goes to the workspace drive. The app stops all recordings if a camera fails or disk space falls below 1 GB, retains partial files, and marks the take interrupted.

All cameras request 30 fps. The UI shows measured acquisition rates; actual rates and frame gaps depend on USB transport and disk throughput. Start/stop is coordinated in software, **not hardware synchronized**. Files can have different frame counts and slightly different first/last timestamps. Use native timestamps and the CSV clock observations for alignment; do not pair files solely by frame index. The preview refreshes independently at about 12 fps without resizing the recorded images.

The recorder now detects timestamp gaps over 50 ms at the requested 30 fps, displays the ZED gap count live, and saves gap locations and estimated missing frame intervals in `recording.json`. Gap estimates are distinct from SDK counters and observed frame-number jumps. A complete take means files were finalized, not that every requested frame arrived. The playback validator reports native stream gaps separately and only returns `passed: true` when playback, frame rate and continuity checks all pass. It does not insert duplicate frames or fabricate missing images.

The user take `20260922_172956_461334` contains an actual 66.63 ms jump between SVO indices 312 and 313. Tests reproduced occasional gaps with the ZED alone, and more frequent gaps with both RealSense cameras recording; manual exposure and a finer Windows timer did not eliminate them. Windows currently places all three cameras on the same Intel USB controller even though another controller is present. Controller/USB load is a suspect, not a proven sole cause; moving the ZED to the other controller requires a new capture test. Existing recordings cannot recover frames that were never captured.

The recorder additionally needs the installed ZED SDK and its matching Python API (`pyzed`). The tested setup is ZED SDK 5.3.1 with the `pyzed` 5.3 wheel for Python 3.14, pinned in `environment/requirements-sdk.txt`. On a different ZED SDK or Python version, use the SDK's `get_python_api.py` installer to obtain a matching wheel. The SDK is optional for the calibration app.

For a short live test and playback validation:

```powershell
.\.venv-sdk\Scripts\python.exe -m mchoi.recorder --test-seconds 6 --output <data>\recorder_live_tests
.\.venv-sdk\Scripts\python.exe -m mchoi.recorder.validate <data>\recorder_live_tests\<take_folder>
```

The playback check reads both SVO views and both native RealSense recordings, verifies sizes, formats, nonzero metric depth, serials, stored frame counts and timestamp-derived frame rates, and writes `playback_validation.json`. It records any difference between the ZED SDK's reported frame total and the number actually decoded. Recorder logs are `logs/recorder.stdout.log` and `logs/recorder.stderr.log`.

Live UI tests on this PC verified Record/Stop, a second take, continued preview updates, and closing during an active recording. Playback measured approximately 29.97–29.98 fps for RealSense RGB/depth and 28.60–29.43 fps for the ZED at the requested 30 fps. SDK 5.3.1 reported one more indexed SVO frame than it decoded in these tests; the validation report retains both counts. These early tests verified readability and average rate; the stricter continuity check added after the reported ZED gap also rejects individual missing frame intervals.

Native format references: [Stereolabs SVO recording](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording), [RealSense recording and playback](https://github.com/realsenseai/librealsense/blob/master/examples/record-playback/readme.md).

References: [ZED camera modes](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-controls), [D435 specifications](https://www.realsenseai.com/products/stereo-depth-camera-d435/), [RealSense comparison](https://www.realsenseai.com/compare-depth-cameras/), [OpenCV ChArUco calibration](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html), [OpenCV camera projection and stereo reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
