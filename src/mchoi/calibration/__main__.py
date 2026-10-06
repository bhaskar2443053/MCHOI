"""Start the desktop app, or run a finite live-camera diagnostic with --probe."""
import argparse
from pathlib import Path
import sys
import time

import cv2
from mchoi.paths import CONFIG_DIR, data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Open UI without connecting cameras")
    parser.add_argument("--probe", action="store_true", help="Measure all streams and detect the board, then exit")
    parser.add_argument("--probe-seconds", type=float, default=6)
    parser.add_argument("--calibration", type=Path, help="Load a calibration JSON/YAML and open click verification")
    parser.add_argument("--session", type=Path, help="Resume an existing capture session")
    parser.add_argument("--show-roi", action="store_true", help="Show calibration coverage after loading")
    args = parser.parse_args()
    workspace = data_root()
    cv2.setNumThreads(2)
    if args.probe:
        from mchoi.calibration.board import BoardDetector, load_board
        from mchoi.calibration.cameras import CameraRig
        from mchoi.calibration.storage import write_json
        rig = CameraRig()
        try:
            rig.start()
            deadline = time.monotonic()+20
            while not rig.batch() and time.monotonic() < deadline:
                time.sleep(.1)
            time.sleep(max(1, args.probe_seconds))
            metadata = rig.metadata()
            _, board = load_board(CONFIG_DIR/"board_config.json")
            detector = BoardDetector(board)
            detections = {key: detector.detect(value["image"]) for key, value in rig.batch().items()}
            result = {"cameras": metadata, "detections": detections}
            write_json(workspace/"camera_diagnostic.json", result)
            for key, info in metadata.items():
                print(f"{key}: {info['image_size']}, {info['measured_fps']:.2f} fps, USB {info.get('usb', 'UVC')}, "
                      f"{len(detections.get(key, {}).get('ids', []))}/24 corners, error={info['error']}", flush=True)
            if len(detections) != 4 or any(info["error"] for info in metadata.values()):
                return 1
        finally:
            rig.stop()
        return 0
    import tkinter as tk
    from mchoi.calibration.app import CalibrationApp
    root = tk.Tk()
    app = CalibrationApp(root, workspace, args.offline)
    if args.session:
        root.after(50, lambda: app.resume_session(args.session))
    if args.calibration:
        root.after(100, lambda: app.load_calibration_file(args.calibration))
    if args.show_roi:
        root.after(200, app.toggle_coverage)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
