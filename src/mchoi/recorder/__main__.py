"""Open the recording UI or create a finite native recording for validation."""
import argparse
import json
from pathlib import Path
import time

import cv2
from mchoi.paths import data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Recording folder (default: recordings/)")
    parser.add_argument("--offline", action="store_true", help="Open UI without cameras")
    parser.add_argument("--test-seconds", type=float, help="Record a finite test take and exit")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    workspace = data_root()
    if args.test_seconds is not None:
        if not 1 <= args.test_seconds <= 60:
            parser.error("--test-seconds must be between 1 and 60")
        from mchoi.recorder.controller import Recorder
        recorder = Recorder(workspace, args.output)
        try:
            recorder.connect()
            # Let automatic exposure settle before preparing the native files.
            time.sleep(1)
            directory = recorder.start_recording()
            deadline = time.monotonic()+args.test_seconds
            while time.monotonic() < deadline:
                error = recorder.health_error()
                if error:
                    raise RuntimeError(error)
                time.sleep(.1)
            recorder.stop_recording()
            print(json.dumps({"directory": str(directory), "status": recorder.manifest["status"],
                              "error": recorder.error}, indent=2), flush=True)
            return 0 if recorder.manifest["status"] == "complete" else 1
        except BaseException as exc:
            recorder.stop_recording(str(exc) or type(exc).__name__)
            raise
        finally:
            recorder.close()
    import tkinter as tk
    from mchoi.recorder.app import RecordingApp
    root = tk.Tk()
    RecordingApp(root, workspace, args.output, args.offline)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
