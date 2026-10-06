"""Generate PC-clock frame indexes and alignment for a new or existing take."""
import argparse
import json
from pathlib import Path

from mchoi.recorder.timestamps import export_timestamps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tolerance-ms", type=float, default=20)
    args = parser.parse_args()
    print(json.dumps(export_timestamps(args.directory, args.tolerance_ms), indent=2), flush=True)
