"""Index recorded frames on the PC clock and match them without inventing frames."""
from bisect import bisect_left
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import struct

from .devices import load_zed

FRAME_FIELDS = ["camera", "stream", "frame_index", "device_frame_number", "pc_timestamp_ns",
                "pc_local_time", "timestamp_source", "native_timestamp", "native_timestamp_domain", "source_file"]


def local_iso(timestamp_ns):
    seconds, nanos = divmod(int(timestamp_ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, timezone.utc).astimezone()
    offset = dt.strftime("%z")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")+f".{nanos:09d}"+offset[:3]+":"+offset[3:]


def milliseconds_to_ns(value):
    return int(Decimal(str(value))*1_000_000)


def parse_metadata(blob):
    if len(blob) < 9 or blob[:2] != b"\x00\x01":
        raise ValueError("Unsupported native RealSense metadata serialization")
    length = struct.unpack_from("<I", blob, 4)[0]
    if length < 1 or 8+length > len(blob):
        raise ValueError("Truncated native RealSense metadata")
    text = blob[8:8+length-1].decode("utf-8")
    return dict(item.split("=", 1) for item in text.split(";") if "=" in item)


def realsense_frames(path, camera):
    """Use SDK host times persisted for EVERY image, including frames the UI did not observe."""
    frames = []
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True) as db:
        topics = db.execute("SELECT id,name FROM topics WHERE name LIKE '%/image/metadata'").fetchall()
        for topic, name in topics:
            stream = "color" if "/Color_" in name else "depth" if "/Depth_" in name else None
            if stream is None:
                continue
            rows = db.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (topic,)).fetchall()
            image_topic = db.execute("SELECT id FROM topics WHERE name=?", (name.replace("/metadata", "/data"),)).fetchone()
            if image_topic is None:
                raise ValueError("Native image topic is missing for "+name)
            image_times = [r[0] for r in db.execute("SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id", image_topic)]
            if image_times != [r[0] for r in rows]:
                raise ValueError("Not every recorded RealSense image has matching timestamp metadata")
            for index, (_, blob) in enumerate(rows):
                metadata = parse_metadata(blob)
                if "system_time" not in metadata:
                    raise ValueError("Native RealSense frame has no saved PC system timestamp")
                timestamp = milliseconds_to_ns(metadata["system_time"])
                frames.append({"camera": camera, "stream": stream, "frame_index": index,
                               "device_frame_number": metadata.get("Frame number", ""), "pc_timestamp_ns": timestamp,
                               "pc_local_time": local_iso(timestamp), "timestamp_source": "RealSense SDK system_time",
                               "native_timestamp": metadata.get("timestamp", ""),
                               "native_timestamp_domain": metadata.get("timestamp_domain", ""), "source_file": path.name})
    if {r["stream"] for r in frames} != {"color", "depth"}:
        raise ValueError("Both RGB and depth frame timestamps are required")
    return frames


def zed_frames(path, ledger=None):
    sl = load_zed()
    camera = sl.Camera()
    params = sl.InitParameters()
    params.set_from_svo_file(str(path.resolve()))
    params.svo_real_time_mode = False
    params.depth_mode = sl.DEPTH_MODE.NONE
    params.sdk_verbose = 0
    params.maximum_working_resolution = sl.Resolution(64, 36)
    result = camera.open(params)
    if result > sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Cannot index SVO timestamps: "+str(result))
    try:
        def at(index):
            camera.set_svo_position(index)
            result = camera.read()
            if result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                return None
            if result > sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("Cannot read SVO timestamp: "+str(result))
            return camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        timestamps = []
        method = "native_svo_decode"
        # Our live ledger includes every successful grab, even encoder warm-up.
        # Check it against native frame count, first/middle/last frames. This avoids
        # decoding an entire long video just to export its timestamp index.
        if ledger and ledger.exists():
            with ledger.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows and rows[0].get("pc_timestamp_ns"):
                first = at(0)
                final_index = camera.get_svo_number_of_frames()-1
                last = at(final_index)
                if last is None and final_index > 0:
                    final_index -= 1
                    last = at(final_index)
                if first is not None and last is not None:
                    # SVO2 in SDK 5.3 stores microseconds even though live SDK
                    # timestamps can contain sub-microsecond digits.
                    quantum = 1000 if first % 1000 == 0 and last % 1000 == 0 else 1
                    native_precision = [int(r["pc_timestamp_ns"])//quantum*quantum for r in rows]
                    candidate = [t for t in native_precision if first <= t <= last]
                    if len(candidate) == final_index+1 and all(a < b for a, b in zip(candidate, candidate[1:])):
                        samples = {0, final_index//3, final_index//2, final_index*2//3, final_index}
                        if all(at(i) == candidate[i] for i in samples):
                            timestamps = candidate
                            method = "live_sdk_timestamps_checked_against_native_svo"
        if not timestamps:
            camera.set_svo_position(0)
            while True:
                result = camera.read()
                if result == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                if result > sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError("Cannot read SVO timestamp: "+str(result))
                timestamps.append(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        frames = [{"camera": "zed", "stream": stream, "frame_index": i, "device_frame_number": "",
                   "pc_timestamp_ns": timestamp, "pc_local_time": local_iso(timestamp),
                   "timestamp_source": "ZED SDK IMAGE timestamp; PC system clock",
                   "native_timestamp": str(timestamp), "native_timestamp_domain": "PC system clock (Unix ns)",
                   "source_file": path.name}
                  for i, timestamp in enumerate(timestamps) for stream in ("left", "right")]
        return frames, method
    finally:
        camera.close()


def align_frames(frames, tolerance_ms=20):
    if not 0 < tolerance_ms <= 100:
        raise ValueError("Alignment tolerance must be greater than zero and at most 100 ms")
    groups = {}
    for frame in frames:
        groups.setdefault((frame["camera"], frame["stream"]), []).append(frame)
    for key, group in groups.items():
        group.sort(key=lambda r: r["frame_index"])
        if any(a["pc_timestamp_ns"] >= b["pc_timestamp_ns"] for a, b in zip(group, group[1:])):
            raise ValueError(f"Non-monotonic PC timestamps in {key}; clock adjustment or duplicate timestamps require review")
    anchor = groups.pop(("zed", "left"))
    groups.pop(("zed", "right"))
    times = {key: [r["pc_timestamp_ns"] for r in group] for key, group in groups.items()}
    next_unused = {key: 0 for key in groups}
    output = []
    for reference in anchor:
        t = reference["pc_timestamp_ns"]
        row = {"zed_frame_index": reference["frame_index"], "zed_pc_timestamp_ns": t,
               "zed_pc_local_time": reference["pc_local_time"]}
        matched = [t]
        all_matched = True
        for key, group in groups.items():
            label = key[0]+"_"+key[1]
            i = bisect_left(times[key], t)
            candidates = [j for j in (i-1, i) if next_unused[key] <= j < len(group)]
            j = min(candidates, key=lambda j: abs(times[key][j]-t)) if candidates else None
            if j is not None and abs(times[key][j]-t) <= tolerance_ms*1e6:
                entry = group[j]
                row.update({label+"_frame_index": entry["frame_index"], label+"_device_frame_number": entry["device_frame_number"],
                            label+"_pc_timestamp_ns": entry["pc_timestamp_ns"], label+"_offset_ms": (times[key][j]-t)/1e6})
                next_unused[key] = j+1
                matched.append(times[key][j])
            else:
                all_matched = False
                for suffix in ("_frame_index", "_device_frame_number", "_pc_timestamp_ns", "_offset_ms"):
                    row[label+suffix] = ""
        row.update(all_streams_matched=all_matched, group_time_span_ms=(max(matched)-min(matched))/1e6)
        output.append(row)
    return output


def write_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_timestamps(directory, tolerance_ms=20):
    directory = Path(directory)
    frames, method = zed_frames(directory/"zed.svo2", directory/"zed_timestamps.csv")
    for path in sorted(directory.glob("realsense_*.db3")):
        frames.extend(realsense_frames(path, path.stem))
    alignment = align_frames(frames, tolerance_ms)
    frames.sort(key=lambda r: (r["camera"], r["stream"], r["frame_index"]))
    write_csv(directory/"frame_timestamps.csv", FRAME_FIELDS, frames)
    if not alignment:
        raise ValueError("No ZED frames available for time alignment")
    write_csv(directory/"frame_alignment.csv", list(alignment[0]), alignment)
    summary = {"clock": "PC Unix wall clock, nanoseconds", "local_timezone": datetime.now().astimezone().tzname(),
               "alignment": "Nearest unused frame for each stream, anchored on ZED stereo frames",
               "tolerance_ms": tolerance_ms, "timestamp_rows": len(frames), "zed_frames": len(alignment),
               "fully_matched_rows": sum(r["all_streams_matched"] for r in alignment), "zed_index_method": method,
               "hardware_synchronized": False,
               "limitation": "PC timestamps align SDK frame arrival times. Transport latency, sensor readout and exposure timing can differ. Matching timestamps do not prove simultaneous exposure.",
               "unmatched": "Blank cells mean no unused frame fell within tolerance; frames are never duplicated or invented."}
    (directory/"timestamp_alignment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
