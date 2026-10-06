import struct

import pytest

from mchoi.recorder.timestamps import align_frames, local_iso, milliseconds_to_ns, parse_metadata


def frame(camera, stream, index, ns):
    return {"camera": camera, "stream": stream, "frame_index": index, "device_frame_number": index+100,
            "pc_timestamp_ns": ns, "pc_local_time": local_iso(ns)}


def test_alignment_keeps_frame_identity_across_missing_frames():
    times = [0, 33_333_333, 66_666_666, 100_000_000]
    frames = [frame("zed", view, i, t) for view in ("left", "right") for i, t in enumerate(times)]
    frames += [frame("rs", "color", i, t+2_000_000) for i, t in enumerate([times[0], times[2], times[3]])]
    rows = align_frames(frames, 20)
    assert [r["rs_color_frame_index"] for r in rows] == [0, "", 1, 2]
    assert rows[1]["all_streams_matched"] is False
    assert rows[2]["rs_color_offset_ms"] == 2


def test_nearest_frame_is_not_reused_even_inside_overlapping_tolerance():
    frames = [frame("zed", view, i, t) for view in ("left", "right") for i, t in enumerate((0, 33_333_333))]
    frames.append(frame("rs", "color", 0, 16_000_000))
    rows = align_frames(frames, 20)
    assert rows[0]["rs_color_frame_index"] == 0
    assert rows[1]["rs_color_frame_index"] == ""


def test_clock_regression_is_not_silently_sorted_away():
    frames = [frame("zed", view, i, t) for view in ("left", "right") for i, t in enumerate((10, 5))]
    with pytest.raises(ValueError, match="Non-monotonic"):
        align_frames(frames)


def test_native_host_timestamp_preserves_decimal_precision_and_timezone():
    text = b'Frame number=22;system_time=1790084900232.808105;timestamp_domain=System Time;\x00'
    blob = b'\x00\x01\x00\x00'+struct.pack('<I', len(text))+text
    metadata = parse_metadata(blob)
    assert milliseconds_to_ns(metadata["system_time"]) == 1790084900232808105
    iso = local_iso(1790084900232808105)
    assert '.232808105' in iso
    assert iso[-6] in '+-' and iso[-3] == ':'
