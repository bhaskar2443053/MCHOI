"""Read the RGB images of a native RealSense DB3 (ROS 2 / SQLite) recording, read-only."""
import sqlite3
import struct
from pathlib import Path

import numpy as np

from mchoi.recorder.timestamps import parse_metadata, milliseconds_to_ns


def decode_image(blob):
    if blob[:4] != b'\x00\x01\x00\x00':
        raise ValueError('Expected little-endian CDR image')
    offset = 12  # encapsulation, seconds, nanoseconds

    def u32():
        nonlocal offset
        offset = (offset + 3) // 4 * 4
        value = struct.unpack_from('<I', blob, offset)[0]
        offset += 4
        return value

    def string():
        nonlocal offset
        length = u32()
        value = blob[offset:offset + length - 1].decode('utf-8')
        if blob[offset + length - 1] != 0:
            raise ValueError('Malformed CDR string')
        offset += length
        return value

    string()  # frame_id
    height, width = u32(), u32()
    encoding = string()
    big_endian = blob[offset]
    offset += 1
    step, length = u32(), u32()
    if encoding != 'bgr8' or big_endian != 0 or step != width * 3 or length != height * step or offset + length != len(blob):
        raise ValueError(f'Unsupported RGB image {encoding} {width}x{height} step={step}')
    return np.frombuffer(blob, np.uint8, count=length, offset=offset).reshape(height, width, 3)


class ColorReader:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        topic, name = self.db.execute("SELECT id,name FROM topics WHERE name LIKE '%/Color_%/image/data'").fetchone()
        meta_topic = self.db.execute('SELECT id FROM topics WHERE name=?', (name.replace('/data', '/metadata'),)).fetchone()[0]
        # Sort small indices, never multi-gigabyte image blobs in a temporary table.
        self.rows = self.db.execute('SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topic,)).fetchall()
        meta = self.db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (meta_topic,)).fetchall()
        if [r[1] for r in self.rows] != [r[0] for r in meta]:
            raise ValueError('Image/metadata mismatch')
        self.timestamps = [milliseconds_to_ns(parse_metadata(blob)['system_time']) for _, blob in meta]

    def read(self, index):
        blob = self.db.execute('SELECT data FROM messages WHERE id=?', (self.rows[index][0],)).fetchone()[0]
        return decode_image(blob)

    def close(self):
        self.db.close()
