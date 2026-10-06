"""Read exactly n bytes from a decoder subprocess pipe."""


def read_exact(pipe, count):
    data = bytearray()
    while len(data) < count:
        chunk = pipe.read(count - len(data))
        if not chunk:
            raise EOFError(f'ZED stream ended after {len(data)} of {count} bytes; see zed_decode.log')
        data.extend(chunk)
    return data
