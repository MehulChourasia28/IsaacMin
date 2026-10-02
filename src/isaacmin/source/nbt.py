"""Bounded, read-only Java big-endian NBT decoder.

Source names are data. This module never evaluates expressions or runs datapacks.
"""
from __future__ import annotations

import gzip
import io
import struct
import zlib
from pathlib import Path


class SourceError(ValueError):
    """A source format or integrity failure; never interpreted as missing air."""


MAX_NBT_BYTES = 64 * 1024 * 1024


def decompress(data: bytes, codec: int, limit: int = MAX_NBT_BYTES) -> bytes:
    if codec == 3:
        result = data
    elif codec in (1, 2):
        stream = zlib.decompressobj(31 if codec == 1 else 15)
        result = stream.decompress(data, limit + 1)
        if len(result) > limit or stream.unconsumed_tail:
            raise SourceError("Decompressed NBT exceeds size limit")
        result += stream.flush()
        if not stream.eof or stream.unused_data:
            raise SourceError("Incomplete or trailing compressed NBT stream")
    else:
        raise SourceError(f"Unsupported Anvil compression codec {codec}; not empty terrain")
    if len(result) > limit:
        raise SourceError("NBT exceeds size limit")
    return result


class Reader:
    def __init__(self, data: bytes):
        if len(data) > MAX_NBT_BYTES:
            raise SourceError("NBT exceeds size limit")
        self.stream = io.BytesIO(data)
        self.length = len(data)
        self.nodes = 0

    def read(self, count: int) -> bytes:
        if count < 0 or count > self.length:
            raise SourceError("Invalid NBT size")
        result = self.stream.read(count)
        if len(result) != count:
            raise SourceError("Truncated NBT")
        return result

    def number(self, fmt: str):
        return struct.unpack(">" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        data = self.read(self.number("H"))
        # Java Modified UTF-8 allows encoded null and UTF-16 surrogate pairs.
        try:
            decoded = data.replace(b"\xc0\x80", b"\x00").decode("utf-8", "surrogatepass")
            return decoded.encode("utf-16", "surrogatepass").decode("utf-16")
        except UnicodeError as exc:
            raise SourceError("Invalid NBT string encoding") from exc

    def payload(self, tag: int, depth: int = 0):
        self.nodes += 1
        if depth > 128 or self.nodes > 4_000_000:
            raise SourceError("NBT nesting or node limit exceeded")
        if tag in (1, 2, 3, 4, 5, 6):
            return self.number({1: "b", 2: "h", 3: "i", 4: "q", 5: "f", 6: "d"}[tag])
        if tag == 8:
            return self.string()
        if tag in (7, 11, 12):
            count = self.number("i")
            if count < 0:
                raise SourceError("Negative NBT array length")
            if tag == 7:
                return self.read(count)
            fmt, size = ("i", 4) if tag == 11 else ("q", 8)
            data = self.read(count * size)
            return list(struct.unpack(">" + str(count) + fmt, data))
        if tag == 9:
            subtype, count = self.number("B"), self.number("i")
            if count < 0 or count > self.length or (subtype == 0 and count):
                raise SourceError("Invalid NBT list")
            return [self.payload(subtype, depth + 1) for _ in range(count)]
        if tag == 10:
            result = {}
            while True:
                subtype = self.number("B")
                if subtype == 0:
                    return result
                name = self.string()
                if name in result:
                    raise SourceError(f"Duplicate NBT tag {name!r}")
                result[name] = self.payload(subtype, depth + 1)
        raise SourceError(f"Unsupported NBT tag {tag}")


def loads(data: bytes) -> dict:
    reader = Reader(data)
    if reader.number("B") != 10:
        raise SourceError("NBT root must be a compound")
    reader.string()
    result = reader.payload(10)
    if reader.stream.tell() != reader.length:
        raise SourceError("Trailing bytes after NBT root")
    return result


def load(path: Path) -> dict:
    data = Path(path).read_bytes()
    return loads(decompress(data, 1) if data.startswith(b"\x1f\x8b") else data)
