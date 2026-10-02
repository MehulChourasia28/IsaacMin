"""Strict Anvil chunk reader with explicit coverage and palette semantics."""
from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .nbt import SourceError, decompress, loads

REGION_PATTERN = re.compile(r"r\.(-?\d+)\.(-?\d+)\.mca\Z")


@dataclass(frozen=True)
class Location:
    x: int
    z: int
    sector: int
    count: int
    timestamp: int


class Region:
    def __init__(self, path: Path):
        self.path = Path(path)
        match = REGION_PATTERN.fullmatch(self.path.name)
        if not match:
            raise SourceError("Invalid Anvil region filename")
        self.x, self.z = map(int, match.groups())
        size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            header = stream.read(8192)
        if len(header) != 8192 or size % 4096:
            raise SourceError(f"Truncated or unaligned Anvil region {self.path.name}")
        self.locations: dict[tuple[int, int], Location] = {}
        used = {0, 1}
        for slot in range(1024):
            value = struct.unpack_from(">I", header, slot * 4)[0]
            if value == 0:
                continue
            sector, count = value >> 8, value & 255
            if sector < 2 or count == 0 or (sector + count) * 4096 > size:
                raise SourceError(f"Invalid chunk sector in {self.path.name}:{slot}")
            occupied = set(range(sector, sector + count))
            if occupied & used:
                raise SourceError(f"Overlapping chunk sectors in {self.path.name}")
            used.update(occupied)
            cx, cz = self.x * 32 + slot % 32, self.z * 32 + slot // 32
            self.locations[cx, cz] = Location(cx, cz, sector, count, struct.unpack_from(">I", header, 4096 + slot * 4)[0])

    def payload(self, cx: int, cz: int) -> tuple[bytes, dict]:
        location = self.locations.get((cx, cz))
        if location is None:
            raise SourceError(f"Missing chunk ({cx}, {cz}); missing is not air")
        with self.path.open("rb") as stream:
            stream.seek(location.sector * 4096)
            length = struct.unpack(">I", stream.read(4))[0]
            marker = stream.read(1)[0]
            if length < 1 or length > location.count * 4096 - 4:
                raise SourceError(f"Invalid chunk payload length ({cx}, {cz})")
            external = bool(marker & 128)
            if external:
                if length != 1:
                    raise SourceError("External Anvil chunk has inline data")
                external_path = self.path.parent / f"c.{cx}.{cz}.mcc"
                if not external_path.is_file() or external_path.is_symlink():
                    raise SourceError(f"Missing or unsafe external chunk {external_path.name}")
                compressed = external_path.read_bytes()
            else:
                compressed = stream.read(length - 1)
        return decompress(compressed, marker & 127), {"compression": marker & 127, "external": external}

    def chunk(self, cx: int, cz: int) -> "Chunk":
        data, metadata = self.payload(cx, cz)
        return Chunk(loads(data), (cx, cz), metadata)


def unpack_indices(data: list[int] | None, count: int, palette_size: int, minimum_bits: int, padded: bool) -> list[int]:
    if palette_size < 1:
        raise SourceError("Empty block/biome palette")
    if data is None:
        if palette_size != 1:
            raise SourceError("Multi-entry palette without packed indices")
        return [0] * count
    bits = max(minimum_bits, (palette_size - 1).bit_length())
    per_long = 64 // bits
    expected = math.ceil(count / per_long) if padded else math.ceil(count * bits / 64)
    if len(data) != expected:
        raise SourceError(f"Wrong palette storage length {len(data)} != {expected}")
    words = [int(word) & ((1 << 64) - 1) for word in data]
    mask = (1 << bits) - 1
    result = []
    for index in range(count):
        word, shift = divmod(index, per_long) if padded else divmod(index * bits, 64)
        if padded:
            shift *= bits
        value = words[word] >> shift
        if not padded and shift + bits > 64:
            value |= words[word + 1] << (64 - shift)
        value &= mask
        if value >= palette_size:
            raise SourceError(f"Palette index {value} out of bounds {palette_size}")
        result.append(value)
    return result


class Chunk:
    def __init__(self, root: dict, expected: tuple[int, int] | None = None, metadata: dict | None = None):
        self.data_version = int(root.get("DataVersion", 0))
        self.root = root.get("Level", root)
        self.x = int(self.root.get("xPos", self.root.get("x_pos", -2**31)))
        self.z = int(self.root.get("zPos", self.root.get("z_pos", -2**31)))
        if expected and (self.x, self.z) != expected:
            raise SourceError(f"Chunk location mismatch {(self.x, self.z)} != {expected}")
        self.status = str(self.root.get("Status", self.root.get("status", "unknown")))
        self.metadata = metadata or {}
        self.sections = {}
        self.decoded = {}
        for section in self.root.get("sections", self.root.get("Sections", [])):
            y = int(section["Y"])
            if y in self.sections:
                raise SourceError("Duplicate chunk section Y")
            self.sections[y] = section

    @property
    def full(self) -> bool:
        return self.status in ("full", "minecraft:full")

    def section(self, sy: int) -> tuple[list[dict], list[int]]:
        if sy in self.decoded:
            return self.decoded[sy]
        section = self.sections.get(sy)
        if section is None:
            raise SourceError(f"Unstored section Y={sy}; outside decoded coverage")
        states = section.get("block_states")
        if states is not None:
            palette, data = states["palette"], states.get("data")
        elif "Palette" in section:
            palette, data = section["Palette"], section.get("BlockStates")
        elif "Blocks" in section:
            raise SourceError("Legacy numeric block IDs require an explicit version registry")
        elif self.full and ("BlockLight" in section or "SkyLight" in section):
            # Light-only sections are explicitly stored but contain no block data.
            palette, data = [{"Name": "minecraft:air"}], None
        else:
            raise SourceError("Incomplete chunk section has no block states")
        for block in palette:
            if not isinstance(block, dict) or not isinstance(block.get("Name"), str):
                raise SourceError("Invalid named block palette")
        indices = unpack_indices(data, 4096, len(palette), 4, self.data_version >= 2529)
        self.decoded[sy] = (palette, indices)
        return palette, indices

    def block(self, x: int, y: int, z: int) -> dict:
        if x // 16 != self.x or z // 16 != self.z:
            raise SourceError("Block position outside chunk")
        palette, indices = self.section(y // 16)
        return palette[indices[(y % 16) * 256 + (z % 16) * 16 + x % 16]]

    def biomes(self, sy: int) -> tuple[list[str], list[int]] | None:
        biome = self.sections[sy].get("biomes")
        if biome is None:
            return None
        palette = biome.get("palette", [])
        return palette, unpack_indices(biome.get("data"), 64, len(palette), 1, True)


def dimensions(world: Path) -> dict[str, Path]:
    world = Path(world)
    candidates = {}
    for name, path in [("minecraft:overworld", world), ("minecraft:the_nether", world / "DIM-1"), ("minecraft:the_end", world / "DIM1")]:
        if (path / "region").is_dir():
            candidates[name] = path
    if (world / "dimensions").is_dir():
        for path in sorted((world / "dimensions").rglob("region")):
            if not path.is_dir():
                continue
            parts = path.relative_to(world / "dimensions").parts
            if len(parts) < 3:
                continue
            name = parts[0] + ":" + "/".join(parts[1:-1])
            if name in candidates:
                raise SourceError(f"Ambiguous old and relocated dimension {name}")
            candidates[name] = path.parent
    if not candidates:
        raise SourceError("No Java Anvil dimensions found")
    return candidates
