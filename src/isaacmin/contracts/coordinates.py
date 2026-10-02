from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CoordinateFrame:
    origin_blocks: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metres_per_block: float = 1.0

    def __post_init__(self):
        if len(self.origin_blocks) != 3 or not all(math.isfinite(x) for x in self.origin_blocks):
            raise ValueError("A finite three-component source origin is required")
        if not math.isfinite(self.metres_per_block) or self.metres_per_block <= 0:
            raise ValueError("Scale must be finite and positive")

    def to_world(self, minecraft: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = minecraft
        ox, oy, oz = self.origin_blocks
        s = self.metres_per_block
        return s * (x - ox), -s * (z - oz), s * (y - oy)

    def to_source(self, world: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = world
        ox, oy, oz = self.origin_blocks
        s = self.metres_per_block
        return x / s + ox, z / s + oy, -y / s + oz

    def record(self) -> dict:
        ox, oy, oz = self.origin_blocks
        s = self.metres_per_block
        return {"metersPerUnit": 1.0, "upAxis": "Z", "handedness": "right",
                "origin_blocks": list(self.origin_blocks), "metres_per_block": s,
                "source_to_world": [[s, 0, 0, -s * ox], [0, 0, -s, s * oz],
                                    [0, s, 0, -s * oy], [0, 0, 0, 1]],
                "world_to_source": [[1/s, 0, 0, ox], [0, 0, 1/s, oy],
                                    [0, -1/s, 0, oz], [0, 0, 0, 1]],
                "source_cells": "integer minimum corners; centres = index + 0.5",
                "ground_samples": "top face centres; height is face elevation, not block centre"}

