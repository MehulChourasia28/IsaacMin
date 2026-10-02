"""Deterministic biome-constrained candidates grounded against final multi-surface geometry.

Field inference and actual target-renderer qualification remain explicit prerequisites.
This module never invents canopy species or substitutes a conifer for a birch.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

RULE_VERSION = "temperate-understory-2"


class PolylineClearance:
    """Exact metric clearance from the placed XY to an explicit route corridor."""
    def __init__(self, points, corridor_width_m):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1] not in (2,3) or len(points)<2 or not np.isfinite(points).all() or not math.isfinite(corridor_width_m) or corridor_width_m<=0:
            raise ValueError('A finite metric route polyline and positive corridor width are required')
        delta=np.diff(points[:,:2],axis=0);squared=np.sum(delta*delta,axis=1);valid=squared>0
        if not valid.any():raise ValueError('Route polyline has no nonzero segments')
        self.start=points[:-1,:2][valid];self.delta=delta[valid];self.squared=squared[valid]
        self.half_width=corridor_width_m/2

    def __call__(self,x,y):
        point=np.asarray([x,y],float)
        if not np.isfinite(point).all():raise ValueError('Placed route query must be finite')
        t=np.clip(np.sum((point-self.start)*self.delta,axis=1)/self.squared,0,1)
        return float(np.min(np.linalg.norm(point-self.start-t[:,None]*self.delta,axis=1))-self.half_width)


@dataclass(frozen=True)
class EcologyRule:
    asset_id: str
    guild: str
    biomes: tuple[str, ...]
    substrates: tuple[str, ...]
    max_slope: float
    moisture: tuple[float, float]
    canopy: tuple[float, float]
    density_per_m2: float
    minimum_spacing_m: float
    scale: tuple[float, float]


RULES = (
    EcologyRule("grass_medium_01", "meadow_grass", ("plains", "forest", "old_growth_birch_forest"), ("soil",), 32, (0.12, 0.9), (0, 0.65), 1.6, 0.12, (0.85, 1.12)),
    EcologyRule("grass_medium_02", "meadow_grass", ("plains", "forest", "old_growth_birch_forest"), ("soil",), 32, (0.15, 0.9), (0, 0.65), 1.1, 0.14, (0.85, 1.12)),
    EcologyRule("dandelion_01", "meadow_forb", ("plains", "forest", "old_growth_birch_forest"), ("soil",), 27, (0.2, 0.8), (0, 0.45), 0.18, 0.2, (0.85, 1.12)),
    EcologyRule("celandine_01", "spring_woodland_forb", ("forest", "old_growth_birch_forest"), ("soil",), 25, (0.45, 0.85), (0.2, 0.85), 0.08, 0.25, (0.85, 1.12)),
    EcologyRule("fern_02", "woodland_fern", ("forest", "old_growth_birch_forest"), ("soil",), 30, (0.5, 0.9), (0.3, 1), 0.12, 0.45, (0.85, 1.12)),
    EcologyRule("dead_tree_trunk", "deadwood", ("forest", "old_growth_birch_forest"), ("soil",), 15, (0.2, 0.95), (0.35, 1), 0.004, 2.5, (0.95, 1.05)),
    EcologyRule("rock_moss_set_01", "mossy_boulder", ("forest", "old_growth_birch_forest"), ("soil", "rock"), 25, (0.45, 0.95), (0.15, 1), 0.006, 3.0, (0.95, 1.05)),
)


def _random(seed, x, y, channel=0):
    value = hashlib.blake2b(f"{seed}:{x}:{y}:{channel}".encode(), digest_size=8).digest()
    return int.from_bytes(value, "big") / 2**64


def _cluster(seed, x, y):
    """Smooth world-space density patches, stable across tile order and cropping."""
    x, y = x / 9, y / 9
    ix, iy = math.floor(x), math.floor(y)
    tx, ty = x - ix, y - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    return ((1-tx)*(1-ty)*_random(seed, ix, iy, 60) + tx*(1-ty)*_random(seed, ix+1, iy, 60)
            + (1-tx)*ty*_random(seed, ix, iy+1, 60) + tx*ty*_random(seed, ix+1, iy+1, 60))


def rule_violations(rule: EcologyRule, field: dict, *, season="late_spring"):
    required = {"valid", "biome", "substrate", "slope_degrees", "moisture", "canopy", "water_depth_m", "route_distance_m"}
    missing = required - set(field)
    if missing:
        return ["missing_fields:" + ",".join(sorted(missing))]
    failures = []
    if not field["valid"]:
        failures.append("invalid_or_missing_source")
    if str(field["biome"]).removeprefix("minecraft:") not in rule.biomes:
        failures.append("biome_mismatch")
    if field["substrate"] not in rule.substrates:
        failures.append("substrate_mismatch")
    if not math.isfinite(field["slope_degrees"]) or not 0 <= field["slope_degrees"] <= rule.max_slope:
        failures.append("slope")
    for name, bounds in (("moisture", rule.moisture), ("canopy", rule.canopy)):
        if not math.isfinite(field[name]) or not bounds[0] <= field[name] <= bounds[1]:
            failures.append(name)
    if not math.isfinite(field["water_depth_m"]) or field["water_depth_m"] > 0:
        failures.append("submerged")
    if not math.isfinite(field["route_distance_m"]) or field["route_distance_m"] < 0.45:
        failures.append("trail_clearance")
    if rule.guild == "spring_woodland_forb" and season not in ("spring", "late_spring"):
        failures.append("season")
    return failures


def _basis(normal, yaw):
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 0.5 or normal[2] <= 0:
        raise ValueError("Supporting surface normal must be finite and upward")
    z = normal / np.linalg.norm(normal)
    x = np.array([math.cos(yaw), math.sin(yaw), 0.])
    x -= z * np.dot(x, z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def ground_prototype(prototype: dict, x: float, y: float, scale: float, yaw: float,
                     support_query: Callable, tolerance_m=0.02, *, max_support_slope_degrees=None):
    """support_query(x,y,z_hint) returns final mesh z/normal/surface_id/mesh_sha256.

    The hint keeps anchors on the same stacked supporting floor. Queries must return
    None at unknown/open ground. The caller supplies an independently testable final
    geometry query, never source Minecraft heights or pre-displacement colliders.
    """
    support = support_query(x, y, None)
    if support is None:
        return None, "no_support"
    if not all(k in support for k in ("z", "normal", "surface_id", "mesh_sha256")):
        raise ValueError("Final supporting geometry identity is required")
    rotation = _basis(support["normal"], yaw)
    anchors = np.asarray(prototype.get("contact_anchors_local_m", []), dtype=float)
    if anchors.ndim != 2 or anchors.shape[1:] != (3,) or not len(anchors) or not np.isfinite(anchors).all():
        return None, "missing_contact_anchors"
    transformed = anchors @ (rotation * scale).T
    if max_support_slope_degrees is not None and (not math.isfinite(max_support_slope_degrees) or not 0 <= max_support_slope_degrees <= 90):
        raise ValueError("Invalid ecological support-slope bound")
    offsets = []
    support_slopes = []
    for ax, ay, az in transformed:
        local = support_query(x + ax, y + ay, support["z"])
        if local is None or local["surface_id"] != support["surface_id"] or local["mesh_sha256"] != support["mesh_sha256"]:
            return None, "support_boundary"
        normal = np.asarray(local.get("normal", []), dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0:
            return None, "invalid_anchor_support_normal"
        slope = math.degrees(math.acos(float(np.clip(normal[2] / np.linalg.norm(normal), -1, 1))))
        support_slopes.append(slope)
        if max_support_slope_degrees is not None and slope > max_support_slope_degrees:
            return None, "root_footprint_slope_limit"
        offsets.append(float(local["z"] - az))
    z = float(np.median(offsets))
    residuals = z - np.asarray(offsets)
    if not np.isfinite(residuals).all() or np.max(np.abs(residuals)) > tolerance_m:
        return None, "root_contact_spread"
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * scale
    matrix[:3, 3] = [x, y, z]
    return {"world_transform": matrix.tolist(), "supporting_surface_id": support["surface_id"],
            "supporting_mesh_sha256": support["mesh_sha256"], "contact_offset_max_m": float(np.abs(residuals).max()),
            "contact_samples": len(offsets), "grounding_method": "fit_source_base_anchors_to_final_supporting_mesh",
            "maximum_root_support_slope_degrees": max(support_slopes),
            "root_support_slope_limit_degrees": max_support_slope_degrees,
            "target_contact_qualification": "not_run"}, None


def scatter_candidates(bounds_xy, sample_fields: Callable, support_query: Callable, assets: list[dict], *,
                       seed=1729, cell_size_m=0.5, season="late_spring", require_qualified=True,
                       candidate_limit=200000):
    if not 0.1 <= cell_size_m <= 2:
        raise ValueError("Ecology sampling cell spacing must be 0.1..2m")
    xmin, ymin, xmax, ymax = bounds_xy
    if not all(math.isfinite(v) for v in bounds_xy) or xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid world-space ecology bounds")
    if (math.ceil(xmax/cell_size_m)-math.floor(xmin/cell_size_m)) * (math.ceil(ymax/cell_size_m)-math.floor(ymin/cell_size_m)) > candidate_limit:
        raise ValueError("Region exceeds bounded ecology candidate budget; tile with unchanged world coordinates")
    catalogue = {a["asset_id"]: a for a in assets}
    selected = [r for r in RULES if r.asset_id in catalogue and (not require_qualified or catalogue[r.asset_id].get("isaac_qualification") == "pass")]
    gaps = [{"asset_id": r.asset_id, "reason": "unavailable_or_not_isaac_qualified"} for r in RULES if r not in selected]
    instances, rejected = [], Counter()

    @lru_cache(maxsize=candidate_limit * 2)
    def proposal(ix, iy):
        x = (ix + _random(seed, ix, iy, 1)) * cell_size_m
        y = (iy + _random(seed, ix, iy, 2)) * cell_size_m
        field = sample_fields(x, y)
        eligible = [r for r in selected if not rule_violations(r, field, season=season)]
        if not eligible:
            return x, y, field, None, "no_ecologically_valid_asset"
        density = sum(r.density_per_m2 for r in eligible)
        probability = 1 - math.exp(-density * cell_size_m**2 * (0.15 + 1.7 * _cluster(seed, x, y)))
        if _random(seed, ix, iy, 3) > probability:
            return x, y, field, None, "density"
        choose = _random(seed, ix, iy, 4) * density
        rule = eligible[-1]
        for candidate in eligible:
            choose -= candidate.density_per_m2
            if choose <= 0:
                rule = candidate
                break
        return x, y, field, rule, None

    def spacing_layer(rule):
        # Grass and forbs share a planting layer. Grass alongside a rock or log
        # must not suppress that rarer guild across its whole spacing radius.
        return rule.guild if rule.guild in ("deadwood", "mossy_boulder") else "ground_flora"

    for iy in range(math.floor(ymin/cell_size_m), math.ceil(ymax/cell_size_m)):
        for ix in range(math.floor(xmin/cell_size_m), math.ceil(xmax/cell_size_m)):
            x, y, field, rule, failure = proposal(ix, iy)
            if not xmin <= x < xmax or not ymin <= y < ymax:
                continue
            if rule is None:
                if failure != "density":
                    rejected[failure] += 1
                continue
            # Only genuine ecological proposals suppress neighbours. Empty cells
            # cannot accidentally remove nearly all low-density logs and rocks.
            spacing = max(r.minimum_spacing_m for r in selected if spacing_layer(r) == spacing_layer(rule))
            radius = max(1, math.ceil(spacing/cell_size_m))
            priority = _random(seed, ix, iy, 8)
            conflict = False
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if not (dx or dy) or _random(seed, ix+dx, iy+dy, 8) >= priority:
                        continue
                    nx, ny, _, neighbour, _ = proposal(ix+dx, iy+dy)
                    if neighbour is None or spacing_layer(neighbour) != spacing_layer(rule):
                        continue
                    if math.hypot(nx-x, ny-y) < max(rule.minimum_spacing_m, neighbour.minimum_spacing_m):
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                rejected["spacing"] += 1
                continue
            asset = catalogue[rule.asset_id]
            prototypes = asset.get("objects", [])
            if not prototypes:
                rejected["missing_normalized_prototype"] += 1
                continue
            prototype = prototypes[min(len(prototypes)-1, int(_random(seed, ix, iy, 5)*len(prototypes)))]
            scale = rule.scale[0] + _random(seed, ix, iy, 6) * (rule.scale[1]-rule.scale[0])
            yaw = _random(seed, ix, iy, 7)*2*math.pi
            grounded, failure = ground_prototype(prototype, x, y, scale, yaw, support_query, max_support_slope_degrees=rule.max_slope)
            if failure:
                rejected[failure] += 1
                continue
            identity = hashlib.sha256(f"{seed}:{ix}:{iy}:{rule.asset_id}:{RULE_VERSION}".encode()).hexdigest()[:24]
            instances.append({"instance_id": identity, "asset_id": rule.asset_id, "object_name": prototype["object_name"],
                              "asset_revision": asset.get("output_sha256"), "blend_path": asset["output_blend"],
                              "tile_owner_world_cell": [ix, iy], "placement_rule_version": RULE_VERSION,
                              "source_biome": field["biome"], "guild": rule.guild, "field_evidence": field,
                              "synthesized": True, "scale": scale, "seed": seed, **grounded})
    return {"schema_version": 1, "status": "candidate" if not require_qualified else "qualified_assets_placement_candidate",
            "qualification": "not_run", "rule_version": RULE_VERSION, "seed": seed, "season": season,
            "instances": instances, "asset_counts": dict(Counter(i["asset_id"] for i in instances)),
            "rejections": dict(rejected), "coverage_gaps": gaps,
            "limitations": ["Species-level grass/fern identification absent from provider metadata",
                            "Canopy generation, source-root placement and target qualification are separate gates",
                            "Anchor fit does not establish full-mesh interpenetration or Isaac material/contact qualification"]}
