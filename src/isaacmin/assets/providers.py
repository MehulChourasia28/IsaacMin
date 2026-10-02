"""Provider schemas captured from real API responses, with explicit qualifications."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlencode, urlsplit

from PIL import Image

from .network import PublicClient, ServiceError, atomic_json, digest, extract_zip, safe_relative, utcnow

POLY_HOSTS = ("api.polyhaven.com", "polyhaven.com", "dl.polyhaven.org")
AMBIENT_HOSTS = ("ambientcg.com", "ambientcg.com", "acg-download.struffelproductions.com")
LICENCES = {"poly_haven": "https://polyhaven.com/license", "ambientcg": "https://docs.ambientcg.com/license/"}


def parse_poly_catalogue(data):
    if not isinstance(data, dict) or not data:
        raise ServiceError("schema_drift", "Poly Haven catalogue must be an object keyed by asset ID")
    for asset_id, item in data.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id) or not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ServiceError("schema_drift", "Poly Haven asset identity/name changed")
    return data


def parse_ambient(data):
    if not isinstance(data, dict) or not isinstance(data.get("assets"), list) or type(data.get("totalResults")) is not int:
        raise ServiceError("schema_drift", "ambientCG v3 envelope changed")
    for asset in data["assets"]:
        if not isinstance(asset, dict) or not isinstance(asset.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", asset["id"]):
            raise ServiceError("schema_drift", "ambientCG v3 asset identity changed")
        if not isinstance(asset.get("downloads"), list):
            raise ServiceError("schema_drift", "ambientCG v3 downloads must be an array (observed live schema)")
        for entry in asset["downloads"]:
            if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in ("attributes", "extension", "url")) or type(entry.get("size")) is not int:
                raise ServiceError("schema_drift", "ambientCG v3 download record changed")
    return data["assets"]


def poly_variant(files, role, resolution, fmt):
    try:
        record = files[role][resolution][fmt]
    except (KeyError, TypeError):
        raise ServiceError("unavailable_variant", f"Requested Poly Haven {role}/{resolution}/{fmt} is unavailable") from None
    if not isinstance(record, dict) or not isinstance(record.get("url"), str) or type(record.get("size")) is not int:
        raise ServiceError("schema_drift", "Poly Haven file record requires URL and size")
    if "include" in record and not isinstance(record["include"], dict):
        raise ServiceError("schema_drift", "Poly Haven dependency include must be an object")
    return record


def _filename(url):
    p = urlsplit(url)
    return safe_relative(unquote(p.path.rsplit("/", 1)[-1]))


def decode_image(path):
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return {"status": "not_run", "reason": "Requires Blender/OpenImageIO for this format"}
    try:
        with Image.open(path) as im:
            im.load()
            return {"status": "pass", "width": im.width, "height": im.height, "mode": im.mode, "format": im.format}
    except Exception:
        raise ServiceError("corrupt_image", "Image failed full Pillow decode") from None


def _poly_record(asset_id, meta, snapshot, files_snapshot, kind):
    dimensions = meta.get("dimensions")
    return {"schema_version": 1, "provider": "poly_haven", "asset_id": asset_id, "kind": kind,
            "source_page": "https://polyhaven.com/a/" + asset_id, "authors": meta.get("authors", {}),
            "licence": "CC0-1.0", "licence_url": LICENCES["poly_haven"], "attribution": "Assets from Poly Haven",
            "discovery_response_sha256": snapshot["response_sha256"], "file_response_sha256": files_snapshot["response_sha256"],
            "upstream_revision": meta.get("files_hash"), "retrieved_at_utc": utcnow(),
            "scene_use": "provider-edited sky-only lighting; no ground-scene representation" if asset_id.endswith('_puresky') else "candidate_requires_target_qualification",
            "metadata": meta, "original_dimensions": dimensions,
            "dimensions_m": [x / 1000 for x in dimensions] if dimensions else None,
            "dimension_confidence": "provider millimetres; model collection bounds require per-object verification",
            "qualification": {"status": "not_run", "renderer": "isaac", "approved_uses": []},
            "normalization": {"status": "not_run", "reason": "Needs native Blender inspection and contact anchor validation"}, "files": []}


class PolyHaven:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.client = PublicClient(self.workspace / "cache/assets/poly_haven/metadata", POLY_HOSTS)

    def catalogue(self, kind, refresh=False):
        if kind not in ("textures", "models", "hdris"):
            raise ServiceError("invalid_type", "Unknown Poly Haven catalogue type")
        data, snap = self.client.metadata("https://api.polyhaven.com/assets?" + urlencode({"type": kind}), refresh=refresh)
        return parse_poly_catalogue(data), snap

    def acquire(self, asset_id, kind, selections, *, refresh=False):
        catalogue, snapshot = self.catalogue(kind, refresh)
        if asset_id not in catalogue:
            raise ServiceError("unknown_asset", "Requested Poly Haven ID is absent from the live catalogue")
        files, files_snapshot = self.client.metadata("https://api.polyhaven.com/files/" + asset_id, refresh=refresh)
        record = _poly_record(asset_id, catalogue[asset_id], snapshot, files_snapshot, kind)
        folder = self.workspace / "assets/masters/poly_haven" / asset_id / str(catalogue[asset_id].get("files_hash", "unversioned"))
        for role, resolution, fmt in selections:
            item = files[role] if resolution is None else poly_variant(files, role, resolution, fmt)
            variant_folder = folder / (resolution or "original")
            target = variant_folder / _filename(item["url"])
            downloaded = self.client.download(item, target)
            downloaded.update(role=role, resolution=resolution, format=fmt, image_validation=decode_image(target))
            record["files"].append(downloaded)
            for name, dependency in item.get("include", {}).items():
                path = variant_folder / safe_relative(name)
                dep = self.client.download(dependency, path)
                dep.update(role="model_dependency", dependency_of=str(target), provider_relative_path=name,
                           image_validation=decode_image(path))
                record["files"].append(dep)
        record["selected_variants"] = [list(x) for x in selections]
        atomic_json(folder / "asset_manifest.json", record)
        return record


class AmbientCG:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.client = PublicClient(self.workspace / "cache/assets/ambientcg/metadata", AMBIENT_HOSTS)

    def discover(self, query, *, limit=100, offset=0, refresh=False):
        if not 1 <= limit <= 500 or offset < 0:
            raise ServiceError("invalid_query", "ambientCG pagination is out of range")
        url = "https://ambientcg.com/api/v3/assets?" + urlencode({"type": "material", "q": query, "limit": limit, "offset": offset,
              "include": "type,title,tags,dimensions,maps,technique,downloads,url"})
        obj, snapshot = self.client.metadata(url, refresh=refresh)
        return parse_ambient(obj), snapshot

    def asset_by_id(self, asset_id, *, refresh=False):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',asset_id):
            raise ServiceError('invalid_query','Invalid ambientCG identifier')
        url='https://ambientcg.com/api/v3/assets?id='+asset_id+'&include=title,tags,dimensions,maps,technique,downloads,url'
        data,snapshot=self.client.metadata(url,refresh=refresh)
        candidates=parse_ambient(data)
        matches=[a for a in candidates if a['id']==asset_id]
        if len(matches)!=1:
            raise ServiceError('unknown_asset','Exact ambientCG ID is not in the live response')
        return matches[0],snapshot

    def acquire(self, asset, snapshot, attributes="1K-PNG"):
        options = [x for x in asset["downloads"] if x["attributes"] == attributes and x["extension"] == "zip"]
        if len(options) != 1:
            raise ServiceError("unavailable_variant", "ambientCG requested lossless qualification variant is unavailable")
        selected = options[0]
        identity = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:16]
        folder = self.workspace / "assets/masters/ambientcg" / asset["id"] / identity
        target = folder / (asset["id"] + "_" + attributes + ".zip")
        downloaded = self.client.download(selected, target)
        extracted = folder / attributes
        index = folder / "extracted_manifest.json"
        if not extracted.exists():
            members = extract_zip(target, extracted)
            atomic_json(index, members)
        elif not index.exists():
            raise ServiceError("unvalidated_extraction", "Existing extraction has no validated file manifest")
        else:
            members = json.loads(index.read_text())
            for member in members:
                if digest(extracted / safe_relative(member["path"])) != member["sha256"]:
                    raise ServiceError("corrupt_cache", "Extracted asset file changed")
        for member in members:
            member["image_validation"] = decode_image(extracted / member["path"])
        record = {"schema_version": 1, "provider": "ambientcg", "protocol": "v3", "asset_id": asset["id"],
                  "kind": asset.get("type", "unclassified_provider_asset"), "source_page": asset["url"], "authors": ["ambientCG / Lennart Demes"],
                  "licence": "CC0-1.0", "licence_url": LICENCES["ambientcg"], "metadata": asset,
                  "discovery_response_sha256": snapshot["response_sha256"], "upstream_revision": identity,
                  "retrieved_at_utc": utcnow(), "files": [downloaded], "extracted_directory": str(extracted),
                  "extracted_files": members, "selected_variants": [attributes],
                  "physical_dimensions": asset.get("dimensions"), "dimension_confidence": "unknown_when_zero; target scale calibration required",
                  "normalization": {"status": "not_run"}, "qualification": {"status": "not_run", "renderer": "isaac", "approved_uses": []}}
        atomic_json(folder / "asset_manifest.json", record)
        return record


def bootstrap_assets(workspace: Path, refresh=False):
    """Small genuine shortlist for qualification, never a realism claim or a full biome."""
    workspace = Path(workspace).resolve()
    previous_path = workspace / "state/asset_catalogue.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    report = {"schema_version": 1, "created_at_utc": utcnow(), "status": "running", "assets": previous.get("assets", []), "failures": [],
              "scope": "temperate forest/mountain qualification shortlist; source ecological interpretation remains independent",
              "target_qualification": "not_run", "master_upgrade_policy": "Acquire 4K/8K only after useful target qualification or measured texel-density need",
              "provider_credit": "Assets from Poly Haven; additional assets from ambientCG"}
    report_path = workspace / "state/asset_catalogue.json"
    poly, ambient = PolyHaven(workspace), AmbientCG(workspace)
    for endpoint in ("types", "taxonomy"):
        try:
            value, snap = poly.client.metadata("https://api.polyhaven.com/" + endpoint, refresh=refresh)
            atomic_json(workspace / ("evidence/services/provider_fixtures/poly_" + endpoint + ".json"), value)
        except ServiceError as e:
            report["failures"].append({"operation": endpoint, **e.record()})
    # IDs are selected only after catalogue lookup, never used to invent download paths.
    selection = [
        ("forest_ground_04", "textures", [("Diffuse", "1k", "png"), ("nor_gl", "1k", "png"), ("Rough", "1k", "png"), ("Displacement", "1k", "png")]),
        ("pine_sapling_small", "models", [("blend", "1k", "blend")]),
        ("fern_02", "models", [("blend", "1k", "blend")]),
        ("rock_moss_set_01", "models", [("blend", "1k", "blend")]),
        ("forest_slope", "hdris", [("hdri", "2k", "hdr"), ("tonemapped", None, "jpg")]),
    ]
    source_ir = workspace / "artifacts/source/region_initial/world_ir.json"
    if source_ir.exists():
        biomes = json.loads(source_ir.read_text()).get("semantics", {}).get("surface_biomes", {})
        report["source_biome_evidence"] = {"path": str(source_ir), "sha256": digest(source_ir), "biomes": biomes}
        if set(b.removeprefix("minecraft:") for b in biomes) <= {"plains", "forest", "old_growth_birch_forest", "birch_forest"}:
            selection = [item for item in selection if item[0] != "pine_sapling_small"]
            selection += [(name, "models", [("blend", "1k", "blend")]) for name in
                          ("grass_medium_01", "grass_medium_02", "dandelion_01", "celandine_01", "dead_tree_trunk")]
            selection += [(name, "hdris", [("hdri", "2k", "hdr"), ("tonemapped", None, "jpg")]) for name in ("birchwood", "alps_field")]
            selection += [(name, "hdris", [("hdri", "4k", "exr"), ("tonemapped", None, "jpg")]) for name in
                          ("kloofendal_48d_partly_cloudy_puresky", "kloofendal_overcast_puresky")]
            selection += [(name, "textures", [(role, "2k", "png") for role in ("Diffuse", "nor_gl", "Rough", "Displacement")]) for name in
                          ("brown_mud_dry", "leafy_grass", "rock_boulder_dry", "rock_face_03")]
            report["scope"] = "source-derived temperate plains/deciduous forest acquisition candidates; historical pine remains diagnostic-only"
    for asset_id, kind, variants in selection:
        try:
            record = poly.acquire(asset_id, kind, variants, refresh=refresh)
            report["assets"] = [a for a in report["assets"] if (a["provider"], a["asset_id"]) != (record["provider"], record["asset_id"])]
            report["assets"].append(record)
        except ServiceError as e:
            report["failures"].append({"provider": "poly_haven", "asset_id": asset_id, **e.record()})
        atomic_json(report_path, report)
    try:
        candidates, snapshot = ambient.discover("forest", limit=30, refresh=refresh)
        atomic_json(workspace / "evidence/services/provider_fixtures/ambient_v3_forest.json", snapshot)
        ranked = sorted(candidates, key=lambda x: (-len(set(x.get("tags", [])) & {"soil", "natural", "forest", "ground", "earth"}), x["id"]))
        if not ranked:
            raise ServiceError("coverage_gap", "No ambientCG forest material candidates returned")
        ambient.client.budget_bytes = max(0, 30 * 1024**3 - poly.client.downloaded)
        record = ambient.acquire(ranked[0], snapshot)
        report["assets"] = [a for a in report["assets"] if (a["provider"], a["asset_id"]) != (record["provider"], record["asset_id"])]
        report["assets"].append(record)
    except ServiceError as e:
        report["failures"].append({"provider": "ambientcg", **e.record()})
    if any(name=='grass_medium_01' for name,_,_ in selection):
        for asset_id in ('LeafSet016','Bark012'):
            try:
                master,snapshot=ambient.asset_by_id(asset_id,refresh=refresh)
                record=ambient.acquire(master,snapshot,attributes='2K-PNG')
                report['assets']=[a for a in report['assets'] if (a['provider'],a['asset_id'])!=(record['provider'],record['asset_id'])]
                report['assets'].append(record)
            except ServiceError as e:
                report['failures'].append({'provider':'ambientcg','asset_id':asset_id,**e.record()})
    report["status"] = "external_tool_verified" if not report["failures"] else "partial"
    report["coverage_gaps"] = ["No target-renderer-qualified assets", "Canopy age/species diversity incomplete", "No qualified contact anchors",
                               "No target displacement amplitude/foliage transmission calibration", "Master texture density pending actual camera projections"]
    atomic_json(report_path, report)
    credits = ["# Asset credits", "", "Assets from Poly Haven. Additional assets from ambientCG.", "",
               "These are acquired qualification candidates; no asset is yet qualified in Isaac.", ""]
    for a in report["assets"]:
        credits += [f"- [{a['asset_id']}]({a['source_page']}) — {a['provider']}; authors: {', '.join(a['authors'])}; [CC0-1.0]({a['licence_url']})."]
    (workspace / "ASSET_CREDITS.md").write_text("\n".join(credits) + "\n")
    return report
