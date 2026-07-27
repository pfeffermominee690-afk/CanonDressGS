from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "canondressgs.multi_outfit_leave_one_out.v1"
SOURCE_SCHEMA_VERSION = "canondressgs.full_dataset.v1"
DEFAULT_FORBIDDEN_FORWARD_FIELDS = (
    "outfit_id",
    "cloth_id",
    "target_rgb",
    "target_edit_rgb",
    "target_base_rgb",
    "target_foreground_mask",
    "target_clothing_mask",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
    "oracle_residual",
    "teacher",
    "teacher_residual",
)
REFERENCE_ASSET_FIELDS = ("rgb", "foreground_mask", "clothing_mask")
TARGET_ASSET_FIELDS = (
    "target_edit_rgb",
    "target_base_rgb",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_base_foreground_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_asset(source_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (source_root / path).resolve()


def _shape(value: Any, expected: tuple[int, ...], name: str) -> None:
    def dimensions(item: Any) -> tuple[int, ...]:
        if not isinstance(item, list):
            return ()
        if not item:
            return (0,)
        child = dimensions(item[0])
        if any(dimensions(part) != child for part in item):
            return (-1,)
        return (len(item), *child)

    if dimensions(value) != expected:
        raise ValueError(f"{name} must have shape {expected}")


def _geometry_contract(condition: Mapping[str, Any]) -> dict[str, Any]:
    _shape(condition.get("pose"), (165,), "pose")
    _shape(condition.get("Rh_raw"), (3,), "Rh_raw")
    _shape(condition.get("R_global"), (3, 3), "R_global")
    _shape(condition.get("Th"), (3,), "Th")
    _shape(condition.get("K"), (3, 3), "K")
    _shape(condition.get("w2c"), (4, 4), "w2c")
    width, height = int(condition["width"]), int(condition["height"])
    if width <= 0 or height <= 0:
        raise ValueError("condition resolution must be positive")
    return {
        "pose_shape": [165],
        "Rh_shape": [3, 3],
        "Th_shape": [3],
        "K_shape": [3, 3],
        "w2c_shape": [4, 4],
        "width": width,
        "height": height,
    }


def _geometry(condition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pose": condition["pose"],
        "Rh": condition["R_global"],
        "Rh_raw": condition["Rh_raw"],
        "Th": condition["Th"],
        "camera": {
            "K": condition["K"],
            "w2c": condition["w2c"],
            "width": int(condition["width"]),
            "height": int(condition["height"]),
        },
    }


def _schema_signature(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _schema_signature(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return ["list"] if not value else ["list", _schema_signature(value[0])]
    return type(value).__name__


def validate_multi_outfit_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected multi-outfit manifest schema")
    protocol = manifest["protocol"]
    outfits = list(protocol["outfits"])
    conditions = list(protocol["conditions"])
    episodes = list(manifest["episodes"])
    expected_count = len(outfits) * len(conditions)
    if len(episodes) != expected_count:
        raise ValueError(f"episode count {len(episodes)} != {expected_count}")

    outfit_counts = Counter(item["outfit_id"] for item in episodes)
    target_counts = Counter(
        (item["outfit_id"], item["target_condition_id"]) for item in episodes
    )
    if any(outfit_counts[outfit] != len(conditions) for outfit in outfits):
        raise ValueError("outfit episode counts are not balanced")
    if any(target_counts[(outfit, condition)] != 1 for outfit in outfits for condition in conditions):
        raise ValueError("each outfit/condition must be target exactly once")

    reference_counts = protocol["reference_counts"]
    for episode in episodes:
        references = list(episode["reference_condition_ids"])
        target = episode["target_condition_id"]
        if target in references or len(references) != len(set(references)):
            raise ValueError(f"reference-target overlap or duplicate reference: {episode['episode_id']}")
        if len(references) != int(reference_counts[episode["outfit_id"]]):
            raise ValueError(f"wrong reference count: {episode['episode_id']}")

    signatures: dict[str, Any] = {}
    for episode in episodes:
        signatures.setdefault(episode["outfit_id"], _schema_signature(episode))
        if signatures[episode["outfit_id"]] != _schema_signature(episode):
            raise ValueError("episode schema varies within an outfit")
    if len({canonical_sha256(value) for value in signatures.values()}) != 1:
        raise ValueError("episode schema is not identical across outfits")

    diagnostics = manifest["diagnostics"]
    if diagnostics["missing_assets"] != 0 or diagnostics["duplicate_assets"] != 0:
        raise ValueError("asset validation did not close")
    if not diagnostics["camera_resolution_contract_consistent"]:
        raise ValueError("camera/resolution contract is inconsistent")
    return {
        "episode_count": len(episodes),
        "per_outfit_episode_counts": dict(sorted(outfit_counts.items())),
        "reference_target_overlap": 0,
        "missing_assets": 0,
        "duplicate_assets": 0,
        "schema_identical_across_outfits": True,
        "camera_resolution_contract_consistent": True,
    }


def build_multi_outfit_manifest(
    source_manifest_path: str | Path,
    *,
    outfit_ids: Sequence[str],
    condition_ids: Sequence[str],
    views: Mapping[str, str],
    reference_counts: Mapping[str, int] | None = None,
    splits: Mapping[str, Sequence[str]] | None = None,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    source_path = Path(source_manifest_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("source manifest must satisfy canondressgs.full_dataset.v1")
    outfits = list(outfit_ids)
    conditions = list(condition_ids)
    if not outfits or len(outfits) != len(set(outfits)):
        raise ValueError("outfit_ids must be a non-empty unique sequence")
    if len(conditions) < 2 or len(conditions) != len(set(conditions)):
        raise ValueError("condition_ids must contain at least two unique values")
    if set(views) != set(conditions):
        raise ValueError("views must name every configured condition exactly once")
    counts = {
        outfit: len(conditions) - 1 if reference_counts is None else int(reference_counts[outfit])
        for outfit in outfits
    }
    if any(value < 1 or value > len(conditions) - 1 for value in counts.values()):
        raise ValueError("reference count must be in [1, condition_count - 1]")

    source_conditions = {item["condition_id"]: item for item in source["conditions"]}
    missing_conditions = sorted(set(conditions).difference(source_conditions))
    if missing_conditions:
        raise ValueError(f"source manifest lacks conditions: {missing_conditions}")
    source_outfits = {item["outfit_id"]: item for item in source["outfits"]}
    missing_outfits = sorted(set(outfits).difference(source_outfits))
    if missing_outfits:
        raise ValueError(f"source manifest lacks outfits: {missing_outfits}")

    contracts = [_geometry_contract(source_conditions[item]) for item in conditions]
    contract_consistent = len({canonical_sha256(item) for item in contracts}) == 1
    if not contract_consistent:
        raise ValueError("selected conditions do not share the same camera/resolution contract")

    source_root = source_path.parent
    registry: dict[str, dict[str, Any]] = {}
    logical_to_asset: dict[str, str] = {}
    missing_assets: list[str] = []
    duplicate_assets: list[str] = []

    def register(outfit: str, condition: str, field: str, raw_path: str) -> str:
        logical_id = f"{outfit}/{condition}/{field}"
        path = _resolve_asset(source_root, raw_path)
        if not path.is_file():
            missing_assets.append(logical_id)
            return logical_id
        digest = sha256_file(path)
        asset_id = f"sha256:{digest}"
        locator = {
                "outfit_id": outfit,
                "condition_id": condition,
                "field": field,
        }
        if asset_id not in registry:
            registry[asset_id] = {
                "asset_id": asset_id,
                "source_locators": [locator],
                "basenames": [path.name],
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        else:
            registry[asset_id]["source_locators"].append(locator)
            if path.name not in registry[asset_id]["basenames"]:
                registry[asset_id]["basenames"].append(path.name)
        logical_to_asset[logical_id] = asset_id
        return asset_id

    observation_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for outfit in outfits:
        records = {item["condition_id"]: item for item in source_outfits[outfit]["observations"]}
        if set(conditions).difference(records):
            raise ValueError(f"{outfit} does not contain every configured condition")
        for condition in conditions:
            observation = records[condition]
            observation_by_key[(outfit, condition)] = observation
            required = set(REFERENCE_ASSET_FIELDS) | set(TARGET_ASSET_FIELDS)
            missing_fields = sorted(required.difference(observation))
            if missing_fields:
                raise ValueError(f"{outfit}/{condition} lacks asset fields: {missing_fields}")
            for field in sorted(required):
                register(outfit, condition, field, observation[field])

    if missing_assets:
        raise FileNotFoundError(f"missing assets: {missing_assets}")
    if duplicate_assets:
        raise ValueError(f"duplicate physical assets: {duplicate_assets}")

    episodes: list[dict[str, Any]] = []
    for condition in conditions:
        for outfit in outfits:
            candidates = [value for value in conditions if value != condition]
            references = candidates[: counts[outfit]]
            reference_records = []
            for reference_condition in references:
                reference_records.append({
                    "condition_id": reference_condition,
                    "view": views[reference_condition],
                    "assets": {
                        field: logical_to_asset[f"{outfit}/{reference_condition}/{field}"]
                        for field in REFERENCE_ASSET_FIELDS
                    },
                    **_geometry(source_conditions[reference_condition]),
                })
            target_assets = {
                field: logical_to_asset[f"{outfit}/{condition}/{field}"]
                for field in TARGET_ASSET_FIELDS
            }
            episodes.append({
                "episode_id": f"{outfit}__target_{condition}",
                "outfit_id": outfit,
                "split": next(
                    (name for name, values in (splits or {"seen": outfits}).items() if outfit in values),
                    "unspecified",
                ),
                "target_condition_id": condition,
                "target_view": views[condition],
                "reference_condition_ids": references,
                "references": reference_records,
                "target": {
                    "condition_id": condition,
                    "view": views[condition],
                    "rgb": target_assets["target_edit_rgb"],
                    "masks": {
                        key: value for key, value in target_assets.items() if key.endswith("mask")
                    },
                    "base_render": target_assets["target_base_rgb"],
                    "protected_mask": target_assets["target_protected_mask"],
                    "trusted_masks": {
                        "preserve": target_assets["target_preserve_mask"],
                        "transition": target_assets["target_transition_mask"],
                        "base_foreground": target_assets["target_base_foreground_mask"],
                    },
                    **_geometry(source_conditions[condition]),
                },
                "forward_contract": {
                    "allowed": [
                        "references.*",
                        "target.pose",
                        "target.Rh",
                        "target.Th",
                        "target.camera",
                    ],
                    "forbidden_fields": list(DEFAULT_FORBIDDEN_FORWARD_FIELDS),
                    "target_images_used_only_after_render": True,
                },
                "provenance": {
                    "source_schema": SOURCE_SCHEMA_VERSION,
                    "source_manifest_sha256": sha256_file(source_path),
                    "source_outfit_id": outfit,
                    "source_condition_id": condition,
                    "generated_target": False,
                    "oracle_or_teacher_used": False,
                },
            })

    split_values = splits or {"seen": outfits, "validation": [], "unseen": [], "held_out": []}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "multi_outfit_leave_one_out_protocol",
        "protocol": {
            "outfits": outfits,
            "conditions": conditions,
            "views": dict(views),
            "reference_counts": counts,
            "episode_order": [item["episode_id"] for item in episodes],
            "splits": {key: list(value) for key, value in split_values.items()},
            "dynamic_outfit_count": True,
        },
        "source": {
            "manifest_basename": source_path.name,
            "manifest_sha256": sha256_file(source_path),
            "config_sha256": config_sha256,
            "absolute_paths_embedded": False,
        },
        "asset_registry": registry,
        "episodes": episodes,
        "forbidden_forward_fields": list(DEFAULT_FORBIDDEN_FORWARD_FIELDS),
        "diagnostics": {
            "missing_assets": len(missing_assets),
            "duplicate_assets": len(duplicate_assets),
            "reference_target_overlap": 0,
            "camera_resolution_contract": contracts[0],
            "camera_resolution_contract_consistent": contract_consistent,
            "episode_schema_identical_across_outfits": True,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    manifest["validation"] = validate_multi_outfit_manifest(manifest)
    return manifest


def build_from_config(config_path: str | Path, source_override: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest_config = config["manifest"]
    source = Path(source_override) if source_override is not None else Path(config["inputs"]["source_manifest"])
    return build_multi_outfit_manifest(
        source,
        outfit_ids=manifest_config["outfits"],
        condition_ids=manifest_config["conditions"],
        views=manifest_config["views"],
        reference_counts=manifest_config.get("reference_counts"),
        splits=manifest_config.get("splits"),
        config_sha256=sha256_file(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable multi-outfit leave-one-out manifest.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = build_from_config(args.config, args.source_manifest)
    summary = manifest["validation"]
    if not args.check_only:
        output = args.output
        if output is None:
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            output = Path(config["inputs"]["generated_manifest"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = {**summary, "output": str(output), "file_sha256": sha256_file(output)}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
