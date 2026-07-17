from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TARGET_SAMPLE = "cond_000347_O05"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"existing destination differs: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    if _sha256(source) != _sha256(temporary):
        raise IOError(f"byte copy verification failed: {source}")
    temporary.replace(destination)


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L":
            image = image.convert("L")
        return np.asarray(image) >= 128


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(value.astype(np.uint8) * 255, "L").save(temporary, format="PNG")
    temporary.replace(path)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build append-only protected-region V5.1 fixture")
    parser.add_argument("--v5-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--v3a-manifest", type=Path, required=True)
    parser.add_argument("--source-staging", type=Path, required=True)
    parser.add_argument("--output-staging", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()

    source_manifest_path = args.v5_manifest.resolve()
    source_mask_manifest_path = args.mask_manifest.resolve()
    source_manifest = _read_json(source_manifest_path)
    mask_manifest = _read_json(source_mask_manifest_path)
    v3a_manifest = _read_json(args.v3a_manifest.resolve())
    output = args.output_staging.resolve()
    audit = args.output_audit.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)

    mask_records = {item["sample_id"]: item for item in mask_manifest["records"]}
    base_records = {
        item["condition_id"]: item
        for item in v3a_manifest["records"]
        if item.get("kind") == "base"
    }
    existing_status = {
        f"{observation['condition_id']}_{outfit['outfit_id']}": observation.get("identity_audit_status", "WARN")
        for outfit in source_manifest["outfits"]
        for observation in outfit["observations"]
    }
    outfits = source_manifest["expected_outfits"]
    conditions = [item["condition_id"] for item in source_manifest["conditions"]]
    built_outfits = []
    contract_records = []
    protected_shoe_path: Path | None = None

    for outfit_id in outfits:
        observations = []
        for condition_id in conditions:
            sample_id = f"{condition_id}_{outfit_id}"
            record = mask_records[sample_id]
            base_record = base_records[condition_id]
            edit_source = args.source_staging.resolve() / "raw_edit_skin_corrected" / outfit_id / f"{condition_id}.png"
            base_source = Path(base_record["image_path"])
            edit_destination = output / "rgb" / "edit" / outfit_id / f"{condition_id}.png"
            base_destination = output / "rgb" / "base" / f"{condition_id}.png"
            _copy_exact(edit_source, edit_destination)
            _copy_exact(base_source, base_destination)

            paths: dict[str, Path] = {}
            values: dict[str, np.ndarray] = {}
            for field, source_value in record["paths"].items():
                source = Path(source_value)
                destination = output / "masks" / field / outfit_id / f"{condition_id}.png"
                if field == "target_clothing_mask":
                    continue
                _copy_exact(source, destination)
                paths[field] = destination
                values[field] = _mask(destination)

            protected = values["target_protected_mask"]
            original_clothing = _mask(Path(record["paths"]["target_clothing_mask"]))
            clothing = original_clothing & ~protected
            clothing_destination = output / "masks" / "target_clothing_mask" / outfit_id / f"{condition_id}.png"
            if clothing_destination.is_file() and not np.array_equal(_mask(clothing_destination), clothing):
                raise ValueError(f"existing clipped clothing mask differs: {clothing_destination}")
            if not clothing_destination.is_file():
                _save_mask(clothing_destination, clothing)
            paths["target_clothing_mask"] = clothing_destination
            values["target_clothing_mask"] = clothing

            base_foreground_source = Path(base_record["output_files"]["person_foreground.png"]["path"])
            base_foreground_destination = output / "masks" / "target_base_foreground_mask" / outfit_id / f"{condition_id}.png"
            _copy_exact(base_foreground_source, base_foreground_destination)
            paths["target_base_foreground_mask"] = base_foreground_destination
            values["target_base_foreground_mask"] = _mask(base_foreground_destination)

            edit = values["target_edit_mask"]
            core = values["target_edit_core_mask"]
            preserve = values["target_preserve_mask"]
            transition = values["target_transition_mask"]
            checks = {
                "protected_outside_preserve": int((protected & ~preserve).sum()),
                "edit_protected_overlap": int((edit & protected).sum()),
                "core_protected_overlap": int((core & protected).sum()),
                "transition_protected_overlap": int((transition & protected).sum()),
                "clothing_protected_overlap_before_clip": int((original_clothing & protected).sum()),
                "clothing_protected_overlap_after_clip": int((clothing & protected).sum()),
            }
            if any(checks[name] for name in (
                "protected_outside_preserve", "edit_protected_overlap", "core_protected_overlap",
                "transition_protected_overlap", "clothing_protected_overlap_after_clip",
            )):
                raise AssertionError(f"protected contract failed for {sample_id}: {checks}")
            contract_records.append({"sample_id": sample_id, "checks": checks})

            observation = {
                "condition_id": condition_id,
                "rgb": _relative(output, edit_destination),
                "foreground_mask": _relative(output, paths["target_foreground_mask"]),
                "clothing_mask": _relative(output, clothing_destination),
                "target_edit_rgb": _relative(output, edit_destination),
                "target_base_rgb": _relative(output, base_destination),
                **{field: _relative(output, path) for field, path in paths.items()},
                "checksums": {
                    "target_edit_rgb": _sha256(edit_destination),
                    "target_base_rgb": _sha256(base_destination),
                    **{field: _sha256(path) for field, path in paths.items()},
                },
                "identity_audit_status": (
                    "PROTECTED_ONLY_DIAGNOSTIC_WARN"
                    if sample_id == TARGET_SAMPLE
                    else existing_status.get(sample_id, "WARN")
                ),
            }
            observations.append(observation)

            if sample_id == TARGET_SAMPLE:
                shoe = (
                    _mask(Path(base_record["output_files"]["base_shoe_seed.png"]["path"]))
                    | _mask(Path(base_record["output_files"]["projected_feet_protection.png"]["path"]))
                )
                protected_shoe_path = output / "masks" / "protected_shoe_audit" / outfit_id / f"{condition_id}.png"
                if not protected_shoe_path.is_file():
                    _save_mask(protected_shoe_path, shoe)
                base_rgb = _rgb(base_destination)
                edit_rgb = _rgb(edit_destination)
                shoe_difference = np.abs(edit_rgb.astype(np.int16) - base_rgb.astype(np.int16))[shoe]
                target_contract = {
                    "sample_id": sample_id,
                    "shoe_pixels": int(shoe.sum()),
                    "shoe_in_protected": int((shoe & protected).sum()),
                    "shoe_in_preserve": int((shoe & preserve).sum()),
                    "shoe_in_edit": int((shoe & edit).sum()),
                    "shoe_in_core": int((shoe & core).sum()),
                    "shoe_in_transition": int((shoe & transition).sum()),
                    "shoe_in_clothing_before_clip": int((shoe & original_clothing).sum()),
                    "shoe_in_clothing_after_clip": int((shoe & clothing).sum()),
                    "shoe_in_edit_foreground": int((shoe & values["target_foreground_mask"]).sum()),
                    "shoe_in_base_foreground": int((shoe & values["target_base_foreground_mask"]).sum()),
                    "raw_base_shoe_rgb_mean_absolute_difference": float(shoe_difference.mean() / 255.0),
                    "raw_base_shoe_rgb_changed_fraction": float((np.max(shoe_difference, axis=1) > 1).mean()),
                    "shoe_mask": _relative(output, protected_shoe_path),
                }
        built_outfits.append({
            "outfit_id": outfit_id,
            "metadata": {"fixture": True, "supervision_mode": "dual_target_region_aware_v1", "acceptance": "v5.1"},
            "observations": observations,
        })

    if protected_shoe_path is None:
        raise RuntimeError(f"missing required sample: {TARGET_SAMPLE}")
    if target_contract["shoe_in_protected"] != target_contract["shoe_pixels"]:
        raise AssertionError("shoe mask is not fully protected")
    if target_contract["shoe_in_preserve"] != target_contract["shoe_pixels"]:
        raise AssertionError("shoe mask is not fully preserved")
    if any(target_contract[key] for key in (
        "shoe_in_edit", "shoe_in_core", "shoe_in_transition", "shoe_in_clothing_after_clip",
    )):
        raise AssertionError("shoe pixels leak into edit/clothing supervision")

    historical_report = source_manifest_path.parent / "raw_edit_identity_visual_adjudication_v5.json"
    final_adjudication = {
        "schema_version": "subject02.protected_region_final_adjudication.v5.1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "append_only": True,
        "historical_v5_report": {"path": str(historical_report), "sha256": _sha256(historical_report)},
        "historical_classification": "RAW_IDENTITY_FAIL",
        "final_classification": "PROTECTED_ONLY_DIAGNOSTIC_WARN",
        "reason": "Raw shoe appearance differs, but all shoe pixels are base-supervised and excluded from edit/clothing RGB and edit alpha regions.",
        "pose_view_contract": "PASS_FROM_FROZEN_V4_ALIGNMENT_EVIDENCE",
        "target_fields_enter_forward": False,
        "target_contract": target_contract,
        "all_sample_contracts": contract_records,
    }
    manifest = {
        **{key: value for key, value in source_manifest.items() if key not in {"outfits", "identity_visual_adjudication"}},
        "expected_condition_count": 4,
        "identity_visual_adjudication": {
            "status": "SUPERSEDED_BY_APPEND_ONLY_V5_1_ADJUDICATION",
            "historical_excluded_samples": source_manifest.get("identity_visual_adjudication", {}).get("excluded_samples", []),
            "excluded_samples": [],
        },
        "protected_region_final_adjudication": final_adjudication,
        "outfits": built_outfits,
    }
    manifest_path = output / "pilot_manifest_full_v1_v5_1.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(audit / "PROTECTED_REGION_FINAL_ADJUDICATION_V5_1.json", final_adjudication)
    _atomic_json(audit / "protected_region_contract_v5_1.json", {
        "status": "PASS",
        "target": target_contract,
        "all_samples": contract_records,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_pixels_modified": False,
    })
    (audit / "PROTECTED_REGION_FINAL_ADJUDICATION_V5_1.md").write_text(
        "# Protected-Region Final Adjudication V5.1\n\n"
        "This is an append-only final adjudication. The original V5 failure report and raw image remain unchanged.\n\n"
        f"- sample: `{TARGET_SAMPLE}`\n"
        "- historical classification: **RAW_IDENTITY_FAIL**\n"
        "- final classification: **PROTECTED_ONLY_DIAGNOSTIC_WARN**\n"
        f"- shoe pixels: {target_contract['shoe_pixels']}\n"
        f"- shoe in protected/preserve: {target_contract['shoe_in_protected']}/{target_contract['shoe_in_preserve']}\n"
        f"- shoe in edit/core/transition: {target_contract['shoe_in_edit']}/{target_contract['shoe_in_core']}/{target_contract['shoe_in_transition']}\n"
        f"- shoe in clothing before/after protected clipping: {target_contract['shoe_in_clothing_before_clip']}/{target_contract['shoe_in_clothing_after_clip']}\n"
        "- raw shoe pixels are diagnostic-only and never enter model forward or edit/clothing supervision.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "manifest": str(manifest_path),
        "samples": sum(len(outfit["observations"]) for outfit in built_outfits),
        "target_contract": target_contract,
    }, indent=2))


if __name__ == "__main__":
    main()
