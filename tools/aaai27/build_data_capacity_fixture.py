from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage


OUTFITS = ("O01", "O02", "O03", "O04", "O06", "O07", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = dict(zip(CONDITIONS, ("front", "back", "left", "right"), strict=True))
EXPECTED_SEGMENTER = "mattmdjaga/segformer_b2_clothes"
EXPECTED_REVISION = "584abc1e1d260e23c0fc627c5217a09b2b461046"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("L")) >= 128


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"))


def save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, "L").save(path)


def morph(value: np.ndarray, radius: int, operation: str) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    code = cv2.MORPH_DILATE if operation == "dilate" else cv2.MORPH_ERODE
    return cv2.morphologyEx(value.astype(np.uint8), code, kernel) > 0


def relative_or_absolute(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if tuple(value["candidate_outfits"]) != OUTFITS:
        raise ValueError("candidate outfits differ from the preregistered contract")
    conditions = tuple(item["condition_id"] for item in value["conditions"])
    if conditions != CONDITIONS:
        raise ValueError("condition order differs from the preregistered contract")
    segmentation = value["segmentation"]
    if segmentation["model"] != EXPECTED_SEGMENTER or segmentation["revision"] != EXPECTED_REVISION:
        raise ValueError("frozen SegFormer identity changed")
    if not segmentation["local_files_only"] or float(segmentation["garment_core_probability_min"]) != 0.35:
        raise ValueError("frozen SegFormer execution contract changed")
    return value


def raw_target(raw_root: Path, outfit: str, condition: str) -> Path:
    return raw_root / outfit / condition / "raw_direct_edit.png"


def prepare_segmentation(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve()
    destination = output / "contract" / "segformer_input_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    source = args.condition_assets_root.resolve()
    items = []
    for outfit in OUTFITS:
        for condition in CONDITIONS:
            target = raw_target(args.raw_root.resolve(), outfit, condition)
            hand = source / "output/condition_part_masks_v3/hands" / f"{condition}.png"
            feet = source / "output/condition_part_masks_v3/feet" / f"{condition}.png"
            condition_mask = source / "input/masks" / f"{condition}.png"
            for path in (target, hand, feet, condition_mask):
                if not path.is_file():
                    raise FileNotFoundError(path)
            with Image.open(target) as image:
                image.load()
                if image.size != (1024, 1536) or image.convert("RGB").getbbox() is None:
                    raise ValueError(f"invalid raw target: {target}")
            items.append({
                "item_id": f"generated_{condition}_{outfit}",
                "kind": "donor",
                "condition_id": condition,
                "outfit_id": outfit,
                "image_path": str(target),
                "source_condition_mask_path": str(condition_mask),
                "projected_hand_mask_path": str(hand),
                "projected_feet_mask_path": str(feet),
                "image_sha256": sha256(target),
            })
    if len(items) != 28 or len({item["item_id"] for item in items}) != 28:
        raise AssertionError("SegFormer input must contain 28 unique targets")
    atomic_json(destination, {
        "schema_version": "canondressgs.aaai27.segformer_input.v1",
        "status": "READY",
        "model": config["segmentation"]["model"],
        "revision": config["segmentation"]["revision"],
        "local_files_only": True,
        "garment_core_probability_min": 0.35,
        "items": items,
    })
    atomic_json(output / "contract" / "input_raw_sha256.json", {
        item["item_id"]: item["image_sha256"] for item in items
    })
    print(json.dumps({"status": "READY", "items": len(items), "manifest": str(destination)}, indent=2))


def derive_radius(records: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    widths = []
    for record in records:
        files = record["output_files"]
        soft_key = "donor_selected_garment_soft.npy"
        soft = np.load(files[soft_key]["path"]).astype(np.float32)
        binary = soft >= 0.5
        band = (soft > 0.05) & (soft < 0.95)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        if perimeter > 0:
            widths.append(float(band.sum() / perimeter))
    median = float(np.median(widths)) if widths else 2.0
    radius = int(np.clip(math.ceil(median), 2, 5))
    return radius, {
        "method": "V5.3 frozen antialias-band estimator, ceil median clamped to [2,5]",
        "per_mask_estimates": widths,
        "median_estimated_antialias_width_pixels": median,
        "selected_global_radius_pixels": radius,
    }


def thumb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    value = image.copy()
    value.thumbnail(size, Image.Resampling.LANCZOS)
    return value


def grid(path: Path, items: list[tuple[str, Image.Image]], columns: int, cell: tuple[int, int]) -> None:
    label = 24
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (columns * cell[0], rows * (cell[1] + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(items):
        x, y = (index % columns) * cell[0], (index // columns) * (cell[1] + label)
        draw.text((x + 4, y + 4), name, fill="black")
        preview = thumb(image.convert("RGB"), cell)
        canvas.paste(preview, (x + (cell[0] - preview.width) // 2, y + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def overlay(image: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]]) -> Image.Image:
    value = image.astype(np.float32).copy()
    for region, color in masks:
        value[region] = 0.55 * value[region] + 0.45 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(value, 0, 255).astype(np.uint8), "RGB")


def bbox(value: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(value)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def build_fixture(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve()
    dataset = output / "dataset"
    manifest_path = dataset / "aaai_gate_28_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    segmenter = read_json(args.segmentation_manifest.resolve())
    if segmenter.get("status") != "SUCCESS" or int(segmenter.get("item_count", -1)) != 28:
        raise ValueError("SegFormer run must contain 28 successful records")
    if not segmenter.get("local_files_only") or segmenter.get("selected_strategy") != "C":
        raise ValueError("SegFormer must use frozen local strategy C")
    records = {record["item_id"].removeprefix("generated_"): record for record in segmenter["records"]}
    base_manifest_path = args.base_fixture.resolve()
    base_root = base_manifest_path.parent
    base_fixture = read_json(base_manifest_path)
    base_observations = {item["condition_id"]: item for item in base_fixture["outfits"][0]["observations"]}
    conditions = base_fixture["conditions"]
    radius, radius_evidence = derive_radius(list(records.values()))
    visual_records = read_json(args.visual_gate_records.resolve())
    visual_by_sample = {item["sample_id"]: item for item in visual_records["records"]}
    input_hashes = read_json(output / "contract" / "input_raw_sha256.json")
    mask_records, outfits, contact, support_rows = [], [], [], []
    distance_maps: dict[str, np.ndarray] = {}

    for condition, base_obs in base_observations.items():
        base_fg = mask(relative_or_absolute(base_root, base_obs["target_base_foreground_mask"]))
        distance_maps[condition] = ndimage.distance_transform_edt(~base_fg)

    for outfit in OUTFITS:
        observations = []
        for condition in CONDITIONS:
            sample_id = f"{condition}_{outfit}"
            record = records[sample_id]
            files = record["output_files"]
            raw_path = raw_target(args.raw_root.resolve(), outfit, condition)
            if sha256(raw_path) != input_hashes[f"generated_{sample_id}"]:
                raise RuntimeError(f"raw target changed after generation: {sample_id}")
            base_obs = base_observations[condition]
            base_rgb_path = relative_or_absolute(base_root, base_obs["target_base_rgb"])
            base_fg = mask(relative_or_absolute(base_root, base_obs["target_base_foreground_mask"]))
            old = mask(relative_or_absolute(base_root, base_obs["target_old_clothing_mask"]))
            protected = mask(relative_or_absolute(base_root, base_obs["target_protected_mask"]))
            raw_fg = mask(Path(files["person_foreground.png"]["path"]))
            clothing_raw = mask(Path(files["donor_selected_garment_mask.png"]["path"]))
            raw_skin = mask(Path(files["donor_arm_leg_skin_mask.png"]["path"]))
            if any(value.shape != base_fg.shape for value in (old, protected, raw_fg, clothing_raw, raw_skin)):
                raise ValueError(f"mask shape mismatch: {sample_id}")
            safe = clothing_raw & raw_fg & ~protected
            change = old | safe
            support = morph(change, radius, "dilate")
            core = morph(change, radius, "erode")
            transition = support & ~core & ~protected
            edit = support & ~protected
            preserve = ~support | protected
            edit_core = core & ~protected
            revealed = old & morph(raw_skin, 1, "erode") & ~safe & ~morph(protected, 1, "dilate")
            values = {
                "target_edit_mask": edit,
                "target_edit_core_mask": edit_core,
                "target_preserve_mask": preserve,
                "target_transition_mask": transition,
                "target_protected_mask": protected,
                "target_foreground_mask": raw_fg,
                "target_base_foreground_mask": base_fg,
                "target_clothing_mask": safe,
                "target_clothing_mask_raw": clothing_raw,
                "target_old_clothing_mask": old,
                "target_revealed_skin_mask": revealed,
            }
            paths: dict[str, Path] = {}
            for name, value in values.items():
                path = dataset / "masks" / name / outfit / f"{condition}.png"
                save_mask(path, value)
                paths[name] = path
            edit_rgb = dataset / "rgb/edit" / outfit / f"{condition}.png"
            edit_rgb.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_path, edit_rgb)
            if sha256(edit_rgb) != sha256(raw_path):
                raise RuntimeError("byte-exact raw target copy failed")
            checks = {
                "edit_protected_overlap": int((edit & protected).sum()),
                "protected_outside_preserve": int((protected & ~preserve).sum()),
                "edit_preserve_overlap": int((edit & preserve).sum()),
                "edit_preserve_uncovered": int((~(edit | preserve)).sum()),
                "core_outside_edit": int((edit_core & ~edit).sum()),
                "transition_outside_edit": int((transition & ~edit).sum()),
                "safe_clothing_formula_mismatch": int(np.logical_xor(safe, clothing_raw & raw_fg & ~protected).sum()),
                "clothing_protected_overlap": int((safe & protected).sum()),
                "clothing_outside_foreground": int((safe & ~raw_fg).sum()),
            }
            if any(checks.values()):
                raise AssertionError(f"V5.3 mask contract failed: {sample_id}: {checks}")
            raw_sha = sha256(raw_path)
            observation = {
                "condition_id": condition,
                "rgb": str(edit_rgb),
                "foreground_mask": str(paths["target_foreground_mask"]),
                "clothing_mask": str(paths["target_clothing_mask"]),
                "target_edit_rgb": str(edit_rgb),
                "target_base_rgb": str(base_rgb_path),
                **{name: str(path) for name, path in paths.items()},
                "checksums": {
                    "raw_direct_edit": raw_sha,
                    "target_edit_rgb": sha256(edit_rgb),
                    "target_base_rgb": sha256(base_rgb_path),
                    **{name: sha256(path) for name, path in paths.items()},
                },
                "identity_audit_status": visual_by_sample[sample_id].get(
                    "visual_status", visual_by_sample[sample_id].get("status")
                ),
                "raw_generation_provider": "CODEX_IMAGE_GENERATION_SKILL",
            }
            observations.append(observation)
            pixel_counts = {name: int(value.sum()) for name, value in values.items()}
            mask_records.append({
                "sample_id": sample_id, "outfit_id": outfit, "condition_id": condition,
                "view": VIEWS[condition], "radius_pixels": radius,
                "pixel_counts": pixel_counts, "checks": checks,
                "paths": {name: str(path) for name, path in paths.items()},
                "checksums": {name: sha256(path) for name, path in paths.items()},
                "raw_target_sha256": raw_sha,
            })
            new_silhouette = raw_fg & ~base_fg
            base_box = bbox(base_fg)
            target_box = bbox(raw_fg)
            diagonal = math.hypot(base_box[2] - base_box[0], base_box[3] - base_box[1]) if base_box else 1.0
            distances = distance_maps[condition][new_silhouette]
            support_rows.append({
                "sample_id": sample_id, "outfit_id": outfit, "condition_id": condition, "view": VIEWS[condition],
                "revealed_skin_ratio": float(revealed.sum() / max(raw_fg.sum(), 1)),
                "new_silhouette_ratio": float(new_silhouette.sum() / max(raw_fg.sum(), 1)),
                "support_shift_mean_normalized": float(distances.mean() / diagonal) if distances.size else 0.0,
                "support_shift_max_normalized": float(distances.max() / diagonal) if distances.size else 0.0,
                "old_clothing_removal_ratio": float((old & ~safe).sum() / max(old.sum(), 1)),
                "foreground_bbox_area_ratio": float(((target_box[2]-target_box[0])*(target_box[3]-target_box[1])) / max((base_box[2]-base_box[0])*(base_box[3]-base_box[1]), 1)) if base_box and target_box else math.nan,
                "clothing_area_ratio": float(safe.sum() / max(raw_fg.sum(), 1)),
                "protected_area_ratio": float(protected.sum() / protected.size),
                "alpha_difference_ratio": float(np.logical_xor(raw_fg, base_fg).mean()),
            })
            raw_image = rgb(raw_path)
            contact.extend([
                (sample_id + " raw", Image.fromarray(raw_image, "RGB")),
                (sample_id + " edit/protected", overlay(raw_image, [(edit, (255, 0, 0)), (protected, (255, 0, 255))])),
                (sample_id + " clothing", overlay(raw_image, [(safe, (0, 255, 0)), (revealed, (255, 160, 0))])),
                (sample_id + " silhouette", overlay(raw_image, [(new_silhouette, (0, 160, 255)), (base_fg, (40, 220, 40))])),
            ])
        outfits.append({
            "outfit_id": outfit,
            "metadata": {"fixture": True, "supervision_mode": "dual_target_region_aware_v1"},
            "observations": observations,
        })
    fixture = {
        "schema_version": "canondressgs.full_dataset.v1",
        "dataset_kind": "regression",
        "fixture_id": "AAAI_GATE_28",
        "fixture_mode": True,
        "expected_outfits": list(OUTFITS),
        "expected_condition_count": 4,
        "supervision_mode": "dual_target_region_aware_v1",
        "conditions": conditions,
        "splits": {"train": list(OUTFITS), "val": [], "test": []},
        "identity_visual_adjudication": {"excluded_samples": [], "generation_visual_gate": "PASS"},
        "outfits": outfits,
    }
    atomic_json(manifest_path, fixture)
    atomic_json(output / "masks" / "dual_target_mask_manifest.json", {
        "schema_version": "canondressgs.aaai27.dual_target_masks.v1",
        "status": "PASS", "record_count": len(mask_records),
        "segmentation": config["segmentation"], "radius_evidence": radius_evidence,
        "records": mask_records,
    })
    write_csv(output / "support_compatibility" / "support_compatibility_metrics.csv", support_rows)
    summaries = []
    for outfit in OUTFITS:
        rows = [row for row in support_rows if row["outfit_id"] == outfit]
        summaries.append({
            "outfit_id": outfit,
            "revealed_skin_ratio_mean": float(np.mean([row["revealed_skin_ratio"] for row in rows])),
            "revealed_skin_ratio_max": max(row["revealed_skin_ratio"] for row in rows),
            "new_silhouette_ratio_mean": float(np.mean([row["new_silhouette_ratio"] for row in rows])),
            "new_silhouette_ratio_max": max(row["new_silhouette_ratio"] for row in rows),
            "support_shift_mean_normalized": float(np.mean([row["support_shift_mean_normalized"] for row in rows])),
            "support_shift_max_normalized": max(row["support_shift_max_normalized"] for row in rows),
            "risk": "PENDING_ACTUAL_IMAGE_INSPECTION",
        })
    atomic_json(output / "support_compatibility" / "support_compatibility_summary.json", {
        "status": "PENDING_ACTUAL_IMAGE_INSPECTION", "outfits": summaries,
    })
    grid(output / "support_compatibility" / "support_compatibility_contact_sheet.png", contact, 4, (256, 384))
    atomic_text(output / "support_compatibility" / "SUPPORT_COMPATIBILITY_AUDIT.md", "\n".join([
        "# Support Compatibility Audit", "",
        "Quantitative metrics are complete. Risk labels remain pending actual opening of the contact sheet; no hidden numeric threshold is substituted for the preregistered qualitative silhouette/support criteria.",
    ]))
    print(json.dumps({"status": "BUILT", "samples": len(mask_records), "radius": radius, "manifest": str(manifest_path)}, indent=2))


def finalize_support(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    output = args.output.resolve()
    summary_path = output / "support_compatibility" / "support_compatibility_summary.json"
    summary = read_json(summary_path)
    decisions = read_json(args.support_decisions.resolve())
    if set(decisions) != set(OUTFITS):
        raise ValueError("support decisions must cover the seven preregistered outfits")
    by_outfit = {item["outfit_id"]: item for item in summary["outfits"]}
    for outfit, decision in decisions.items():
        if decision.get("risk") not in {"LOW_SUPPORT_RISK", "MEDIUM_SUPPORT_RISK", "HIGH_SUPPORT_RISK"}:
            raise ValueError(f"invalid support risk: {outfit}")
        if not decision.get("images_actually_opened") or not decision.get("observations"):
            raise ValueError(f"support decision lacks visual evidence: {outfit}")
        if by_outfit[outfit]["revealed_skin_ratio_max"] > float(config["support_compatibility"]["revealed_skin_ratio_low_max"]):
            if decision["risk"] == "LOW_SUPPORT_RISK":
                raise ValueError(f"{outfit} exceeds preregistered revealed-skin LOW threshold")
        by_outfit[outfit].update(decision)
    summary["status"] = "COMPLETE"
    summary["inspection_method"] = "actual image opening with Codex view_image"
    atomic_json(summary_path, summary)
    lines = ["# Support Compatibility Audit", "", "Actual contact-sheet inspection completed.", ""]
    for outfit in OUTFITS:
        item = by_outfit[outfit]
        lines.append(f"- {outfit}: **{item['risk']}** — {item['observations']}")
    atomic_text(output / "support_compatibility" / "SUPPORT_COMPATIBILITY_AUDIT.md", "\n".join(lines))
    print(json.dumps({"status": "COMPLETE", "risks": {key: value["risk"] for key, value in by_outfit.items()}}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and audit the preregistered AAAI 28-image V5.3 fixture")
    parser.add_argument("--phase", required=True, choices=("prepare-segmentation", "build", "finalize-support"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--condition-assets-root", type=Path)
    parser.add_argument("--base-fixture", type=Path)
    parser.add_argument("--segmentation-manifest", type=Path)
    parser.add_argument("--visual-gate-records", type=Path)
    parser.add_argument("--support-decisions", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = validate_config(args.config.resolve())
    if args.phase == "prepare-segmentation":
        prepare_segmentation(args, config)
    elif args.phase == "build":
        build_fixture(args, config)
    else:
        finalize_support(args, config)


if __name__ == "__main__":
    main()
