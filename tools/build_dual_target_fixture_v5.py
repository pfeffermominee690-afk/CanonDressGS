from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity


CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
OUTFITS = ("O00", "O01", "O05")
VIEWS = {
    "cond_000000": "front",
    "cond_000318": "back",
    "cond_000017": "left",
    "cond_000347": "right",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.size != (1024, 1536):
            raise ValueError(f"{path}: expected 1024x1536, got {image.size}")
        return np.asarray(image.convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.size != (1024, 1536):
            raise ValueError(f"{path}: expected 1024x1536, got {image.size}")
        return np.asarray(image.convert("L")) >= 128


def _save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(value), 0, 255).astype(np.uint8), "RGB").save(path)


def _save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value.astype(np.uint8) * 255, "L").save(path)


def _morph(mask: np.ndarray, radius: int, op: str) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    code = cv2.MORPH_DILATE if op == "dilate" else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), code, kernel) > 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_metrics(base_mask: np.ndarray, edit_mask: np.ndarray) -> dict[str, float | list[int] | None]:
    base_box, edit_box = _bbox(base_mask), _bbox(edit_mask)
    if base_box is None or edit_box is None:
        return {
            "base_bbox": list(base_box) if base_box else None,
            "edit_bbox": list(edit_box) if edit_box else None,
            "bbox_center_residual_diag": math.nan,
            "bbox_scale_ratio": math.nan,
        }
    base_center = np.array([(base_box[0] + base_box[2]) / 2, (base_box[1] + base_box[3]) / 2])
    edit_center = np.array([(edit_box[0] + edit_box[2]) / 2, (edit_box[1] + edit_box[3]) / 2])
    diagonal = math.hypot(1024, 1536)
    base_height = max(base_box[3] - base_box[1], 1)
    edit_height = edit_box[3] - edit_box[1]
    return {
        "base_bbox": list(base_box),
        "edit_bbox": list(edit_box),
        "bbox_center_residual_diag": float(np.linalg.norm(edit_center - base_center) / diagonal),
        "bbox_scale_ratio": float(edit_height / base_height),
    }


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _region_metrics(
    base: np.ndarray,
    edit: np.ndarray,
    mask: np.ndarray,
    ssim_map: np.ndarray,
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {
            "pixel_count": 0,
            "mean_rgb_absolute_difference": 0.0,
            "max_rgb_absolute_difference": 0.0,
            "changed_pixel_fraction": 0.0,
            "lab_delta_e_median": 0.0,
            "lab_delta_e_p95": 0.0,
            "ssim": 1.0,
        }
    difference = np.abs(edit.astype(np.int16) - base.astype(np.int16))
    changed = np.max(difference, axis=2) > 1
    delta = np.linalg.norm(_lab(edit) - _lab(base), axis=2)
    values = ssim_map[mask]
    return {
        "pixel_count": count,
        "mean_rgb_absolute_difference": float(difference[mask].mean() / 255.0),
        "max_rgb_absolute_difference": float(difference[mask].max() / 255.0),
        "changed_pixel_fraction": float(changed[mask].mean()),
        "lab_delta_e_median": float(np.median(delta[mask])),
        "lab_delta_e_p95": float(np.percentile(delta[mask], 95)),
        "ssim": float(values.mean()),
    }


def _robust_skin_statistics(rgb: np.ndarray, mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    interior = _morph(mask, 2, "erode")
    lab = _lab(rgb)
    pixels = lab[interior]
    rgb_pixels = rgb[interior].astype(np.float32)
    if len(pixels) < 32:
        raise ValueError("subject02 skin reference has fewer than 32 reliable pixels")
    luminance = pixels[:, 0]
    lo, hi = np.percentile(luminance, (10, 90))
    keep = (luminance >= lo) & (luminance <= hi)
    pixels, rgb_pixels = pixels[keep], rgb_pixels[keep]
    median = np.median(pixels, axis=0)
    delta = np.linalg.norm(pixels - median, axis=1)
    sorted_pixels = np.sort(pixels, axis=0)
    trim = max(1, int(len(sorted_pixels) * 0.1))
    trimmed = sorted_pixels[trim:-trim] if len(sorted_pixels) > 2 * trim else sorted_pixels
    stats = {
        "reliable_pixel_count": int(len(pixels)),
        "lab_median": median.tolist(),
        "lab_trimmed_mean": trimmed.mean(axis=0).tolist(),
        "lab_p10": np.percentile(pixels, 10, axis=0).tolist(),
        "lab_p50": np.percentile(pixels, 50, axis=0).tolist(),
        "lab_p90": np.percentile(pixels, 90, axis=0).tolist(),
        "rgb_median": np.median(rgb_pixels, axis=0).tolist(),
        "rgb_trimmed_mean": np.sort(rgb_pixels, axis=0)[trim:-trim].mean(axis=0).tolist(),
        "rgb_p10": np.percentile(rgb_pixels, 10, axis=0).tolist(),
        "rgb_p50": np.percentile(rgb_pixels, 50, axis=0).tolist(),
        "rgb_p90": np.percentile(rgb_pixels, 90, axis=0).tolist(),
        "intra_identity_delta_e_median": float(np.median(delta)),
        "intra_identity_delta_e_p95": float(np.percentile(delta, 95)),
    }
    return stats, pixels


def _correct_revealed_skin(
    raw: np.ndarray,
    mask: np.ndarray,
    reference_pixels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = raw.copy()
    raw_lab = _lab(raw)
    values = raw_lab[mask]
    ref_median = np.median(reference_pixels, axis=0)
    ref_delta = np.linalg.norm(reference_pixels - ref_median, axis=1)
    if len(values) < 32:
        return output, {
            "applied": False,
            "reason": "fewer_than_32_revealed_skin_pixels",
            "before_median_delta_e": 0.0,
            "before_p95_delta_e": 0.0,
            "after_median_delta_e": 0.0,
            "after_p95_delta_e": 0.0,
            "subject02_intra_skin_median_delta_e": float(np.median(ref_delta)),
            "subject02_intra_skin_p95_delta_e": float(np.percentile(ref_delta, 95)),
            "improvement_ratio_median": 0.0,
        }
    raw_median = np.median(values, axis=0)
    raw_iqr = np.maximum(np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0), 1.0)
    ref_iqr = np.maximum(
        np.percentile(reference_pixels, 75, axis=0) - np.percentile(reference_pixels, 25, axis=0),
        1.0,
    )
    chroma_scale = np.clip(ref_iqr[1:] / raw_iqr[1:], 0.75, 1.25)
    corrected_values = values.copy()
    corrected_values[:, 1:] = (values[:, 1:] - raw_median[1:]) * chroma_scale + ref_median[1:]
    ref_luminance_range = max(float(np.percentile(reference_pixels[:, 0], 90) - np.percentile(reference_pixels[:, 0], 10)), 1.0)
    luminance_shift = float(np.clip(ref_median[0] - raw_median[0], -0.5 * ref_luminance_range, 0.5 * ref_luminance_range))
    corrected_values[:, 0] = values[:, 0] + luminance_shift
    lower = np.percentile(reference_pixels, 1, axis=0)
    upper = np.percentile(reference_pixels, 99, axis=0)
    corrected_values = np.clip(corrected_values, lower, upper)
    corrected_lab = raw_lab.copy()
    corrected_lab[mask] = corrected_values
    corrected_rgb = cv2.cvtColor(np.clip(corrected_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / 2.0, 0.0, 1.0)[..., None]
    alpha[~mask] = 0.0
    output = np.rint(raw.astype(np.float32) * (1 - alpha) + corrected_rgb.astype(np.float32) * alpha).astype(np.uint8)
    before = np.linalg.norm(values - ref_median, axis=1)
    after_values = _lab(output)[mask]
    after = np.linalg.norm(after_values - ref_median, axis=1)
    before_median = float(np.median(before))
    after_median = float(np.median(after))
    return output, {
        "applied": True,
        "method": "Lab chroma median shift plus robust scale; bounded L shift; reference percentile clipping; two-pixel interior feather",
        "raw_lab_median": raw_median.tolist(),
        "reference_lab_median": ref_median.tolist(),
        "chroma_scale_ab": chroma_scale.tolist(),
        "luminance_shift": luminance_shift,
        "before_median_delta_e": before_median,
        "before_p95_delta_e": float(np.percentile(before, 95)),
        "after_median_delta_e": after_median,
        "after_p95_delta_e": float(np.percentile(after, 95)),
        "subject02_intra_skin_median_delta_e": float(np.median(ref_delta)),
        "subject02_intra_skin_p95_delta_e": float(np.percentile(ref_delta, 95)),
        "improvement_ratio_median": float(1.0 - after_median / max(before_median, 1e-8)),
    }


def _thumb(value: np.ndarray, size: tuple[int, int] = (256, 384)) -> Image.Image:
    return ImageOps.contain(Image.fromarray(value), size, Image.Resampling.LANCZOS)


def _mask_rgb(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    value = np.zeros((*mask.shape, 3), dtype=np.uint8)
    value[mask] = color
    return value


def _overlay(rgb: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    value = rgb.astype(np.float32).copy()
    for mask, color in masks:
        value[mask] = 0.55 * value[mask] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.clip(value, 0, 255).astype(np.uint8)


def _contact_sheet(
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]],
    path: Path,
    cell_size: tuple[int, int] = (256, 384),
) -> None:
    if not rows:
        return
    columns = max(len(items) for _, items in rows)
    header, label = 28, 24
    canvas = Image.new("RGB", (columns * cell_size[0], len(rows) * (cell_size[1] + label + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (row_name, items) in enumerate(rows):
        y = row_index * (cell_size[1] + label + header)
        draw.text((4, y + 4), row_name, fill="black")
        for column, (name, image) in enumerate(items):
            x = column * cell_size[0]
            draw.text((x + 4, y + header + 2), name, fill="black")
            thumb = _thumb(image, cell_size)
            canvas.paste(thumb, (x + (cell_size[0] - thumb.width) // 2, y + header + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _derive_radius(records: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    widths = []
    for record in records:
        soft_path = Path(record["output_files"]["base_selected_garment_soft.npy"]["path"])
        soft = np.load(soft_path).astype(np.float32)
        binary = soft >= 0.5
        band = (soft > 0.05) & (soft < 0.95)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        if perimeter > 0:
            widths.append(float(band.sum() / perimeter))
    median_width = float(np.median(widths)) if widths else 2.0
    radius = int(np.clip(math.ceil(median_width), 2, 5))
    return radius, {
        "method": "ceil(median(soft 0.05-0.95 band pixels / binary contour perimeter)), clamped to [2,5]",
        "per_mask_estimates": widths,
        "median_estimated_antialias_width_pixels": median_width,
        "selected_global_radius_pixels": radius,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SUBJECT02 V5 dual-target fixture")
    parser.add_argument("--v4-audit", type=Path, required=True)
    parser.add_argument("--v4-staging", type=Path, required=True)
    parser.add_argument("--v3a-manifest", type=Path, required=True)
    parser.add_argument("--generated-mask-manifest", type=Path, required=True)
    parser.add_argument("--condition-contract", type=Path, required=True)
    parser.add_argument("--v3-condition-contract", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    args = parser.parse_args()
    audit, staging = args.audit_dir.resolve(), args.staging_dir.resolve()
    audit.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    raw_manifest = _read_json(args.v4_audit / "direct_edit_raw_manifest_v4.json")
    if raw_manifest.get("status") != "SUCCESS" or len(raw_manifest["records"]) != 12:
        raise ValueError("V4 raw manifest must contain 12 successful records")
    base_manifest = _read_json(args.v3a_manifest)
    generated_manifest = _read_json(args.generated_mask_manifest)
    condition_contract = _read_json(args.condition_contract)
    v3_contract = _read_json(args.v3_condition_contract)
    base_by_condition = {
        record["condition_id"]: record
        for record in base_manifest["records"]
        if record["kind"] == "base"
    }
    generated_by_sample = {
        record["item_id"].removeprefix("generated_"): record
        for record in generated_manifest["records"]
    }
    raw_by_sample = {record["sample_id"]: record for record in raw_manifest["records"]}
    condition_by_id = {record["condition_id"]: record for record in condition_contract["conditions"]}
    v3_by_id = {record["condition_id"]: record for record in v3_contract["conditions"]}
    radius, radius_evidence = _derive_radius(list(base_by_condition.values()) + list(generated_by_sample.values()))

    subject_skin_stats: dict[str, Any] = {
        "schema_version": "subject02.skin_reference.v5",
        "created_at": _now(),
        "identity_source": "subject02 base only",
        "jay_skin_used": False,
        "views": {},
    }
    base_cache: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        record = base_by_condition[condition]
        files = record["output_files"]
        base = _read_rgb(Path(record["image_path"]))
        old = _read_mask(Path(files["base_selected_garment_mask.png"]["path"]))
        face = _read_mask(Path(files["base_face_seed.png"]["path"]))
        hair = _read_mask(Path(files["base_hair_seed.png"]["path"]))
        skin = _read_mask(Path(files["base_arm_leg_skin_mask.png"]["path"]))
        hands = _read_mask(Path(files["projected_hand_protection.png"]["path"]))
        shoes = _read_mask(Path(files["base_shoe_seed.png"]["path"])) | _read_mask(Path(files["projected_feet_protection.png"]["path"]))
        foreground = _read_mask(Path(files["person_foreground.png"]["path"]))
        protected = face | hair | hands | shoes
        reliable_skin = (face | hands | skin) & ~hair & ~shoes & ~_morph(old, 2, "dilate")
        stats, reference_pixels = _robust_skin_statistics(base, reliable_skin)
        subject_skin_stats["views"][VIEWS[condition]] = {"condition_id": condition, **stats}
        base_cache[condition] = {
            "record": record,
            "rgb": base,
            "old": old,
            "face": face,
            "hair": hair,
            "skin": skin,
            "hands": hands,
            "shoes": shoes,
            "foreground": foreground,
            "protected": protected,
            "reference_pixels": reference_pixels,
        }
    _write_json(audit / "subject02_skin_reference_statistics_v5.json", subject_skin_stats)

    identity_rows: list[dict[str, Any]] = []
    skin_rows: list[dict[str, Any]] = []
    mask_records: list[dict[str, Any]] = []
    identity_contact, mask_contact, revealed_contact, skin_contact = [], [], [], []
    manifest_outfits: list[dict[str, Any]] = []
    for outfit in OUTFITS:
        observations = []
        for condition in CONDITIONS:
            sample = f"{condition}_{outfit}"
            base_data = base_cache[condition]
            base = base_data["rgb"]
            generated_record = generated_by_sample[sample]
            raw_record = raw_by_sample[sample]
            raw = _read_rgb(Path(raw_record["output_path"]))
            generated_files = generated_record["output_files"]
            raw_foreground = _read_mask(Path(generated_files["person_foreground.png"]["path"]))
            new = _read_mask(Path(generated_files["base_selected_garment_mask.png"]["path"]))
            raw_face = _read_mask(Path(generated_files["base_face_seed.png"]["path"]))
            raw_hair = _read_mask(Path(generated_files["base_hair_seed.png"]["path"]))
            raw_skin = _read_mask(Path(generated_files["base_arm_leg_skin_mask.png"]["path"]))
            raw_hands = _read_mask(Path(generated_files["projected_hand_protection.png"]["path"]))
            raw_shoes = _read_mask(Path(generated_files["base_shoe_seed.png"]["path"])) | _read_mask(Path(generated_files["projected_feet_protection.png"]["path"]))
            old, protected = base_data["old"], base_data["protected"]
            new = new & ~protected
            change = old | new
            support = _morph(change, radius, "dilate")
            core = _morph(change, radius, "erode")
            transition = support & ~core
            edit = support & ~protected
            preserve = ~support | protected
            edit_core = core & ~protected
            transition_target = transition & ~protected
            revealed = old & _morph(raw_skin, 1, "erode") & ~new & ~_morph(protected, 1, "dilate")
            corrected, correction = _correct_revealed_skin(raw, revealed, base_data["reference_pixels"]) if outfit == "O00" else (raw.copy(), {"applied": False, "reason": "outfit_rule_no_revealed_skin_correction"})

            safe_background = ~(base_data["foreground"] | raw_foreground)
            half = np.indices(base_data["hands"].shape)[1] < 512
            regions = {
                "face": base_data["face"],
                "hair": base_data["hair"],
                "left_hand": base_data["hands"] & half,
                "right_hand": base_data["hands"] & ~half,
                "shoes": base_data["shoes"],
                "background": safe_background,
                "base_visible_skin": (base_data["face"] | base_data["hands"] | base_data["skin"]) & ~old,
                "non_edit_foreground": base_data["foreground"] & ~support & ~protected,
            }
            _, ssim_map = structural_similarity(base, raw, channel_axis=2, data_range=255, full=True)
            if ssim_map.ndim == 3:
                ssim_map = ssim_map.mean(axis=2)
            raw_region_masks = {
                "face": raw_face,
                "hair": raw_hair,
                "left_hand": raw_hands & half,
                "right_hand": raw_hands & ~half,
                "shoes": raw_shoes,
            }
            for region, region_mask in regions.items():
                bbox_values = _bbox_metrics(region_mask, raw_region_masks.get(region, region_mask))
                identity_rows.append({
                    "sample_id": sample,
                    "condition_id": condition,
                    "view": VIEWS[condition],
                    "outfit_id": outfit,
                    "region": region,
                    **_region_metrics(base, raw, region_mask, ssim_map),
                    **bbox_values,
                })

            corrected_path = staging / "raw_edit_skin_corrected" / outfit / f"{condition}.png"
            _save_rgb(corrected_path, corrected)
            field_paths = {
                "target_edit_mask": staging / "dual_target_masks_v5" / outfit / "target_edit_mask" / f"{condition}.png",
                "target_edit_core_mask": staging / "dual_target_masks_v5" / outfit / "target_edit_core_mask" / f"{condition}.png",
                "target_preserve_mask": staging / "dual_target_masks_v5" / outfit / "target_preserve_mask" / f"{condition}.png",
                "target_transition_mask": staging / "dual_target_masks_v5" / outfit / "target_transition_mask" / f"{condition}.png",
                "target_protected_mask": staging / "dual_target_masks_v5" / outfit / "target_protected_mask" / f"{condition}.png",
                "target_foreground_mask": staging / "dual_target_masks_v5" / outfit / "target_foreground_mask" / f"{condition}.png",
                "target_base_foreground_mask": staging / "dual_target_masks_v5" / outfit / "target_base_foreground_mask" / f"{condition}.png",
                "target_clothing_mask": staging / "dual_target_masks_v5" / outfit / "target_clothing_mask" / f"{condition}.png",
                "target_old_clothing_mask": staging / "dual_target_masks_v5" / outfit / "target_old_clothing_mask" / f"{condition}.png",
                "target_revealed_skin_mask": staging / "revealed_skin_masks_v5" / outfit / f"{condition}.png",
            }
            mask_values = {
                "target_edit_mask": edit,
                "target_edit_core_mask": edit_core,
                "target_preserve_mask": preserve,
                "target_transition_mask": transition_target,
                "target_protected_mask": protected,
                "target_foreground_mask": raw_foreground,
                "target_base_foreground_mask": base_data["foreground"],
                "target_clothing_mask": new,
                "target_old_clothing_mask": old,
                "target_revealed_skin_mask": revealed,
            }
            for name, path in field_paths.items():
                _save_mask(path, mask_values[name])

            mask_checks = {
                "edit_protected_overlap": int((edit & protected).sum()),
                "protected_outside_preserve": int((protected & ~preserve).sum()),
                "edit_preserve_overlap": int((edit & preserve).sum()),
                "edit_preserve_uncovered": int((~(edit | preserve)).sum()),
                "core_outside_edit": int((edit_core & ~edit).sum()),
                "transition_outside_edit": int((transition_target & ~edit).sum()),
                "clothing_outside_foreground_ratio": float((new & ~raw_foreground).sum() / max(int(new.sum()), 1)),
                "clothing_protected_overlap": int((new & protected).sum()),
            }
            if any(mask_checks[key] for key in ("edit_protected_overlap", "protected_outside_preserve", "edit_preserve_overlap", "edit_preserve_uncovered", "core_outside_edit", "transition_outside_edit", "clothing_protected_overlap")):
                raise AssertionError(f"dual-target mask contract failed for {sample}: {mask_checks}")
            mask_record = {
                "sample_id": sample,
                "condition_id": condition,
                "view": VIEWS[condition],
                "outfit_id": outfit,
                "radius_pixels": radius,
                "pixel_counts": {name: int(value.sum()) for name, value in mask_values.items()},
                "fractions": {name: float(value.mean()) for name, value in mask_values.items()},
                "checks": mask_checks,
                "paths": {name: str(path.resolve()) for name, path in field_paths.items()},
                "checksums": {name: _sha256(path) for name, path in field_paths.items()},
            }
            mask_records.append(mask_record)
            correction.update({
                "sample_id": sample,
                "condition_id": condition,
                "view": VIEWS[condition],
                "outfit_id": outfit,
                "revealed_skin_pixels": int(revealed.sum()),
                "corrected_path": str(corrected_path.resolve()),
                "corrected_sha256": _sha256(corrected_path),
            })
            skin_rows.append(correction)

            condition_info = condition_by_id[condition]
            observation = {
                "condition_id": condition,
                "rgb": str(corrected_path.resolve()),
                "foreground_mask": str(field_paths["target_foreground_mask"].resolve()),
                "clothing_mask": str(field_paths["target_clothing_mask"].resolve()),
                "target_edit_rgb": str(corrected_path.resolve()),
                "target_base_rgb": str(Path(base_data["record"]["image_path"]).resolve()),
                **{name: str(path.resolve()) for name, path in field_paths.items()},
                "checksums": {
                    "raw_direct_edit": raw_record["output_sha256"],
                    "target_edit_rgb": _sha256(corrected_path),
                    "target_base_rgb": _sha256(Path(base_data["record"]["image_path"])),
                    **{name: _sha256(path) for name, path in field_paths.items()},
                },
            }
            observations.append(observation)

            identity_contact.append((sample, [
                ("subject02 base", base),
                ("raw direct edit", raw),
                ("identity regions", _overlay(raw, [(protected, (255, 0, 0)), (safe_background, (0, 0, 255))])),
                ("absolute difference x3", np.clip(np.abs(raw.astype(np.int16) - base.astype(np.int16)) * 3, 0, 255).astype(np.uint8)),
            ]))
            mask_contact.append((sample, [
                ("raw edit", raw),
                ("edit red / preserve blue", _overlay(raw, [(edit, (255, 0, 0)), (preserve, (0, 0, 255))])),
                ("core green / transition yellow", _overlay(raw, [(edit_core, (0, 255, 0)), (transition_target, (255, 255, 0))])),
                ("protected magenta", _overlay(raw, [(protected, (255, 0, 255))])),
            ]))
            revealed_contact.append((sample, [
                ("raw edit", raw),
                ("revealed skin", _overlay(raw, [(revealed, (255, 0, 0))])),
                ("revealed mask", _mask_rgb(revealed, (255, 255, 255))),
            ]))
            if outfit == "O00":
                skin_contact.append((sample, [
                    ("subject02 base", base),
                    ("raw O00", raw),
                    ("revealed mask", _mask_rgb(revealed, (255, 255, 255))),
                    ("skin corrected", corrected),
                    ("correction x4", np.clip(np.abs(corrected.astype(np.int16) - raw.astype(np.int16)) * 4, 0, 255).astype(np.uint8)),
                ]))
        manifest_outfits.append({
            "outfit_id": outfit,
            "metadata": {"fixture": True, "supervision_mode": "dual_target_region_aware_v1"},
            "observations": observations,
        })

    _write_csv(audit / "raw_edit_identity_metrics_v5.csv", identity_rows)
    _write_csv(audit / "skin_color_correction_metrics_v5.csv", skin_rows)
    _write_json(audit / "dual_target_mask_manifest_v5.json", {
        "schema_version": "subject02.dual_target_masks.v5",
        "created_at": _now(),
        "radius_evidence": radius_evidence,
        "records": mask_records,
        "status": "PASS" if all(not any(record["checks"][key] for key in ("edit_protected_overlap", "protected_outside_preserve", "edit_preserve_overlap", "edit_preserve_uncovered", "core_outside_edit", "transition_outside_edit", "clothing_protected_overlap")) for record in mask_records) else "FAIL",
    })
    _contact_sheet(identity_contact, audit / "raw_edit_identity_contact_sheet_v5.png")
    _contact_sheet(mask_contact, audit / "dual_target_mask_contact_sheet_v5.png")
    _contact_sheet(revealed_contact, audit / "revealed_skin_contact_sheet_v5.png")
    _contact_sheet(skin_contact, audit / "skin_color_correction_contact_sheet_v5.png")

    conditions = []
    for condition in CONDITIONS:
        item = condition_by_id[condition]
        original = v3_by_id[condition]
        conditions.append({
            "condition_id": condition,
            "source_frame_id": str(original["source_frame"]),
            "source_camera_id": str(original["source_camera"]),
            "pose": item["pose"],
            "Rh_raw": item["Rh_raw"],
            "R_global": item["R_global"],
            "Th": item["Th"],
            "K": item["K"],
            "w2c": item["w2c"],
            "c2w": item["c2w"],
            "width": item["width"],
            "height": item["height"],
            "background": item["background"],
            "conventions": {
                "pose": item["pose_convention"],
                "camera": item["camera_derivation"],
                "supervision": "dual_target_region_aware_v1",
            },
            "source_checksum": item["source_checksum"],
        })
    fixture = {
        "schema_version": "canondressgs.full_dataset.v1",
        "dataset_kind": "regression",
        "fixture_mode": True,
        "expected_outfits": list(OUTFITS),
        "expected_condition_count": 4,
        "supervision_mode": "dual_target_region_aware_v1",
        "conditions": conditions,
        "splits": {"train": list(OUTFITS), "val": [], "test": []},
        "outfits": manifest_outfits,
    }
    _write_json(audit / "pilot_manifest_full_v1_v5.json", fixture)
    _write_json(audit / "v5_builder_manifest.json", {
        "created_at": _now(),
        "status": "BUILT_PENDING_VISUAL_AND_CODE_ACCEPTANCE",
        "input_files": {
            "v4_raw_manifest": {"path": str((args.v4_audit / "direct_edit_raw_manifest_v4.json").resolve()), "sha256": _sha256(args.v4_audit / "direct_edit_raw_manifest_v4.json")},
            "v3a_manifest": {"path": str(args.v3a_manifest.resolve()), "sha256": _sha256(args.v3a_manifest)},
            "generated_mask_manifest": {"path": str(args.generated_mask_manifest.resolve()), "sha256": _sha256(args.generated_mask_manifest)},
            "condition_contract": {"path": str(args.condition_contract.resolve()), "sha256": _sha256(args.condition_contract)},
        },
        "outputs": {"audit": str(audit), "staging": str(staging)},
        "record_count": 12,
        "imagegen_calls": 0,
        "jay_pixels_copied": False,
        "pixel_composite_target_created": False,
    })
    (audit / "RAW_EDIT_IDENTITY_AUDIT_V5.md").write_text(
        "# Raw Direct Edit Identity Audit V5\n\n"
        "- raw_direct_edit_generation: **PASS**\n"
        "- identity_safe_pixel_composite: **FAIL (V4 historical adjudication; not used by V5)**\n"
        "- region_aware_training_supervision: **PENDING**\n"
        "- visual adjudication: **PENDING_ACTUAL_IMAGE_OPEN**\n\n"
        "Metrics are in `raw_edit_identity_metrics_v5.csv`; visual status is intentionally deferred until the generated contact sheet and all raw edits are opened.\n",
        encoding="utf-8",
    )
    (audit / "SKIN_COLOR_CORRECTION_V5.md").write_text(
        "# Subject02 Revealed-Skin Color Correction V5\n\n"
        "Only O00 newly revealed skin is adjusted. Geometry, local luminance structure, face, hair, hands, shoes, clothing, and background are not replaced. The reference distribution is computed independently per view from subject02 base pixels; Jay skin is never read as a target. See `skin_color_correction_metrics_v5.csv` and `subject02_skin_reference_statistics_v5.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "BUILT_PENDING_VISUAL_AND_CODE_ACCEPTANCE",
        "samples": 12,
        "radius_pixels": radius,
        "audit": str(audit),
        "staging": str(staging),
    }, indent=2))


if __name__ == "__main__":
    main()
