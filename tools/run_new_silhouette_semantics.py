from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = "canondressgs.new_silhouette_semantics.v1"
OUTFITS = ("O01", "O08")
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = dict(zip(CONDITIONS, ("front", "back", "left", "right")))
CATEGORIES = (
    "TRUSTED_GARMENT_EXPANSION",
    "OLD_GARMENT_REMOVAL",
    "PROTECTED_IDENTITY_DIFFERENCE",
    "TARGET_BODY_OR_POSE_DRIFT",
    "GARMENT_TRANSITION_UNCERTAIN",
    "BACKGROUND_OR_GENERATION_ARTIFACT",
    "SEGMENTATION_UNCERTAIN",
)
CATEGORY_COLORS = {
    "TRUSTED_GARMENT_EXPANSION": (0, 220, 80),
    "OLD_GARMENT_REMOVAL": (0, 130, 255),
    "PROTECTED_IDENTITY_DIFFERENCE": (255, 40, 180),
    "TARGET_BODY_OR_POSE_DRIFT": (255, 210, 0),
    "GARMENT_TRANSITION_UNCERTAIN": (150, 80, 255),
    "BACKGROUND_OR_GENERATION_ARTIFACT": (255, 40, 40),
    "SEGMENTATION_UNCERTAIN": (40, 220, 255),
}
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
    "SUBJECT02-NEW-SILHOUETTE-SEMANTICS-001/attempt_001"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("L")) >= 128


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"))


def save_rgb(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(value, dtype=np.uint8), "RGB").save(path)


def morph(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    code = cv2.MORPH_DILATE if operation == "dilate" else cv2.MORPH_ERODE
    return cv2.morphologyEx(mask.astype(np.uint8), code, kernel) > 0


def bbox_diagonal(mask: np.ndarray) -> float:
    y, x = np.nonzero(mask)
    if not len(x):
        return 0.0
    return math.hypot(int(x.max() - x.min() + 1), int(y.max() - y.min() + 1))


def non_largest_components(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels > 0) & (labels != largest)


def distance_to(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.nan, dtype=np.float32)
    return distance_transform_edt(~mask).astype(np.float32)


def manifest_records(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: dict[tuple[str, str], Mapping[str, Any]] = {}
    for outfit in payload["outfits"]:
        outfit_id = outfit["outfit_id"]
        if outfit_id not in OUTFITS:
            continue
        for observation in outfit["observations"]:
            condition = observation["condition_id"]
            if condition in CONDITIONS:
                key = (outfit_id, condition)
                if key in records:
                    raise ValueError(f"duplicate manifest record: {key}")
                records[key] = observation
    expected = {(outfit, condition) for outfit in OUTFITS for condition in CONDITIONS}
    if set(records) != expected:
        raise ValueError(f"manifest protocol mismatch: {sorted(expected ^ set(records))}")
    return records


def trusted_masks(masks: Mapping[str, np.ndarray], ratio: float, minimum: int) -> dict[str, Any]:
    target = masks["target_fg"]
    base = masks["base_fg"]
    safe = masks["safe_clothing"]
    raw_clothing = masks["raw_clothing"]
    old = masks["old_clothing"]
    protected = masks["protected"]
    core = masks["core"]
    transition = masks["transition"]
    target_artifact = non_largest_components(target)
    prediction_artifact = non_largest_components(masks["pred_fg"])
    artifact = target_artifact | prediction_artifact
    radius = max(minimum, int(math.ceil(bbox_diagonal(base) * ratio)))
    evidence = core | safe | old
    base_neighborhood = base & morph(evidence, radius, "dilate")
    support = morph(evidence | base_neighborhood, radius, "dilate")
    expand_raw = target & ~base
    remove_raw = base & ~target
    trusted_expand = expand_raw & safe & ~protected & ~artifact & support
    trusted_remove = remove_raw & old & ~protected
    uncertain = (expand_raw | remove_raw) & ~(trusted_expand | trusted_remove) & ~protected
    return {
        "support_radius_pixels": radius,
        "garment_support_band": support,
        "raw_expansion": expand_raw,
        "raw_removal": remove_raw,
        "trusted_expansion": trusted_expand,
        "trusted_removal": trusted_remove,
        "silhouette_uncertain": uncertain,
        "artifact": artifact,
        "target_artifact": target_artifact,
        "prediction_artifact": prediction_artifact,
        "raw_clothing": raw_clothing,
        "transition": transition,
    }


def classify_errors(error: np.ndarray, kind: str, masks: Mapping[str, np.ndarray], derived: Mapping[str, Any]) -> dict[str, np.ndarray]:
    remaining = error.copy()
    categories = {name: np.zeros_like(error) for name in CATEGORIES}

    def assign(name: str, candidate: np.ndarray) -> None:
        nonlocal remaining
        selected = remaining & candidate
        categories[name] = selected
        remaining &= ~selected

    # Explicit priority: identity safety, artifacts, trusted garment signal,
    # transition, segmentation uncertainty, then residual body/pose drift.
    assign("PROTECTED_IDENTITY_DIFFERENCE", masks["protected"] | morph(masks["protected"], 1, "dilate"))
    assign("BACKGROUND_OR_GENERATION_ARTIFACT", derived["artifact"])
    if kind == "FN":
        assign("TRUSTED_GARMENT_EXPANSION", derived["trusted_expansion"])
    else:
        assign("OLD_GARMENT_REMOVAL", derived["trusted_removal"])
    assign("GARMENT_TRANSITION_UNCERTAIN", masks["transition"] | derived["silhouette_uncertain"] & morph(masks["transition"], 1, "dilate"))
    segmentation = derived["raw_clothing"] & ~masks["safe_clothing"]
    assign("SEGMENTATION_UNCERTAIN", segmentation | derived["silhouette_uncertain"] & derived["raw_clothing"])
    assign("TARGET_BODY_OR_POSE_DRIFT", remaining)
    if remaining.any():
        raise AssertionError("silhouette error classification was not exhaustive")
    total = sum(value.astype(np.uint8) for value in categories.values())
    if np.any(total[error] != 1) or np.any(total[~error] != 0):
        raise AssertionError("silhouette error categories are not exactly-one disjoint")
    return categories


def overlay(image: np.ndarray, categories: Mapping[str, np.ndarray]) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for name, color in CATEGORY_COLORS.items():
        mask = categories[name]
        output[mask] = output[mask] * 0.25 + np.asarray(color, dtype=np.float32) * 0.75
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def binary_overlay(image: np.ndarray, entries: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for mask, color in entries:
        output[mask] = output[mask] * 0.25 + np.asarray(color, dtype=np.float32) * 0.75
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def contact_sheet(path: Path, panels: list[tuple[str, np.ndarray]], columns: int = 4) -> None:
    cell = (256, 384)
    label = 24
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * cell[0], rows * (cell[1] + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, value) in enumerate(panels):
        x = index % columns * cell[0]
        y = index // columns * (cell[1] + label)
        draw.text((x + 3, y + 3), name, fill="black")
        thumb = ImageOps.contain(Image.fromarray(value), cell, Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (cell[0] - thumb.width) // 2, y + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError("unexpected new-silhouette config schema")
    return value


def run_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.output.exists():
        raise FileExistsError(f"append-only output already exists: {args.output}")
    for name in (
        "contract", "input_audit", "silhouette_error_audit", "mask_semantics", "gradient_analysis",
        "S0_historical_t5", "S1_v6_1_semantics", "S2_v6_1_underfill_if_eligible",
        "visual_acceptance", "final_adjudication",
    ):
        (args.output / name).mkdir(parents=True, exist_ok=False)
    status = args.output / "RUN_STATUS.json"
    atomic_json(status, {"status": "AUDIT_RUNNING", "optimizer_steps": 0, "started_at": now()})
    try:
        if git("branch", "--show-current") != config["research_branch"]:
            raise ValueError("audit must run on the registered research branch")
        source = Path(config["source_objective_output"])
        source_aaai = Path(config["source_aaai_output"])
        manifest = Path(config["source_manifest"])
        boundary_contract = json.loads(Path(config["source_boundary_contract"]).read_text(encoding="utf-8"))
        registered_radius = int(boundary_contract["radius_evidence"]["selected_global_radius_pixels"])
        if registered_radius != int(config["support_band"]["source_boundary_radius_pixels"]):
            raise ValueError("pre-existing boundary width contract drifted")
        ratio = float(config["support_band"]["base_bbox_diagonal_ratio"])
        expected_ratio = registered_radius / math.hypot(
            int(config["support_band"]["source_resolution_width"]),
            int(config["support_band"]["source_resolution_height"]),
        )
        if not math.isclose(ratio, expected_ratio, rel_tol=0, abs_tol=1e-15):
            raise ValueError("support radius ratio is not derived from the registered resolution/boundary contract")
        records = manifest_records(manifest)
        rows: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}
        panels: list[tuple[str, np.ndarray]] = []
        source_files: list[dict[str, Any]] = []
        s0: dict[str, Any] = {"rerun": False, "source": str(source), "outfits": {}}
        for outfit in OUTFITS:
            source_run = source / "causal_matrix" / "T5" / outfit
            metrics = json.loads((source_run / "metrics.json").read_text(encoding="utf-8"))
            final_status = json.loads((source_run / "FINAL_STATUS.json").read_text(encoding="utf-8"))
            checkpoint = source_run / "checkpoints" / "step_001000.pth"
            s0["outfits"][outfit] = {
                "historical_final_status": final_status,
                "historical_metrics_path": str(source_run / "metrics.json"),
                "historical_metrics_sha256": sha256(source_run / "metrics.json"),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "optimizer_steps": metrics["optimizer_steps"],
                "per_view": metrics["final_per_view"],
            }
            summaries[outfit] = {}
            for condition in CONDITIONS:
                view = VIEWS[condition]
                render = source_run / "renders" / "final" / view
                record = records[(outfit, condition)]
                paths = {
                    "target_rgb": render / "edit_target.png",
                    "target_fg": render / "edit_foreground_mask.png",
                    "base_fg": render / "base_foreground_mask.png",
                    "safe_clothing": render / "clothing_mask_safe.png",
                    "old_clothing": render / "old_clothing_mask.png",
                    "protected": render / "protected_mask.png",
                    "core": render / "edit_core_mask.png",
                    "transition": render / "transition_mask.png",
                    "alpha": render / "alpha.png",
                    "raw_clothing": Path(record["target_clothing_mask_raw"]),
                }
                for name, path in paths.items():
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    source_files.append({"outfit": outfit, "condition": condition, "field": name, "path": str(path), "sha256": sha256(path)})
                target_rgb = read_rgb(paths["target_rgb"])
                masks = {name: read_mask(path) for name, path in paths.items() if name not in {"target_rgb", "alpha"}}
                alpha_u8 = np.asarray(Image.open(paths["alpha"]).convert("L"))
                masks["pred_fg"] = alpha_u8 >= 128
                derived = trusted_masks(masks, ratio, int(config["support_band"]["minimum_radius_pixels"]))
                fn = masks["target_fg"] & ~masks["pred_fg"]
                fp = masks["pred_fg"] & ~masks["target_fg"]
                classifications = {
                    "FN": classify_errors(fn, "FN", masks, derived),
                    "FP": classify_errors(fp, "FP", masks, derived),
                }
                union = masks["target_fg"] | masks["pred_fg"]
                intersection = masks["target_fg"] & masks["pred_fg"]
                raw_iou = float(intersection.sum() / max(union.sum(), 1))
                historical = next(row for row in metrics["final_per_view"] if row["view"] == view)["silhouette_iou"]
                if not math.isclose(raw_iou, float(historical), rel_tol=0, abs_tol=2e-6):
                    raise AssertionError(f"persisted PNG silhouette metric drift: {outfit}/{view}: {raw_iou} vs {historical}")
                diagonal = max(bbox_diagonal(masks["base_fg"]), 1.0)
                distances = {
                    "base_foreground": distance_to(masks["base_fg"]),
                    "target_garment_core": distance_to(masks["core"]),
                    "old_clothing": distance_to(masks["old_clothing"]),
                }
                summary = {
                    "condition_id": condition,
                    "view": view,
                    "raw_silhouette_iou": raw_iou,
                    "historical_silhouette_iou": historical,
                    "raw_fn_pixels": int(fn.sum()),
                    "raw_fp_pixels": int(fp.sum()),
                    "raw_union_pixels": int(union.sum()),
                    "support_radius_pixels": derived["support_radius_pixels"],
                    "base_bbox_diagonal_pixels": diagonal,
                    "active_pixels": {
                        "trusted_expansion": int(derived["trusted_expansion"].sum()),
                        "trusted_removal": int(derived["trusted_removal"].sum()),
                        "silhouette_uncertain": int(derived["silhouette_uncertain"].sum()),
                    },
                    "errors": {},
                }
                combined = {name: np.zeros_like(fn) for name in CATEGORIES}
                for kind, error in (("FN", fn), ("FP", fp)):
                    total_error = int(error.sum())
                    category_summary = {}
                    y, x = np.nonzero(error)
                    for yy, xx in zip(y.tolist(), x.tolist()):
                        category = next(name for name in CATEGORIES if classifications[kind][name][yy, xx])
                        rows.append({
                            "outfit_id": outfit,
                            "condition_id": condition,
                            "view": view,
                            "error_type": kind,
                            "x": xx,
                            "y": yy,
                            "category": category,
                            "distance_to_base_foreground_over_bbox_diagonal": float(distances["base_foreground"][yy, xx] / diagonal),
                            "distance_to_target_garment_core_over_bbox_diagonal": float(distances["target_garment_core"][yy, xx] / diagonal),
                            "distance_to_old_clothing_over_bbox_diagonal": float(distances["old_clothing"][yy, xx] / diagonal),
                        })
                    for name in CATEGORIES:
                        selected = classifications[kind][name]
                        combined[name] |= selected
                        count = int(selected.sum())
                        values = {}
                        for distance_name, distance in distances.items():
                            normalized = distance[selected] / diagonal
                            finite = normalized[np.isfinite(normalized)]
                            values[f"distance_to_{distance_name}_over_bbox_diagonal_mean"] = float(finite.mean()) if len(finite) else None
                            values[f"distance_to_{distance_name}_over_bbox_diagonal_median"] = float(np.median(finite)) if len(finite) else None
                        category_summary[name] = {
                            "pixels": count,
                            "fraction_of_error_type": count / max(total_error, 1),
                            "iou_union_contribution": count / max(int(union.sum()), 1),
                            **values,
                        }
                    summary["errors"][kind] = {"pixels": total_error, "categories": category_summary}
                fn_total = max(int(fn.sum()), 1)
                trusted_fraction = int(classifications["FN"]["TRUSTED_GARMENT_EXPANSION"].sum()) / fn_total
                drift_count = sum(int(classifications["FN"][name].sum()) for name in (
                    "PROTECTED_IDENTITY_DIFFERENCE", "TARGET_BODY_OR_POSE_DRIFT", "BACKGROUND_OR_GENERATION_ARTIFACT",
                ))
                summary["trusted_fn_fraction"] = trusted_fraction
                summary["drift_fn_fraction"] = drift_count / fn_total
                summaries[outfit][view] = summary
                category_image = overlay(target_rgb, combined)
                raw_image = binary_overlay(target_rgb, [(fn, (255, 0, 0)), (fp, (0, 110, 255))])
                trusted_image = binary_overlay(target_rgb, [
                    (fn & derived["trusted_expansion"], (0, 220, 80)),
                    (fp & derived["trusted_removal"], (0, 130, 255)),
                ])
                artifact_image = binary_overlay(target_rgb, [(derived["artifact"], (255, 40, 40))])
                prefix = args.output / "silhouette_error_audit" / outfit / view
                save_rgb(prefix / "mask_category_overlay.png", category_image)
                save_rgb(prefix / "raw_fn_fp_overlay.png", raw_image)
                save_rgb(prefix / "trusted_fn_fp_overlay.png", trusted_image)
                save_rgb(prefix / "artifact_overlay.png", artifact_image)
                panels.extend([
                    (f"{outfit} {view} categories", category_image),
                    (f"{outfit} {view} raw FN/FP", raw_image),
                    (f"{outfit} {view} trusted", trusted_image),
                    (f"{outfit} {view} artifact", artifact_image),
                ])
        causal = {}
        for view in ("back", "right"):
            item = summaries["O08"][view]
            trusted = item["trusted_fn_fraction"]
            drift = item["drift_fn_fraction"]
            if trusted >= float(config["causal_adjudication"]["trusted_true_min"]):
                case = "S-TRUE"
            elif trusted >= float(config["causal_adjudication"]["trusted_mixed_min"]):
                case = "S-MIXED"
            elif drift > float(config["causal_adjudication"]["drift_dominant_min_exclusive"]):
                case = "S-DRIFT"
            else:
                case = "UNRESOLVED"
            causal[view] = {"trusted_fn_fraction": trusted, "drift_fn_fraction": drift, "case": case}
        write_csv(args.output / "silhouette_error_audit/silhouette_error_classification.csv", rows)
        summary_payload = {
            "schema_version": SCHEMA,
            "silhouette_threshold": config["silhouette_threshold"],
            "classification_priority": [
                "PROTECTED_IDENTITY_DIFFERENCE", "BACKGROUND_OR_GENERATION_ARTIFACT",
                "TRUSTED_GARMENT_EXPANSION_or_OLD_GARMENT_REMOVAL", "GARMENT_TRANSITION_UNCERTAIN",
                "SEGMENTATION_UNCERTAIN", "TARGET_BODY_OR_POSE_DRIFT",
            ],
            "categories_exhaustive": True,
            "categories_disjoint": True,
            "soft_target_garment_probability_available": False,
            "raw_binary_segmentation_evidence_used": True,
            "soft_probability_limitation": "The frozen dataset persists raw and safe binary garment masks but no soft probability tensor; no probability was guessed or regenerated.",
            "support_band_contract": {**config["support_band"], "verified_source_radius_pixels": registered_radius},
            "per_view": summaries,
            "O08_back_right_causal_adjudication": causal,
        }
        atomic_json(args.output / "silhouette_error_audit/silhouette_error_summary.json", summary_payload)
        contact_sheet(args.output / "silhouette_error_audit/silhouette_error_audit_contact_sheet.png", panels, 4)
        audit_lines = [
            "# Silhouette error audit", "", "S0 was not rerun. Pixels come from the persisted T5 alpha and frozen target masks.",
            "The threshold remains `alpha >= 0.5`; recomputed PNG IoU agrees with the historical float report within 2e-6.", "",
            "The frozen dataset has raw/safe binary clothing masks but no soft probability tensor. No probability was guessed or regenerated.", "",
            "## O08 causal adjudication", "",
            "| view | FN | FP | trusted FN fraction | drift FN fraction | case |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for view in ("back", "right"):
            item = summaries["O08"][view]
            decision = causal[view]
            audit_lines.append(
                f"| {view} | {item['raw_fn_pixels']} | {item['raw_fp_pixels']} | {decision['trusted_fn_fraction']:.6f} | {decision['drift_fn_fraction']:.6f} | {decision['case']} |"
            )
        audit_lines += ["", "Category overlays are inspection evidence only and never enter Oracle forward."]
        atomic_text(args.output / "silhouette_error_audit/SILHOUETTE_ERROR_AUDIT.md", "\n".join(audit_lines))
        atomic_json(args.output / "S0_historical_t5/S0_HISTORICAL_T5.json", s0)
        atomic_text(args.output / "S0_historical_t5/README.md", "# S0 historical T5\n\nCitation-only baseline. No renderer or optimizer was invoked and historical Case F was not rewritten.")
        atomic_json(args.output / "input_audit/input_manifest.json", {
            "task_id": config["task_id"], "created_at": now(), "git_branch": git("branch", "--show-current"),
            "git_head": git("rev-parse", "HEAD"), "git_status_short": git("status", "--short"),
            "source_objective_output": str(source), "source_aaai_output": str(source_aaai),
            "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
            "boundary_contract": {"path": config["source_boundary_contract"], "sha256": sha256(Path(config["source_boundary_contract"]))},
            "source_files": source_files, "optimizer_steps": 0, "renderer_invoked": False,
        })
        shutil.copy2(args.config, args.output / "contract/config_resolved.yaml")
        atomic_json(args.output / "contract/contract.json", {
            "append_only": True, "outfits": list(OUTFITS), "conditions": list(CONDITIONS),
            "S0_rerun": False, "silhouette_threshold": .5, "target_masks_supervision_only": True,
            "bounds_modified": False, "residual_composition_modified": False, "non_silhouette_v6_modified": False,
        })
        result = {
            "status": "PIXEL_AUDIT_COMPLETE",
            "optimizer_steps": 0,
            "renderer_invoked": False,
            "O08_back_right_causal_adjudication": causal,
            "next_phase": "IMPLEMENT_V6_1_AFTER_AUDIT",
        }
        atomic_json(status, result)
        print(json.dumps(result, indent=2))
    except Exception as error:
        atomic_json(status, {
            "status": "AUDIT_FAILED", "optimizer_steps": 0, "exception_type": type(error).__name__,
            "exception": str(error), "traceback": traceback.format_exc(),
        })
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the registered new-silhouette semantics study")
    parser.add_argument("--phase", required=True, choices=("audit",))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research/subject02_new_silhouette_semantics_v1.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_audit(args, config)


if __name__ == "__main__":
    main()
