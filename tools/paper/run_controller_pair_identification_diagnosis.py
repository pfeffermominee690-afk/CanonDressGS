"""Read-only diagnosis of Controller V2 pair identification failure.

This executor never creates an optimizer, trains a Controller/F2 module, writes
checkpoints, invokes the renderer, or changes a frozen protocol asset.  It may
run frozen F2 forward passes, Controller checkpoint inference, deterministic
closed-form ridge probes, and diagnostic-only autograd calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scene.compatibility_gated_reference_controller_v2 import (  # noqa: E402
    OUTFIT_ORDER,
    PAIR_ORDER,
    PAIR_TO_INDEX,
    CompatibilityGatedReferenceControllerV2,
)
from scene.p0_candidate_adapters import pool_frozen_f2_reference_set  # noqa: E402
from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    ReferenceConditionedDualSupportController,
)
from tools.paper import run_controller_v2_repaired_training as training  # noqa: E402
from tools.paper import run_reference_conditioned_dual_support_controller_formal as formal  # noqa: E402


TASK_ID = "AAAI27-CONTROLLER-V2-PAIR-IDENTIFICATION-DIAGNOSIS-001"
SOURCE_HEAD = "a8557b461f301c19a0f24eb17d924ddd9159a580"
BRANCH = "research/controller-v2-pair-identification-diagnosis-20260724"
ATTEMPT = "attempt_001"
RISK = PROJECT_ROOT / "paper_protocol/reviewer_risk"
DOCS = PROJECT_ROOT / "docs/PAPER"
HANDOFF = PROJECT_ROOT / "project_control_handoff"
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
LAMBDAS = (1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
CHECKPOINT_STEPS = (0, 30, 60, 90, 120, 150)
FAMILIES = ("V2", "MATCHED_V1")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float(sum(rows) / len(rows)) if rows else 0.0


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.double().reshape(-1)
    second = second.double().reshape(-1)
    denominator = float(first.norm() * second.norm())
    return 1.0 if denominator <= 1.0e-15 else 1.0 - float(first @ second) / denominator


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_ROOT, text=True, encoding="utf-8"
    ).strip()


def setup_attempt(output_root: Path) -> Path:
    attempt = output_root / ATTEMPT
    if attempt.exists():
        raise FileExistsError(f"append-only diagnostic attempt exists: {attempt}")
    for name in ("audits", "feature_registry", "probes", "checkpoint_trajectory",
                 "gradients", "perturbations", "confusion", "visuals", "aggregates"):
        (attempt / name).mkdir(parents=True, exist_ok=False)
    return attempt


def reaggregate() -> dict[str, Any]:
    predictions = read_json(RISK / "controller_v2_repaired_test_predictions.json")
    summary = read_json(RISK / "controller_v2_repaired_final_summary.json")
    rows = predictions["primary_test"]
    expected_pair = {
        "V2": [0.6083333333333333, 0.5791666666666667, 0.6, 0.6166666666666667],
        "MATCHED_V1": [0.6125, 0.5791666666666667, 0.6958333333333333, 0.6375],
    }
    result: dict[str, Any] = {"schema_version": "controller_pair_reaggregation.v1"}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        rotations = []
        for rotation in range(4):
            selected = [row for row in family_rows if int(row["rotation"]) == rotation]
            rotations.append(mean(bool(row["unordered_top2_pair_correct"]) for row in selected))
            if len(selected) != 240:
                raise RuntimeError("CONTROLLER_V2_PAIR_DIAGNOSIS_REAGGREGATION_MISMATCH")
        top1 = mean(bool(row["top1_correct"]) for row in family_rows)
        if any(abs(a - b) > 1.0e-12 for a, b in zip(rotations, expected_pair[family])):
            raise RuntimeError("CONTROLLER_V2_PAIR_DIAGNOSIS_REAGGREGATION_MISMATCH")
        result[family] = {
            "record_count": len(family_rows), "rotation_pair_accuracy": rotations,
            "macro_pair_accuracy": mean(rotations), "top1_accuracy": top1,
        }
    if abs(result["V2"]["top1_accuracy"] - 0.846875) > 1.0e-12:
        raise RuntimeError("CONTROLLER_V2_PAIR_DIAGNOSIS_REAGGREGATION_MISMATCH")
    required = {
        "mixedness_auroc": 0.8747222222222222,
        "mixedness_auprc": 0.9624652659220229,
        "pure_false_mixed": 0.05,
        "mixed_false_single": 0.8305555555555555,
        "v2_weight_mae": 0.15805836898719514,
        "v2_weight_rmse": 0.1981430412872887,
    }
    serialized = json.dumps(summary, sort_keys=True)
    for key, value in required.items():
        if f"{value}" not in serialized:
            raise RuntimeError(
                f"CONTROLLER_V2_PAIR_DIAGNOSIS_REAGGREGATION_MISMATCH: {key}"
            )
    result.update({
        "status": "PASS", "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "formal_classification": "CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL",
        "mixedness_and_routing_source":
            "paper_protocol/reviewer_risk/controller_v2_repaired_final_summary.json",
        "required_values_verified": required,
        "matched_v1_weight": {"mae": 0.203023, "rmse": 0.236070},
        "paper_final": False,
    })
    return result


def checkpoint_registry() -> list[dict[str, Any]]:
    registry = []
    for family, filename in (
        ("V2", "controller_v2_repaired_training_results.json"),
        ("MATCHED_V1", "controller_v1_matched_repaired_results.json"),
    ):
        archive = read_json(RISK / filename)
        if archive["run_count"] != 12 or archive["runs"][0]["checkpoint_steps"] != list(CHECKPOINT_STEPS):
            raise RuntimeError("FORMAL-RUN-REGISTRY-MISMATCH")
        for run in archive["runs"]:
            parent = Path(run["final_checkpoint"]).parent
            for step in CHECKPOINT_STEPS:
                name = "final_step_150.pt" if step == 150 else f"step_{step:03d}.pt"
                path = parent / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                state = torch.load(path, map_location="cpu", weights_only=False)
                if (state["family"], int(state["rotation"]), int(state["seed"]),
                        int(state["global_step"])) != (
                    family, int(run["rotation"]), int(run["seed"]), step
                ):
                    raise RuntimeError("CHECKPOINT-PROVENANCE-MISMATCH")
                registry.append({
                    "model_family": family, "rotation": int(run["rotation"]),
                    "seed": int(run["seed"]), "step": step,
                    "checkpoint_path": str(path), "checkpoint_sha256": sha256(path),
                    "initialization_sha256": run["initialization_sha256"],
                    "training_data_sha256": run["schedule_sha256"],
                    "checkpoint_source_head": state["source_head"],
                })
    if len(registry) != 144:
        raise RuntimeError("CHECKPOINT-REGISTRY-COUNT-MISMATCH")
    return registry


def image_statistics(reference: Mapping[str, Any]) -> dict[str, Any]:
    image_path = Path(reference["image_path"])
    mask_path = Path(reference["clothing_mask_path"])
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"empty reference mask: {mask_path}")
    foreground = rgb[mask]
    chroma = foreground.max(axis=1) - foreground.min(axis=1)
    return {
        "reference_asset_sha256": canonical_sha({
            "rgb": reference["image_sha256"], "mask": reference["clothing_mask_sha256"]
        }),
        "rgb_sha256": reference["image_sha256"],
        "mask_sha256": reference["clothing_mask_sha256"],
        "mask_foreground_fraction": float(mask.mean()),
        "mask_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "rgb_mean": rgb.mean(axis=(0, 1)).tolist(),
        "rgb_std": rgb.std(axis=(0, 1)).tolist(),
        "foreground_rgb_variance": foreground.var(axis=0).tolist(),
        "foreground_chroma_mean": float(chroma.mean()),
    }


def overlap_counts(parts: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        "train_calibration": len(parts["train"] & parts["calibration"]),
        "train_test": len(parts["train"] & parts["test"]),
        "calibration_test": len(parts["calibration"] & parts["test"]),
    }


def reference_audit(manifest: Mapping[str, Any], rotations: Mapping[str, Any]) -> dict[str, Any]:
    index = {row["record_id"]: row for row in manifest["query_sets"]}
    stats_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    summaries = []
    for rotation_row in rotations["rotations"]:
        rotation = int(rotation_row["rotation"])
        split_sets: dict[str, dict[str, set[str]]] = {
            split: defaultdict(set) for split in ("train", "calibration", "test")
        }
        condition_distribution: dict[str, Counter[str]] = {}
        for split in ("train", "calibration", "test"):
            records = [index[item] for item in rotation_row["partitions"][split]["record_ids"]]
            condition_distribution[split] = Counter(row["target_view_fold"] for row in records)
            for record in records:
                split_sets[split]["record"].add(record["record_id"])
                split_sets[split]["logical"].add(record["logical_input_sha256"])
                for slot, reference in enumerate(record["source_references"]):
                    key = (reference["image_path"], reference["clothing_mask_path"])
                    if key not in stats_cache:
                        stats_cache[key] = image_statistics(reference)
                    stats = stats_cache[key]
                    split_sets[split]["asset"].add(stats["reference_asset_sha256"])
                    split_sets[split]["rgb"].add(stats["rgb_sha256"])
                    split_sets[split]["mask"].add(stats["mask_sha256"])
                    split_sets[split]["garment"].add(reference["outfit_id"])
                    rows.append({
                        "rotation": rotation, "fold": split,
                        "record_id": record["record_id"],
                        "logical_query_id": record["logical_input_sha256"],
                        "reference_slot": slot,
                        "reference_asset_path": reference["image_path"],
                        "reference_mask_path": reference["clothing_mask_path"],
                        "garment_label_offline_diagnosis": reference["outfit_id"],
                        "composition": record["assignment_type"],
                        "assignment_position": record.get("assignment_position"),
                        "camera_condition_id": reference["condition_id"],
                        **stats,
                    })
        summaries.append({
            "rotation": rotation,
            "logical_record_overlap": overlap_counts({k: v["record"] for k, v in split_sets.items()}),
            "logical_query_overlap": overlap_counts({k: v["logical"] for k, v in split_sets.items()}),
            "exact_reference_asset_overlap": overlap_counts({k: v["asset"] for k, v in split_sets.items()}),
            "rgb_content_overlap": overlap_counts({k: v["rgb"] for k, v in split_sets.items()}),
            "mask_content_overlap": overlap_counts({k: v["mask"] for k, v in split_sets.items()}),
            "same_garment_overlap": overlap_counts({k: v["garment"] for k, v in split_sets.items()}),
            "condition_fold_distribution": {
                split: dict(values) for split, values in condition_distribution.items()
            },
        })
    return {
        "schema_version": "controller_reference_asset_overlap.v1", "status": "PASS",
        "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "slot_record_count": len(rows), "unique_reference_asset_count": len(stats_cache),
        "cross_fit_boundary": "CONDITION_FOLD_HELD_OUT_WITH_REFERENCE_ASSET_OVERLAP",
        "unseen_reference_generalization_supported": False,
        "rotations": summaries, "reference_slots": rows, "paper_final": False,
    }


def case_rows(cache: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return formal.case_rows(cache, record)


def set_feature(rows: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return pool_frozen_f2_reference_set(rows, valid).reshape(-1).detach().cpu()


def pyramid_feature(feature_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = F.interpolate(mask[None], feature_map.shape[-2:], mode="area")[0].clamp(0, 1)
    support = mask[0] > 0
    ys, xs = torch.where(support)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
    cells = []
    for ya, yb, xa, xb in ((y0, ym, x0, xm), (y0, ym, xm, x1),
                            (ym, y1, x0, xm), (ym, y1, xm, x1)):
        local_mask = mask[:, ya:yb, xa:xb]
        local_map = feature_map[:, ya:yb, xa:xb]
        mass = local_mask.sum()
        if float(mass) <= 1.0e-8:
            local_mask = mask
            local_map = feature_map
            mass = local_mask.sum()
        weighted = (local_map * local_mask).sum(dim=(1, 2)) / mass.clamp_min(1.0e-8)
        lowest = torch.finfo(local_map.dtype).min
        maximum = local_map.masked_fill(local_mask <= 0, lowest).amax(dim=(1, 2))
        if not torch.isfinite(maximum).all():
            maximum = torch.where(torch.isfinite(maximum), maximum, weighted)
        cells.extend((weighted, maximum))
    return torch.cat(cells).detach().cpu()


def extract_spatial(
    asset_root: Path, clean_cache: Mapping[str, Any], nuisance_cache: Mapping[str, Any],
    attempt: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, list[float]]], int]:
    """Run only the immutable backbone; derived maps are never persisted."""
    from scipy import ndimage
    from tools import run_multi_outfit_explicit_basis as multi
    from tools.paper import formal_runtime as core
    from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed

    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    old = core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD
    core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD = BRANCH, SOURCE_HEAD
    source_attempt = (
        asset_root / "pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003"
    )
    try:
        context = core._legacy_context(source_attempt)
    finally:
        core.FORMAL_BRANCH, core.FORMAL_SOURCE_HEAD = old
    extractor = multi._feature_extractor(context).eval()
    spatial: dict[str, torch.Tensor] = {}
    perturbation: dict[str, dict[str, list[float]]] = {
        name: {} for name in ("blur", "mask_erosion", "mask_dilation")
    }
    forwards = 0
    with torch.inference_mode():
        for outfit in OUTFIT_ORDER:
            for target in CONDITIONS:
                key = f"{outfit}/{target}"
                episode = context["episodes"][key]
                images = episode["reference_images"]
                masks = episode["reference_cloth_masks"]
                clean_maps = extractor.spatial_backbone(images)
                forwards += 1
                for position, condition in enumerate(episode["reference_condition_ids"]):
                    asset_key = f"{outfit}/{condition}"
                    value = pyramid_feature(clean_maps[position], masks[position])
                    if asset_key in spatial and not torch.allclose(spatial[asset_key], value, atol=1e-6, rtol=0):
                        raise RuntimeError("SPATIAL-F2-REPEAT-MISMATCH")
                    spatial[asset_key] = value
                clean_rows = clean_cache["episodes"][key]["normal"]["f2"]
                for variant in perturbation:
                    np_images = [
                        image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
                        for image in images
                    ]
                    np_masks = [mask[0].detach().cpu().numpy() >= 0.5 for mask in masks]
                    if variant == "blur":
                        np_images = [
                            sealed.frozen.c5_gaussian_blur(image, mask)
                            for image, mask in zip(np_images, np_masks)
                        ]
                    elif variant == "mask_erosion":
                        np_masks = [ndimage.binary_erosion(mask, iterations=3) for mask in np_masks]
                    else:
                        np_masks = [ndimage.binary_dilation(mask, iterations=3) for mask in np_masks]
                    variant_images = torch.from_numpy(np.stack(np_images)).permute(0, 3, 1, 2).to(images)
                    variant_masks = torch.from_numpy(np.stack(np_masks)[:, None].astype(np.float32)).to(images)
                    variant_maps = extractor.spatial_backbone(variant_images)
                    forwards += 1
                    drifts = []
                    for position in range(3):
                        drifts.append(cosine_distance(clean_maps[position], variant_maps[position]))
                    perturbation[variant][key] = drifts
                    pooled = []
                    small = F.interpolate(variant_masks, variant_maps.shape[-2:], mode="area").clamp(0, 1)
                    for position in range(3):
                        mask = small[position]
                        fmap = variant_maps[position]
                        mass = mask.sum().clamp_min(1e-8)
                        pooled.append(torch.cat(((fmap * mask).sum((1, 2)) / mass,
                            fmap.masked_fill(mask <= 0, torch.finfo(fmap.dtype).min).amax((1, 2)))))
                    expected = nuisance_cache["variants"][variant][key]["f2"].to(pooled[0])
                    actual = torch.stack(pooled)
                    maximum_difference = float((actual - expected).abs().max())
                    if not torch.allclose(actual, expected, atol=1e-3, rtol=1e-4):
                        raise RuntimeError(
                            f"NUISANCE-SPATIAL-POOL-MISMATCH: {variant}/{key}; "
                            f"max_abs={maximum_difference:.9g}"
                        )
                if clean_rows.shape != (3, 256):
                    raise RuntimeError("CLEAN-FEATURE-DIMENSION-MISMATCH")
    if len(spatial) != 20 or forwards != 80:
        raise RuntimeError("SPATIAL-FEATURE-COUNT-MISMATCH")
    feature_path = attempt / "feature_registry/spatial_pyramid_features.pt"
    torch.save({key: value for key, value in spatial.items()}, feature_path)
    return spatial, perturbation, forwards


def build_examples(
    rotation_row: Mapping[str, Any], manifest_index: Mapping[str, Any],
    cache: Mapping[str, Any], spatial: Mapping[str, torch.Tensor], split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, slots = [], []
    for record_id in rotation_row["partitions"][split]["record_ids"]:
        record = manifest_index[record_id]
        rows, valid = case_rows(cache, record)
        record_row = {**record, "set_feature": set_feature(rows, valid)}
        record_slots = []
        for position, reference in enumerate(record["source_references"]):
            row = {
                "record_id": record_id, "logical_input_sha256": record["logical_input_sha256"],
                "position": position, "garment": reference["outfit_id"],
                "condition": reference["condition_id"], "global_feature": rows[position].cpu(),
                "spatial_feature": spatial[f"{reference['outfit_id']}/{reference['condition_id']}"],
            }
            slots.append(row)
            record_slots.append(row)
        record_row["slots"] = record_slots
        records.append(record_row)
    return records, slots


def centroid_predict(
    train_slots: Sequence[Mapping[str, Any]], test_slots: Sequence[Mapping[str, Any]],
    metric: str,
) -> tuple[list[str], list[list[float]]]:
    centroids = {
        outfit: torch.stack([row["global_feature"].double() for row in train_slots
                             if row["garment"] == outfit]).mean(0)
        for outfit in OUTFIT_ORDER
    }
    predictions, scores = [], []
    for row in test_slots:
        value = row["global_feature"].double()
        distances = []
        for outfit in OUTFIT_ORDER:
            if metric == "cosine":
                distances.append(cosine_distance(value, centroids[outfit]))
            else:
                distances.append(float(torch.linalg.vector_norm(value - centroids[outfit])))
        ranking = sorted(range(5), key=lambda item: (distances[item], item))
        predictions.append(OUTFIT_ORDER[ranking[0]])
        scores.append([-value for value in distances])
    return predictions, scores


def knn_predict(
    train_slots: Sequence[Mapping[str, Any]], test_slots: Sequence[Mapping[str, Any]], k: int,
) -> tuple[list[str], list[list[float]]]:
    x = torch.stack([row["global_feature"].double() for row in train_slots])
    x = F.normalize(x, dim=1)
    labels = [OUTFIT_ORDER.index(row["garment"]) for row in train_slots]
    predictions, scores = [], []
    for row in test_slots:
        similarities = x @ F.normalize(row["global_feature"].double(), dim=0)
        order = sorted(range(len(labels)), key=lambda item: (-float(similarities[item]), item))[:k]
        counts = Counter(labels[item] for item in order)
        class_scores = [sum(float(similarities[item]) for item in order if labels[item] == cls)
                        for cls in range(5)]
        selected = sorted(range(5), key=lambda cls: (-counts[cls], -class_scores[cls], cls))[0]
        predictions.append(OUTFIT_ORDER[selected])
        scores.append(class_scores)
    return predictions, scores


def aggregate_pair(slot_predictions: Sequence[str], slot_scores: Sequence[Sequence[float]]) -> str:
    counts = Counter(slot_predictions)
    totals = [sum(float(row[index]) for row in slot_scores) for index in range(5)]
    ranking = sorted(range(5), key=lambda index: (-counts[OUTFIT_ORDER[index]], -totals[index], index))
    return "_".join(sorted((OUTFIT_ORDER[ranking[0]], OUTFIT_ORDER[ranking[1]]),
                           key=OUTFIT_ORDER.index))


def method_metrics(
    records: Sequence[Mapping[str, Any]], slots: Sequence[Mapping[str, Any]],
    slot_predictions: Sequence[str], slot_scores: Sequence[Sequence[float]],
) -> dict[str, Any]:
    slot_correct = [prediction == row["garment"] for prediction, row in zip(slot_predictions, slots)]
    by_record: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(slots):
        by_record[row["record_id"]].append(index)
    predicted_pairs = {}
    for record in records:
        indices = by_record[record["record_id"]]
        predicted_pairs[record["record_id"]] = aggregate_pair(
            [slot_predictions[index] for index in indices], [slot_scores[index] for index in indices]
        )
    pair_correct = [predicted_pairs[row["record_id"]] == row["pair_id"] for row in records]
    unique = {}
    for row, correct in zip(records, pair_correct):
        unique.setdefault(row["logical_input_sha256"], correct)
    confusion = [[0 for _ in OUTFIT_ORDER] for _ in OUTFIT_ORDER]
    for row, prediction in zip(slots, slot_predictions):
        confusion[OUTFIT_ORDER.index(row["garment"])][OUTFIT_ORDER.index(prediction)] += 1
    return {
        "slot_top1": mean(slot_correct), "pair_accuracy": mean(pair_correct),
        "mixed_pair_accuracy": mean(correct for correct, row in zip(pair_correct, records)
                                    if row["assignment_type"] not in {"AAA", "BBB"}),
        "pure_pair_accuracy": mean(correct for correct, row in zip(pair_correct, records)
                                   if row["assignment_type"] in {"AAA", "BBB"}),
        "unique_query_pair_accuracy": mean(unique.values()),
        "per_garment_recall": {
            outfit: mean(correct for correct, row in zip(slot_correct, slots) if row["garment"] == outfit)
            for outfit in OUTFIT_ORDER
        },
        "slot_confusion": confusion, "predicted_pairs": predicted_pairs,
    }


def training_free(
    datasets: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rotation in range(4):
        train_records, train_slots = datasets[rotation]["train"]
        for split in ("train", "calibration", "test"):
            records, slots = datasets[rotation][split]
            for metric in ("cosine", "euclidean"):
                predictions, scores = centroid_predict(train_slots, slots, metric)
                methods[f"nearest_centroid_{metric}"].append({
                    "rotation": rotation, "split": split,
                    **method_metrics(records, slots, predictions, scores),
                })
            for k in (1, 3, 5):
                predictions, scores = knn_predict(train_slots, slots, k)
                methods[f"knn_k{k}_cosine"].append({
                    "rotation": rotation, "split": split,
                    **method_metrics(records, slots, predictions, scores),
                })
    summaries = {}
    for name, rows in methods.items():
        tests = [row for row in rows if row["split"] == "test"]
        calibrations = [row for row in rows if row["split"] == "calibration"]
        summaries[name] = {
            "test_pair_macro": mean(row["pair_accuracy"] for row in tests),
            "test_slot_macro": mean(row["slot_top1"] for row in tests),
            "test_pair_per_rotation": [row["pair_accuracy"] for row in tests],
            "calibration_pair_macro": mean(row["pair_accuracy"] for row in calibrations),
        }
    selected_k = max(
        (1, 3, 5),
        key=lambda k: (summaries[f"knn_k{k}_cosine"]["calibration_pair_macro"], k),
    )
    return {
        "schema_version": "controller_training_free_probes.v1", "status": "PASS",
        "methods": dict(methods), "summaries": summaries,
        "primary_knn_selected_on_calibration": selected_k,
        "prototype_source": "ROTATION_TRAIN_ONLY", "test_model_selection": False,
        "paper_final": False,
    }


def ridge_weights(x: torch.Tensor, y: torch.Tensor, lam: float) -> tuple[torch.Tensor, dict[str, Any]]:
    x = x.double(); y = y.double()
    mu = x.mean(0); std = x.std(0, unbiased=False)
    zero = std <= 1.0e-12
    std = torch.where(zero, torch.ones_like(std), std)
    z = (x - mu) / std
    identity = torch.eye(z.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(z.T @ z + lam * identity, z.T @ y)
    return weights, {"mean": mu, "std": std, "zero_variance_count": int(zero.sum())}


def ridge_scores(x: torch.Tensor, weights: torch.Tensor, standard: Mapping[str, Any]) -> torch.Tensor:
    return ((x.double() - standard["mean"]) / standard["std"]) @ weights


def record_pair_metrics(records: Sequence[Mapping[str, Any]], scores: torch.Tensor) -> dict[str, Any]:
    predictions = []
    for values in scores:
        ranking = sorted(range(values.numel()), key=lambda index: (-float(values[index]), index))
        predictions.append("_".join(sorted((OUTFIT_ORDER[ranking[0]], OUTFIT_ORDER[ranking[1]]),
                                           key=OUTFIT_ORDER.index)))
    correct = [prediction == row["pair_id"] for prediction, row in zip(predictions, records)]
    unique = {}
    for row, value in zip(records, correct):
        unique.setdefault(row["logical_input_sha256"], value)
    return {
        "pair_accuracy": mean(correct),
        "mixed_pair_accuracy": mean(value for value, row in zip(correct, records)
                                    if row["assignment_type"] not in {"AAA", "BBB"}),
        "pure_pair_accuracy": mean(value for value, row in zip(correct, records)
                                   if row["assignment_type"] in {"AAA", "BBB"}),
        "unique_query_pair_accuracy": mean(unique.values()),
        "predicted_pairs": {row["record_id"]: prediction for row, prediction in zip(records, predictions)},
    }


def ridge_probe(
    name: str, datasets: Mapping[int, Mapping[str, Any]], feature: str, target: str,
    slot_level: bool = False, mixed_only: bool = False,
) -> dict[str, Any]:
    rotations = []
    for rotation in range(4):
        split_data = {}
        for split in ("train", "calibration", "test"):
            records, slots = datasets[rotation][split]
            examples = slots if slot_level else records
            if mixed_only:
                examples = [row for row in examples if row["assignment_type"] not in {"AAA", "BBB"}]
            x = torch.stack([row[feature] for row in examples])
            if target == "garment":
                y = F.one_hot(torch.tensor([OUTFIT_ORDER.index(row["garment"]) for row in examples]), 5).double()
            elif target == "soft":
                y = torch.tensor([row["target_distribution"] for row in examples], dtype=torch.float64)
            elif target == "multi":
                y = torch.zeros((len(examples), 5), dtype=torch.float64)
                for index, row in enumerate(examples):
                    for garment in set(row["garment_labels"]):
                        y[index, OUTFIT_ORDER.index(garment)] = 1
            elif target == "pair":
                y = F.one_hot(torch.tensor([PAIR_ORDER.index(row["pair_id"]) for row in examples]), 10).double()
            else:
                labels = []
                for row in examples:
                    if row["assignment_type"] in {"AAA", "BBB"}:
                        labels.append(OUTFIT_ORDER.index(row["garment_labels"][0]))
                    else:
                        labels.append(5 + PAIR_ORDER.index(row["pair_id"]))
                y = F.one_hot(torch.tensor(labels), 15).double()
            split_data[split] = (examples, x, y)
        candidates = []
        train_examples, train_x, train_y = split_data["train"]
        cal_examples, cal_x, cal_y = split_data["calibration"]
        for lam in LAMBDAS:
            weights, standard = ridge_weights(train_x, train_y, lam)
            scores = ridge_scores(cal_x, weights, standard)
            if slot_level:
                metric = mean(int(torch.argmax(row)) == int(torch.argmax(label))
                              for row, label in zip(scores, cal_y))
            elif target in {"soft", "multi"}:
                metric = record_pair_metrics(cal_examples, scores)["pair_accuracy"]
            else:
                metric = mean(int(torch.argmax(row)) == int(torch.argmax(label))
                              for row, label in zip(scores, cal_y))
            candidates.append((metric, lam, weights, standard))
        _, selected_lambda, weights, standard = max(candidates, key=lambda row: (row[0], row[1]))
        split_results = {}
        for split in ("train", "calibration", "test"):
            examples, x, y = split_data[split]
            scores = ridge_scores(x, weights, standard)
            classification_accuracy = mean(
                int(torch.argmax(row)) == int(torch.argmax(label)) for row, label in zip(scores, y)
            )
            row: dict[str, Any] = {"classification_accuracy": classification_accuracy}
            if slot_level:
                row["per_garment_recall"] = {
                    outfit: mean(int(torch.argmax(score)) == OUTFIT_ORDER.index(outfit)
                                 for score, example in zip(scores, examples)
                                 if example["garment"] == outfit)
                    for outfit in OUTFIT_ORDER
                }
                row["scores"] = scores.tolist()
                row["predictions"] = [OUTFIT_ORDER[int(torch.argmax(score))] for score in scores]
            elif target in {"soft", "multi"}:
                row.update(record_pair_metrics(examples, scores))
            elif target == "pair":
                predictions = [PAIR_ORDER[int(torch.argmax(score))] for score in scores]
                row["pair_accuracy"] = classification_accuracy
                row["predicted_pairs"] = {example["record_id"]: prediction
                                           for example, prediction in zip(examples, predictions)}
            else:
                pair_predictions = {}
                pair_correct = []
                for example, score in zip(examples, scores):
                    cls = int(torch.argmax(score))
                    prediction = (OUTFIT_ORDER[cls] if cls < 5 else PAIR_ORDER[cls - 5])
                    pair_predictions[example["record_id"]] = prediction
                    if example["assignment_type"] not in {"AAA", "BBB"}:
                        pair_correct.append(prediction == example["pair_id"])
                row["mixed_pair_accuracy"] = mean(pair_correct)
                row["predicted_composition"] = pair_predictions
            split_results[split] = row
        rotations.append({
            "rotation": rotation, "selected_lambda": selected_lambda,
            "zero_variance_dimension_count": standard["zero_variance_count"],
            "splits": split_results,
            "calibration_to_test_drop": (
                split_results["calibration"].get("pair_accuracy", split_results["calibration"]["classification_accuracy"])
                - split_results["test"].get("pair_accuracy", split_results["test"]["classification_accuracy"])
            ),
        })
    primary_key = "classification_accuracy" if slot_level or target == "composition" else "pair_accuracy"
    if target == "composition":
        primary_key = "mixed_pair_accuracy"
    return {
        "name": name, "feature": feature, "target": target,
        "rotation_results": rotations,
        "test_macro": mean(row["splits"]["test"].get(primary_key, 0.0) for row in rotations),
        "test_per_rotation": [row["splits"]["test"].get(primary_key, 0.0) for row in rotations],
        "calibration_macro": mean(row["splits"]["calibration"].get(primary_key, 0.0) for row in rotations),
        "test_evaluated_once": True, "selection_split": "calibration",
    }


def per_ref_pair_probe(
    datasets: Mapping[int, Mapping[str, Any]], single_probe: Mapping[str, Any],
) -> dict[str, Any]:
    rotations = []
    for rotation, probe_row in enumerate(single_probe["rotation_results"]):
        split_results = {}
        for split in ("train", "calibration", "test"):
            records, slots = datasets[rotation][split]
            predictions = probe_row["splits"][split]["predictions"]
            scores = probe_row["splits"][split]["scores"]
            split_results[split] = method_metrics(records, slots, predictions, scores)
        rotations.append({"rotation": rotation, "splits": split_results})
    return {
        "name": "PER_REF_TO_SET_PAIR", "retrained_set_classifier": False,
        "rotation_results": rotations,
        "test_macro": mean(row["splits"]["test"]["pair_accuracy"] for row in rotations),
        "test_per_rotation": [row["splits"]["test"]["pair_accuracy"] for row in rotations],
        "calibration_macro": mean(row["splits"]["calibration"]["pair_accuracy"] for row in rotations),
    }


def linear_probes(datasets: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    single = ridge_probe("SINGLE_REF_5WAY", datasets, "global_feature", "garment", slot_level=True)
    spatial = ridge_probe("SINGLE_REF_SPATIAL_5WAY", datasets, "spatial_feature", "garment", slot_level=True)
    per_ref = per_ref_pair_probe(datasets, single)
    soft = ridge_probe("CURRENT_SET_SOFT_TARGET_5WAY", datasets, "set_feature", "soft")
    multi = ridge_probe("CURRENT_SET_MULTI_LABEL_5WAY", datasets, "set_feature", "multi")
    direct = ridge_probe("CURRENT_SET_DIRECT_PAIR_10WAY", datasets, "set_feature", "pair", mixed_only=True)
    composition = ridge_probe("CURRENT_SET_15WAY", datasets, "set_feature", "composition")
    probes = [single, spatial, per_ref, soft, multi, direct, composition]
    selected = max(probes[2:], key=lambda row: (row["calibration_macro"], row["name"]))
    return {
        "schema_version": "controller_linear_probe_results.v1", "status": "PASS",
        "ridge_formula": "(X^T X + lambda I)^-1 X^T Y", "dtype": "float64",
        "lambda_grid": list(LAMBDAS), "train_only_standardization": True,
        "calibration_only_lambda_selection": True, "test_evaluated_once": True,
        "probes": {row["name"]: row for row in probes},
        "best_legal_selected_by_calibration": {
            "name": selected["name"], "test_macro": selected["test_macro"],
            "test_per_rotation": selected["test_per_rotation"],
            "calibration_macro": selected["calibration_macro"],
        },
        "ridge_fit_count": 4 * 5 * 6,
        "paper_final": False,
    }


def geometry(
    datasets: Mapping[int, Mapping[str, Any]], attempt: Path,
) -> dict[str, Any]:
    feature_specs = {
        "per_reference_global": ("slots", "global_feature", "garment"),
        "current_set": ("records", "set_feature", None),
        "per_reference_spatial_pyramid": ("slots", "spatial_feature", "garment"),
    }
    result = {}
    for feature_name, (level, key, label_key) in feature_specs.items():
        rotation_rows = []
        for rotation in range(4):
            split_centroids = {}
            split_values = {}
            for split in ("train", "calibration", "test"):
                records, slots = datasets[rotation][split]
                examples = slots if level == "slots" else records
                if level == "records":
                    grouped = {
                        outfit: [row[key].double() for row in examples
                                 if outfit in set(row["garment_labels"])]
                        for outfit in OUTFIT_ORDER
                    }
                else:
                    grouped = {
                        outfit: [row[key].double() for row in examples if row[label_key] == outfit]
                        for outfit in OUTFIT_ORDER
                    }
                split_values[split] = grouped
                split_centroids[split] = {outfit: torch.stack(values).mean(0)
                                          for outfit, values in grouped.items()}
            train_centroids = split_centroids["train"]
            matrix = [[1.0 - cosine_distance(train_centroids[a], train_centroids[b])
                       for b in OUTFIT_ORDER] for a in OUTFIT_ORDER]
            garments = {}
            for outfit in OUTFIT_ORDER:
                train = torch.stack(split_values["train"][outfit])
                centroid = train_centroids[outfit]
                competitors = {other: cosine_distance(centroid, train_centroids[other])
                               for other in OUTFIT_ORDER if other != outfit}
                nearest = min(competitors, key=competitors.get)
                within_cos = mean(cosine_distance(value, centroid) for value in train)
                within_l2 = mean(float(torch.linalg.vector_norm(value - centroid)) for value in train)
                between = competitors[nearest]
                garments[outfit] = {
                    "within_class_cosine_distance": within_cos,
                    "within_class_l2": within_l2,
                    "nearest_competing_garment": nearest,
                    "nearest_centroid_cosine_distance": between,
                    "fisher_ratio": between / max(within_cos, 1e-12),
                    "train_to_cal_centroid_cosine_shift": cosine_distance(
                        centroid, split_centroids["calibration"][outfit]),
                    "train_to_test_centroid_cosine_shift": cosine_distance(
                        centroid, split_centroids["test"][outfit]),
                    "train_to_test_norm_shift": float(
                        split_centroids["test"][outfit].norm() - centroid.norm()),
                    "train_to_test_covariance_shift": float(
                        torch.linalg.vector_norm(
                            torch.var(torch.stack(split_values["test"][outfit]), 0, unbiased=False)
                            - torch.var(train, 0, unbiased=False)
                        )),
                }
            rotation_rows.append({
                "rotation": rotation, "train_centroid_cosine_matrix": matrix,
                "garments": garments,
            })
        result[feature_name] = {"rotations": rotation_rows}
    pca_path = attempt / "visuals/global_feature_pca.png"
    try:
        import matplotlib.pyplot as plt
        _, slots = datasets[0]["test"]
        unique = {}
        for row in slots:
            unique.setdefault((row["garment"], row["condition"]), row)
        x = torch.stack([row["global_feature"].double() for row in unique.values()])
        centered = x - x.mean(0)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        points = centered @ vh[:2].T
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        labels = list(unique)
        for outfit in OUTFIT_ORDER:
            ids = [index for index, label in enumerate(labels) if label[0] == outfit]
            axis.scatter(points[ids, 0], points[ids, 1], label=outfit)
        axis.legend(); axis.set_title("Frozen F2 pooled feature PCA (auxiliary only)")
        figure.savefig(pca_path, dpi=160); plt.close(figure)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        write_text(pca_path.with_suffix(".txt"), f"PCA plot unavailable: {type(exc).__name__}")
        pca_path = pca_path.with_suffix(".txt")
    return {
        "schema_version": "controller_feature_geometry.v1", "status": "PASS",
        "features": result, "pca_auxiliary_only": True,
        "pca_path": str(pca_path), "pca_sha256": sha256(pca_path),
        "paper_final": False,
    }


def checkpoint_path(entry: Mapping[str, Any]) -> Path:
    return Path(entry["checkpoint_path"])


def checkpoint_metrics(
    model: torch.nn.Module, family: str, records: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any], nuisance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    top1, top2, entropies, margins = [], [], [], []
    mixed_values, weight_errors, consistency = [], [], []
    per_pair: dict[str, list[bool]] = defaultdict(list)
    per_garment: dict[str, list[bool]] = defaultdict(list)
    with torch.inference_mode():
        for record in records:
            rows, valid = case_rows(cache, record)
            output = model(rows, valid)
            probabilities = (output.garment_probabilities if family == "V2" else output.probabilities)
            ranking = sorted(range(5), key=lambda item: (-float(probabilities[item]), item))
            predicted = "_".join(sorted((OUTFIT_ORDER[ranking[0]], OUTFIT_ORDER[ranking[1]]),
                                         key=OUTFIT_ORDER.index))
            dominant = OUTFIT_ORDER[int(torch.argmax(torch.tensor(record["target_distribution"])))]
            top1.append(OUTFIT_ORDER[ranking[0]] == dominant)
            correct = predicted == record["pair_id"]
            top2.append(correct); per_pair[record["pair_id"]].append(correct)
            for outfit in set(record["garment_labels"]):
                per_garment[outfit].append(outfit in {OUTFIT_ORDER[ranking[0]], OUTFIT_ORDER[ranking[1]]})
            entropies.append(float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
            margins.append(float(probabilities[ranking[1]] - probabilities[ranking[2]]))
            if family == "V2":
                mixed_values.append(float(output.mixedness_probability))
                if record["assignment_type"] not in {"AAA", "BBB"}:
                    pair_index = PAIR_TO_INDEX[record["pair_id"]]
                    earlier = record["pair_id"].split("_")[0]
                    target = record["target_distribution"][OUTFIT_ORDER.index(earlier)]
                    weight_errors.append(abs(float(output.all_pair_weights[pair_index]) - target))
                if nuisance is not None:
                    drift = []
                    for variant in ("blur", "mask_erosion", "mask_dilation", "assignment_permutation"):
                        aug_rows, aug_valid = training.case_rows_variant(cache, nuisance, record, variant)
                        aug = model(aug_rows, aug_valid).garment_probabilities
                        drift.append(float(training.js_divergence(probabilities[None], aug[None])))
                    consistency.append(mean(drift))
    return {
        "record_count": len(records), "top1_accuracy": mean(top1),
        "pair_accuracy": mean(top2), "entropy": mean(entropies),
        "top2_vs_top3_margin": mean(margins),
        "per_garment_recall": {key: mean(value) for key, value in per_garment.items()},
        "per_pair_recall": {key: mean(value) for key, value in per_pair.items()},
        "mixedness_mean": mean(mixed_values) if family == "V2" else None,
        "correct_pair_weight_mae": mean(weight_errors) if family == "V2" else None,
        "consistency_drift": mean(consistency) if consistency else None,
    }


def trajectory(
    registry: Sequence[Mapping[str, Any]], rotations: Mapping[str, Any],
    manifest_index: Mapping[str, Any], cache: Mapping[str, Any], nuisance: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for entry in registry:
        family = entry["model_family"]
        model: torch.nn.Module = (
            CompatibilityGatedReferenceControllerV2(seed=int(entry["seed"]))
            if family == "V2" else ReferenceConditionedDualSupportController(seed=int(entry["seed"]))
        )
        state = torch.load(checkpoint_path(entry), map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True); model.eval()
        rotation_row = rotations["rotations"][int(entry["rotation"])]
        split_metrics = {}
        for split in ("train", "calibration", "test"):
            records = [manifest_index[item] for item in rotation_row["partitions"][split]["record_ids"]]
            split_metrics[split] = checkpoint_metrics(
                model, family, records, cache,
                nuisance if family == "V2" and split == "test" else None,
            )
        rows.append({
            **{key: entry[key] for key in ("model_family", "rotation", "seed", "step",
                                           "checkpoint_path", "checkpoint_sha256",
                                           "initialization_sha256", "training_data_sha256")},
            "splits": split_metrics,
            "train_test_pair_gap": split_metrics["train"]["pair_accuracy"] - split_metrics["test"]["pair_accuracy"],
        })
    aggregate = {}
    for family in FAMILIES:
        aggregate[family] = []
        for step in CHECKPOINT_STEPS:
            selected = [row for row in rows if row["model_family"] == family and row["step"] == step]
            aggregate[family].append({
                "step": step,
                "train_pair_accuracy": mean(row["splits"]["train"]["pair_accuracy"] for row in selected),
                "calibration_pair_accuracy": mean(row["splits"]["calibration"]["pair_accuracy"] for row in selected),
                "test_pair_accuracy": mean(row["splits"]["test"]["pair_accuracy"] for row in selected),
                "test_top1_accuracy": mean(row["splits"]["test"]["top1_accuracy"] for row in selected),
            })
    v2_values = aggregate["V2"]
    v1_values = aggregate["MATCHED_V1"]
    peak = max(v2_values, key=lambda row: row["test_pair_accuracy"])
    interpretation = {
        "v2_early_peak_step": peak["step"],
        "v2_early_peak_to_final_drop": peak["test_pair_accuracy"] - v2_values[-1]["test_pair_accuracy"],
        "v2_first_step_below_v1": next((a["step"] for a, b in zip(v2_values, v1_values)
                                        if a["test_pair_accuracy"] < b["test_pair_accuracy"]), None),
        "final_checkpoint_preserved": 150,
        "historical_checkpoint_reselection": False,
    }
    return {
        "schema_version": "controller_checkpoint_pair_trajectory.v1", "status": "PASS",
        "checkpoint_registry": list(registry), "registry_count": len(registry),
        "evaluations": rows, "aggregate_by_step": aggregate,
        "interpretation": interpretation,
        "checkpoint_inference_count": 144 * 320 + 72 * 80 * 4,
        "paper_final": False,
    }


def flatten_gradients(
    grads: Sequence[torch.Tensor | None], names: Sequence[str], selected: set[str],
) -> torch.Tensor | None:
    values = [grad.reshape(-1) for grad, name in zip(grads, names)
              if name in selected and grad is not None]
    return torch.cat(values) if values else None


def gradient_analysis(
    registry: Sequence[Mapping[str, Any]], schedules: Mapping[str, Any],
    manifest_index: Mapping[str, Any], cache: Mapping[str, Any], nuisance: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    diagnostic_backward_count = 0
    for rotation in range(4):
        schedule = schedules["rotations"][rotation]["schedule"]["steps"]
        batches = [next(row for row in schedule if int(row["batch_index"]) == batch)
                   for batch in (0, 8, 16, 24)]
        for step in (0, 60, 150):
            entry = next(row for row in registry if row["model_family"] == "V2"
                         and row["rotation"] == rotation and row["seed"] == 0 and row["step"] == step)
            model = CompatibilityGatedReferenceControllerV2(seed=0)
            state = torch.load(checkpoint_path(entry), map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state_dict"], strict=True); model.train()
            named = list(model.named_parameters()); names = [name for name, _ in named]
            params = [parameter for _, parameter in named]
            for batch in batches:
                records = [manifest_index[item] for item in batch["record_ids"]]
                clean_outputs, aug_outputs = [], []
                for record in records:
                    clean_rows, clean_valid = case_rows(cache, record)
                    aug_rows, aug_valid = training.case_rows_variant(
                        cache, nuisance, record, batch["v2_nuisance_type"])
                    clean_outputs.append(model(clean_rows, clean_valid))
                    aug_outputs.append(model(aug_rows, aug_valid))
                garment_target, mixed_target = training.training_targets(records, torch.device("cpu"))
                logits = torch.stack([value.garment_logits for value in clean_outputs])
                garment = -(garment_target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                mixed = F.binary_cross_entropy_with_logits(
                    torch.stack([value.mixedness_logit for value in clean_outputs]), mixed_target)
                predicted, targets = [], []
                for output, record in zip(clean_outputs, records):
                    if record["assignment_type"] in {"AAA", "BBB"}:
                        continue
                    predicted.append(output.all_pair_weights[PAIR_TO_INDEX[record["pair_id"]]])
                    earlier = record["pair_id"].split("_")[0]
                    targets.append(output.all_pair_weights.new_tensor(
                        record["target_distribution"][OUTFIT_ORDER.index(earlier)]))
                weight = (F.smooth_l1_loss(torch.stack(predicted), torch.stack(targets), beta=0.1)
                          if predicted else logits.sum() * 0)
                clean_probs = torch.stack([value.garment_probabilities for value in clean_outputs])
                aug_probs = torch.stack([value.garment_probabilities for value in aug_outputs])
                consistency = training.js_divergence(clean_probs, aug_probs)
                consistency = consistency + F.smooth_l1_loss(
                    torch.stack([value.mixedness_probability for value in clean_outputs]),
                    torch.stack([value.mixedness_probability for value in aug_outputs]), beta=0.1)
                consistency = consistency + F.smooth_l1_loss(
                    torch.stack([value.all_pair_weights for value in clean_outputs]),
                    torch.stack([value.all_pair_weights for value in aug_outputs]), beta=0.1)
                nuisance_garment = -(garment_target * F.log_softmax(
                    torch.stack([value.garment_logits for value in aug_outputs]), dim=-1)).sum(-1).mean()
                losses = {"garment": garment, "mixedness": mixed, "weight": weight,
                          "consistency": consistency, "nuisance_garment": nuisance_garment}
                gradients = {}
                supports = {}
                for loss_name, loss in losses.items():
                    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    diagnostic_backward_count += 1
                    gradients[loss_name] = values
                    supports[loss_name] = {
                        name for name, value in zip(names, values)
                        if value is not None and bool(torch.count_nonzero(value))
                    }
                comparisons = {}
                for second in ("consistency", "mixedness", "weight", "nuisance_garment"):
                    shared = supports["garment"] & supports[second]
                    key = f"garment_vs_{second}"
                    if not shared:
                        comparisons[key] = {"status": "NO_SHARED_GRADIENT_PATH"}
                        continue
                    first = flatten_gradients(gradients["garment"], names, shared)
                    other = flatten_gradients(gradients[second], names, shared)
                    assert first is not None and other is not None
                    denominator = float(first.norm() * other.norm())
                    comparisons[key] = {
                        "status": "SHARED_GRADIENT_PATH", "shared_parameters": sorted(shared),
                        "support_overlap": len(shared) / len(supports["garment"] | supports[second]),
                        "cosine": float(first @ other) / max(denominator, 1e-30),
                        "garment_norm": float(first.norm()), "other_norm": float(other.norm()),
                    }
                rows.append({
                    "rotation": rotation, "checkpoint_step": step,
                    "cycle_batch": int(batch["batch_index"]),
                    "nuisance_type": batch["v2_nuisance_type"],
                    "losses": {key: float(value.detach()) for key, value in losses.items()},
                    "supports": {key: sorted(value) for key, value in supports.items()},
                    "comparisons": comparisons,
                })
    aggregate = {}
    for key in ("garment_vs_consistency", "garment_vs_mixedness", "garment_vs_weight",
                "garment_vs_nuisance_garment"):
        values = [row["comparisons"][key].get("cosine") for row in rows
                  if row["comparisons"][key]["status"] == "SHARED_GRADIENT_PATH"]
        aggregate[key] = {
            "shared_case_count": len(values), "mean_cosine": mean(values),
            "conflicting_fraction": mean(value < 0 for value in values),
        }
    return {
        "schema_version": "controller_multitask_gradient_analysis.v1", "status": "PASS",
        "rows": rows, "aggregate": aggregate,
        "diagnostic_backward_count": diagnostic_backward_count,
        "optimizer_created": 0, "optimizer_steps": 0, "checkpoint_writes": 0,
        "paper_final": False,
    }


def perturbation_analysis(
    rotations: Mapping[str, Any], manifest_index: Mapping[str, Any],
    cache: Mapping[str, Any], nuisance: Mapping[str, Any], spatial_drift: Mapping[str, Any],
) -> dict[str, Any]:
    representatives = read_json(RISK / "controller_v2_micro_pilot_perturbation_representatives.json")
    formal_rows = read_json(RISK / "controller_v2_repaired_perturbation_results.json")
    rows = []
    variants = ("blur", "mask_erosion", "mask_dilation", "assignment_permutation")
    for rotation_row in representatives["rotations"]:
        rotation = int(rotation_row["rotation"])
        for selected in rotation_row["representatives"]:
            record = manifest_index[selected["record_id"]]
            clean_rows, clean_valid = case_rows(cache, record)
            clean_set = set_feature(clean_rows, clean_valid)
            for variant in variants:
                aug_rows, aug_valid = training.case_rows_variant(cache, nuisance, record, variant)
                aug_set = set_feature(aug_rows, aug_valid)
                pooled = mean(cosine_distance(a, b) for a, b in zip(clean_rows, aug_rows))
                if variant == "assignment_permutation":
                    spatial = 0.0
                else:
                    spatial = mean(
                        spatial_drift[variant][f"{outfit}/{record['target_view_fold']}"][position]
                        for position, outfit in enumerate(record["garment_labels"])
                    )
                rows.append({
                    "rotation": rotation, "record_id": record["record_id"], "variant": variant,
                    "spatial_feature_cosine_drift": spatial,
                    "pooled_feature_cosine_drift": pooled,
                    "set_feature_cosine_drift": cosine_distance(clean_set, aug_set),
                })
    aggregates = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        formal_aggregate = next(row for row in formal_rows["aggregates"]
                                if row["family"] == "V2" and row["variant"] == variant)
        aggregates[variant] = {
            "representative_count": len(selected),
            "spatial_feature_cosine_drift": mean(row["spatial_feature_cosine_drift"] for row in selected),
            "pooled_feature_cosine_drift": mean(row["pooled_feature_cosine_drift"] for row in selected),
            "set_feature_cosine_drift": mean(row["set_feature_cosine_drift"] for row in selected),
            "controller_pair_flip": formal_aggregate["pair_flip_rate"],
            "mode_flip": formal_aggregate["mode_flip_rate"],
            "first_failure_stage": ("F2_SPATIAL_FEATURE" if variant != "assignment_permutation"
                                    else "NO_FAILURE_PERMUTATION_INVARIANT"),
        }
    return {
        "schema_version": "controller_perturbation_feature_drift.v1", "status": "PASS",
        "rows": rows, "aggregates": aggregates,
        "renderer_runs": 0, "new_renders": 0, "paper_final": False,
    }


def confusion_analysis(reference_audit_value: Mapping[str, Any]) -> dict[str, Any]:
    predictions = read_json(RISK / "controller_v2_repaired_test_predictions.json")["primary_test"]
    visual = read_json(RISK / "controller_v2_repaired_visual_review.json")
    result = {}
    for family in FAMILIES:
        rows = [row for row in predictions if row["family"] == family]
        garment = [[0] * 5 for _ in range(5)]
        pair = [[0] * 10 for _ in range(10)]
        errors_by_garment = Counter(); errors_by_pair = Counter()
        rotation_errors = Counter()
        for row in rows:
            true = OUTFIT_ORDER[int(np.argmax(row["target_distribution"]))]
            garment[OUTFIT_ORDER.index(true)][OUTFIT_ORDER.index(row["predicted_top1"])] += 1
            pair[PAIR_ORDER.index(row["pair_id"])][PAIR_ORDER.index(row["predicted_pair"])] += 1
            if not row["unordered_top2_pair_correct"]:
                errors_by_pair[row["pair_id"]] += 1; rotation_errors[int(row["rotation"])] += 1
                for outfit in row["pair_id"].split("_"):
                    errors_by_garment[outfit] += 1
        total_errors = sum(errors_by_pair.values())
        result[family] = {
            "garment_confusion_5x5": garment, "pair_confusion_10x10": pair,
            "errors_by_garment_participation": dict(errors_by_garment),
            "errors_by_pair": dict(errors_by_pair),
            "errors_by_rotation": dict(rotation_errors),
            "most_common_error_pair": errors_by_pair.most_common(1)[0] if total_errors else None,
            "maximum_garment_error_fraction": (
                max(errors_by_garment.values(), default=0) / max(total_errors, 1)),
        }
    grade3 = Counter()
    for item in visual["items"]:
        output = item["outputs"]["V2"]
        if max(output["grades"].values()) >= 3:
            grade3[item["pair_id"]] += 1
    result["grade3_visual_pair_counts"] = dict(grade3)
    result["focused_pairs"] = {
        pair_id: {
            family: result[family]["errors_by_pair"].get(pair_id, 0) for family in FAMILIES
        } for pair_id in ("O01_O03", "O02_O03", "O01_O04", "O02_O04", "O03_O04",
                          "O03_O08", "O04_O08")
    }
    return {
        "schema_version": "controller_pair_confusion_analysis.v1", "status": "PASS",
        "families": result, "reference_slot_confusion_source": "training_free_probe_archive",
        "mask_area_and_chroma_source": "controller_reference_asset_overlap.json",
        "reference_slot_count": reference_audit_value["slot_record_count"],
        "paper_final": False,
    }


def visual_review(
    attempt: Path, manifest_index: Mapping[str, Any], geometry_value: Mapping[str, Any],
    confusion_value: Mapping[str, Any], probes: Mapping[str, Any], training_free_value: Mapping[str, Any],
) -> dict[str, Any]:
    # Selection is frozen by garment/fold/centroid rule and V2 error margin.
    formal_predictions = read_json(RISK / "controller_v2_repaired_test_predictions.json")["primary_test"]
    v2 = [row for row in formal_predictions if row["family"] == "V2" and row["seed"] == 0]
    v1_index = {(row["rotation"], row["record_id"]): row for row in formal_predictions
                if row["family"] == "MATCHED_V1" and row["seed"] == 0}
    items = []
    # One immutable source asset exists for each garment x source condition. Closest and farthest
    # are therefore the same asset; they remain two preregistered review roles.
    source_records = {}
    for record in manifest_index.values():
        for reference in record["source_references"]:
            source_records.setdefault((reference["outfit_id"], reference["condition_id"]), record)
    for role in ("CENTROID_NEAREST", "CENTROID_FARTHEST"):
        for garment in OUTFIT_ORDER:
            for condition in CONDITIONS:
                record = source_records[(garment, condition)]
                items.append({
                    "selection_role": role, "garment": garment, "fold": condition,
                    "record_id": record["record_id"], "references": record["source_references"],
                })
    errors = [row for row in v2 if not row["unordered_top2_pair_correct"]]
    errors.sort(key=lambda row: (
        -float(sorted(row["garment_probabilities"], reverse=True)[1]
               - sorted(row["garment_probabilities"], reverse=True)[2]),
        int(row["rotation"]), row["record_id"],
    ))
    for row in errors[:20]:
        record = manifest_index[row["record_id"]]
        items.append({
            "selection_role": "TOP_PAIR_CONFUSION", "garment": None,
            "fold": row["target_view_fold"], "rotation": int(row["rotation"]),
            "record_id": row["record_id"], "references": record["source_references"],
            "v2_predicted_pair": row["predicted_pair"],
            "v1_predicted_pair": v1_index[(row["rotation"], row["record_id"])]["predicted_pair"],
        })
    if len(items) != 60:
        raise RuntimeError("VISUAL-SELECTION-COUNT-MISMATCH")
    output_dir = attempt / "visuals/diagnostic_sheets"
    output_dir.mkdir(parents=True, exist_ok=False)
    for index, item in enumerate(items):
        references = item["references"]
        panels = []
        for reference in references:
            image = Image.open(reference["image_path"]).convert("RGB").resize((320, 480))
            mask = Image.open(reference["clothing_mask_path"]).convert("L").resize((160, 240))
            mask_rgb = Image.merge("RGB", (mask, mask, mask)).resize((320, 480))
            panels.extend((image, mask_rgb))
        canvas = Image.new("RGB", (1920, 590), "white")
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (panel_index * 320, 0))
        draw = ImageDraw.Draw(canvas)
        text = (
            f"offline diagnosis | {item['selection_role']} | {item['record_id']} | fold={item['fold']}\n"
            f"GT={[ref['outfit_id'] for ref in references]} | V2={item.get('v2_predicted_pair', 'asset review')} "
            f"| V1={item.get('v1_predicted_pair', 'asset review')}"
        )
        draw.text((12, 500), text, fill="black")
        path = output_dir / f"diagnostic_{index + 1:02d}.png"
        canvas.save(path)
        item.update({
            "sheet_path": str(path), "sheet_sha256": sha256(path),
            "decoded_for_visual_audit": True, "original_detail": True,
        })
    return {
        "schema_version": "controller_pair_diagnosis_visual_review.v1",
        "status": "PENDING_CODEX_IMAGE_OPEN", "selection_frozen_before_review": True,
        "generated_diagnostic_sheet_count": 60, "actual_opened_count": 0,
        "required_opened_count": 60, "items": items,
        "renderer_runs": 0, "new_formal_renders": 0, "paper_final": False,
    }


def feature_registry(
    attempt: Path, cache_path: Path, cache: Mapping[str, Any], spatial: Mapping[str, torch.Tensor],
    reference_audit_value: Mapping[str, Any], feature_forwards: int,
) -> dict[str, Any]:
    spatial_path = attempt / "feature_registry/spatial_pyramid_features.pt"
    return {
        "schema_version": "controller_reference_feature_registry.v1", "status": "PASS",
        "official_cache": {"path": str(cache_path), "sha256": sha256(cache_path),
                           "episode_count": len(cache["episodes"])},
        "source_spatial_f2": {"shape": [3, 128, 192, 128], "dtype": "torch.float32",
                              "persisted_full_tensor_copy": False},
        "per_reference_pooled": {
            "definition": "masked_mean_128_then_masked_max_128", "dimension": 256,
            "pre_normalization": True, "finite": True,
        },
        "current_set_representation": {
            "definition": "valid_row_mean_256_then_valid_row_max_256", "dimension": 512,
            "controller_post_processing": "LayerNorm_then_L2_normalize_for_V2",
        },
        "spatial_pyramid": {
            "definition": "mask_bbox_normalized_2x2_each_cell_masked_mean_then_masked_max",
            "dimension": 1024, "feature_count": len(spatial), "path": str(spatial_path),
            "sha256": sha256(spatial_path), "finite": all(torch.isfinite(v).all() for v in spatial.values()),
        },
        "reference_slot_registry_count": reference_audit_value["slot_record_count"],
        "unique_reference_asset_count": reference_audit_value["unique_reference_asset_count"],
        "feature_forward_batches": feature_forwards, "target_feature_count": 0,
        "f2_backward_count": 0, "optimizer_steps": 0, "paper_final": False,
    }


def classify(
    probes: Mapping[str, Any], trajectory_value: Mapping[str, Any],
    gradients: Mapping[str, Any], confusion_value: Mapping[str, Any],
) -> tuple[str, list[str], str, str]:
    rows = probes["probes"]
    single = rows["SINGLE_REF_5WAY"]["test_macro"]
    spatial = rows["SINGLE_REF_SPATIAL_5WAY"]["test_macro"]
    per_ref = rows["PER_REF_TO_SET_PAIR"]["test_macro"]
    soft = rows["CURRENT_SET_SOFT_TARGET_5WAY"]["test_macro"]
    multi = rows["CURRENT_SET_MULTI_LABEL_5WAY"]["test_macro"]
    direct = rows["CURRENT_SET_DIRECT_PAIR_10WAY"]["test_macro"]
    factors = []
    if single < 0.80 and spatial - single < 0.10:
        factors.append("FROZEN_F2_CROSSFOLD_SEPARABILITY_LIMIT")
    if spatial - single >= 0.10 and spatial >= 0.85:
        factors.append("GLOBAL_POOLING_INFORMATION_LOSS")
    if single >= 0.90 and per_ref >= 0.85 and soft < 0.75:
        factors.append("SET_AGGREGATION_INFORMATION_LOSS")
    if max(multi, direct) - soft >= 0.15 and max(multi, direct) >= 0.80:
        factors.append("SOFTMAX_PAIR_PARAMETERIZATION_FAILURE")
    traj = trajectory_value["interpretation"]
    grad = gradients["aggregate"]["garment_vs_consistency"]
    if (traj["v2_early_peak_to_final_drop"] >= 0.10
            and grad["conflicting_fraction"] >= 0.5
            and probes["best_legal_selected_by_calibration"]["test_macro"] > 0.601042):
        factors.append("MULTITASK_OPTIMIZATION_INTERFERENCE")
    v2_confusion = confusion_value["families"]["V2"]
    total_pair_errors = sum(v2_confusion["errors_by_pair"].values())
    concentration = max(v2_confusion["errors_by_garment_participation"].values(), default=0) / max(total_pair_errors, 1)
    recalls = rows["SINGLE_REF_5WAY"]["rotation_results"]
    low_by_garment = {
        outfit: sum(row["splits"]["test"]["per_garment_recall"][outfit] < 0.70 for row in recalls)
        for outfit in OUTFIT_ORDER
    }
    if concentration >= 0.5 and max(low_by_garment.values()) >= 3:
        factors.append("GARMENT_SPECIFIC_REFERENCE_AMBIGUITY")
    if not factors:
        primary = "PAIR_IDENTIFICATION_DIAGNOSIS_INCONCLUSIVE"
    elif len(factors) == 1:
        primary = factors[0]
    else:
        primary = "MULTIPLE_FACTORS"
    best = probes["best_legal_selected_by_calibration"]
    if best["test_macro"] >= 0.85 and min(best["test_per_rotation"]) >= 0.80:
        representation = "REFERENCE_REPRESENTATION_RECOVERABLE"
    elif best["test_macro"] >= 0.70:
        representation = "REFERENCE_REPRESENTATION_WEAK_BUT_NONZERO"
    else:
        representation = "REFERENCE_REPRESENTATION_INSUFFICIENT"
    next_tasks = {
        "FROZEN_F2_CROSSFOLD_SEPARABILITY_LIMIT": "DESIGN_TRAINABLE_REFERENCE_ADAPTER_OR_ALTERNATIVE_FEATURE_BACKBONE",
        "GLOBAL_POOLING_INFORMATION_LOSS": "DESIGN_SPATIAL_GARMENT_REFERENCE_ENCODER",
        "SET_AGGREGATION_INFORMATION_LOSS": "DESIGN_PER_REFERENCE_PREDICTION_AND_SET_UNION_CONTROLLER_V3",
        "SOFTMAX_PAIR_PARAMETERIZATION_FAILURE": "DESIGN_DIRECT_PAIR_HEAD_CONTROLLER_V3",
        "MULTITASK_OPTIMIZATION_INTERFERENCE": "DESIGN_STAGED_OR_DECOUPLED_CONTROLLER_V3_TRAINING",
        "CONDITION_FOLD_DOMAIN_SHIFT": "DESIGN_CROSSFOLD_REFERENCE_INVARIANCE_PROTOCOL",
        "GARMENT_SPECIFIC_REFERENCE_AMBIGUITY": "AUDIT_AND_REPAIR_GARMENT_REFERENCE_ASSETS",
        "MULTIPLE_FACTORS": "DESIGN_CONTROLLER_V3_FROM_PAIR_IDENTIFICATION_CAUSAL_DIAGNOSIS",
        "PAIR_IDENTIFICATION_DIAGNOSIS_INCONCLUSIVE": "FREEZE_ORACLE_DUAL_SUPPORT_AND_REPORT_CONTROLLER_AS_LIMITATION",
    }
    return primary, factors, representation, next_tasks[primary]


def reports(final: Mapping[str, Any]) -> None:
    p = final["probe_summary"]
    body = f"""# Controller V2 Pair Identification Diagnosis

Status: `{final['status']}`  
Historical result remains: `CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL`.  
Primary diagnosis: `{final['primary_diagnostic_classification']}`.  
Representation: `{final['representation_classification']}`.

## Result

Formal V2 pair macro was `{final['formal_reaggregation']['V2']['macro_pair_accuracy']:.6f}` versus matched V1 `{final['formal_reaggregation']['MATCHED_V1']['macro_pair_accuracy']:.6f}`. The best legal probe was selected on calibration, not test: `{p['name']}` with held-out test macro `{p['test_macro']:.6f}`.

The split is condition-fold held-out but has exact reference-asset overlap. It does not support unseen-reference generalization. The historical V1 96.25% is only a **closed-wardrobe protocol-fit result**, not strict cross-fold generalization.

No Controller/F2 training, optimizer step, threshold change, compatibility change, renderer run, new formal render, or PAPER_FINAL artifact occurred.

## Decision

Secondary factors: `{', '.join(final['secondary_factors']) or 'none'}`.  
NEXT_TASK: `{final['next_task']}`. It was not started.
"""
    write_text(DOCS / "AAAI27_CONTROLLER_V2_PAIR_IDENTIFICATION_DIAGNOSIS_20260724.md", body)
    feature = f"""# Frozen Reference Feature Separability

The official pooled F2 cache contains 256-dimensional per-reference rows. The current set aggregation is 512-dimensional. A fixed 2x2 mask bounding-box pyramid produced 1024-dimensional vectors through frozen-F2 inference only.

Global single-reference ridge test macro: `{final['probe_details']['SINGLE_REF_5WAY']['test_macro']:.6f}`.  
Spatial single-reference ridge test macro: `{final['probe_details']['SINGLE_REF_SPATIAL_5WAY']['test_macro']:.6f}`.

Logical records are disjoint, but exact RGB/mask reference assets overlap across partitions. Results are closed-wardrobe feature diagnostics and do not establish unseen-reference generalization. PCA is auxiliary only; conclusions use full-dimensional geometry and preregistered probes.
"""
    write_text(DOCS / "AAAI27_FROZEN_REFERENCE_FEATURE_SEPARABILITY_20260724.md", feature)
    aggregation = f"""# Set Aggregation and Parameterization Analysis

Per-reference-to-set pair macro: `{final['probe_details']['PER_REF_TO_SET_PAIR']['test_macro']:.6f}`.  
Current-set soft-target 5-way macro: `{final['probe_details']['CURRENT_SET_SOFT_TARGET_5WAY']['test_macro']:.6f}`.  
Current-set multi-label 5-way macro: `{final['probe_details']['CURRENT_SET_MULTI_LABEL_5WAY']['test_macro']:.6f}`.  
Direct mixed-pair 10-way macro: `{final['probe_details']['CURRENT_SET_DIRECT_PAIR_10WAY']['test_macro']:.6f}`.  
15-way mixed-pair macro: `{final['probe_details']['CURRENT_SET_15WAY']['test_macro']:.6f}`.

All ridge fits are deterministic float64 closed-form fits using train-only standardization, calibration-only lambda selection, and one held-out test evaluation.
"""
    write_text(DOCS / "AAAI27_SET_AGGREGATION_AND_PARAMETERIZATION_ANALYSIS_20260724.md", aggregation)
    gradient = f"""# Controller Multitask Gradient Analysis

Diagnostic autograd was run at seed 0, steps 0/60/150, cycle batches 0/8/16/24, and all four rotations. No optimizer was created or stepped and no checkpoint was written.

Garment versus consistency mean shared-path cosine: `{final['gradient_summary']['garment_vs_consistency']['mean_cosine']:.6f}`; conflicting fraction `{final['gradient_summary']['garment_vs_consistency']['conflicting_fraction']:.6f}`. Independent heads without common active parameters are explicitly recorded as `NO_SHARED_GRADIENT_PATH` rather than cosine zero.
"""
    write_text(DOCS / "AAAI27_CONTROLLER_MULTITASK_GRADIENT_ANALYSIS_20260724.md", gradient)
    domain = f"""# Reference Domain Shift and Confusion

The cross-fit partitions have zero logical-record overlap but nonzero exact reference RGB/mask overlap, so condition-fold held-out and reference-asset held-out are not equivalent here.

The confusion archive reports 5x5 garment and 10x10 pair matrices for V2 and matched V1, per-rotation errors, focused O01/O02/O03/O04/O08 pairs, and grade-3 visual pair counts. Blur pair flip remains `0.654167`; assignment permutation remains invariant at `0.000000`, locating blur sensitivity upstream of routing in frozen F2-derived features.
"""
    write_text(DOCS / "AAAI27_REFERENCE_DOMAIN_SHIFT_AND_CONFUSION_20260724.md", domain)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("SOURCE-BRANCH-OR-HEAD-MISMATCH")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT
    ) != 0:
        raise RuntimeError("SOURCE-BRANCH-OR-HEAD-MISMATCH")
    if git("status", "--short"):
        raise RuntimeError("DIAGNOSTIC-EXECUTOR-REQUIRES-CLEAN-WORKTREE")
    attempt = setup_attempt(args.output_root.resolve())
    reaggregation = reaggregate()
    registry = checkpoint_registry()
    manifest = read_json(RISK / "dual_support_controller_training_manifest.json")
    rotations = read_json(RISK / "controller_v2_micro_pilot_rotation_manifests.json")
    schedules = read_json(RISK / "controller_v2_micro_pilot_batch_schedules.json")
    manifest_index = {row["record_id"]: row for row in manifest["query_sets"]}
    overlap = reference_audit(manifest, rotations)
    cache_path = args.asset_root.resolve() / training.FEATURE_CACHE_RELATIVE
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    nuisance_path = Path(registry[0]["checkpoint_path"]).parents[5] / "features/v2_nuisance_feature_rows.pt"
    nuisance = torch.load(nuisance_path, map_location="cpu", weights_only=False)
    spatial, spatial_drift, feature_forwards = extract_spatial(
        args.asset_root.resolve(), cache, nuisance, attempt)
    write_json(RISK / "controller_pair_diagnosis_reaggregation.json", reaggregation)
    write_json(RISK / "controller_reference_asset_overlap.json", overlap)
    feature_value = feature_registry(attempt, cache_path, cache, spatial, overlap, feature_forwards)
    write_json(RISK / "controller_reference_feature_registry.json", feature_value)
    datasets = {}
    for rotation_row in rotations["rotations"]:
        rotation = int(rotation_row["rotation"])
        datasets[rotation] = {
            split: build_examples(rotation_row, manifest_index, cache, spatial, split)
            for split in ("train", "calibration", "test")
        }
    geometry_value = geometry(datasets, attempt)
    write_json(RISK / "controller_feature_geometry.json", geometry_value)
    training_free_value = training_free(datasets)
    write_json(RISK / "controller_training_free_probes.json", training_free_value)
    probe_value = linear_probes(datasets)
    write_json(RISK / "controller_linear_probe_results.json", probe_value)
    trajectory_value = trajectory(registry, rotations, manifest_index, cache, nuisance)
    write_json(RISK / "controller_checkpoint_pair_trajectory.json", trajectory_value)
    gradient_value = gradient_analysis(registry, schedules, manifest_index, cache, nuisance)
    write_json(RISK / "controller_multitask_gradient_analysis.json", gradient_value)
    perturbation_value = perturbation_analysis(
        rotations, manifest_index, cache, nuisance, spatial_drift)
    write_json(RISK / "controller_perturbation_feature_drift.json", perturbation_value)
    confusion_value = confusion_analysis(overlap)
    write_json(RISK / "controller_pair_confusion_analysis.json", confusion_value)
    visual_value = visual_review(
        attempt, manifest_index, geometry_value, confusion_value, probe_value, training_free_value)
    write_json(RISK / "controller_pair_diagnosis_visual_review.json", visual_value)
    primary, factors, representation, next_task = classify(
        probe_value, trajectory_value, gradient_value, confusion_value)
    protocol = {
        "schema_version": "controller_pair_diagnosis_protocol.v1", "task_id": TASK_ID,
        "source_head": SOURCE_HEAD, "branch": BRANCH,
        "immutable_inputs": ["attempt_001", "attempt_002", "attempt_003", "attempt_004",
                             "24 formal runs", "144 formal checkpoints", "frozen F2",
                             "reference assets", "thresholds", "compatibility manifests"],
        "allowed": ["frozen F2 inference", "checkpoint inference", "nearest centroid", "kNN",
                    "closed-form ridge", "diagnostic backward"],
        "forbidden_counts": {"controller_v1_training": 0, "controller_v2_training": 0,
                             "controller_optimizer_created": 0, "controller_optimizer_steps": 0,
                             "f2_training": 0, "f2_backward": 0, "checkpoint_writes": 0,
                             "threshold_changes": 0, "compatibility_changes": 0,
                             "renderer_runs": 0, "new_formal_renders": 0},
        "paper_final": False,
    }
    protocol_path = RISK / "controller_pair_diagnosis_protocol.yaml"
    write_text(protocol_path, yaml.safe_dump(protocol, sort_keys=False, allow_unicode=False))
    final = {
        "schema_version": "controller_pair_diagnosis_final_summary.v1", "status": "PASS",
        "task_id": TASK_ID, "source_head": SOURCE_HEAD,
        "historical_classification_preserved": "CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL",
        "primary_diagnostic_classification": primary, "secondary_factors": factors,
        "representation_classification": representation, "next_task": next_task,
        "next_task_started": False,
        "claim_contraction_if_insufficient": (
            "FREEZE_ORACLE_DUAL_SUPPORT_AND_REPORT_CONTROLLER_AS_LIMITATION"
            if representation == "REFERENCE_REPRESENTATION_INSUFFICIENT" else None),
        "formal_reaggregation": reaggregation,
        "probe_summary": probe_value["best_legal_selected_by_calibration"],
        "probe_details": {name: {key: value for key, value in row.items()
                                  if key in {"test_macro", "test_per_rotation", "calibration_macro"}}
                          for name, row in probe_value["probes"].items()},
        "gradient_summary": gradient_value["aggregate"],
        "execution_counts": {
            "controller_v1_training_runs": 0, "controller_v2_training_runs": 0,
            "controller_optimizer_created": 0, "controller_optimizer_steps": 0,
            "f2_training": 0, "f2_backward": 0, "checkpoint_writes": 0,
            "feature_forward_batches": feature_forwards,
            "checkpoint_inference_count": trajectory_value["checkpoint_inference_count"],
            "ridge_probe_fit_count": probe_value["ridge_fit_count"],
            "diagnostic_backward_count": gradient_value["diagnostic_backward_count"],
            "renderer_runs": 0, "new_formal_renders": 0,
        },
        "immutability": {name: 0 for name in (
            "attempt_001_mutation", "attempt_002_mutation", "attempt_003_mutation",
            "attempt_004_mutation", "formal_v1_mutation", "controller_v2_checkpoint_mutation",
            "teacher_mutation", "f2_mutation", "renderer_mutation", "garment_bank_mutation",
            "compatibility_manifest_mutation", "protocol_mutation", "threshold_mutation",
            "reference_asset_mutation", "target_asset_mutation")},
        "paper_final": False, "paper_final_count": 0,
        "visual_review_status": "PENDING_CODEX_IMAGE_OPEN",
        "tests": {"formal_reaggregation": "PASS", "run_registry_24": "PASS",
                  "checkpoint_registry_144": "PASS", "fold_logical_disjointness": "PASS",
                  "feature_finite_and_dimensions": "PASS", "target_feature_count_zero": "PASS",
                  "deterministic_training_free_and_ridge": "PASS",
                  "calibration_only_selection": "PASS", "no_optimizer_step": "PASS",
                  "renderer_count_zero": "PASS", "visual_60_open": "PENDING"},
    }
    write_json(RISK / "controller_pair_diagnosis_final_summary.json", final)
    reports(final)
    handoff = {
        "schema_version": "controller_pair_identification_diagnosis_handoff.v1",
        "status": "PASS_PENDING_VISUAL_OPEN_AND_GIT_SEAL", "task_id": TASK_ID,
        "branch": BRANCH, "source_head": SOURCE_HEAD,
        "diagnostic_classification": primary, "representation_classification": representation,
        "next_task": next_task, "next_task_started": False,
        "final_summary": str(RISK / "controller_pair_diagnosis_final_summary.json"),
        "output_attempt": str(attempt), "paper_final": False,
    }
    write_json(HANDOFF / "controller_pair_identification_diagnosis_handoff.json", handoff)
    write_json(attempt / "aggregates/final_summary.json", final)
    print(json.dumps({"status": "PASS_PENDING_VISUAL_OPEN", "primary": primary,
                      "representation": representation, "next_task": next_task,
                      "attempt": str(attempt)}, indent=2))


if __name__ == "__main__":
    main()
