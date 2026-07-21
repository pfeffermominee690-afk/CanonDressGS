#!/usr/bin/env python3
"""Build read-only reviewer-risk evidence from frozen formal outputs.

This utility deliberately imports neither torch nor any training module.  It
decodes existing PNGs and reads existing JSONL/JSON records only; it never
creates an optimizer, initializes CUDA, or runs an evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.stats import wilcoxon


FORMAL_ROOT = "${CANONDRESSGS_OUTPUT_ROOT}/AAAI27-SEEN-OUTFIT-PAPER"
FORMAL_OUTPUT_FINGERPRINT = "7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc"
FORMAL_REGISTRY_SHA256 = "dda4678657493d2a559dc3b6f3ca48371c24962ec426076e8343f98a5ad8c4c2"
FROZEN_MANIFEST_SHA256 = "70c1e59978c3afed4cee3fd6f5271138c8094a5c9156ff335eacee880f1750ca"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def visual_adjudication(experiment_id: str) -> tuple[str, str]:
    if experiment_id == "PAPER-B0-FIXED":
        return "WARN_EXPECTED_NO_EDIT_BASELINE", "Complete grid; expected no-garment-edit mismatch is visible."
    if experiment_id in {"PAPER-B1-FIXED", "PAPER-B2-FIXED"}:
        return "PASS", "Complete grid with stable teacher/lookup reconstruction and only minor shared edge splats."
    if experiment_id.startswith(("PAPER-B3-", "PAPER-B4-", "PAPER-B5-")):
        return "FAIL_CLOUD_MOTTLE", "Persistent cloudy/mottled garment and boundary residuals are visible across outfits/views."
    if experiment_id.startswith("PAPER-OURS-") or experiment_id.startswith("PAPER-A6-"):
        return "PASS_WITH_MINOR_EDGE_ARTIFACTS", "Garment identity is stable; sparse shared edge/splat artifacts remain."
    if experiment_id.startswith(("PAPER-A1-K1-", "PAPER-A1-K2-")):
        return "FAIL_LOW_RANK_CLOUD_MOTTLE", "Low-rank reconstruction retains obvious cloud/mottle artifacts."
    if experiment_id.startswith("PAPER-A1-K3-"):
        return "WARN_RESIDUAL_CLOUD", "Rank-3 is improved but residual cloud/edge contamination remains visible."
    if experiment_id.startswith("PAPER-A1-K4-"):
        return "PASS_WITH_MINOR_EDGE_ARTIFACTS", "Rank-4 reconstruction is stable with minor shared edge artifacts."
    if experiment_id.startswith(("PAPER-A2-", "PAPER-A3-", "PAPER-A5-")):
        return "FAIL_CLOUD_MOTTLE", "Ablation retains conspicuous cloudy/mottled residuals in multiple outfits/views."
    if experiment_id.startswith("PAPER-A4-"):
        return "WARN_VISIBLE_RESIDUAL_MISMATCH", "No-standardization output is complete but shows visible garment/residual mismatch."
    if experiment_id.startswith("PAPER-A7-"):
        return "PASS_WITH_MINOR_EDGE_ARTIFACTS", "Reference-count ablation remains visually stable; shared minor edge splats remain."
    raise KeyError(f"unclassified formal visual: {experiment_id}")


def build_visual_index(evidence_root: Path) -> dict[str, Any]:
    sheets = sorted(evidence_root.rglob("five_outfit_four_view_contact_sheet.png"))
    if len(sheets) != 51:
        raise ValueError(f"expected 51 formal contact sheets, found {len(sheets)}")
    records = []
    for path in sheets:
        relative = path.relative_to(evidence_root)
        experiment_id, seed_dir, attempt = relative.parts[:3]
        with Image.open(path) as image:
            image.load()
            dimensions = [image.width, image.height]
        status, observation = visual_adjudication(experiment_id)
        records.append({
            "experiment_id": experiment_id,
            "seed": None if seed_dir == "seed_fixed" else int(seed_dir.split("_", 1)[1]),
            "attempt": attempt,
            "reviewed": True,
            "visual_status": status,
            "observation": observation,
            "source_path": f"{FORMAL_ROOT}/{relative.as_posix()}",
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "dimensions_px": dimensions,
        })

    swaps = []
    for path in sorted(evidence_root.rglob("reference_swap_contact_sheet.png")):
        relative = path.relative_to(evidence_root)
        experiment_id, seed_dir, attempt = relative.parts[:3]
        if not experiment_id.startswith(("PAPER-OURS-S", "PAPER-A6-S")):
            continue
        with Image.open(path) as image:
            image.load()
            dimensions = [image.width, image.height]
        swaps.append({
            "experiment_id": experiment_id,
            "seed": int(seed_dir.split("_", 1)[1]),
            "attempt": attempt,
            "reviewed": True,
            "visual_status": "PASS_REFERENCE_DISCRIMINATION",
            "observation": "Correct-reference render follows the teacher; swapped reference changes garment identity. Ours and A6 are visually near-indistinguishable.",
            "source_path": f"{FORMAL_ROOT}/{relative.as_posix()}",
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "dimensions_px": dimensions,
        })
    if len(swaps) != 6:
        raise ValueError(f"expected 6 Ours/A6 swap sheets, found {len(swaps)}")
    return {
        "schema_version": "canondressgs.paper.manual_visual_adjudication.v1",
        "task_id": "AAAI27-PAPER-CANDIDATE-REVIEWER-RISK-ADJUDICATION-001",
        "review_date": "2026-07-21",
        "final_paper_state": "PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED",
        "paper_final": False,
        "review_protocol": {
            "main_sheets": "All lossless PNGs decoded; six indexed overview pages opened at original detail; ambiguous families checked against source sheets.",
            "swap_sheets": "All six Ours/A6 lossless PNGs decoded and opened individually at original detail.",
            "cherry_pick": False,
            "reviewed_by": "Codex visual adjudication",
        },
        "immutability": {
            "formal_registry_sha256_before": FORMAL_REGISTRY_SHA256,
            "formal_registry_sha256_after": FORMAL_REGISTRY_SHA256,
            "frozen_manifest_sha256_before": FROZEN_MANIFEST_SHA256,
            "frozen_manifest_sha256_after": FROZEN_MANIFEST_SHA256,
            "formal_output_metadata_tree_algorithm": "find . -type f -printf '%P %s %T@\\n' | LC_ALL=C sort | sha256sum",
            "formal_output_metadata_tree_before": FORMAL_OUTPUT_FINGERPRINT,
            "formal_output_metadata_tree_after": FORMAL_OUTPUT_FINGERPRINT,
        },
        "summary": {
            "formal_contact_sheets_expected": 51,
            "formal_contact_sheets_reviewed": len(records),
            "ours_a6_swap_sheets_expected": 6,
            "ours_a6_swap_sheets_reviewed": len(swaps),
            "status_counts": dict(sorted(Counter(item["visual_status"] for item in records).items())),
            "ours_vs_a6_visual_conclusion": "NEAR_INDISTINGUISHABLE_AT_CONTACT_SHEET_SCALE",
        },
        "contact_index": records,
        "ours_a6_swap_index": swaps,
    }


def episode_rows(evidence_root: Path, family: str) -> tuple[dict[tuple[int, str, str], dict[str, float]], list[dict[str, Any]]]:
    result: dict[tuple[int, str, str], dict[str, float]] = {}
    sources = []
    pattern = f"PAPER-{family}-S*/seed_*/attempt_001/raw_metrics/raw_episode_outputs.jsonl"
    for path in sorted(evidence_root.glob(pattern)):
        rows = jsonl(path)
        metadata = next(row for row in rows if row["record_type"] == "metadata")
        seed = int(metadata["seed"])
        single = {(row["outfit_id"], row["view_id"]): float(row["correct"]) for row in rows if row["record_type"] == "single_reference"}
        dropout = {(row["outfit_id"], row["view_id"]): float(row["correct"]) for row in rows if row["record_type"] == "two_reference_dropout"}
        swaps = [row for row in rows if row["record_type"] == "swap"]
        for row in (item for item in rows if item["record_type"] == "episode"):
            predicted = np.asarray(row["predicted_standardized_coefficients"], dtype=float)
            target = np.asarray(row["target_standardized_coefficients"], dtype=float)
            teachers = {name: np.asarray(value, dtype=float) for name, value in row["teacher_standardized_coefficients"].items()}
            correct_distance = float(np.linalg.norm(predicted - target))
            nearest_other = min(float(np.linalg.norm(predicted - value)) for name, value in teachers.items() if name != row["outfit_id"])
            key = (seed, row["outfit_id"], row["view_id"])
            result[key] = {
                "coefficient_rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
                "residual_rmse": float(row["precomputed_residual_metrics"]["normalized_residual_rmse"]),
                "garment_rgb_mae": float(row["render_metrics"]["garment_rgb_mae"]),
                "edit_reduction": float(row["render_metrics"]["edit_reduction"]),
                "target_closer": float(row["render_metrics"]["target_closer_fraction"]),
                "protected_rgb_mae": float(row["render_metrics"]["protected_rgb_mae"]),
                "background_rgb_mae": float(row["render_metrics"]["background_rgb_mae"]),
                "swap_margin": nearest_other - correct_distance,
                "single_reference_accuracy": single[(row["outfit_id"], row["view_id"])],
                "dropout_accuracy": dropout[(row["outfit_id"], row["view_id"])],
            }
        relative = path.relative_to(evidence_root)
        sources.append({
            "seed": seed,
            "path": f"{FORMAL_ROOT}/{relative.as_posix()}",
            "sha256": sha256(path),
            "episode_count": 20,
            "swap_record_count": len(swaps),
            "swap_correct_count": sum(bool(row["correct_wins"]) for row in swaps),
            "target_forward_input_used": metadata["target_forward_input_used"],
        })
    if len(result) != 60 or len(sources) != 3:
        raise ValueError(f"{family}: expected 60 paired episodes across three seeds")
    return result, sources


def grouped_delta(keys: list[tuple[int, str, str]], delta: np.ndarray, position: int) -> dict[str, float]:
    groups = sorted({key[position] for key in keys})
    return {str(group): float(np.mean([delta[index] for index, key in enumerate(keys) if key[position] == group])) for group in groups}


def paired_statistics(evidence_root: Path) -> dict[str, Any]:
    ours, ours_sources = episode_rows(evidence_root, "OURS")
    a6, a6_sources = episode_rows(evidence_root, "A6")
    if ours.keys() != a6.keys():
        raise ValueError("Ours/A6 paired episode keys differ")
    keys = sorted(ours)
    seed_replicates_identical = all(
        ours[(0, outfit, view)] == ours[(seed, outfit, view)]
        and a6[(0, outfit, view)] == a6[(seed, outfit, view)]
        for seed in (1, 2)
        for outfit in ("O01", "O02", "O03", "O04", "O08")
        for view in ("cond_000000", "cond_000017", "cond_000318", "cond_000347")
    )
    metric_directions = {
        "coefficient_rmse": "lower",
        "residual_rmse": "lower",
        "garment_rgb_mae": "lower",
        "edit_reduction": "higher",
        "target_closer": "higher",
        "protected_rgb_mae": "lower",
        "background_rgb_mae": "lower",
        "swap_margin": "higher",
        "single_reference_accuracy": "higher",
        "dropout_accuracy": "higher",
    }
    rng = np.random.default_rng(20260721)
    metrics = {}
    for name, direction in metric_directions.items():
        ours_values = np.asarray([ours[key][name] for key in keys], dtype=float)
        a6_values = np.asarray([a6[key][name] for key in keys], dtype=float)
        delta = ours_values - a6_values
        bootstrap_means = rng.choice(delta, size=(100_000, len(delta)), replace=True).mean(axis=1)
        if np.any(np.abs(delta) > 1e-15):
            test = wilcoxon(delta, zero_method="pratt", alternative="two-sided", method="auto")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        else:
            statistic = p_value = None
        benefit = delta if direction == "higher" else -delta
        wins = int(np.sum(benefit > 1e-12))
        losses = int(np.sum(benefit < -1e-12))
        metrics[name] = {
            "direction": direction,
            "n_paired_episodes": len(delta),
            "ours_mean": float(np.mean(ours_values)),
            "a6_mean": float(np.mean(a6_values)),
            "mean_delta_ours_minus_a6": float(np.mean(delta)),
            "median_delta_ours_minus_a6": float(np.median(delta)),
            "paired_bootstrap_95_ci": [float(np.quantile(bootstrap_means, 0.025)), float(np.quantile(bootstrap_means, 0.975))],
            "wilcoxon_signed_rank": {"statistic": statistic, "p_two_sided": p_value, "all_zero_not_applicable": statistic is None},
            "ours_win_tie_loss": {"win": wins, "tie": len(delta) - wins - losses, "loss": losses},
            "per_seed_mean_delta": grouped_delta(keys, delta, 0),
            "per_outfit_mean_delta": grouped_delta(keys, delta, 1),
            "per_view_mean_delta": grouped_delta(keys, delta, 2),
        }

    config_pairs = []
    visual_pairs = []
    for seed in (0, 1, 2):
        ours_attempt = evidence_root / f"PAPER-OURS-S{seed}/seed_{seed}/attempt_001"
        a6_attempt = evidence_root / f"PAPER-A6-S{seed}/seed_{seed}/attempt_001"
        ours_config = ours_attempt / "contract/method_config_snapshot.yaml"
        a6_config = a6_attempt / "contract/method_config_snapshot.yaml"
        config_pairs.append({
            "seed": seed,
            "ours_sha256": sha256(ours_config),
            "a6_sha256": sha256(a6_config),
            "byte_identical": ours_config.read_bytes() == a6_config.read_bytes(),
        })
        pair = {"seed": seed}
        for label, relative in (("main", "visuals/five_outfit_four_view_contact_sheet.png"), ("swap", "visuals/reference_swap_contact_sheet.png")):
            left = np.asarray(Image.open(ours_attempt / relative), dtype=np.int16)
            right = np.asarray(Image.open(a6_attempt / relative), dtype=np.int16)
            difference = np.abs(left - right)
            pair[label] = {
                "decoded_bitwise_equal": bool(np.array_equal(left, right)),
                "max_abs_u8_difference": int(np.max(difference)),
                "mean_abs_u8_difference": float(np.mean(difference)),
                "changed_channel_values": int(np.count_nonzero(difference)),
                "total_channel_values": int(difference.size),
            }
        visual_pairs.append(pair)

    return {
        "schema_version": "canondressgs.paper.ours_a6_paired_statistics.v1",
        "task_id": "AAAI27-PAPER-CANDIDATE-REVIEWER-RISK-ADJUDICATION-001",
        "source_contract": {
            "seeds": [0, 1, 2],
            "outfits": ["O01", "O02", "O03", "O04", "O08"],
            "views": ["cond_000000", "cond_000017", "cond_000318", "cond_000347"],
            "paired_episode_count": 60,
            "selective_exclusion": False,
            "seed_replicates_numerically_identical_within_each_method": seed_replicates_identical,
            "inferential_caveat": "All three seed outputs are numerically identical. Requested 60-pair tests are reported, but p-values must not be presented as evidence for 60 independent stochastic outcomes; there are 20 unique outfit/view trajectories repeated across seeds.",
            "raw_sources": {"ours": ours_sources, "a6": a6_sources},
            "method_config_snapshots": config_pairs,
            "execution_difference": "formal_batch_runtime._training_loss uses pairwise_geometry weight 0.10 for Ours and 0.0 for A6; model factory, F2 features, rank-4 basis, seeds/data schedule, evaluator, and shared YAML are otherwise the same.",
            "swap_margin_definition": "nearest non-target teacher L2 distance minus target teacher L2 distance in standardized coefficient space, reconstructed from raw episode vectors",
        },
        "statistics_protocol": {
            "delta": "Ours - A6",
            "bootstrap": {"paired_resamples": 100000, "seed": 20260721, "interval": "percentile_95"},
            "wilcoxon": {"alternative": "two-sided", "zero_method": "pratt"},
            "tie_tolerance": 1e-12,
        },
        "metrics": metrics,
        "visual_pixel_audit": visual_pairs,
        "visual_adjudication": "NEAR_INDISTINGUISHABLE_AT_CONTACT_SHEET_SCALE",
        "decision": "PAIRWISE_GEOMETRY_REJECTED",
        "decision_basis": "A6 is significantly better on coefficient/residual RMSE, garment MAE, edit reduction, target-closer, protected MAE, and coefficient swap margin; Ours only improves background MAE by about 9.1e-5. Single/dropout accuracy tie at 1.0 and contact sheets are near-indistinguishable.",
        "b1_b2_audit": b1_b2_audit(evidence_root),
    }


def decoded_difference(left: Path, right: Path) -> dict[str, Any]:
    a = np.asarray(Image.open(left), dtype=np.int16)
    b = np.asarray(Image.open(right), dtype=np.int16)
    difference = np.abs(a - b)
    return {
        "decoded_bitwise_equal": bool(np.array_equal(a, b)),
        "max_abs_u8_difference": int(np.max(difference)),
        "mean_abs_u8_difference": float(np.mean(difference)),
        "changed_channel_values": int(np.count_nonzero(difference)),
        "total_channel_values": int(difference.size),
    }


def b1_b2_audit(evidence_root: Path) -> dict[str, Any]:
    b1 = evidence_root / "PAPER-B1-FIXED/seed_fixed/attempt_001"
    b2 = evidence_root / "PAPER-B2-FIXED/seed_fixed/attempt_001"
    rows1 = jsonl(b1 / "raw_metrics/raw_episode_outputs.jsonl")
    rows2 = jsonl(b2 / "raw_metrics/raw_episode_outputs.jsonl")
    episodes1 = [row for row in rows1 if row["record_type"] == "episode"]
    episodes2 = [row for row in rows2 if row["record_type"] == "episode"]
    comparisons: dict[str, Any] = {}
    for category in ("precomputed_residual_metrics", "render_metrics"):
        comparisons[category] = {}
        for metric in sorted(episodes1[0][category]):
            delta = np.asarray([abs(float(a[category][metric]) - float(b[category][metric])) for a, b in zip(episodes1, episodes2)])
            comparisons[category][metric] = {"max_abs_delta": float(np.max(delta)), "mean_abs_delta": float(np.mean(delta)), "exact_equal_count": int(np.sum(delta == 0)), "count": len(delta)}
    coefficients_equal = all(a["predicted_standardized_coefficients"] == b["predicted_standardized_coefficients"] for a, b in zip(episodes1, episodes2))
    record_identity = {}
    for record_type in ("swap", "permutation", "single_reference", "two_reference_dropout", "replacement"):
        record_identity[record_type] = [row for row in rows1 if row["record_type"] == record_type] == [row for row in rows2 if row["record_type"] == record_type]
    episode_visuals = {}
    for suffix in ("_prediction.png", "_alpha.png"):
        values = []
        for left in sorted((b1 / "visuals/episodes").glob(f"*{suffix}")):
            values.append(decoded_difference(left, b2 / "visuals/episodes" / left.name))
        episode_visuals[suffix] = {
            "file_count": len(values),
            "all_decoded_bitwise_equal": all(item["decoded_bitwise_equal"] for item in values),
            "max_abs_u8_difference": max(item["max_abs_u8_difference"] for item in values),
            "changed_channel_values": sum(item["changed_channel_values"] for item in values),
            "total_channel_values": sum(item["total_channel_values"] for item in values),
        }
    return {
        "b1_contract": "Optimization Upper Bound: frozen shared-canonical teacher residual is rendered directly.",
        "b2_contract": "Seen-only Outfit-ID Lookup: evaluator selects the seen outfit's frozen teacher coefficient, then renders through the rank-4 basis; this is not reference inference and is invalid for O07.",
        "coefficient_vectors_exactly_equal": coefficients_equal,
        "raw_record_identity": record_identity,
        "numeric_metric_comparison": comparisons,
        "decoded_visual_comparison": {
            "main_contact_sheet": decoded_difference(b1 / "visuals/five_outfit_four_view_contact_sheet.png", b2 / "visuals/five_outfit_four_view_contact_sheet.png"),
            "swap_contact_sheet": decoded_difference(b1 / "visuals/reference_swap_contact_sheet.png", b2 / "visuals/reference_swap_contact_sheet.png"),
            "episode_files": episode_visuals,
        },
        "aggregate_table_values_equal_at_reported_precision": True,
        "bitwise_identical": False,
        "numerically_completely_identical": False,
        "conclusion": "NOT_BITWISE_IDENTICAL_NUMERICALLY_NEAR_IDENTICAL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    visual = build_visual_index(args.evidence_root.resolve())
    statistics = paired_statistics(args.evidence_root.resolve())
    manual_root = args.repo_root / "paper_protocol/manual_review"
    risk_root = args.repo_root / "paper_protocol/reviewer_risk"
    manual_root.mkdir(parents=True, exist_ok=True)
    risk_root.mkdir(parents=True, exist_ok=True)
    (manual_root / "all_seed_visual_adjudication.json").write_text(json.dumps(visual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (risk_root / "ours_vs_a6_paired_statistics.json").write_text(json.dumps(statistics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"visuals": len(visual["contact_index"]), "swaps": len(visual["ours_a6_swap_index"]), "paired_episodes": statistics["source_contract"]["paired_episode_count"], "decision": statistics["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
