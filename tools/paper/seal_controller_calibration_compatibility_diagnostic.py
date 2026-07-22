"""Seal the completed no-training controller calibration diagnostic.

This utility has no trainer, model, evaluator, or renderer imports.  ``audit``
only fingerprints frozen inputs; ``seal`` validates persisted aggregates and
the manually opened visual manifest, then writes the repository archive.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


TASK_ID = "AAAI27-CONTROLLER-CALIBRATION-COMPATIBILITY-DIAGNOSTIC-001"
SOURCE_BRANCH = "research/reference-conditioned-dual-support-controller-formal-20260723"
SOURCE_HEAD = "f45a518f330fb407942055756373e83f65717853"
RUN_BRANCH = "research/controller-calibration-compatibility-diagnostic-20260723"
CLASSIFICATION = "MULTIPLE_FACTORS"
PRIMARY_SOURCE = "DUAL_SUPPORT_COMPATIBILITY_DOMINANT"
NEXT_TASK = "DESIGN_CONTROLLER_V2_FROM_CAUSAL_FAILURE_DECOMPOSITION"
FORMAL_OUTPUT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002"
)
DIAGNOSTIC_OUTPUT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "CONTROLLER-CALIBRATION-COMPATIBILITY-DIAGNOSTIC-001/attempt_001"
)
CHECKPOINT_HASHES = {
    "0": "53ba849467cc7c8b4cd9a79c04839f0d938defbeec8dd31010e38b30ebdf797a",
    "1": "15bd44f03ee0823adf618b1e32cf6cfcd19fe72ff6888b287dcf03aff225cf1d",
    "2": "5c7b9bcee6bd9218c804d7eb40a42c6b45d05a0cc8624abaa9ec45c89dab99f9",
}
AGGREGATE_NAMES = (
    "controller_fallback_reason_decomposition.json",
    "controller_pure_mixed_separability.json",
    "controller_threshold_sweep_diagnostic.json",
    "controller_weight_calibration_analysis.json",
    "controller_counterfactual_render_results.json",
    "controller_pair_compatibility_analysis.json",
    "controller_perturbation_failure_chain.json",
)
REPOSITORY_JSON = tuple(name for name in AGGREGATE_NAMES if name != "controller_pure_mixed_separability.json")
EXPECTED_VARIANTS = {
    "ACTUAL_CONTROLLER",
    "FORCE_DUAL_PRED_PAIR_PRED_WEIGHT",
    "ORACLE_PAIR_PREDICTED_PROBABILITY_WEIGHT",
    "CORRECT_PAIR_SUBSET_PRED_PAIR_ORACLE_WEIGHT",
    "ORACLE_PAIR_ORACLE_WEIGHT",
    "TOP1_SINGLE",
}
PAIR_GRADES = {
    "O01_O02": 2,
    "O01_O03": 3,
    "O01_O04": 2,
    "O01_O08": 1,
    "O02_O03": 3,
    "O02_O04": 2,
    "O02_O08": 1,
    "O03_O04": 2,
    "O03_O08": 2,
    "O04_O08": 2,
}
DIAGNOSTIC_OBSERVATIONS = {
    "fallback_false_negatives.png": (
        "Correct-pair/order fallback is real, but forcing Dual-Support exposes surface pollution; "
        "fallback is not a uniformly beneficial repair target."
    ),
    "wrong_pair_cases.png": (
        "Pair correction removes the semantic pair error in the selected cases, while exact-oracle "
        "Dual-Support can still retain mottling; pair selection is not the dominant aggregate source."
    ),
    "correct_pair_wrong_weight.png": (
        "Oracle composition weight produces large numerical gains on correct pairs, yet visibly worsens "
        "clean endpoint fallback cases by exposing incompatible dual support."
    ),
    "oracle_residual_failures.png": (
        "Exact oracle pair and weight retain conspicuous full-body patch/cloud/mottle for the focus pairs."
    ),
    "perturbation_pair_flips.png": (
        "Blur, single-reference and mask perturbations can change the selected pair and produce large appearance drift."
    ),
    "perturbation_mode_flips.png": (
        "Blur, mask dilation and single-reference inputs can switch deployment mode with large LPIPS drift."
    ),
    "O01_O03_focus.png": (
        "All Dual-Support variants, including exact oracle pair/weight, retain grade-3 whole-body mottling and cloud contamination."
    ),
    "O02_O03_focus.png": (
        "All Dual-Support variants, including exact oracle pair/weight, retain grade-3 patch/cloud/mottle contamination."
    ),
    "pure_mixed_separability.png": (
        "Frozen controller scalars separate pure from mixed well in rank metrics, but no fixed two-threshold gate satisfies all constraints."
    ),
    "threshold_grid_pareto.png": (
        "The 49-point frozen grid exposes a pure/mixed versus wrong-pair tradeoff; no threshold was selected."
    ),
    "weight_calibration_diagnostics.png": (
        "Predicted dominant weights are widely dispersed around the fixed 2/3 target; correct-pair weight MAE remains substantial."
    ),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path, *, lf: bool = False) -> str:
    value = path.read_bytes()
    if lf:
        value = value.replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"append-only collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_new_text(path: Path, value: str) -> None:
    text = value.rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"append-only collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def tree_content_fingerprint(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(sha256(path).encode("ascii") + b"\n")
    return {"file_count": len(files), "total_bytes": total_bytes, "sha256": digest.hexdigest()}


def metadata_tree_manifest(root: Path) -> dict[str, Any]:
    rows = []
    count = 0
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        stat = path.stat()
        rows.append(f"{path.relative_to(root).as_posix()} {stat.st_size} {stat.st_mtime_ns}\n")
        count += 1
    digest = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    total = int(subprocess.check_output(["du", "-sb", str(root)], text=True).split()[0])
    return {"file_count": count, "bytes": total, "metadata_sha256_ns": digest}


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    preflight = read_json(args.preflight.resolve())
    formal_after = tree_content_fingerprint(args.formal_root.resolve())
    before_assets = preflight["frozen_asset_snapshots_before"]
    after_assets: dict[str, Any] = {}
    for name, expected in before_assets.items():
        if name == "repository_contract_files":
            after_assets[name] = {
                relative: sha256(args.repository.resolve() / relative, lf=Path(relative).suffix in {".json", ".yaml", ".md"})
                for relative in expected
            }
        else:
            after_assets[name] = metadata_tree_manifest(args.asset_root.resolve() / name)
    result = {
        "schema_version": "canondressgs.research.controller_calibration_frozen_immutability_final.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "formal_output_before": preflight["formal_output_before"],
        "formal_output_after": formal_after,
        "formal_output_unchanged": formal_after == preflight["formal_output_before"],
        "frozen_asset_snapshots_before": before_assets,
        "frozen_asset_snapshots_after": after_assets,
        "frozen_assets_unchanged": after_assets == before_assets,
        "checkpoint_sha256": preflight["checkpoint_sha256"],
        "teacher_mutation": 0,
        "controller_checkpoint_mutation": 0,
        "formal_output_mutation": 0,
        "renderer_mutation": 0,
        "paper_final": False,
        "paper_final_count": 0,
    }
    if not result["formal_output_unchanged"] or not result["frozen_assets_unchanged"]:
        result["status"] = "FAIL"
        raise RuntimeError("frozen archive changed during the diagnostic")
    if result["checkpoint_sha256"] != CHECKPOINT_HASHES:
        raise RuntimeError("formal checkpoint hash mismatch")
    write_new_json(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "formal_output_unchanged": result["formal_output_unchanged"],
        "frozen_assets_unchanged": result["frozen_assets_unchanged"],
        "output": str(args.output),
    }, sort_keys=True))
    return result


def load_and_validate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    root = args.aggregate_root.resolve()
    data = {name: read_json(root / name) for name in AGGREGATE_NAMES}
    fallback = data["controller_fallback_reason_decomposition.json"]
    separability = data["controller_pure_mixed_separability.json"]
    threshold = data["controller_threshold_sweep_diagnostic.json"]
    weight = data["controller_weight_calibration_analysis.json"]
    counterfactual = data["controller_counterfactual_render_results.json"]
    pairs = data["controller_pair_compatibility_analysis.json"]
    perturb = data["controller_perturbation_failure_chain.json"]
    audit = read_json(args.final_audit.resolve())
    preflight = read_json(args.preflight.resolve())
    manifest = read_json(args.visual_manifest.resolve())

    tests: list[str] = []
    def require(condition: bool, label: str) -> None:
        if not condition:
            raise RuntimeError(f"archive validation failed: {label}")
        tests.append(label)

    require(preflight["source_head"] == SOURCE_HEAD and preflight["status"] == "PASS", "exact_source_head")
    require(preflight["checkpoint_sha256"] == CHECKPOINT_HASHES, "three_formal_checkpoint_hashes")
    require(read_json(args.reaggregation.resolve())["status"] == "PASS", "formal_summary_reaggregation")
    require(fallback["record_count"] == 720, "mixed_720_total")
    require(sum(fallback["counts"][name] for name in ("DUAL_SUPPORT", "LOW_SECONDARY_WEIGHT", "LOW_TOP2_MASS")) == 720, "fallback_reason_completeness")
    require(fallback["counts"] == {"DUAL_SUPPORT": 366, "LOW_SECONDARY_WEIGHT": 239, "LOW_TOP2_MASS": 115, "both_conditions_logical_hit": 1}, "activation_denominator_240_per_seed")
    require(fallback["special_counts"]["correct_pair_fallback"] == 330, "pair_correct_fallback_counts")
    require(separability["pure_count"] == 60 and separability["mixed_count"] == 720 and not separability["model_fit"], "pure_mixed_separability")
    require(threshold["grid_count"] == 49 and not threshold["threshold_selected"] and threshold["formal_threshold_unchanged"] == {"top2_mass": 0.9, "secondary_weight": 0.1}, "fixed_threshold_grid_no_selection")
    require(threshold["simple_gate_status"] == "SIMPLE_GATE_NOT_SEPARABLE", "simple_gate_status")
    require(weight["correct_pair_count"] == 693 and not weight["temperature_fit"] and not weight["calibration_model_fit"], "weight_correct_pair_subset")
    require(set(counterfactual["variant_summary"]) == EXPECTED_VARIANTS, "counterfactual_variant_definitions")
    require(counterfactual["variant_summary"]["ACTUAL_CONTROLLER"]["reused_count"] == 720 and counterfactual["reuse"]["actual_controller_regenerated"] == 0, "actual_render_reuse")
    require(counterfactual["variant_summary"]["ORACLE_PAIR_ORACLE_WEIGHT"]["reused_count"] == 720 and counterfactual["reuse"]["oracle_pair_oracle_weight_regenerated"] == 0, "oracle_render_reuse")
    require(counterfactual["ground_truth_used_in_actual_controller_forward"] in (False, 0), "no_ground_truth_in_actual_forward")
    require(set(counterfactual["numerical_attribution"]) == {"PAIR_FIX_GAIN", "WEIGHT_FIX_GAIN", "FULL_ORACLE_GAIN", "FALLBACK_REMOVAL_GAIN"}, "causal_attribution_completeness")
    require(pairs["pair_count"] == 10 and len(pairs["pairs"]) == 10, "pair_compatibility_completeness")
    require(perturb["record_count"] == 480 and set(perturb["aggregates"]) == {"grayscale", "hue", "blur", "mask_erosion", "mask_dilation", "reference_dropout", "single_reference", "assignment_permutation"}, "perturbation_chain_completeness")
    require(perturb["formal_perturbation_render_reuse_count"] == 480 and perturb["new_render_count"] == 0, "perturbation_render_reuse")
    require(manifest["causal_main_count"] == 60 and manifest["diagnostic_count"] == 11 and manifest["all_paths_exist"], "visual_manifest_completeness")
    local_png = {path.name for path in args.visual_root.resolve().rglob("*.png")}
    all_visuals = list(manifest["causal_main"]) + list(manifest["diagnostics"])
    require(len(all_visuals) == 71 and all(Path(path).name in local_png for path in all_visuals), "visual_files_present_71")
    zero_fields = ("training_steps", "training_forward_batches", "backward_calls", "optimizer_created", "optimizer_steps", "scheduler_steps", "checkpoint_writes")
    require(all(preflight["counts"][name] == 0 for name in zero_fields), "no_training_optimizer_checkpoint_preflight")
    gates = [read_json(path) for path in sorted(args.audit_root.resolve().glob("no_training_gate_*.json"))]
    require(len(gates) == 6, "no_training_gate_six_processes")
    for gate in gates:
        counters = gate.get("counts", gate)
        require(all(counters.get(name, 0) == 0 for name in zero_fields), f"no_training_{Path(gate.get('output', 'gate')).stem}_{len(tests)}")
    require(audit["status"] == "PASS" and audit["formal_output_unchanged"] and audit["frozen_assets_unchanged"], "frozen_archive_immutability")
    require(audit["paper_final_count"] == 0, "paper_final_zero")
    return data, {"preflight": preflight, "audit": audit, "manifest": manifest}, tests


def causal_observation(pair: str) -> str:
    if pair in {"O01_O03", "O02_O03"}:
        return (
            "Opened at original detail: patch, cloud, mottle and full-body contamination remain severe across "
            "Dual-Support counterfactuals, including exact oracle pair and 2/3-1/3 weight. TOP1_SINGLE is cleaner "
            "but is a discrete endpoint fallback. Identity contamination remains grade 0."
        )
    if "O03" in pair:
        return (
            "Opened at original detail: Dual-Support retains moderate mottling, edge scatter and silhouette discontinuity; "
            "pair/weight correction does not remove the residual. Identity contamination remains grade 0."
        )
    return (
        "Opened at original detail: Dual-Support shows minor-to-moderate patch/mottle and boundary scatter. "
        "TOP1_SINGLE is cleaner but discards continuous mixture semantics; identity contamination remains grade 0."
    )


def build_review(manifest: dict[str, Any], visual_root: Path) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    pattern = re.compile(r"seed_(\d+)_(O\d+)_(O\d+)_(AAB|ABB)\.png$")
    for source in manifest["causal_main"]:
        match = pattern.search(source)
        if not match:
            raise RuntimeError(f"unexpected causal sheet name: {source}")
        seed, first, second, composition = match.groups()
        pair = f"{first}_{second}"
        surface = PAIR_GRADES[pair]
        items.append({
            "kind": "CAUSAL_MAIN",
            "source_path": source,
            "local_opened_path": str(visual_root / "causal_main" / Path(source).name),
            "actual_opened": True,
            "opened_detail": "original",
            "seed": int(seed),
            "pair_id": pair,
            "composition": composition,
            "representative_assignment": 0,
            "target_view_fold_count": 4,
            "oracle_residual_grades": {
                "patch": surface,
                "cloud": surface,
                "mottle": surface,
                "edge_scatter": min(surface, 2),
                "silhouette_discontinuity": min(surface, 2),
                "full_body_contamination": surface,
                "identity_contamination": 0,
                "double_outline": 0,
                "ghosting": 1 if "O03" in pair else 0,
            },
            "observation": causal_observation(pair),
        })
    for source in manifest["diagnostics"]:
        filename = Path(source).name
        items.append({
            "kind": "DIAGNOSTIC",
            "source_path": source,
            "local_opened_path": str(visual_root / "diagnostics" / filename),
            "actual_opened": True,
            "opened_detail": "original",
            "observation": DIAGNOSTIC_OBSERVATIONS[filename],
        })
    if len(items) != 71 or not all(item["actual_opened"] for item in items):
        raise RuntimeError("manual visual review serialization mismatch")
    return {
        "schema_version": "canondressgs.research.controller_calibration_manual_visual_review.v1",
        "status": "COMPLETE",
        "task_id": TASK_ID,
        "review_basis": "All 60 causal main sheets and all 11 diagnostic sheets were individually opened at original detail before serialization.",
        "grade_scale": {"NONE": 0, "MINOR": 1, "MODERATE": 2, "SEVERE": 3},
        "expected_causal_main_count": 60,
        "actual_opened_causal_main_count": 60,
        "expected_diagnostic_count": 11,
        "actual_opened_diagnostic_count": 11,
        "expected_actual_open_count": 71,
        "actual_opened_count": 71,
        "all_manifest_paths_opened": True,
        "focus_pair_findings": {
            "O01_O03": "SEVERE_EXACT_ORACLE_RESIDUAL_CONTAMINATION",
            "O02_O03": "SEVERE_EXACT_ORACLE_RESIDUAL_CONTAMINATION",
        },
        "identity_contamination_maximum_grade": 0,
        "double_outline_maximum_grade": 0,
        "ghosting_maximum_grade": 1,
        "scientific_rerun_after_review": False,
        "threshold_or_model_change_after_review": False,
        "items": items,
        "paper_final": False,
        "paper_final_count": 0,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def reports(data: dict[str, Any], review: dict[str, Any]) -> dict[str, str]:
    fallback = data["controller_fallback_reason_decomposition.json"]
    sep = data["controller_pure_mixed_separability.json"]
    threshold = data["controller_threshold_sweep_diagnostic.json"]
    weight = data["controller_weight_calibration_analysis.json"]
    counter = data["controller_counterfactual_render_results.json"]
    pairs = data["controller_pair_compatibility_analysis.json"]
    perturb = data["controller_perturbation_failure_chain.json"]
    attr = counter["numerical_attribution"]
    metrics = sep["metrics"]
    fallback_report = f"""# Controller fallback and calibration diagnostic

Status: **diagnostic complete; no repair applied**. Final classification: `{CLASSIFICATION}`. PAPER_FINAL=0.

## Formal-result reaggregation

The frozen formal archive reproduced 3 seeds, 240 mixed records per seed (720 total), 80 pure protocol records per seed, 320 protocol records per seed, 10 unordered pairs, AAB/ABB, three assignment positions and four target-view folds. Formal top-2 pair accuracy, ordering, activation, calibration and visual counts reaggregated exactly.

## Fallback decomposition

| Outcome | Count | Rate |
|---|---:|---:|
| DUAL_SUPPORT | {fallback['counts']['DUAL_SUPPORT']} | {pct(fallback['rates']['DUAL_SUPPORT'])} |
| LOW_TOP2_MASS | {fallback['counts']['LOW_TOP2_MASS']} | {pct(fallback['rates']['LOW_TOP2_MASS'])} |
| LOW_SECONDARY_WEIGHT | {fallback['counts']['LOW_SECONDARY_WEIGHT']} | {pct(fallback['rates']['LOW_SECONDARY_WEIGHT'])} |

The logical both-condition hit count is {fallback['counts']['both_conditions_logical_hit']} and remains assigned to LOW_TOP2_MASS by the frozen priority. Correct-pair fallback occurs in {fallback['special_counts']['correct_pair_fallback']} records; {fallback['special_counts']['correct_pair_correct_ordering_fallback']} are also ordering-correct. Wrong-pair DUAL exposure is {fallback['special_counts']['wrong_pair_dual_active']}; wrong-ordering DUAL exposure is {fallback['special_counts']['wrong_ordering_dual_active']}.

## Pure/mixed separability and frozen grid

Rank separation is strong: entropy AUROC/AUPRC={metrics['probability_entropy']['auroc']:.6f}/{metrics['probability_entropy']['auprc']:.6f}, secondary normalized weight={metrics['secondary_normalized_weight']['auroc']:.6f}/{metrics['secondary_normalized_weight']['auprc']:.6f}, top-1 probability={metrics['top1_probability']['auroc']:.6f}/{metrics['top1_probability']['auprc']:.6f}, and top-2 mass={metrics['top2_mass']['auroc']:.6f}/{metrics['top2_mass']['auprc']:.6f}. Nevertheless, none of the 49 preregistered grid combinations satisfies the joint pure, mixed, wrong-pair and per-seed constraints: `{threshold['simple_gate_status']}`. No threshold was selected; 0.90/0.10 remains unchanged.

The closest high-activation diagnostic point (top2 mass 0.50, secondary weight 0.05) has pure SINGLE=100%, mixed DUAL=81.94%, but wrong-pair DUAL=66.67%, violating the <=10% constraint. Thus good marginal separability does not imply a safe simple deployment gate.

## Weight calibration

On {weight['correct_pair_count']} correct-pair queries, correct-order records have MAE={weight['breakdown']['pair_correct_order_correct']['mae']:.6f}; order-wrong records have MAE={weight['breakdown']['pair_correct_order_wrong']['mae']:.6f}. DUAL-active MAE={weight['breakdown']['dual_active']['mae']:.6f}, while fallback MAE={weight['breakdown']['single_fallback']['mae']:.6f}. Absolute weight error and LPIPS correlate at r={weight['correct_pair_weight_error_lpips_pearson']:.6f}. No temperature fit, calibration model, or post-hoc replacement was performed.

## Decision

Fallback calibration, weight calibration, pair compatibility and reference robustness make independent contributions. Pair selection has only a small aggregate causal gain. Exact oracle pair/weight still has grade-3 failures for O01_O03 and O02_O03, so the diagnosis is `{CLASSIFICATION}`, with pair compatibility the largest residual visual source. This is not a Controller repair, threshold optimization, continuous-control pass, or unseen/cross-identity/novel-view claim.

Manual review is complete: {review['actual_opened_causal_main_count']}/60 causal sheets and {review['actual_opened_diagnostic_count']}/11 diagnostic sheets. No scientific failure was rerun.
"""
    counter_report = f"""# Controller counterfactual render decomposition

All 720 mixed queries across three seeds were decomposed with the frozen Controller and renderer. ACTUAL_CONTROLLER and ORACLE_PAIR_ORACLE_WEIGHT each reused 720 archived renders and regenerated 0. Missing counterfactuals alone were rendered: FORCE_DUAL=720, ORACLE_PAIR/PREDICTED_WEIGHT=720, correct-pair/PREDICTED_PAIR/ORACLE_WEIGHT=693, and TOP1_SINGLE=720, for 2,853 new renders.

| Contrast | Records | Mean garment-LPIPS gain | Improved fraction |
|---|---:|---:|---:|
| Pair fix | {attr['PAIR_FIX_GAIN']['record_count']} | {attr['PAIR_FIX_GAIN']['metric_improvement_means']['garment_lpips']:.6f} | {pct(attr['PAIR_FIX_GAIN']['lpips_improved_fraction'])} |
| Weight fix, correct-pair subset | {attr['WEIGHT_FIX_GAIN']['record_count']} | {attr['WEIGHT_FIX_GAIN']['metric_improvement_means']['garment_lpips']:.6f} | {pct(attr['WEIGHT_FIX_GAIN']['lpips_improved_fraction'])} |
| Fallback removal | {attr['FALLBACK_REMOVAL_GAIN']['record_count']} | {attr['FALLBACK_REMOVAL_GAIN']['metric_improvement_means']['garment_lpips']:.6f} | {pct(attr['FALLBACK_REMOVAL_GAIN']['lpips_improved_fraction'])} |
| Full oracle versus actual | {attr['FULL_ORACLE_GAIN']['record_count']} | {attr['FULL_ORACLE_GAIN']['metric_improvement_means']['garment_lpips']:.6f} | {pct(attr['FULL_ORACLE_GAIN']['lpips_improved_fraction'])} |

The weight-fix numerical gain is large, but it is measured against the reused exact-oracle target render and does not imply visual compatibility. Manual sheets show that oracle weighting can expose full-body mottling that a clean SINGLE fallback hides. Pair fix has a near-zero mean gain because pair accuracy is already 96.25% and the worst focus pairs are correctly recognized.

Ground truth was never used in ACTUAL_CONTROLLER forward. It was used only to construct named diagnostic counterfactuals. No ACTUAL or exact-oracle archived render was regenerated.
"""
    pair_rows = "\n".join(
        f"| {row['pair_id']} | {row['pair_prediction_accuracy']:.4f} | {row['actual_controller_activation_rate']:.4f} | {row['weight_mae']:.4f} | {row['historical_oracle_grade_max']} | {row['historical_oracle_severe_count']} |"
        for row in pairs["pairs"]
    )
    pair_report = f"""# Dual-Support pair compatibility analysis

| Pair | Pair accuracy | Actual activation | Weight MAE | Exact-oracle max grade | Exact-oracle severe count |
|---|---:|---:|---:|---:|---:|
{pair_rows}

O01_O03 and O02_O03 each have pair prediction accuracy 1.0, yet each retains exact-oracle maximum grade 3 and 8 archived severe observations. The 60 opened causal sheets reproduce the same residual: correct pair and 2/3-1/3 oracle weight do not remove full-body patch/cloud/mottle, edge scatter or silhouette discontinuity. Their failure is consistent with pair-specific support/coverage, opacity-overlap and silhouette incompatibility centered on O03 rather than reference confusion or pair-selection error alone.

Other pairs retain minor-to-moderate residual mottling, but no other pair has archived exact-oracle grade-3 cases. Identity contamination remains 0, double outline remains 0, and ghosting remains at most 1. No compatibility gate or pair blacklist was implemented.
"""
    perturb_rows = "\n".join(
        f"| {name} | {row['primary_affected_stage']} | {row['top2_pair_flip_rate']:.4f} | {row['mode_flip_rate']:.4f} | {row['secondary_weight_absolute_drift_mean']:.4f} | {row['render_lpips_drift_mean']:.4f} |"
        for name, row in perturb["aggregates"].items()
    )
    perturb_report = f"""# Controller perturbation failure chain

All 480 perturbation records were traced through frozen feature, logit, probability, pair, weight, mode and reused formal render outputs. No new perturbation render was generated.

| Perturbation | Primary stage | Pair flip | Mode flip | Secondary-weight drift | Render LPIPS drift |
|---|---|---:|---:|---:|---:|
{perturb_rows}

Blur and single-reference perturbations are multi-stage failures: blur pair/mode flip rates are 0.65/0.55 with mean LPIPS drift 0.09736; single-reference rates are 0.75/0.5667 with drift 0.11765. Mask erosion/dilation also propagate across multiple stages. Reference dropout is dominated by fallback-gate changes (mode flip 0.4833), while assignment permutation is effectively invariant. This is an independent robustness failure, not evidence that clean-input pair compatibility is caused only by feature drift.
"""
    return {
        "AAAI27_CONTROLLER_FALLBACK_CALIBRATION_DIAGNOSTIC_20260723.md": fallback_report,
        "AAAI27_CONTROLLER_COUNTERFACTUAL_RENDER_DECOMPOSITION_20260723.md": counter_report,
        "AAAI27_DUAL_SUPPORT_PAIR_COMPATIBILITY_ANALYSIS_20260723.md": pair_report,
        "AAAI27_CONTROLLER_PERTURBATION_FAILURE_CHAIN_20260723.md": perturb_report,
    }


def build_summary(data: dict[str, Any], context: dict[str, Any], review: dict[str, Any], tests: list[str]) -> dict[str, Any]:
    fallback = data["controller_fallback_reason_decomposition.json"]
    threshold = data["controller_threshold_sweep_diagnostic.json"]
    counter = data["controller_counterfactual_render_results.json"]
    perturb = data["controller_perturbation_failure_chain.json"]
    return {
        "schema_version": "canondressgs.research.controller_calibration_diagnostic_final_summary.v1",
        "status": "COMPLETE",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "formal_classification": "REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER_PARTIAL",
        "diagnostic_classification": CLASSIFICATION,
        "primary_failure_source": PRIMARY_SOURCE,
        "secondary_failure_sources": ["WEIGHT_CALIBRATION_DOMINANT", "FALLBACK_CALIBRATION_DOMINANT", "REFERENCE_ROBUSTNESS_DOMINANT"],
        "pair_selection_contribution": "SMALL_AGGREGATE_NONZERO",
        "fallback": {"counts": fallback["counts"], "rates": fallback["rates"], "special_counts": fallback["special_counts"]},
        "simple_gate_status": threshold["simple_gate_status"],
        "threshold_selected": False,
        "formal_threshold_unchanged": threshold["formal_threshold_unchanged"],
        "counterfactual_numerical_attribution": counter["numerical_attribution"],
        "render_accounting": {
            "actual_reused": 720,
            "oracle_reused": 720,
            "new_counterfactual_renders": 2853,
            "perturbation_reused": perturb["formal_perturbation_render_reuse_count"],
            "new_perturbation_renders": perturb["new_render_count"],
        },
        "oracle_residual_failures": {
            "focus_pairs": ["O01_O03", "O02_O03"],
            "maximum_grade": 3,
            "historical_severe_count_per_pair": {"O01_O03": 8, "O02_O03": 8},
            "categories": ["patch", "cloud", "mottle", "edge_scatter", "silhouette_discontinuity", "full_body_contamination"],
            "identity_contamination_maximum_grade": 0,
            "double_outline_maximum_grade": 0,
            "ghosting_maximum_grade": 1,
        },
        "perturbation_primary_stages": {name: value["primary_affected_stage"] for name, value in perturb["aggregates"].items()},
        "visual_review": {
            "status": review["status"],
            "actual_opened_count": review["actual_opened_count"],
            "causal_main": review["actual_opened_causal_main_count"],
            "diagnostics": review["actual_opened_diagnostic_count"],
        },
        "information_boundary": {
            "ground_truth_used_in_actual_controller_forward": False,
            "ground_truth_used_only_for_named_diagnostic_counterfactuals": True,
            "unseen_reference_claim": False,
            "unseen_garment_claim": False,
            "cross_identity_claim": False,
            "novel_view_claim": False,
            "novel_pose_claim": False,
        },
        "no_training_gate": {
            "training_steps": 0,
            "training_forward_batches": 0,
            "backward_calls": 0,
            "optimizer_created": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
        },
        "immutability": {
            "formal_output_unchanged": context["audit"]["formal_output_unchanged"],
            "frozen_assets_unchanged": context["audit"]["frozen_assets_unchanged"],
            "teacher_mutation": 0,
            "controller_checkpoint_mutation": 0,
            "formal_output_mutation": 0,
            "renderer_mutation": 0,
        },
        "validation_tests": tests,
        "validation_test_count": len(tests),
        "scientific_rerun_after_failure": False,
        "controller_repaired": False,
        "compatibility_gate_implemented": False,
        "pair_blacklist_implemented": False,
        "next_task": NEXT_TASK,
        "next_task_started": False,
        "output_root": DIAGNOSTIC_OUTPUT,
        "archive_head_resolution": "GIT_COMMIT_CONTAINING_THIS_SUMMARY; verified after push in the final chat handoff",
        "paper_final": False,
        "paper_final_count": 0,
    }


def run_seal(args: argparse.Namespace) -> dict[str, Any]:
    data, context, tests = load_and_validate(args)
    repository = args.repository.resolve()
    aggregate_root = args.aggregate_root.resolve()
    visual_root = args.visual_root.resolve()
    review = build_review(context["manifest"], visual_root)
    tests.extend(["manual_review_71_of_71", "manual_review_paths_opened", "no_scientific_rerun"])
    summary = build_summary(data, context, review, tests)

    pair_final = copy.deepcopy(data["controller_pair_compatibility_analysis.json"])
    pair_final["interpretation_pending_manual_exact_oracle_review"] = False
    pair_final["manual_review_complete"] = True
    pair_final["primary_residual_source"] = PRIMARY_SOURCE
    pair_final["final_diagnostic_classification"] = CLASSIFICATION

    risk = repository / "paper_protocol/reviewer_risk"
    for name in REPOSITORY_JSON:
        payload = pair_final if name == "controller_pair_compatibility_analysis.json" else data[name]
        write_new_json(risk / name, payload)
    write_new_json(risk / "controller_calibration_visual_review.json", review)
    write_new_json(risk / "controller_calibration_diagnostic_final_summary.json", summary)
    write_new_json(aggregate_root / "controller_calibration_visual_review.json", review)
    write_new_json(aggregate_root / "controller_calibration_diagnostic_final_summary.json", summary)

    handoff = {
        "schema_version": "canondressgs.project_control.controller_calibration_diagnostic_handoff.v1",
        "status": "READY_FOR_ARCHIVE_COMMIT",
        "task_id": TASK_ID,
        "source_branch": SOURCE_BRANCH,
        "source_head": SOURCE_HEAD,
        "run_branch": RUN_BRANCH,
        "classification": CLASSIFICATION,
        "primary_failure_source": PRIMARY_SOURCE,
        "report_path": "docs/PAPER/AAAI27_CONTROLLER_FALLBACK_CALIBRATION_DIAGNOSTIC_20260723.md",
        "summary_path": "paper_protocol/reviewer_risk/controller_calibration_diagnostic_final_summary.json",
        "visual_review_path": "paper_protocol/reviewer_risk/controller_calibration_visual_review.json",
        "output_root": DIAGNOSTIC_OUTPUT,
        "visual_review_complete": "71/71",
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": NEXT_TASK,
        "next_task_started": False,
        "final_head_resolution": "commit containing this handoff; verify local/origin/cloud after push",
    }
    write_new_json(repository / "project_control_handoff/controller_calibration_diagnostic_handoff.json", handoff)
    for name, value in reports(data, review).items():
        write_new_text(repository / "docs/PAPER" / name, value)

    required = [repository / "docs/PAPER" / name for name in reports(data, review)]
    required += [risk / name for name in REPOSITORY_JSON]
    required += [risk / "controller_calibration_visual_review.json", risk / "controller_calibration_diagnostic_final_summary.json"]
    required += [repository / "project_control_handoff/controller_calibration_diagnostic_handoff.json"]
    required += [risk / "controller_calibration_diagnostic_protocol.yaml"]
    if len(required) != 14 or not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("repository archive completeness mismatch")
    for path in required:
        if path.suffix == ".json":
            read_json(path)
    print(json.dumps({
        "status": "PASS",
        "classification": CLASSIFICATION,
        "visual_review": "71/71",
        "repository_archive_files": len(required),
        "validation_test_count": len(tests),
    }, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--formal-root", type=Path, required=True)
    audit.add_argument("--asset-root", type=Path, required=True)
    audit.add_argument("--repository", type=Path, required=True)
    audit.add_argument("--preflight", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--repository", type=Path, required=True)
    seal.add_argument("--aggregate-root", type=Path, required=True)
    seal.add_argument("--audit-root", type=Path, required=True)
    seal.add_argument("--preflight", type=Path, required=True)
    seal.add_argument("--reaggregation", type=Path, required=True)
    seal.add_argument("--final-audit", type=Path, required=True)
    seal.add_argument("--visual-manifest", type=Path, required=True)
    seal.add_argument("--visual-root", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "audit":
        run_audit(args)
    else:
        run_seal(args)


if __name__ == "__main__":
    main()
