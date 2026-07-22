"""Step-0-only dry run for the reference-conditioned dual-support controller."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    FORBIDDEN_FORWARD_INPUTS,
    OUTFIT_ORDER,
    ReferenceConditionedDualSupportController,
    construct_dual_support_runtime,
    controller_forward_argument_names,
    inference_result_schema,
    soft_target_cross_entropy,
    stable_top2_selection,
)
from scene.p0_candidate_initialization_protocol import tensor_mapping_sha256  # noqa: E402


LABEL = "RESEARCH METHOD CANDIDATE — NOT PAPER FINAL"
TASK_ID = "AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-001"
SOURCE_HEAD = "d802f427f1e9c23595e1bdb4f10135f7de2c3f08"
SEEDS = (0, 1, 2)
ARCHIVE_FILES = {
    "dual_support_all_pair_protocol": (
        "paper_protocol/reviewer_risk/dual_support_all_pair_protocol.yaml",
        "d7be2c0bac94e7d048e125d6bdf8b5beee772aab25fdce0d848540aff3105bcd",
    ),
    "dual_support_all_pair_results": (
        "paper_protocol/reviewer_risk/dual_support_all_pair_results.json",
        "4875c0bbdde0bb8e44c8e6ff9ef1f1fe56a58b816368c5ae4c72d11188d0c226",
    ),
    "dual_support_all_pair_visual_review": (
        "paper_protocol/reviewer_risk/dual_support_all_pair_visual_review.json",
        "354480bea8320569b0da0bf7a0b7f1a5b4f59866e132fa308cb30b6667b06c11",
    ),
    "dual_support_all_pair_final_summary": (
        "paper_protocol/reviewer_risk/dual_support_all_pair_final_summary.json",
        "0bf795eedba9e0555374a9080e6db7b9f6a69b917eba512d9b0453ff2929064b",
    ),
    "dual_support_all_pair_report": (
        "docs/PAPER/AAAI27_DUAL_SUPPORT_ALL_PAIR_EVALUATION_20260722.md",
        "71d0121e261f045aab5fc0511783e9114e06ce7893f2c1518463afa772e5be3c",
    ),
    "dual_support_interface_report": (
        "docs/PAPER/AAAI27_REFERENCE_CONDITIONED_DUAL_SUPPORT_INTERFACE_20260722.md",
        "019e61e9200f609efd94b9ec02693ce7e16e1ba53a449a18bfe0673c691b38b9",
    ),
}


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"append-only dry run refuses existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(path: Path) -> dict[str, Any]:
    rows = []
    total_size = 0
    files = sorted((entry for entry in path.rglob("*") if entry.is_file()), key=lambda value: value.as_posix())
    for item in files:
        stat = item.stat()
        total_size += stat.st_size
        rows.append(f"{item.relative_to(path).as_posix()} {stat.st_size} {stat.st_mtime_ns}\n")
    metadata = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    try:
        allocated = int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])
    except (FileNotFoundError, subprocess.CalledProcessError):
        allocated = total_size
    return {"file_count": len(files), "bytes": allocated, "metadata_sha256_ns": metadata}


def archive_snapshot(formal_root: Path, all_pair_root: Path) -> dict[str, Any]:
    repo = {}
    for name, (relative, expected) in ARCHIVE_FILES.items():
        path = PROJECT_ROOT / relative
        actual = sha256(path)
        repo[name] = {"path": relative, "expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected}
    return {
        "repository_archive": repo,
        "repository_archive_pass": all(row["pass"] for row in repo.values()),
        "formal_output_tree": tree_manifest(formal_root),
        "dual_support_all_pair_output_tree": tree_manifest(all_pair_root),
    }


def case_rows(cache: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    rows, validity = [], []
    target = record["target_view_fold"]
    for position, (outfit, condition) in enumerate(zip(record["garment_labels"], record["reference_condition_ids"])):
        legal = [view for view in ("cond_000000", "cond_000318", "cond_000017", "cond_000347") if view != target]
        if legal[position] != condition or condition == target:
            raise RuntimeError("CONTROLLER-DRY-RUN-TARGET-LEAKAGE")
        episode = cache["episodes"][f"{outfit}/{target}"]["normal"]
        rows.append(episode["f2"][position])
        validity.append(episode["valid"][position])
    return torch.stack(rows), torch.stack(validity)


def selected_cases(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    pure = [row for row in manifest["formal_pure_endpoint_episodes"] if row["target_view_fold"] == "cond_000000"]
    aab = [row for row in manifest["query_sets"] if row["assignment_type"] == "AAB"][:3]
    abb = [row for row in manifest["query_sets"] if row["assignment_type"] == "ABB"][:3]
    if len(pure) != 5 or len(aab) != 3 or len(abb) != 3:
        raise RuntimeError("CONTROLLER-DRY-RUN-CASE-COUNT-MISMATCH")
    return pure + aab + abb


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def fresh_probe(seed: int, feature_cache: Path, training_manifest: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    if cache.get("schema_version") != "canondressgs.paper_frozen_reference_rows.v1":
        raise RuntimeError("CONTROLLER-DRY-RUN-FEATURE-CACHE-MISMATCH")
    manifest = json.loads(training_manifest.read_text(encoding="utf-8"))
    first_row = cache["episodes"]["O01/cond_000000"]["normal"]["f2"]
    input_dim = 2 * int(first_row.shape[1])
    model = ReferenceConditionedDualSupportController(seed=seed, input_dim=input_dim).eval()
    endpoint_bank = {outfit: f"FROZEN_TEACHER_ENDPOINT/{outfit}" for outfit in OUTFIT_ORDER}
    results = []
    losses = []
    with torch.no_grad():
        for record in selected_cases(manifest):
            rows, valid = case_rows(cache, record)
            distribution = model(rows, valid)
            selection = stable_top2_selection(distribution.probabilities)
            runtime = construct_dual_support_runtime(selection, endpoint_bank)
            target = torch.tensor(record["target_distribution"], dtype=distribution.logits.dtype)
            losses.append(soft_target_cross_entropy(distribution.logits, target))
            schema = inference_result_schema(distribution, selection, runtime)
            results.append({
                "record_id": record["record_id"],
                "assignment_type": record["assignment_type"],
                "logits_sha256": tensor_sha(distribution.logits),
                "probabilities_sha256": tensor_sha(distribution.probabilities),
                "top1_outfit": schema["top1_outfit"],
                "top2_outfit": schema["top2_outfit"],
                "mode": schema["mode"],
                "runtime_branch_count": len(runtime.branches),
                "target_forward_leakage": schema["target_forward_leakage"],
            })
        loss = torch.stack(losses).mean()
    guards = {
        "low_top2_mass": stable_top2_selection(torch.tensor([0.40, 0.30, 0.10, 0.10, 0.10])).as_dict(),
        "low_secondary_weight": stable_top2_selection(torch.tensor([0.91, 0.08, 0.005, 0.003, 0.002])).as_dict(),
        "tie": stable_top2_selection(torch.tensor([0.45, 0.45, 0.10, 0.0, 0.0])).as_dict(),
    }
    payload = {
        "schema_version": "canondressgs.research.dual_support_controller_fresh_probe.v1",
        "seed": seed,
        "input_dim": input_dim,
        "parameter_count": model.parameter_count,
        "initialization_sha256": tensor_mapping_sha256(model.state_dict()),
        "step0_loss": float(loss),
        "step0_loss_sha256": tensor_sha(loss),
        "cases": results,
        "guards": guards,
        "counts": {"forward": len(results), "backward": 0, "optimizer_step": 0, "scheduler_step": 0, "checkpoint_write": 0, "formal_render": 0, "formal_metrics": 0},
        "paper_final": False,
    }
    return payload


def run_fresh_process(seed: int, feature_cache: Path, training_manifest: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--fresh-probe",
        "--seed",
        str(seed),
        "--feature-cache",
        str(feature_cache),
        "--training-manifest",
        str(training_manifest),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def exact_probe_signature(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "initialization_sha256": probe["initialization_sha256"],
        "step0_loss_sha256": probe["step0_loss_sha256"],
        "cases": probe["cases"],
        "guards": probe["guards"],
    }


def forward_boundary_artifact() -> dict[str, Any]:
    signature = inspect.signature(ReferenceConditionedDualSupportController.forward)
    arguments = [name for name in signature.parameters if name != "self"]
    forbidden_overlap = sorted(set(arguments).intersection(FORBIDDEN_FORWARD_INPUTS))
    return {
        "schema_version": "canondressgs.research.dual_support_controller_forward_boundary.v1",
        "label": LABEL,
        "prediction_forward_arguments": arguments,
        "declared_forward_arguments": list(controller_forward_argument_names()),
        "allowed": ["reference_rgb", "reference_clothing_masks", "frozen_F2_features", "reference_validity", "fixed_normalization"],
        "forbidden": list(FORBIDDEN_FORWARD_INPUTS),
        "loss_target_only": ["garment_labels", "soft_distribution"],
        "downstream_render_only": ["target_pose", "target_camera"],
        "forbidden_argument_overlap": forbidden_overlap,
        "ground_truth_outfit_id_in_forward": False,
        "teacher_endpoint_id_in_forward": False,
        "target_pose_camera_in_controller": False,
        "target_forward_leakage": len(forbidden_overlap),
        "status": "PASS" if not forbidden_overlap and arguments == list(controller_forward_argument_names()) else "FAIL",
        "paper_final": False,
        "paper_final_count": 0,
    }


def evaluator_contract_artifact() -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.research.dual_support_controller_evaluator_contract.v1",
        "label": LABEL,
        "execution_in_this_task": False,
        "pure_endpoint_task": {
            "correct": 20,
            "swaps": 80,
            "checks": ["endpoint_accuracy", "single_reference", "dropout", "color_counterfactual", "endpoint_parity", "single_fallback_rate"],
        },
        "mixed_reference_task": {
            "query_sets_per_seed": 320,
            "seeds": [0, 1, 2],
            "total_seed_queries": 960,
            "checks": ["top2_pair_accuracy", "AAB_ABB_ordering", "mixture_calibration", "dual_support_activation_rate", "fallback_rate", "LPIPS", "silhouette_IoU", "boundary_F_score", "ghosting", "identity_contamination"],
        },
        "perturbations": ["grayscale", "hue", "blur", "mask_morphology", "mode_switching_stability"],
        "preregistered_success": {
            "pure_top1_accuracy": "20/20",
            "pure_single_endpoint_rate_min": 0.95,
            "mixed_top2_pair_accuracy_min": 0.90,
            "AAB_ABB_ordering_accuracy_min": 0.85,
            "mixed_dual_support_activation_rate_min": 0.80,
            "macro_LPIPS": "BETTER_THAN_FULL_LINEAR",
            "silhouette_IoU": "NOT_LOWER_THAN_FULL_LINEAR",
            "severe_artifact_reduction_vs_FULL_LINEAR_min": 0.50,
            "identity_contamination_max": 0,
            "grade3_ghosting_pairs_max": 1,
            "ground_truth_id_or_manual_alpha": 0,
            "frozen_mutation": 0,
        },
        "thresholds_frozen_before_formal_results": True,
        "formal_metrics_executed": 0,
        "formal_renders_executed": 0,
        "paper_final": False,
        "paper_final_count": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"append-only smoke output root already exists: {output_root}")
    manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    if manifest["counts"]["pair_fold_query_sets"] != 320 or manifest["counts"]["retained_formal_pure_endpoint_episodes"] != 20:
        raise RuntimeError("CONTROLLER-DRY-RUN-MANIFEST-MISMATCH")
    before = archive_snapshot(args.formal_root.resolve(), args.all_pair_root.resolve())
    if not before["repository_archive_pass"]:
        raise RuntimeError("CONTROLLER-DRY-RUN-ARCHIVE-MISMATCH")
    output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(output_root / "audits/frozen_before.json", before)

    seed_audits = []
    for seed in SEEDS:
        first = run_fresh_process(seed, args.feature_cache.resolve(), args.training_manifest.resolve())
        second = run_fresh_process(seed, args.feature_cache.resolve(), args.training_manifest.resolve())
        exact = exact_probe_signature(first) == exact_probe_signature(second)
        seed_root = output_root / f"seed_{seed}"
        atomic_json(seed_root / "fresh_process_1.json", first)
        atomic_json(seed_root / "fresh_process_2.json", second)
        audit = {
            "seed": seed,
            "same_seed_fresh_process_exact": exact,
            "initialization_bitwise_exact": first["initialization_sha256"] == second["initialization_sha256"],
            "fixed_batch_logits_bitwise_exact": [row["logits_sha256"] for row in first["cases"]] == [row["logits_sha256"] for row in second["cases"]],
            "distribution_bitwise_exact": [row["probabilities_sha256"] for row in first["cases"]] == [row["probabilities_sha256"] for row in second["cases"]],
            "top2_choice_exact": [(row["top1_outfit"], row["top2_outfit"]) for row in first["cases"]] == [(row["top1_outfit"], row["top2_outfit"]) for row in second["cases"]],
            "initialization_sha256": first["initialization_sha256"],
            "status": "PASS" if exact else "FAIL",
        }
        atomic_json(seed_root / "determinism_audit.json", audit)
        seed_audits.append(audit)

    initialization_hashes = [row["initialization_sha256"] for row in seed_audits]
    model = ReferenceConditionedDualSupportController(seed=0, input_dim=512)
    candidate_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    optimizer_provenance = {
        "schema_version": "canondressgs.research.dual_support_controller_optimizer_provenance.v1",
        "label": LABEL,
        "controller_candidate_optimizer": {
            "created": True,
            "instances_created": 1,
            "class": "torch.optim.Adam",
            "candidate_parameter_count": model.parameter_count,
            "parameter_membership_exact": optimizer_ids == candidate_ids,
            "zero_grad_calls": 0,
            "steps": 0,
            "state_saves": 0,
        },
        "legacy_context_optimizer": {"created": False, "instances_created": 0, "zero_grad_calls": 0, "steps": 0, "state_saves": 0},
        "candidate_frozen_parameter_overlap": 0,
        "candidate_legacy_parameter_overlap": 0,
        "backward_calls": 0,
        "scheduler_created": False,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "formal_training_steps": 0,
        "status": "PASS" if optimizer_ids == candidate_ids else "FAIL",
        "paper_final": False,
        "paper_final_count": 0,
    }
    del optimizer
    boundary = forward_boundary_artifact()
    evaluator = evaluator_contract_artifact()

    after = archive_snapshot(args.formal_root.resolve(), args.all_pair_root.resolve())
    frozen_unchanged = before == after
    atomic_json(output_root / "audits/frozen_after.json", after)
    atomic_json(output_root / "audits/optimizer_provenance.json", optimizer_provenance)
    atomic_json(output_root / "audits/forward_boundary.json", boundary)
    atomic_json(output_root / "audits/evaluator_contract.json", evaluator)

    all_exact = all(row["same_seed_fresh_process_exact"] for row in seed_audits)
    cross_unique = len(set(initialization_hashes)) == len(SEEDS)
    guard_probe = json.loads((output_root / "seed_0/fresh_process_1.json").read_text(encoding="utf-8"))["guards"]
    guards_pass = (
        guard_probe["low_top2_mass"]["fallback_reason"] == "LOW_TOP2_MASS"
        and guard_probe["low_secondary_weight"]["fallback_reason"] == "LOW_SECONDARY_WEIGHT"
        and guard_probe["tie"]["top1_outfit"] == "O01"
        and guard_probe["tie"]["top2_outfit"] == "O02"
    )
    classifications = {
        "CONTROLLER_ADAPTER": "PASS",
        "SOFT_TARGET_DATASET": "PASS" if manifest["duplicate_detection"]["pass"] and manifest["target_exclusion"]["pass"] else "FAIL",
        "TOP2_SELECTION": "PASS" if guards_pass else "FAIL",
        "SINGLE_ENDPOINT_FALLBACK": "PASS" if guards_pass else "FAIL",
        "DUAL_SUPPORT_RUNTIME_INTERFACE": "PASS",
        "FORWARD_BOUNDARY": boundary["status"],
        "NO_FORMAL_TRAINING_GATE": "PASS" if optimizer_provenance["status"] == "PASS" and frozen_unchanged else "FAIL",
    }
    all_pass = all(value == "PASS" for value in classifications.values()) and all_exact and cross_unique and frozen_unchanged
    summary = {
        "schema_version": "canondressgs.research.dual_support_controller_dry_run_summary.v1",
        "label": LABEL,
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "status": "PASS" if all_pass else "FAIL",
        "controller": {"path": "scene/reference_conditioned_dual_support_controller.py", "input_dim": 512, "logit_count": 5, "parameter_count": model.parameter_count, "initialization_policy": "RANDOM_SEEDED_INITIALIZATION", "seeds": list(SEEDS)},
        "dataset_counts": manifest["counts"],
        "dry_run_cases_per_fresh_process": {"pure": 5, "AAB": 3, "ABB": 3, "low_top2_mass_guard": 1, "low_secondary_weight_guard": 1, "tie_guard": 1},
        "same_seed_fresh_process_exact": all_exact,
        "cross_seed_initialization_hashes_unique": cross_unique,
        "cross_seed_unique_initialization_count": len(set(initialization_hashes)),
        "seed_audits": seed_audits,
        "frozen_before": before,
        "frozen_after": after,
        "frozen_unchanged": frozen_unchanged,
        "classifications": classifications,
        "counts": {"formal_training": 0, "backward": 0, "candidate_optimizer_step": 0, "legacy_optimizer_step": 0, "scheduler_step": 0, "checkpoint_write": 0, "formal_render": 0, "formal_metrics": 0},
        "paper_final": False,
        "paper_final_count": 0,
        "next_task": "TRAIN_AND_EVALUATE_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER" if all_pass else "REPAIR_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER",
        "next_task_started": False,
    }
    archive = output_root / "audits/archive"
    atomic_json(archive / "dual_support_controller_forward_boundary.json", boundary)
    atomic_json(archive / "dual_support_controller_optimizer_provenance.json", optimizer_provenance)
    atomic_json(archive / "dual_support_controller_evaluator_contract.json", evaluator)
    atomic_json(archive / "dual_support_controller_dry_run_summary.json", summary)
    atomic_json(output_root / "audits/dry_run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-probe", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--formal-root", type=Path)
    parser.add_argument("--all-pair-root", type=Path)
    args = parser.parse_args()
    if args.fresh_probe:
        if args.seed not in SEEDS:
            raise ValueError("fresh-probe seed must be 0, 1, or 2")
        print(json.dumps(fresh_probe(args.seed, args.feature_cache.resolve(), args.training_manifest.resolve()), sort_keys=True))
        return
    if args.output_root is None or args.formal_root is None or args.all_pair_root is None:
        parser.error("main dry run requires --output-root, --formal-root, and --all-pair-root")
    summary = run(args)
    print(json.dumps({"status": summary["status"], "classifications": summary["classifications"], "output_root": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
