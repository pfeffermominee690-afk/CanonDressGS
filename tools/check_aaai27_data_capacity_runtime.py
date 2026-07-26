from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_OUTFITS = ("O01", "O02", "O03", "O04", "O06", "O07", "O08")
EXPECTED_CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static/runtime-contract checks for the AAAI capacity gate")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_tree(path: Path) -> tuple[str, ast.AST]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def function_names(tree: ast.AST) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def tests(root: Path) -> dict[str, Any]:
    builder = root / "tools/aaai27/build_data_capacity_fixture.py"
    runner = root / "tools/aaai27/run_fixed_open_capacity_oracle.py"
    config_path = root / "configs/aaai27/subject02_data_capacity_gate_v1.yaml"
    builder_source, builder_tree = load_tree(builder)
    runner_source, runner_tree = load_tree(runner)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checks = {
        "exact_outfits": tuple(config["candidate_outfits"]) == EXPECTED_OUTFITS,
        "exact_conditions": tuple(item["condition_id"] for item in config["conditions"]) == EXPECTED_CONDITIONS,
        "segformer_frozen": (
            config["segmentation"]["model"] == "mattmdjaga/segformer_b2_clothes"
            and config["segmentation"]["revision"] == "584abc1e1d260e23c0fc627c5217a09b2b461046"
            and config["segmentation"]["local_files_only"] is True
            and config["segmentation"]["garment_core_probability_min"] == 0.35
        ),
        "builder_phases_present": {"prepare_segmentation", "build_fixture", "finalize_support"}.issubset(function_names(builder_tree)),
        "safe_clothing_formula_present": "clothing_raw & raw_fg & ~protected" in builder_source,
        "raw_hash_rechecked": "raw target changed after generation" in builder_source,
        "visual_gate_status_schema_compatible": (
            '"visual_status", visual_by_sample[sample_id].get("status")' in builder_source
        ),
        "no_raw_postprocess": "shutil.copyfile(raw_path, edit_rgb)" in builder_source,
        "fixed_open_oracle_used": "FixedOpenGaussianOracle" in runner_source,
        "no_anchor_oracle": "AnchorResidualOracle" not in runner_source,
        "no_trainable_gate": "gate_parameter_count" in runner_source and "fixed_open_oracle_contract" in runner_source,
        "exact_480_steps": "range(1, 481)" in runner_source,
        "round_robin_exact": "CONDITIONS[(step - 1) % 4]" in runner_source,
        "target_loss_only_declared": '"target_fields_loss_only": True' in runner_source,
        "base_200k_enforced": runner_source.count("200000") >= 3,
        "no_image_conditioner": "image_conditioning_used" in runner_source and "ImageConditioned" not in runner_source,
        "no_teacher": '"teacher_used": False' in runner_source,
        "acceptance_5_and_2_percent": config["numeric_acceptance"]["pass_reduction_min"] == 0.05 and config["numeric_acceptance"]["warn_reduction_min"] == 0.02,
        "selection_uses_frozen_function": "select_benchmark" in runner_source,
        "remaining_generation_not_started": '"remaining_targets_generated": 0' in runner_source,
        "formal_training_forbidden": '"formal_training_allowed": False' in runner_source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"status": "PASS" if not failed else "FAIL", "passed": len(checks) - len(failed), "total": len(checks), "failed": failed, "checks": checks}


def main() -> int:
    args = parse_args()
    result = tests(args.repo_root.resolve())
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
