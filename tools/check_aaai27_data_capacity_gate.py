#!/usr/bin/env python3
"""Check the frozen AAAI-27 28-image data/capacity gate contract.

These checks are intentionally runnable before generation. They freeze the
candidate/condition set, reuse criteria, generation budget, V5.3 mask rule,
fixed-open Oracle contract, numerical thresholds, benchmark replacement rules,
and long-term branch immutability.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_protected_region_fixture_v5_1 import build_safe_clothing_mask  # noqa: E402
from tools.check_o00_arm_support_closure import (  # noqa: E402
    test_o00_fixed_open_oracle_has_no_gate_parameters,
    test_o00_oracle_uses_r2_rotation_path,
    test_o00_oracle_uses_shared_canonical_residual,
    test_o00_oracle_uses_v5_3_loss,
)
from tools.run_aaai27_data_capacity_preflight import (  # noqa: E402
    CANDIDATES,
    CONDITIONS,
    EXPECTED_LONG_HEAD,
    existing_target_is_reusable,
    generation_gate_status,
    select_benchmark,
)


EXPECTED_CANDIDATES = ("O01", "O02", "O03", "O04", "O06", "O07", "O08")
EXPECTED_CONDITIONS = (
    ("cond_000000", "front"),
    ("cond_000318", "back"),
    ("cond_000017", "left"),
    ("cond_000347", "right"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aaai27/subject02_data_capacity_gate_v1.yaml"),
    )
    parser.add_argument(
        "--gate-manifest",
        type=Path,
        default=Path("artifacts/aaai27_sprint/target_generation_gate_28_manifest.json"),
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gate_uses_exact_seven_candidate_outfits(context: dict[str, Any]) -> None:
    assert CANDIDATES == EXPECTED_CANDIDATES
    assert tuple(context["config"]["candidate_outfits"]) == EXPECTED_CANDIDATES
    assert tuple(context["manifest"]["candidate_outfits"]) == EXPECTED_CANDIDATES


def test_gate_uses_exact_four_conditions(context: dict[str, Any]) -> None:
    assert CONDITIONS == EXPECTED_CONDITIONS
    actual = tuple((item["condition_id"], item["view"]) for item in context["config"]["conditions"])
    assert actual == EXPECTED_CONDITIONS
    assert context["manifest"]["canonical_conditions"] == {
        "front": "cond_000000",
        "back": "cond_000318",
        "left": "cond_000017",
        "right": "cond_000347",
    }


def test_existing_o01_targets_are_reused_only_if_verified(context: dict[str, Any]) -> None:
    checks = {"base": True, "clay": True, "garment_reference": True}
    dimensions = {"width": 1024, "height": 1536, "mode": "RGB"}
    verified = {
        "status": "SUCCESS",
        "model_id": "gpt-image-2",
        "generation_backend": "openai_images_edit",
    }
    assert existing_target_is_reusable(verified, dimensions, checks, True)
    assert not existing_target_is_reusable({**verified, "model_id": "UNEXPOSED_PLATFORM_IMAGE_MODEL"}, dimensions, checks, True)
    assert not existing_target_is_reusable({**verified, "generation_backend": "codex_builtin_imagegen"}, dimensions, checks, True)
    assert not existing_target_is_reusable(verified, dimensions, checks, False)


def test_generation_call_budget_is_capped(context: dict[str, Any]) -> None:
    contract = context["config"]["generation"]
    assert contract["primary_call_max"] == 28
    assert contract["retry_call_max"] == 4
    assert contract["total_call_max"] == 32
    assert contract["primary_call_max"] + contract["retry_call_max"] == contract["total_call_max"]


def test_raw_direct_edit_is_immutable(context: dict[str, Any]) -> None:
    generation = context["config"]["generation"]
    assert generation["first_valid_response_is_immutable"] is True
    assert generation["candidate_sampling_forbidden"] is True


def test_protected_only_change_is_not_hard_failure(context: dict[str, Any]) -> None:
    assert generation_gate_status({"face_local_change", "shoe_color_change"}) == "PROTECTED_ONLY_DIAGNOSTIC_WARN"
    assert generation_gate_status({"pose_changed"}) == "FAIL"
    assert context["config"]["generation"]["protected_only_diagnostic_warn_is_nonblocking"] is True


def test_gate_fixture_has_28_unique_samples(context: dict[str, Any]) -> None:
    records = context["manifest"]["records"]
    ids = [record["record_id"] for record in records]
    pairs = {(record["outfit_id"], record["condition_id"]) for record in records}
    expected = {(outfit, condition) for outfit in EXPECTED_CANDIDATES for condition, _ in EXPECTED_CONDITIONS}
    assert len(records) == len(ids) == len(set(ids)) == 28
    assert pairs == expected
    assert context["config"]["fixture"]["expected_sample_count"] == 28


def test_safe_clothing_excludes_protected(context: dict[str, Any]) -> None:
    raw = np.array([[1, 1, 1], [0, 1, 1]], dtype=np.uint8)
    foreground = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8)
    protected = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    before = (raw.copy(), foreground.copy(), protected.copy())
    safe = build_safe_clothing_mask(raw, foreground, protected)
    assert np.array_equal(safe, np.array([[1, 0, 0], [0, 1, 0]], dtype=bool))
    assert not np.any(safe & protected.astype(bool))
    assert not np.any(safe & ~foreground.astype(bool))
    assert all(np.array_equal(item, frozen) for item, frozen in zip((raw, foreground, protected), before))


def test_fixed_open_oracle_has_no_trainable_gate(context: dict[str, Any]) -> None:
    assert context["config"]["oracle"]["train_gate"] is False
    assert context["config"]["oracle"]["fixed_gates"] == {"geometry": 1.0, "appearance": 1.0, "opacity": 1.0}
    test_o00_fixed_open_oracle_has_no_gate_parameters()


def test_fixed_open_oracle_uses_shared_canonical_field(context: dict[str, Any]) -> None:
    assert context["config"]["oracle"]["shared_canonical_field"] is True
    assert context["config"]["oracle"]["condition_specific_residuals"] is False
    test_o00_oracle_uses_shared_canonical_residual()


def test_fixed_open_oracle_uses_r2_rotation(context: dict[str, Any]) -> None:
    assert "delta_rotvec" in context["config"]["oracle"]["optimized_attributes"]
    test_o00_oracle_uses_r2_rotation_path()


def test_fixed_open_oracle_uses_v5_3_loss(context: dict[str, Any]) -> None:
    assert context["config"]["oracle"]["loss_contract"] == "v5_3_region_aware_dual_target"
    test_o00_oracle_uses_v5_3_loss()


def test_outfit_gate_thresholds_are_frozen(context: dict[str, Any]) -> None:
    numeric = context["config"]["numeric_acceptance"]
    assert numeric == {
        "first_window": [0, 19],
        "last_window": [441, 480],
        "slope_window": [401, 480],
        "pass_reduction_min": 0.05,
        "warn_reduction_min": 0.02,
        "pass_requires_both_slopes_nonpositive": True,
        "protected_final_max": 0.005,
        "protected_and_preserve_relative_max": 1.10,
        "finite_required": True,
    }
    oracle = context["config"]["oracle"]
    assert oracle["seed"] == 20260718 and oracle["steps"] == 480
    assert oracle["updates_per_condition"] == 120


def test_outfit_replacement_rules_are_deterministic(context: dict[str, Any]) -> None:
    passed = {outfit: "OUTFIT_GATE_PASS" for outfit in EXPECTED_CANDIDATES}
    primary = select_benchmark(passed)
    assert primary["train_outfits"] == ["O01", "O02", "O04", "O06"]
    assert primary["unseen_outfits"] == ["O03", "O08"]
    assert primary["reserve_enabled"] is False and primary["target_count"] == 192
    replacement = select_benchmark({**passed, "O03": "OUTFIT_GATE_FAIL"})
    assert replacement["unseen_outfits"] == ["O08", "O07"]


def test_failed_outfit_cannot_enter_benchmark(context: dict[str, Any]) -> None:
    status = {outfit: "OUTFIT_GATE_PASS" for outfit in EXPECTED_CANDIDATES}
    status["O02"] = "OUTFIT_GATE_FAIL"
    result = select_benchmark(status)
    assert "O02" not in result["selected_outfits"]
    status["O02"] = "OUTFIT_GATE_RESERVE"
    result = select_benchmark(status)
    assert "O02" not in result["selected_outfits"]


def test_five_outfit_fallback_requires_unseen_pass(context: dict[str, Any]) -> None:
    status = {outfit: "OUTFIT_GATE_FAIL" for outfit in EXPECTED_CANDIDATES}
    for outfit in ("O01", "O02", "O04", "O06", "O03"):
        status[outfit] = "OUTFIT_GATE_PASS"
    valid = select_benchmark(status)
    assert valid["status"] == "GO" and valid["target_count"] == 160 and valid["five_outfit_fallback"]
    status["O03"] = "OUTFIT_GATE_FAIL"
    status["O07"] = "OUTFIT_GATE_RESERVE"
    invalid = select_benchmark(status)
    assert invalid["status"] == "NO_GO" and invalid["target_count"] == 0


def test_long_term_branch_remains_unchanged(context: dict[str, Any]) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "refs/heads/pipeline/full-dressable-20260715"],
        cwd=context["repo_root"],
        text=True,
        encoding="utf-8",
    ).strip()
    assert head == EXPECTED_LONG_HEAD


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    manifest_path = args.gate_manifest if args.gate_manifest.is_absolute() else repo / args.gate_manifest
    context = {
        "repo_root": repo,
        "config": yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "manifest": read_json(manifest_path),
    }
    tests: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value) and value.__module__ == __name__
    ]
    results = []
    for name, test in tests:
        test(context)
        print(f"PASS {name}")
        results.append({"name": name, "status": "PASS"})
    payload = {"status": "PASS", "passed": len(results), "failed": 0, "tests": results}
    if args.output_json:
        output = args.output_json if args.output_json.is_absolute() else repo / args.output_json
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(f"PASS {len(results)}/{len(results)} AAAI data/capacity gate contract checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
