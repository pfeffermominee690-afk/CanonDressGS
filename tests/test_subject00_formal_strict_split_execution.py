from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "tools/second_identity/run_subject00_formal_strict_split.py"
)
PROTOCOL_ROOT = REPO_ROOT / "paper_protocol/second_identity"
ARTIFACTS = {
    "subject00_formal_training_contract.json": "contract_content_sha256",
    "subject00_formal_training_schedule.json": "schedule_content_sha256",
    "subject00_formal_train_record_manifest.json": "manifest_content_sha256",
    "subject00_formal_evaluation_manifests.json": "manifests_content_sha256",
    "subject00_formal_visual_review_manifest.json": "manifest_content_sha256",
    "subject00_formal_resource_budget.json": "budget_content_sha256",
    "subject00_formal_protocol_final_summary.json": "summary_content_sha256",
}


def canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_sealed(path: Path, key: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value[key]
    unsigned = dict(value)
    del unsigned[key]
    assert canonical_sha(unsigned) == claimed
    return value


def runner_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            result[target.id] = node.value
    return result


def test_protocol_seals_and_counts_remain_frozen() -> None:
    values = {
        name: load_sealed(PROTOCOL_ROOT / name, key)
        for name, key in ARTIFACTS.items()
    }
    schedule = values["subject00_formal_training_schedule.json"]
    records = values["subject00_formal_train_record_manifest.json"]
    evaluation = values["subject00_formal_evaluation_manifests.json"]
    visual = values["subject00_formal_visual_review_manifest.json"]
    resource = values["subject00_formal_resource_budget.json"]
    assert schedule["optimizer_steps"] == 101_245
    assert schedule["pass_count"] == 5
    assert schedule["new_checkpoint_steps"] == [
        20_249,
        40_498,
        60_747,
        80_996,
        100_000,
        101_245,
    ]
    assert len(schedule["steps"]) == 101_245
    assert records["record_count"] == len(records["records"]) == 20_249
    assert sum(
        evaluation["quadrants"][name]["valid_count"]
        for name in evaluation["quadrant_order"]
    ) == 29_954
    assert evaluation["full_evaluation"]["full_evaluation_png_count"] == 0
    assert visual["query_count"] == len(visual["queries"]) == 96
    assert resource["resource_gate"]["minimum_free_bytes"] == 32_212_254_720


def test_runner_freezes_execution_identity_and_paths() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = runner_assignments(tree)
    assert ast.literal_eval(assignments["TASK_ID"]) == (
        "AAAI27-SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001"
    )
    assert ast.literal_eval(assignments["PROTOCOL_HEAD"]) == (
        "4993f5c865ec19895f35811fa399fc4a1834c6a7"
    )
    assert ast.literal_eval(assignments["TARGET_BRANCH"]) == (
        "research/mmlphuman-subject00-formal-strict-split-experiment-20260725"
    )
    assert "SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001" in source
    assert "SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001" in source
    assert "conflicting non-protocol output root exists" in source


def test_runner_has_atomic_checkpoint_and_exact_resume_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "preflight",
        "save_checkpoint_atomic",
        "load_complete_checkpoint",
        "validate_resume_log",
        "train_phase",
        "roundtrip_phase",
        "evaluate_phase",
        "git_blob",
    }.issubset(functions)
    assert "os.replace(temporary, path)" in source
    assert '"atomically_sealed": True' in source
    assert '"checkpoint_overwrite": 0' in source
    assert '"repeated_optimizer_steps": 0' in source
    assert "resume log must end exactly at the sealed checkpoint" in source
    assert "checkpoint overwrite forbidden" in source
    assert "historical Subject02 source closure changed" in source
    assert "formal runtime differs from executed medium source" in source


def test_runner_preserves_streaming_evaluation_and_capacity_metrics() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"full_evaluation_png_count": 0' in source
    assert "final_full_query_records.jsonl" in source
    assert "standard_deviation" in source
    for name in (
        "foreground_rgb_variance",
        "foreground_chroma",
        "gt_pred_color_histogram_l1",
        "body_conforming_bias_score",
        "sleeve_bulk_roi_absolute_coverage_error",
        "hem_boundary_vertical_error_px",
        "face_hand_roi_lpips",
        "face_hand_roi_rgb_mae",
    ):
        assert name in source
    assert "projected fitted per-frame SMPL-X joints" in source


def test_runner_never_restores_medium_checkpoint_for_training() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "MEDIUM_FINAL_PATH" not in source
    assert "torch.load(CANARY_STEP0_PATH" in source
    assert "restore_model_from_checkpoint" in source
