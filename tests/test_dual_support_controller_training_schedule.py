from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json"
CYCLE_PATH = ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_cycle.json"
SCHEDULE_PATH = ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_schedule_300_steps.json"
SUMMARY_PATH = ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_schedule_summary.json"
RUNTIME_PATH = ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_schedule_runtime.json"
SOURCE_HEAD = "174655ce6aabcae5d60ce45f3b4be319eb71e914"

from scene.dual_support_controller_training_schedule import (
    ASSIGNMENT_POSITION_ORDER,
    FOLD_ORDER,
    OUTFIT_ORDER,
    PAIR_ORDER,
    build_schedule,
    canonical_sha256,
    step_lookup,
)


MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
BUNDLE = build_schedule(MANIFEST)
CYCLE = json.loads(CYCLE_PATH.read_text(encoding="utf-8"))
SCHEDULE = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
SUMMARY = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _cycle_record_ids() -> list[str]:
    return [
        record["record_id"]
        for batch in BUNDLE.scientific_cycle["batches"]
        for record in batch["records"]
    ]


def _schedule_record_ids() -> list[str]:
    return [
        record["record_id"]
        for step in BUNDLE.scientific_schedule["steps"]
        for record in step["records"]
    ]


def test_training_scope_is_exactly_320_protocol_records() -> None:
    assert len(MANIFEST["query_sets"]) == 320
    assert set(_cycle_record_ids()) == {row["record_id"] for row in MANIFEST["query_sets"]}


def test_formal_pure_20_are_evaluation_only() -> None:
    assert len(MANIFEST["formal_pure_endpoint_episodes"]) == 20
    assert SUMMARY["formal_pure_isolation"]["role"] == "FORMAL_PURE_EVAL_ONLY"


def test_formal_pure_have_zero_training_exposure() -> None:
    formal_ids = {row["record_id"] for row in MANIFEST["formal_pure_endpoint_episodes"]}
    assert formal_ids.isdisjoint(_schedule_record_ids())
    assert SUMMARY["counts"]["formal_pure_training_exposures"] == 0


def test_consistent_duplicates_are_preserved() -> None:
    duplicate_ids = {row["record_id"] for row in MANIFEST["query_sets"] if row["duplicate_of"]}
    assert len(duplicate_ids) == 80
    assert duplicate_ids.issubset(_cycle_record_ids())


def test_outfit_order_is_frozen() -> None:
    assert OUTFIT_ORDER == ("O01", "O02", "O03", "O04", "O08")
    assert tuple(BUNDLE.scientific_cycle["outfit_order"]) == OUTFIT_ORDER


def test_all_10_pairs_are_present() -> None:
    assert tuple(MANIFEST["pairs"]) == PAIR_ORDER
    assert {row["pair_id"] for row in MANIFEST["query_sets"]} == set(PAIR_ORDER)


def test_fold_order_is_frozen() -> None:
    assert tuple(MANIFEST["frozen_target_view_order"]) == FOLD_ORDER
    assert [batch["target_view_fold"] for batch in BUNDLE.scientific_cycle["batches"][:8]] == list(FOLD_ORDER) * 2


def test_each_dominant_queue_has_64_records() -> None:
    counts = Counter(record["dominant_outfit"] for batch in BUNDLE.scientific_cycle["batches"] for record in batch["records"])
    assert counts == Counter({outfit: 64 for outfit in OUTFIT_ORDER})


def test_each_dominant_fold_has_16_records() -> None:
    counts = Counter(
        (record["dominant_outfit"], batch["target_view_fold"])
        for batch in BUNDLE.scientific_cycle["batches"]
        for record in batch["records"]
    )
    assert set(counts.values()) == {16}
    assert len(counts) == 20


def test_counterpart_order_is_deterministic() -> None:
    o01_fold0 = [
        BUNDLE.scientific_cycle["batches"][4 * slot]["records"][0]["pair_id"]
        for slot in range(16)
    ]
    assert o01_fold0 == [pair for pair in ("O01_O02", "O01_O03", "O01_O04", "O01_O08") for _ in range(4)]


def test_pure_precedes_three_assignment_positions() -> None:
    compositions = [
        BUNDLE.scientific_cycle["batches"][4 * slot]["records"][0]["composition"]
        for slot in range(4)
    ]
    assert compositions == [
        "PURE_DOMINANT",
        "MIXED_ASSIGNMENT_POSITION_0",
        "MIXED_ASSIGNMENT_POSITION_1",
        "MIXED_ASSIGNMENT_POSITION_2",
    ]


def test_assignment_positions_are_0_1_2() -> None:
    positions = [
        BUNDLE.scientific_cycle["batches"][4 * slot]["records"][0]["assignment_position"]
        for slot in range(4)
    ]
    assert positions == [None, *ASSIGNMENT_POSITION_ORDER]


def test_cycle_has_64_batches() -> None:
    assert len(BUNDLE.scientific_cycle["batches"]) == 64
    assert [row["batch_index"] for row in BUNDLE.scientific_cycle["batches"]] == list(range(64))


def test_each_batch_has_five_records() -> None:
    assert all(len(batch["records"]) == 5 for batch in BUNDLE.scientific_cycle["batches"])


def test_each_batch_has_one_of_each_dominant_outfit() -> None:
    assert all(tuple(row["dominant_outfit"] for row in batch["records"]) == OUTFIT_ORDER for batch in BUNDLE.scientific_cycle["batches"])


def test_each_batch_uses_one_target_view_fold() -> None:
    index = {row["record_id"]: row for row in MANIFEST["query_sets"]}
    assert all(
        {index[row["record_id"]]["target_view_fold"] for row in batch["records"]}
        == {batch["target_view_fold"]}
        for batch in BUNDLE.scientific_cycle["batches"]
    )


def test_each_cycle_covers_320_records_exactly_once() -> None:
    ids = _cycle_record_ids()
    assert len(ids) == len(set(ids)) == 320


def test_300_step_wrap_rule() -> None:
    assert len(BUNDLE.scientific_schedule["steps"]) == 300
    assert (step_lookup(BUNDLE, 1)["cycle_index"], step_lookup(BUNDLE, 1)["batch_index"]) == (0, 0)
    assert (step_lookup(BUNDLE, 64)["cycle_index"], step_lookup(BUNDLE, 64)["batch_index"]) == (0, 63)
    assert (step_lookup(BUNDLE, 65)["cycle_index"], step_lookup(BUNDLE, 65)["batch_index"]) == (1, 0)
    assert (step_lookup(BUNDLE, 300)["cycle_index"], step_lookup(BUNDLE, 300)["batch_index"]) == (4, 43)


def test_total_sample_exposures_are_1500() -> None:
    assert len(_schedule_record_ids()) == 1500
    exposure = Counter(_schedule_record_ids())
    assert Counter(exposure.values()) == Counter({5: 220, 4: 100})


def test_three_seeds_share_identical_data_order() -> None:
    seed_hashes = SUMMARY["seed_data_order"]
    assert seed_hashes["seed_0"] == seed_hashes["seed_1"] == seed_hashes["seed_2"]
    assert seed_hashes["unique_hash_count"] == 1


def _fresh_process_hash() -> str:
    script = (
        "import json; from pathlib import Path; "
        "from scene.dual_support_controller_training_schedule import build_schedule; "
        "m=json.loads(Path('paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json').read_text(encoding='utf-8')); "
        "b=build_schedule(m); print(b.cycle_sha256+' '+b.schedule_sha256)"
    )
    return subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True).strip()


def test_schedule_hash_is_fresh_process_deterministic() -> None:
    assert _fresh_process_hash() == _fresh_process_hash()
    assert _fresh_process_hash() == f"{BUNDLE.cycle_sha256} {BUNDLE.schedule_sha256}"


def test_scientific_hash_excludes_absolute_paths() -> None:
    serialized = json.dumps(BUNDLE.scientific_schedule, sort_keys=True)
    assert "E:\\" not in serialized
    assert "/root/" not in serialized
    assert canonical_sha256(BUNDLE.scientific_schedule) == BUNDLE.schedule_sha256
    runtime = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    assert runtime["participates_in_scientific_hash"] is False


def test_protocol_and_unique_query_counts_are_reported() -> None:
    assert SUMMARY["counts"]["training_protocol_records"] == 320
    assert SUMMARY["counts"]["protocol_unique_logical_queries"] == 260
    assert len(SUMMARY["protocol_record_exposure"]) == 320
    assert len(SUMMARY["unique_query_exposure"]) == 260


def test_no_model_optimizer_backward_or_checkpoint() -> None:
    paths = [
        ROOT / "scene/dual_support_controller_training_schedule.py",
        ROOT / "tools/paper/build_dual_support_controller_training_schedule.py",
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not imports.intersection({"torch", "gsplat", "pytorch3d"})
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not calls.intersection({"backward", "step", "zero_grad", "save"})
    assert set(SUMMARY["execution_counts"].values()) == {0}


def test_source_controller_archive_is_immutable() -> None:
    protected = [
        "scene/reference_conditioned_dual_support_controller.py",
        "paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json",
        "paper_protocol/reviewer_risk/dual_support_controller_forward_boundary.json",
        "paper_protocol/reviewer_risk/dual_support_controller_evaluator_contract.json",
        "paper_protocol/reviewer_risk/dual_support_controller_dry_run_summary.json",
    ]
    assert subprocess.check_output(
        ["git", "diff", "--name-only", SOURCE_HEAD, "--", *protected], cwd=ROOT, text=True
    ).strip() == ""


def test_paper_final_remains_zero() -> None:
    assert SUMMARY["paper_final_count"] == 0
    assert SUMMARY["paper_final"] is False
    assert json.loads(
        (ROOT / "paper_protocol/reviewer_risk/dual_support_controller_dry_run_summary.json").read_text(encoding="utf-8")
    )["paper_final_count"] == 0
