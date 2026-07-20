from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from tools.paper import aaai27_paper_pipeline as cli
from tools.paper.aggregation import aggregate_evaluations
from tools.paper.evaluate_seen_outfit import ALL_METRICS, FIXED_VIEWS, SEEN_OUTFITS, evaluate_records
from tools.paper.method_adapters import ADAPTER_TYPES, adapter_for
from tools.paper.paper_exports import FIGURE_LAYOUTS, export_figure_layouts, export_tables, validate_figure_layouts
from tools.paper.path_resolver import contains_user_specific_path, resolve_paths
from tools.paper.planning import build_run_plan
from tools.paper.protocol_audit import audit_protocol, load_documents
from tools.paper.registry_state import transition_is_valid, validate_paper_final_evidence
from tools.paper.run_contract import ATTEMPT_DIRECTORIES, assert_asset_report_pass, create_attempt
from tools.paper.synthetic_fixture import MARKER, build_synthetic_records, synthetic_table_source


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml"
MANIFEST_PATH = ROOT / "paper_protocol/frozen_asset_manifest.json"
REGISTRY_PATH = ROOT / "paper_protocol/experiment_registry.yaml"
CONFIG, MANIFEST, REGISTRY = load_documents(CONFIG_PATH, MANIFEST_PATH, REGISTRY_PATH)
FINGERPRINT = REGISTRY["frozen_asset_manifest_sha256"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluations():
    return [evaluate_records(build_synthetic_records(seed, FINGERPRINT), expected_asset_fingerprint=FINGERPRINT) for seed in (0, 1, 2)]


def test_registry_counts_are_internally_consistent():
    audit = audit_protocol(
        CONFIG, MANIFEST, REGISTRY,
        claim_matrix_text=(ROOT / "docs/PAPER/AAAI27_CLAIM_EVIDENCE_MATRIX_SEEN_OUTFIT_20260720.md").read_text(encoding="utf-8"),
        table_figure_plan_text=(ROOT / "docs/PAPER/AAAI27_PAPER_TABLE_AND_FIGURE_PLAN_20260720.md").read_text(encoding="utf-8"),
    )
    assert audit["status"] == "PASS"
    assert audit["claim_matrix_parsed"] and audit["table_figure_plan_parsed"]
    assert (audit["total_count"], audit["executable_count"], audit["not_run_count"], audit["historical_evidence_count"]) == (55, 51, 51, 4)


def test_historical_a8_is_not_executable():
    historical = [item for item in REGISTRY["experiments"] if item["status"] == "HISTORICAL_EVIDENCE"]
    assert len(historical) == 4 and all(item["executable"] is False for item in historical)


def test_historical_a8_has_no_run_command():
    plan = build_run_plan(REGISTRY, CONFIG, resolve_paths(repo_root=ROOT))
    assert not any("PAPER-A8" in row["linux_command"] or "PAPER-A8" in row["windows_command"] for row in plan["runs"])


def test_registry_status_transition_is_valid():
    assert transition_is_valid("NOT_RUN", "PREFLIGHT_PASS")
    assert transition_is_valid("RUNNING", "TRAINED")
    assert transition_is_valid("EVALUATED", "MANUAL_REVIEW_REQUIRED")


def test_not_run_cannot_become_paper_final_directly():
    assert not transition_is_valid("NOT_RUN", "PAPER_FINAL")


def test_paper_final_requires_manual_adjudication():
    try:
        validate_paper_final_evidence({"evaluator_pass": True, "artifacts_complete": True})
    except ValueError:
        pass
    else:
        raise AssertionError("PAPER_FINAL accepted incomplete evidence")


def test_asset_mismatch_blocks_run():
    try:
        assert_asset_report_pass({"status": "PAPER_ASSET_MISMATCH", "failed_assets": ["basis"]})
    except RuntimeError as error:
        assert str(error) == "PAPER_ASSET_MISMATCH"
    else:
        raise AssertionError("asset mismatch did not block")


def test_path_resolver_has_no_user_specific_path():
    files = list((ROOT / "tools/paper").rglob("*.py")) + list((ROOT / "paper_protocol/generated").glob("paper_run_commands_*"))
    forbidden = ("C:" + "\\Users\\", "E:" + "\\model_train\\", "/root/" + "autodl-tmp/")
    assert all(not any(token in path.read_text(encoding="utf-8") for token in forbidden) for path in files)
    assert not contains_user_specific_path("${CANONDRESSGS_REPO_ROOT}/tools/paper")


def test_dry_run_creates_no_optimizer():
    source = inspect.getsource(cli.command_run)
    assert "torch.optim" not in source and "optimizer =" not in source
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "paper_output"
        cli.main(["run", "--dry-run", "--repo-root", str(ROOT), "--output-root", str(output), "--experiment-id", "PAPER-OURS-S0", "--seed", "0"])
        assert not output.exists()


def test_dry_run_does_not_initialize_cuda():
    code = "import sys; from tools.paper.aaai27_paper_pipeline import main; main(['status','--repo-root','.']); assert 'torch' not in sys.modules"
    environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, check=True, capture_output=True)


def test_dry_run_does_not_change_registry():
    before = _sha(REGISTRY_PATH)
    build_run_plan(REGISTRY, CONFIG, resolve_paths(repo_root=ROOT))
    assert _sha(REGISTRY_PATH) == before


def test_run_plan_uses_seeds_0_1_2():
    plan = build_run_plan(REGISTRY, CONFIG, resolve_paths(repo_root=ROOT))
    assert sorted({row["seed"] for row in plan["runs"] if row["seed"] is not None}) == [0, 1, 2]


def test_training_plan_uses_exactly_300_steps():
    plans = [adapter_for(item, CONFIG).build_training_plan() for item in REGISTRY["experiments"] if item["executable"]]
    assert all(plan["steps"] == 300 for plan in plans if plan["optimizer_required"])
    assert all(plan["steps"] == 0 for plan in plans if not plan["optimizer_required"])


def test_b1_is_upper_bound():
    item = next(item for item in REGISTRY["experiments"] if item["experiment_id"] == "PAPER-B1-FIXED")
    assert adapter_for(item, CONFIG).validate_contract()["role"] == "optimization_upper_bound"


def test_b2_is_seen_only():
    item = next(item for item in REGISTRY["experiments"] if item["experiment_id"] == "PAPER-B2-FIXED")
    report = adapter_for(item, CONFIG).validate_contract()
    assert report["seen_only"] and not report["allowed_in_held_out"]


def test_o07_is_not_in_seen_training():
    plan = build_run_plan(REGISTRY, CONFIG, resolve_paths(repo_root=ROOT))
    assert all("O07" not in row["linux_command"] for row in plan["runs"])
    assert CONFIG["permissions"]["train_on_o07"] is False


def test_o06_is_unused():
    assert CONFIG["data"]["unused_reserve"] == "O06"
    assert CONFIG["permissions"]["substitute_o06_for_o07"] is False


def test_evaluator_requires_20_seen_episodes():
    records = build_synthetic_records(0, FINGERPRINT, missing_episode=True)
    try:
        evaluate_records(records, expected_asset_fingerprint=FINGERPRINT)
    except ValueError as error:
        assert "20" in str(error)
    else:
        raise AssertionError("missing episode accepted")


def test_evaluator_requires_80_swaps():
    records = build_synthetic_records(0, FINGERPRINT)
    records.pop(next(index for index, record in enumerate(records) if record["record_type"] == "swap"))
    try:
        evaluate_records(records, expected_asset_fingerprint=FINGERPRINT)
    except ValueError as error:
        assert "80" in str(error)
    else:
        raise AssertionError("missing swap accepted")


def test_evaluator_uses_outfit_macro_average():
    result = aggregate_evaluations(_evaluations())
    assert result["aggregation_order"] == ["episode", "outfit_macro", "seed", "seed_mean_std"]
    assert result["pixel_weighted_micro_average"] is False


def test_evaluator_rejects_missing_seed():
    try:
        aggregate_evaluations(_evaluations()[:2])
    except ValueError as error:
        assert "seeds" in str(error)
    else:
        raise AssertionError("missing seed accepted")


def test_evaluator_rejects_missing_episode():
    evaluations = _evaluations()
    evaluations[0] = copy.deepcopy(evaluations[0]); evaluations[0]["per_episode_metrics"].pop()
    try:
        aggregate_evaluations(evaluations)
    except ValueError as error:
        assert "20" in str(error)
    else:
        raise AssertionError("missing per-episode metric accepted")


def test_evaluator_never_selects_best_seed():
    result = aggregate_evaluations(_evaluations())
    assert result["best_seed_selected"] is False and result["seeds"] == [0, 1, 2]


def test_held_out_is_exported_separately():
    evaluation = _evaluations()[0]
    assert evaluation["held_out_diagnostic"]["included_in_seen_aggregation"] is False
    assert evaluation["held_out_diagnostic"]["status"] == "Held-out Diagnostic — FAIL"


def test_table_missing_metric_is_na():
    aggregate = aggregate_evaluations(_evaluations())
    with tempfile.TemporaryDirectory() as directory:
        outputs = export_tables(synthetic_table_source(aggregate), Path(directory))
        text = Path(outputs["table_1"][1]).read_text(encoding="utf-8")
        assert "N/A" in text


def test_figure_export_uses_fixed_outfit_order():
    assert FIGURE_LAYOUTS["figure_2"]["rows"] == list(SEEN_OUTFITS)


def test_figure_export_uses_fixed_view_order():
    assert FIGURE_LAYOUTS["figure_2"]["views"] == list(FIXED_VIEWS)


def test_no_cherry_pick_visual_selection():
    validate_figure_layouts(FIGURE_LAYOUTS)
    assert FIGURE_LAYOUTS["figure_2"]["columns"][-1] == "B5"


def test_synthetic_fixture_never_enters_registry():
    assert MARKER not in REGISTRY_PATH.read_text(encoding="utf-8")
    assert all(item.get("evidence_class") != MARKER for item in REGISTRY["experiments"])


def test_historical_outputs_immutable():
    historical = next(item for item in MANIFEST["assets"] if item["asset_id"] == "historical_formal_output_tree")
    assert historical["fingerprint"] == "108c8f2a7e327cd76a508c16e1797cd220d169cbff44091fc9cfefbd2911c74c"


def test_adapter_count_is_eight_and_a8_has_none():
    assert len(ADAPTER_TYPES) == 8
    historical = next(item for item in REGISTRY["experiments"] if not item["executable"])
    try:
        adapter_for(historical, CONFIG)
    except ValueError:
        pass
    else:
        raise AssertionError("A8 received executable adapter")


def test_output_attempt_is_append_only_and_complete():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first, second = create_attempt(root), create_attempt(root)
        assert first.name == "attempt_001" and second.name == "attempt_002"
        assert set(path.name for path in first.iterdir()) == set(ATTEMPT_DIRECTORIES)


def test_unified_cli_has_all_ten_commands():
    assert set(cli.COMMANDS) == {"validate", "plan", "run", "resume", "evaluate", "aggregate", "export-tables", "export-figures", "archive", "status"}


def test_evaluator_metric_count_matches_frozen_contract():
    assert len(ALL_METRICS) == 27
    assert _evaluations()[0]["metric_count"] == 27


def test_table_one_to_four_export_all_formats():
    aggregate = aggregate_evaluations(_evaluations())
    with tempfile.TemporaryDirectory() as directory:
        outputs = export_tables(synthetic_table_source(aggregate), Path(directory))
        assert set(outputs) == {"table_1", "table_2", "table_3", "table_4"}
        assert all(len(paths) == 4 and all(Path(path).is_file() for path in paths) for paths in outputs.values())


def test_figure_one_to_six_export_fixture():
    with tempfile.TemporaryDirectory() as directory:
        manifest = export_figure_layouts(Path(directory), synthetic=True)
        assert list(manifest["figures"]) == [f"figure_{index}" for index in range(1, 7)]
        assert all(Path(item["path"]).is_file() for item in manifest["figures"].values())


def test_asset_mismatch_fixture_is_rejected():
    try:
        evaluate_records(build_synthetic_records(0, FINGERPRINT, asset_mismatch=True), expected_asset_fingerprint=FINGERPRINT)
    except ValueError as error:
        assert str(error) == "PAPER_ASSET_MISMATCH"
    else:
        raise AssertionError("asset mismatch fixture accepted")


def test_reference_forward_boundary_is_recorded():
    result = _evaluations()[0]
    assert result["validation"]["target_forward_boundary"] is True
