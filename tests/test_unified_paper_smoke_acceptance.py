from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import yaml
from PIL import Image

from tools.paper.aggregation import aggregate_evaluations, aggregate_smoke_evaluation
from tools.paper.evaluate_seen_outfit import ALL_METRICS
from tools.paper.paper_exports import FIGURE_LAYOUTS, export_figure_layouts, export_tables
from tools.paper.smoke_contract import (
    EXPECTED_SMOKE_IDS,
    SMOKE_ATTEMPT_DIRECTORIES,
    SMOKE_STATES,
    create_smoke_attempt,
    git_blob,
    load_smoke_registry,
    smoke_output_root,
)
from tools.paper.smoke_runtime import run_smoke_experiment
from tools.paper.run_unified_paper_smoke_acceptance import _write_not_run_panel


ROOT = Path(__file__).resolve().parents[1]
SMOKE_REGISTRY_PATH = ROOT / "paper_protocol/smoke/smoke_experiment_registry.yaml"
FORMAL_REGISTRY_PATH = ROOT / "paper_protocol/experiment_registry.yaml"
CONFIG = yaml.safe_load((ROOT / "configs/paper/aaai27_seen_outfit_explicit_basis_v1.yaml").read_text(encoding="utf-8"))
SMOKE_REGISTRY = load_smoke_registry(SMOKE_REGISTRY_PATH)
FORMAL_BEFORE = hashlib.sha256(FORMAL_REGISTRY_PATH.read_bytes()).hexdigest()
_TEMPORARY = tempfile.TemporaryDirectory()
_RUN_ROOT = Path(_TEMPORARY.name) / "AAAI27-SEEN-OUTFIT-PAPER-SMOKE"
_CACHE: dict[str, tuple[dict, Path]] = {}


def _experiment(identifier: str) -> dict:
    item = next(dict(value) for value in SMOKE_REGISTRY["experiments"] if value["smoke_experiment_id"] == identifier)
    item["frozen_asset_fingerprint"] = SMOKE_REGISTRY["frozen_asset_fingerprint"]
    return item


def _run(identifier: str) -> tuple[dict, Path]:
    if identifier not in _CACHE:
        attempt = create_smoke_attempt(_RUN_ROOT / identifier / "seed_0")
        _CACHE[identifier] = (run_smoke_experiment(_experiment(identifier), CONFIG, attempt), attempt)
    return _CACHE[identifier]


def _evaluation(identifier: str) -> dict:
    result, _ = _run(identifier)
    return json.loads(Path(result["evaluated_metrics"]).read_text(encoding="utf-8"))


def test_smoke_registry_is_separate_from_paper_registry():
    assert SMOKE_REGISTRY_PATH != FORMAL_REGISTRY_PATH
    assert "smoke_only" not in FORMAL_REGISTRY_PATH.read_text(encoding="utf-8")


def test_smoke_entries_are_not_paper_final_eligible():
    assert len(SMOKE_REGISTRY["experiments"]) == 10
    assert all(item["smoke_only"] and not item["paper_final_eligible"] for item in SMOKE_REGISTRY["experiments"])


def test_smoke_output_root_is_isolated():
    root = smoke_output_root(Path(_TEMPORARY.name))
    assert root.name == "AAAI27-SEEN-OUTFIT-PAPER-SMOKE"
    assert root.name != "AAAI27-SEEN-OUTFIT-PAPER"


def test_b0_b1_b2_create_no_optimizer():
    for identifier in ("S-B0", "S-B1", "S-B2"):
        result, attempt = _run(identifier)
        assert result["optimizer_created"] is False
        assert not list((attempt / "checkpoints").iterdir())


def test_all_adapter_types_execute_real_forward():
    adapters = set()
    for identifier in EXPECTED_SMOKE_IDS:
        result, _ = _run(identifier)
        assert result["real_forward_executed"] is True
        adapters.add(result["adapter"])
    assert len(adapters) == 8


def test_trainable_adapter_optimizer_whitelist():
    for identifier in ("S-B3", "S-B4", "S-B5", "S-OURS", "S-A1", "S-A5", "S-A7"):
        result, _ = _run(identifier)
        assert set(result["optimizer_parameter_names"]) == {
            "layer_norm.weight", "layer_norm.bias", "linear.weight", "linear.bias",
        }
        assert result["optimizer_parameter_scope"] != "none"


def test_ours_resume_does_not_repeat_steps():
    result, _ = _run("S-OURS")
    resume = result["resume"]
    assert resume["resumed_completed_steps"] == list(range(1, 21))
    assert resume["resume_repeated_steps"] is False


def test_ours_resume_matches_uninterrupted_control():
    result, _ = _run("S-OURS")
    assert result["resume"]["status"] == "PASS"
    assert max(result["resume"]["differences"].values()) == 0.0


def test_evaluator_reads_raw_episode_outputs():
    result, _ = _run("S-B5")
    lines = Path(result["raw_metrics"]).read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["record_type"] == "episode" for line in lines)


def test_real_evaluator_requires_20_correct_cases():
    evaluation = _evaluation("S-OURS")
    assert evaluation["validation"]["seen_episode_count"] == 20


def test_real_evaluator_requires_80_swaps():
    evaluation = _evaluation("S-OURS")
    assert evaluation["validation"]["swap_count"] == 80


def test_smoke_metrics_are_finite_or_na():
    import math
    for identifier in EXPECTED_SMOKE_IDS:
        metrics = _evaluation(identifier)["metrics"]
        assert len(metrics) == 27
        assert all(value is None or math.isfinite(value) for value in metrics.values())


def test_formal_aggregate_rejects_missing_seeds():
    try:
        aggregate_evaluations([_evaluation("S-OURS")])
    except ValueError as error:
        assert "MISSING_REQUIRED_PAPER_SEEDS" in str(error)
    else:
        raise AssertionError("formal aggregate accepted one smoke seed")


def test_smoke_aggregate_has_no_fake_three_seed_stats():
    result = aggregate_smoke_evaluation(_evaluation("S-OURS"))
    assert result["aggregation_order"] == ["episode", "outfit_macro", "seed"]
    assert result["three_seed_statistics_generated"] is False
    assert "aggregate_metrics" not in result


def _table_source() -> dict:
    metrics = {key: None for key in ALL_METRICS}
    return {
        "marker": "SMOKE_ONLY",
        "methods": {key: metrics for key in ("B0", "B1", "B2", "B3", "B4", "B5", "Ours")},
        "ablations": {"S-A1": metrics, "S-A5": metrics, "S-A7": metrics},
        "held_out": {"O07": {"status": "FAIL"}},
    }


def test_smoke_tables_are_watermarked():
    with tempfile.TemporaryDirectory() as directory:
        outputs = export_tables(_table_source(), Path(directory))
        assert all("SMOKE ONLY" in Path(paths[1]).read_text(encoding="utf-8") for paths in outputs.values())


def _figure_source(directory: Path) -> dict:
    source = directory / "source.png"
    Image.new("RGB", (32, 32), (20, 100, 180)).save(source)
    return {
        "figures": {
            name: {"source_paths": [str(source)] * (len(layout["columns"]) * len(layout.get("rows", [])) * len(layout.get("views", [None])))}
            for name, layout in FIGURE_LAYOUTS.items()
        }
    }


def test_smoke_figures_are_watermarked():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = export_figure_layouts(root / "figures", synthetic=False, source_manifest=_figure_source(root), smoke_only=True)
        assert result["marker"] == "SMOKE_ONLY"
        assert all(Image.open(item["path"]).height >= 20 for item in result["figures"].values())


def test_smoke_outputs_never_enter_paper_ready():
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "paper_ready"
        try:
            export_tables(_table_source(), target)
        except ValueError:
            pass
        else:
            raise AssertionError("smoke table entered paper_ready")


def test_formal_registry_fingerprint_is_unchanged():
    assert hashlib.sha256(FORMAL_REGISTRY_PATH.read_bytes()).hexdigest() == FORMAL_BEFORE
    assert git_blob(ROOT, "paper_protocol/experiment_registry.yaml") == SMOKE_REGISTRY["formal_registry_git_blob"]


def test_no_smoke_entry_becomes_paper_final():
    assert "PAPER_FINAL" not in SMOKE_REGISTRY["states"]
    assert all(item["status"] == "NOT_RUN" for item in SMOKE_REGISTRY["experiments"])


def test_target_forward_boundary_in_real_run():
    for identifier in EXPECTED_SMOKE_IDS:
        result, _ = _run(identifier)
        assert result["target_forward_leakage"] is False
        assert _evaluation(identifier)["validation"]["target_forward_boundary"] is True


def test_base_backbone_mmlp_basis_renderer_frozen():
    for identifier in EXPECTED_SMOKE_IDS:
        result, _ = _run(identifier)
        assert result["frozen_gradient_count"] == 0
        assert result["frozen_max_change"] == 0.0
        assert result["frozen_before_sha256"] == result["frozen_after_sha256"]


def test_smoke_attempt_contract_directories_are_complete():
    _, attempt = _run("S-B0")
    assert set(path.name for path in attempt.iterdir() if path.is_dir()) == set(SMOKE_ATTEMPT_DIRECTORIES)


def test_trainable_smoke_has_step_zero_and_final_checkpoint():
    for identifier in ("S-B3", "S-B4", "S-B5", "S-A1", "S-A5", "S-A7"):
        _, attempt = _run(identifier)
        assert (attempt / "checkpoints/checkpoint_step_000000.pth").is_file()
        assert (attempt / "checkpoints/checkpoint_step_000005.pth").is_file()


def test_fixed_adapter_na_is_preserved():
    evaluation = _evaluation("S-B0")
    assert evaluation["metrics"]["standardized_coefficient_rmse"] is None
    assert "standardized_coefficient_rmse" in evaluation["validation"]["not_applicable_metrics"]


def test_smoke_state_machine_has_no_paper_final():
    assert SMOKE_STATES[-1] == "FAILED"
    assert "SMOKE_ACCEPTED" in SMOKE_STATES and "PAPER_FINAL" not in SMOKE_STATES


def test_unrun_smoke_figure_columns_use_explicit_placeholders():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "not_run.png"
        assert Path(_write_not_run_panel(path, "A1 K=1")).is_file()
        assert Image.open(path).getbbox() is not None
