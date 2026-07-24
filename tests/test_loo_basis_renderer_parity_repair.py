from __future__ import annotations

import ast
from pathlib import Path

from tools.paper import run_loo_basis_renderer_parity_repair as repair


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools/paper/run_loo_basis_renderer_parity_repair.py").read_text(
    encoding="utf-8"
)


def test_exact_source_and_repair_governance() -> None:
    assert repair.SOURCE_HEAD == "6ddd3a637b322a48e965bff8bb11a16213264c37"
    assert repair.EXECUTION_HEAD == "a868df3c3811483c5dd07456d8cabfdc3577d5f4"
    assert repair.RESULT_HEAD == "8436a6187da02ca3c21ef546f0940208a02fdac3"
    assert repair.REPAIR_BRANCH == (
        "research/loo-basis-renderer-parity-closure-repair-20260724"
    )


def test_diagnostic_output_is_separate_from_scientific_attempt() -> None:
    assert repair.OUTPUT_NAME == "LOO-BASIS-RENDERER-PARITY-REPAIR-001"
    assert repair.ORIGINAL_OUTPUT_NAME == "LOO-BASIS-ADAPTATION-001"
    assert repair.OUTPUT_NAME != repair.ORIGINAL_OUTPUT_NAME
    assert repair.ATTEMPT_NAME == "attempt_001"
    assert repair.DIAGNOSTIC_LABEL == "DIAGNOSTIC_ONLY_NOT_SCIENTIFIC_EVALUATION"


def test_original_failure_contract_is_frozen() -> None:
    assert repair.RENDER_GATE == 1e-5
    assert repair.EXPECTED_ORIGINAL_MAX_ALPHA == 0.003845691680908203
    assert repair.EXPECTED_ORIGINAL_BASIS_FINGERPRINT == (
        "a7a16acf8c0920c5d4bba80c76be87f343e708cba7a8836d4bba3fee176410b9"
    )
    assert repair.EXPECTED_ORIGINAL_BASIS_SHA256 == (
        "71144cafcdfd070942dc0a808b65216539d0320c3a801a1cde7fb0b75ecd40c6"
    )
    assert repair.EXPECTED_ORIGINAL_NORMALIZATION_SHA256 == (
        "91a4121a58fcf8b839e60284290022b9b2963e4b8a26c339f0dd5e7dab7f1d19"
    )


def test_channel_isolation_matrix_is_complete() -> None:
    assert set(repair.CHANNEL_VARIANTS) == {
        "xyz_only",
        "rotation_only",
        "scale_only",
        "opacity_only",
        "sh0_only",
        "shN_only",
        "geometry_combined",
        "appearance_combined",
        "all_channels",
    }
    assert len(repair.CHANNEL_VARIANTS["all_channels"]) == 6


def test_runner_contains_no_optimizer_or_scientific_attempt_materialization() -> None:
    tree = ast.parse(SOURCE)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "Adam" not in calls
    assert "AdamW" not in calls
    assert "backward" not in calls
    assert "save" not in calls
    assert 'ORIGINAL_OUTPUT_NAME / "attempt_002"' in SOURCE


def test_candidate_keeps_rank_three_and_frozen_renderer_gate() -> None:
    assert 'vh[:3]' in SOURCE
    assert '"rank": 3' in SOURCE
    assert "<= RENDER_GATE" in SOURCE
    assert "RENDER_GATE = 1e-5" in SOURCE


def test_dtype_and_solver_diagnostics_are_preregistered() -> None:
    for token in (
        "N0_float32",
        "N1_float64",
        "N2_float64_zero_sum",
        "orthonormal_projection",
        "lstsq",
        "pinv",
        "svd_coordinates",
    ):
        assert token in SOURCE


def test_all_five_splits_and_twenty_endpoints_are_declared() -> None:
    assert repair.OUTFITS == ("O01", "O02", "O03", "O04", "O08")
    assert "candidate_split_rows" in SOURCE
    assert "candidate_endpoint_pass_count" in SOURCE


def test_preflight_manifest_is_required_before_diagnostic_runtime() -> None:
    require_source = ast.get_source_segment(
        SOURCE,
        next(
            node
            for node in ast.walk(ast.parse(SOURCE))
            if isinstance(node, ast.FunctionDef) and node.name == "require_diagnostic_runtime"
        ),
    )
    assert require_source is not None
    assert "load_manifest(asset_root)" in require_source
    assert "original_immutability(asset_root, manifest)" in require_source


def test_manifest_records_both_frozen_tree_and_file_content_bytes() -> None:
    assert '"total_bytes": tree_apparent_bytes' in SOURCE
    assert '"file_content_bytes": file_content_bytes' in SOURCE
    assert '"aggregate_sha256": canonical_sha(aggregate_rows)' in SOURCE


def test_paper_final_and_execution_counts_remain_zero() -> None:
    assert '"optimizer_creations": 0' in SOURCE
    assert '"optimizer_steps": 0' in SOURCE
    assert '"checkpoint_writes": 0' in SOURCE
    assert '"formal_metrics": 0' in SOURCE
    assert '"PAPER_FINAL": False' in SOURCE
