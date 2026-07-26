from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
from scene.reference_conditioned_dual_support_controller import (
    FORBIDDEN_FORWARD_INPUTS,
    OUTFIT_ORDER,
    ReferenceConditionedDualSupportController,
    construct_dual_support_runtime,
    soft_target_cross_entropy,
    stable_top2_selection,
)
from tools.paper.build_dual_support_controller_dataset import (
    CONDITIONS,
    build_training_manifest,
    target_distribution,
)


ARCHIVE_HASHES = {
    "paper_protocol/reviewer_risk/dual_support_all_pair_protocol.yaml": "34f7e7cd45f3a9d5cb49323f9981936104e11e2b23b1a9f88461524d9b0a8d2e",
    "paper_protocol/reviewer_risk/dual_support_all_pair_results.json": "294e5dd0618cbb2cd7b497ae0e8b1a789b776e0db2b215fb42729929469d3b14",
    "paper_protocol/reviewer_risk/dual_support_all_pair_visual_review.json": "ef1544012754e633c9e24eba7ee60dfbe9e95f36c7a405f835ebd9cf0a9f42fa",
    "paper_protocol/reviewer_risk/dual_support_all_pair_final_summary.json": "3c6aa16a1f4d5324aeaaf426ffef83ba3a008d95308e21f09963a3c50525c5b2",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.fixture()
def training_manifest(tmp_path: Path) -> dict:
    source = {"outfits": []}
    for outfit_index, outfit in enumerate(OUTFIT_ORDER):
        observations = []
        for view_index, condition in enumerate(CONDITIONS):
            token = hashlib.sha256(f"{outfit}/{condition}".encode()).hexdigest()
            observations.append({
                "condition_id": condition,
                "target_edit_rgb": f"/frozen/rgb/{outfit}/{condition}.png",
                "target_clothing_mask": f"/frozen/mask/{outfit}/{condition}.png",
                "checksums": {
                    "target_edit_rgb": token,
                    "target_clothing_mask": hashlib.sha256((token + "/mask").encode()).hexdigest(),
                },
            })
        source["outfits"].append({"outfit_id": outfit, "observations": observations})
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    return build_training_manifest(source, source_path)


def _controller(seed: int = 0) -> ReferenceConditionedDualSupportController:
    return ReferenceConditionedDualSupportController(seed=seed, input_dim=8)


def _references() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11.0, torch.ones(3, 1)


def _endpoint_bank() -> dict[str, str]:
    return {outfit: f"/frozen/{outfit}.pth" for outfit in OUTFIT_ORDER}


def test_controller_uses_reference_only() -> None:
    parameters = list(inspect.signature(ReferenceConditionedDualSupportController.forward).parameters)
    assert parameters == ["self", "reference_f2", "reference_valid"]
    assert not set(parameters).intersection(FORBIDDEN_FORWARD_INPUTS)


def test_soft_targets_match_reference_composition() -> None:
    assert target_distribution(("O01", "O01", "O01")) == [1.0, 0.0, 0.0, 0.0, 0.0]
    logits = torch.zeros(5)
    target = torch.tensor(target_distribution(("O01", "O01", "O02")))
    assert torch.allclose(soft_target_cross_entropy(logits, target), torch.log(torch.tensor(5.0)))


def test_aab_targets_are_two_thirds_one_third(training_manifest: dict) -> None:
    rows = [row for row in training_manifest["query_sets"] if row["assignment_type"] == "AAB"]
    assert len(rows) == 120
    assert all(sorted(value for value in row["target_distribution"] if value) == pytest.approx([1 / 3, 2 / 3]) for row in rows)


def test_abb_targets_are_one_third_two_thirds(training_manifest: dict) -> None:
    rows = [row for row in training_manifest["query_sets"] if row["assignment_type"] == "ABB"]
    assert len(rows) == 120
    assert all(sorted(value for value in row["target_distribution"] if value) == pytest.approx([1 / 3, 2 / 3]) for row in rows)


def test_all_assignment_positions_are_present(training_manifest: dict) -> None:
    assert training_manifest["all_assignment_positions_present"] is True
    assert training_manifest["assignment_position_counts"] == {
        "AAB": {"0": 40, "1": 40, "2": 40},
        "ABB": {"0": 40, "1": 40, "2": 40},
    }


def test_controller_has_five_logits() -> None:
    output = _controller()(*_references())
    assert output.logits.shape == (5,)
    assert output.probabilities.shape == (5,)
    assert _controller().parameter_count == 61


def test_top2_selection_is_stable() -> None:
    selection = stable_top2_selection(torch.tensor([0.45, 0.45, 0.10, 0.0, 0.0]))
    assert (selection.top1_outfit, selection.top2_outfit) == ("O01", "O02")
    assert selection.stable_rank_indices == (0, 1, 2, 3, 4)


def test_top2_weights_are_normalized() -> None:
    selection = stable_top2_selection(torch.tensor([0.50, 0.40, 0.10, 0.0, 0.0]))
    assert selection.mode == "DUAL_SUPPORT"
    assert selection.weight_1 + selection.weight_2 == pytest.approx(1.0)
    runtime = construct_dual_support_runtime(selection, _endpoint_bank())
    assert sum(branch.opacity_weight for branch in runtime.branches) == pytest.approx(1.0)


def test_single_endpoint_fallback_w2_threshold() -> None:
    selection = stable_top2_selection(torch.tensor([0.91, 0.08, 0.005, 0.003, 0.002]))
    assert selection.mode == "SINGLE_ENDPOINT"
    assert selection.fallback_reason == "LOW_SECONDARY_WEIGHT"
    assert len(construct_dual_support_runtime(selection, _endpoint_bank()).branches) == 1


def test_single_endpoint_fallback_top2_mass() -> None:
    selection = stable_top2_selection(torch.tensor([0.40, 0.30, 0.10, 0.10, 0.10]))
    assert selection.mode == "SINGLE_ENDPOINT"
    assert selection.fallback_reason == "LOW_TOP2_MASS"
    assert len(construct_dual_support_runtime(selection, _endpoint_bank()).branches) == 1


def test_dual_support_never_interpolates_geometry() -> None:
    selection = stable_top2_selection(torch.tensor([0.50, 0.40, 0.10, 0.0, 0.0]))
    runtime = construct_dual_support_runtime(selection, _endpoint_bank())
    assert len(runtime.branches) == 2
    assert runtime.geometry_interpolation is False
    assert runtime.geometry_averaging is False
    assert runtime.basis_coefficient_interpolation is False
    assert all(branch.geometry_policy == "IMMUTABLE_ENDPOINT_NO_INTERPOLATION" for branch in runtime.branches)


def test_ground_truth_outfit_id_is_not_in_forward() -> None:
    parameters = inspect.signature(ReferenceConditionedDualSupportController.forward).parameters
    assert "outfit_id" not in parameters
    assert "garment_id" not in parameters


def test_target_pose_camera_are_not_in_controller() -> None:
    parameters = inspect.signature(ReferenceConditionedDualSupportController.forward).parameters
    assert "target_pose" not in parameters
    assert "target_camera" not in parameters


def _fresh_process_hash(seed: int) -> str:
    script = (
        "import json, torch; "
        "from scene.reference_conditioned_dual_support_controller import ReferenceConditionedDualSupportController; "
        "from scene.p0_candidate_initialization_protocol import tensor_mapping_sha256; "
        f"m=ReferenceConditionedDualSupportController(seed={seed},input_dim=8); "
        "x=torch.arange(12,dtype=torch.float32).reshape(3,4)/11; v=torch.ones(3,1); o=m(x,v); "
        "print(json.dumps({'state':tensor_mapping_sha256(m.state_dict()),'logits':o.logits.detach().tolist(),'p':o.probabilities.detach().tolist()}))"
    )
    return subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True).strip()


def test_same_seed_fresh_process_is_exact() -> None:
    assert _fresh_process_hash(0) == _fresh_process_hash(0)


def test_cross_seed_initializations_are_unique() -> None:
    payloads = [json.loads(_fresh_process_hash(seed)) for seed in (0, 1, 2)]
    assert len({row["state"] for row in payloads}) == 3


def test_no_backward_or_optimizer_step() -> None:
    source = (ROOT / "tools/paper/run_reference_conditioned_dual_support_controller_dry_run.py").read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert ".zero_grad(" not in source
    assert "optimizer.step(" not in source
    assert "scheduler.step(" not in source
    assert "torch.save(" not in source


def test_formal_outputs_are_immutable() -> None:
    summary = json.loads((ROOT / "paper_protocol/reviewer_risk/dual_support_all_pair_final_summary.json").read_text(encoding="utf-8"))
    assert summary["frozen_trees_before"]["formal"] == summary["frozen_trees_after"]["formal"]
    assert summary["frozen_trees_after"]["formal"]["file_count"] == 4127
    assert summary["counts"]["formal_output_mutation"] == 0


def test_dual_support_archive_is_immutable() -> None:
    assert {_path: _sha(ROOT / _path) for _path in ARCHIVE_HASHES} == ARCHIVE_HASHES


def test_paper_final_remains_zero() -> None:
    summary = json.loads((ROOT / "paper_protocol/reviewer_risk/dual_support_all_pair_final_summary.json").read_text(encoding="utf-8"))
    assert summary["paper_final_count"] == 0
    assert summary["paper_final"] is False
    protocol = (ROOT / "paper_protocol/reviewer_risk/dual_support_controller_protocol.yaml").read_text(encoding="utf-8")
    assert "paper_final: false" in protocol
