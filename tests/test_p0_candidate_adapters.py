from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import torch

from scene.p0_candidate_adapters import (
    FIXED_VIEWS,
    SEEN_OUTFITS,
    B6ReferenceClassifierHardLookupAdapter,
    B7F2NearestCentroidHardLookupAdapter,
    OursV2CandidateAdapter,
    a5_legacy_endpoint_supervision_loss,
    build_b7_fold_adapter,
    build_candidate_optimizer,
    build_m3_m4_candidate_adapters,
    candidate_parameter_manifest,
)
from scene.p0_candidate_initialization_protocol import selected_state_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "42a28b386f6f32e23e5e16680408820efd8a65c7"


def _f2() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    return torch.rand(3, 256, generator=generator), torch.ones(3, 1)


def _rff() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(53)
    return torch.rand(3, 16, generator=generator), torch.ones(3, 1)


def _b7() -> tuple[B7F2NearestCentroidHardLookupAdapter, dict]:
    rows = {}
    valid = {}
    hashes = {}
    for index, outfit in enumerate(SEEN_OUTFITS):
        generator = torch.Generator().manual_seed(100 + index)
        rows[outfit] = torch.rand(3, 256, generator=generator) + index
        valid[outfit] = torch.ones(3, 1)
        hashes[outfit] = [hashlib.sha256(f"{outfit}/{view}".encode()).hexdigest() for view in FIXED_VIEWS[1:]]
    return build_b7_fold_adapter(
        target_condition=FIXED_VIEWS[0],
        fold_rows=rows,
        fold_validity=valid,
        reference_file_hashes=hashes,
    )


def test_ours_v2_adapter_uses_deterministic_zero_init() -> None:
    first, second = OursV2CandidateAdapter(), OursV2CandidateAdapter()
    assert candidate_parameter_manifest(first)["state_sha256"] == candidate_parameter_manifest(second)["state_sha256"]
    assert torch.count_nonzero(first.predictor.linear.weight) == 0
    assert torch.count_nonzero(first.predictor.linear.bias) == 0


def test_ours_v2_adapter_has_3076_parameters() -> None:
    assert candidate_parameter_manifest(OursV2CandidateAdapter())["parameter_count"] == 3076


def test_ours_v2_uses_smoothl1_only() -> None:
    adapter = OursV2CandidateAdapter()
    result = adapter.training_loss(torch.zeros(5, 4), torch.ones(5, 4))
    assert set(result) == {"total", "coefficient_smooth_l1"}
    assert adapter.loss_contract == "SMOOTHL1_STANDARDIZED_COEFFICIENT_ONLY"


def test_ours_v2_has_no_pairwise_geometry() -> None:
    adapter = OursV2CandidateAdapter()
    assert adapter.pairwise_geometry_weight == 0.0
    assert "pairwise" not in inspect.getsource(adapter.training_loss).lower()


def test_ours_v2_has_no_outfit_id() -> None:
    assert "outfit" not in inspect.signature(OursV2CandidateAdapter.forward).parameters


def test_ours_v2_step0_is_mean_garment() -> None:
    adapter = OursV2CandidateAdapter()
    f2, valid = _f2()
    output = adapter(f2, valid)
    assert torch.count_nonzero(output.standardized_coefficients) == 0
    assert adapter.initial_output_semantics == "MEAN-GARMENT INITIAL PREDICTION"


def test_ours_v2_does_not_load_a6_state() -> None:
    source = inspect.getsource(OursV2CandidateAdapter)
    assert "torch.load" not in source and "load_state_dict" not in source
    assert OursV2CandidateAdapter.loads_a6_checkpoint is False
    assert OursV2CandidateAdapter.copies_a6_state_dict is False


def test_b6_uses_reference_only() -> None:
    parameters = inspect.signature(B6ReferenceClassifierHardLookupAdapter.forward).parameters
    assert set(parameters) == {"self", "reference_f2", "reference_valid"}


def test_b6_label_is_loss_only() -> None:
    assert "outfit_label" in inspect.signature(B6ReferenceClassifierHardLookupAdapter.training_loss).parameters
    assert "outfit_label" not in inspect.signature(B6ReferenceClassifierHardLookupAdapter.forward).parameters


def test_b6_lookup_uses_predicted_class() -> None:
    adapter = B6ReferenceClassifierHardLookupAdapter(seed=0)
    f2, valid = _f2()
    prediction = adapter(f2, valid)
    bank = {outfit: f"residual:{outfit}" for outfit in SEEN_OUTFITS}
    assert adapter.lookup_predicted_teacher(prediction, bank) == f"residual:{prediction.predicted_outfit}"


def test_b6_does_not_use_ground_truth_outfit_id() -> None:
    forward_source = inspect.getsource(B6ReferenceClassifierHardLookupAdapter.forward)
    lookup_source = inspect.getsource(B6ReferenceClassifierHardLookupAdapter.lookup_predicted_teacher)
    assert "ground_truth" not in forward_source + lookup_source
    assert "outfit_label" not in forward_source + lookup_source


def test_b6_tie_and_invalid_class_handling() -> None:
    adapter = B6ReferenceClassifierHardLookupAdapter(seed=0)
    torch.nn.init.zeros_(adapter.linear.weight)
    torch.nn.init.zeros_(adapter.linear.bias)
    f2, valid = _f2()
    assert adapter(f2, valid).predicted_outfit == SEEN_OUTFITS[0]
    for invalid in (-1, 5):
        try:
            adapter.class_index_to_outfit(invalid)
        except IndexError:
            pass
        else:
            raise AssertionError("invalid B6 class was accepted")


def test_b7_has_no_optimizer() -> None:
    adapter, _ = _b7()
    optimizer, report = build_candidate_optimizer(adapter)
    assert optimizer is None and report["created"] is False
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 0


def test_b7_excludes_target_view_from_centroid() -> None:
    adapter, manifest = _b7()
    assert adapter.target_condition not in adapter.reference_condition_ids
    assert manifest["target_condition_excluded"] is True
    assert len(manifest["reference_condition_ids"]) == 3


def test_b7_saves_centroid_manifest() -> None:
    _, manifest = _b7()
    assert set(manifest["outfits"]) == set(SEEN_OUTFITS)
    assert all(len(row["reference_file_hashes"]) == 3 for row in manifest["outfits"].values())
    assert all(len(row["centroid_sha256"]) == 64 for row in manifest["outfits"].values())


def test_b7_lookup_uses_nearest_centroid() -> None:
    adapter, _ = _b7()
    f2, valid = _f2()
    prediction = adapter(f2, valid)
    expected = min(SEEN_OUTFITS, key=lambda outfit: (prediction.squared_distances[outfit], SEEN_OUTFITS.index(outfit)))
    bank = {outfit: f"residual:{outfit}" for outfit in SEEN_OUTFITS}
    assert prediction.predicted_outfit == expected
    assert adapter.lookup_nearest_teacher(prediction, bank) == bank[expected]


def test_m3_m4_share_same_seed_trunk() -> None:
    for seed in (0, 1, 2):
        m3, m4 = build_m3_m4_candidate_adapters(raw_dim=16, seed=seed)
        assert selected_state_sha256(m3.candidate, "trunk") == selected_state_sha256(m4.candidate, "trunk")


def test_m3_m4_heads_are_zero_initialized() -> None:
    m3, m4 = build_m3_m4_candidate_adapters(raw_dim=16, seed=0)
    for adapter in (m3, m4):
        assert torch.count_nonzero(adapter.output_head.weight) == 0
        assert torch.count_nonzero(adapter.output_head.bias) == 0


def test_m3_uses_smoothl1_only() -> None:
    m3, _ = build_m3_m4_candidate_adapters(raw_dim=16, seed=0)
    result = m3.training_loss(torch.zeros(5, 4), torch.ones(5, 4))
    assert set(result) == {"total", "coefficient_smooth_l1"}


def test_m4_matches_a5_supervision_contract() -> None:
    _, m4 = build_m3_m4_candidate_adapters(raw_dim=16, seed=0)
    prediction = torch.linspace(-0.5, 0.5, 20).reshape(5, 4)
    target = torch.linspace(0.75, -0.75, 20).reshape(5, 4)
    expected = a5_legacy_endpoint_supervision_loss(prediction, target)
    actual = m4.training_loss(prediction, target)
    assert set(actual) == {"total", "coefficient_loss", "sign_loss", "absolute_pair_loss"}
    assert all(torch.equal(actual[name], expected[name]) for name in actual)


def test_b5_remains_off_matrix_historical() -> None:
    audit = json.loads((ROOT / "paper_protocol/reviewer_risk/b5_contract_audit.json").read_text(encoding="utf-8"))
    assert audit["actual_b5_contract"]["actual_matrix_cell"] == "OFF_MATRIX_COMPLEX_PLUS_PAIRWISE_GEOMETRY"
    historical = subprocess.check_output(["git", "show", f"{SOURCE_HEAD}:tools/paper/formal_batch_runtime.py"], cwd=ROOT)
    current = (ROOT / "tools/paper/formal_batch_runtime.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(current).hexdigest() == hashlib.sha256(historical).hexdigest()


def test_candidate_parameters_are_not_frozen() -> None:
    candidates = [
        OursV2CandidateAdapter(),
        B6ReferenceClassifierHardLookupAdapter(seed=0),
        *build_m3_m4_candidate_adapters(raw_dim=16, seed=0),
    ]
    assert all(all(parameter.requires_grad for parameter in model.parameters()) for model in candidates)


def test_frozen_parameters_are_not_in_candidate_optimizer() -> None:
    adapter = OursV2CandidateAdapter()
    frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    optimizer, report = build_candidate_optimizer(adapter)
    assert optimizer is not None and report["created"] is True
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(frozen) not in optimizer_ids


def test_target_appearance_is_not_in_prediction_forward() -> None:
    for method in (
        OursV2CandidateAdapter.forward,
        B6ReferenceClassifierHardLookupAdapter.forward,
        B7F2NearestCentroidHardLookupAdapter.forward,
    ):
        names = set(inspect.signature(method).parameters)
        assert not {"target_rgb", "target_mask"}.intersection(names)


def test_target_pose_camera_are_not_in_coefficient_branch() -> None:
    m3, _ = build_m3_m4_candidate_adapters(raw_dim=16, seed=0)
    names = set(inspect.signature(m3.forward).parameters)
    assert "target_pose" not in names and "target_camera" not in names
