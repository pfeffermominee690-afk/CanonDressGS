from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
import yaml

from scene.p0_candidate_adapters import (
    B6ReferenceClassifierHardLookupAdapter,
    OursV2CandidateAdapter,
    build_m3_m4_candidate_adapters,
)
from scene.p0_candidate_initialization_protocol import selected_state_sha256
from tools.paper.formal_batch_runtime import _training_loss
from tools.paper.p0_formal_candidate_runtime import (
    MILESTONES,
    candidate_code_aggregate_sha256,
    forward_training_batch,
    load_registry,
    model_factory,
    nested_sha256,
    optimizer_factory,
    train_candidate,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "paper_protocol/reviewer_risk/p0_formal_candidate_run_registry.yaml"
TRAINING_AUDIT = ROOT / "paper_protocol/reviewer_risk/p0_formal_training_contract_audit.json"
TRUNK_AUDIT = ROOT / "paper_protocol/reviewer_risk/m3_m4_complex_trunk_provenance.json"
M4_AUDIT = ROOT / "paper_protocol/reviewer_risk/m4_a5_supervision_contract_parity.json"


def _cache(feature_dim: int = 4) -> dict:
    episodes = {}
    outfits = ("O01", "O02", "O03", "O04", "O08")
    conditions = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
    for outfit_index, outfit in enumerate(outfits):
        for condition_index, condition in enumerate(conditions):
            base = torch.arange(3 * feature_dim, dtype=torch.float32).reshape(3, feature_dim)
            base = base + outfit_index * 0.25 + condition_index * 0.01
            episodes[f"{outfit}/{condition}"] = {
                "normal": {
                    "f2": base,
                    "rff": torch.cat((base, base[:, :1]), dim=1),
                    "valid": torch.ones(3, 1),
                },
                "zero": {
                    "f2": torch.zeros_like(base),
                    "rff": torch.zeros(3, feature_dim + 1),
                    "valid": torch.ones(3, 1),
                },
                "base": {
                    "f2": torch.ones_like(base),
                    "rff": torch.ones(3, feature_dim + 1),
                    "valid": torch.ones(3, 1),
                },
            }
    return {"episodes": episodes}


def _targets(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        outfit: torch.tensor([index - 2.0, index - 1.0, index + 0.5, index + 1.5], device=device)
        for index, outfit in enumerate(("O01", "O02", "O03", "O04", "O08"))
    }


def test_formal_registry_has_13_authorized_runs() -> None:
    payload = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    validate_registry(payload)
    assert len(payload["runs"]) == 13
    assert all(row["status"] == "AUTHORIZED_NOT_RUN" for row in payload["runs"])


def test_formal_registry_has_3600_planned_steps() -> None:
    payload = load_registry()
    assert sum(row["expected_steps"] for row in payload["runs"]) == 3600
    assert sum(row["expected_steps"] == 0 for row in payload["runs"]) == 1


def test_formal_registry_never_contains_paper_final() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    assert "status: PAPER_FINAL" not in text
    assert yaml.safe_load(text)["paper_final"] is False


def test_optimizer_contract_audit_is_unambiguous() -> None:
    value = json.loads(TRAINING_AUDIT.read_text(encoding="utf-8"))
    assert value["status"] == "PASS"
    assert value["classification"] == "OPTIMIZER_CONTRACT_UNAMBIGUOUS"
    assert value["method_specific_optimizer_difference_found"] is False


def test_optimizer_contract_fields_are_exact() -> None:
    value = json.loads(TRAINING_AUDIT.read_text(encoding="utf-8"))["shared_optimizer_contract"]
    assert value["class"] == "torch.optim.Adam"
    assert value["learning_rate"] == 0.02
    assert value["weight_decay"] == 0.0
    assert value["betas"] == [0.9, 0.999]
    assert value["epsilon"] == 1e-8
    assert value["training_steps"] == 300
    assert value["gradient_clip"]["max_norm"] == 5.0
    assert value["batch_size"] == 5


def test_optimizer_factory_matches_frozen_contract() -> None:
    model = OursV2CandidateAdapter(input_dim=8)
    optimizer, scheduler = optimizer_factory(model)
    group = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.Adam)
    assert group["lr"] == 0.02 and group["weight_decay"] == 0.0
    assert group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
    assert scheduler.get_last_lr() == [0.02]


def test_optimizer_membership_is_all_and_only_candidate_parameters() -> None:
    model = B6ReferenceClassifierHardLookupAdapter(seed=0, input_dim=8)
    optimizer, _ = optimizer_factory(model)
    expected = {id(value) for value in model.parameters() if value.requires_grad}
    actual = {id(value) for group in optimizer.param_groups for value in group["params"]}
    assert actual == expected


def test_complex_trunk_provenance_is_preregistered() -> None:
    value = json.loads(TRUNK_AUDIT.read_text(encoding="utf-8"))
    assert value["status"] == "PASS"
    assert value["classification"] == "PREREGISTERED_P0_COMPLEX_TRUNK"
    assert value["post_hoc_simplification"] is False


def test_complex_parameter_breakdown_is_exact() -> None:
    value = json.loads(TRUNK_AUDIT.read_text(encoding="utf-8"))["parameter_breakdown"]
    assert value["shared_trunk_total"] == 234255
    assert value["zero_initialized_vector_head"] == 516
    assert value["candidate_total"] == 234771


def test_m3_m4_same_seed_trunks_are_bitwise_equal() -> None:
    m3, m4 = build_m3_m4_candidate_adapters(raw_dim=519, seed=2)
    assert selected_state_sha256(m3.candidate, "trunk") == selected_state_sha256(m4.candidate, "trunk")


def test_m3_m4_cross_seed_trunks_are_unique() -> None:
    hashes = []
    for seed in (0, 1, 2):
        m3, _ = build_m3_m4_candidate_adapters(raw_dim=519, seed=seed)
        hashes.append(selected_state_sha256(m3.candidate, "trunk"))
    assert len(set(hashes)) == 3


def test_m4_a5_numeric_parity() -> None:
    m3, m4 = build_m3_m4_candidate_adapters(raw_dim=5, seed=0)
    prediction = torch.randn(5, 4)
    target = torch.randn(5, 4)
    actual = m4.training_loss(prediction, target)
    expected_total, expected_parts = _training_loss("A5_Legacy_Endpoint_Supervision", prediction, target)
    assert torch.equal(actual["total"], expected_total)
    for name, value in expected_parts.items():
        assert torch.equal(actual[name], value)
    assert json.loads(M4_AUDIT.read_text(encoding="utf-8"))["status"] == "PASS"


def test_b6_label_is_only_a_loss_argument() -> None:
    signature = inspect.signature(B6ReferenceClassifierHardLookupAdapter.forward)
    assert set(signature.parameters) == {"self", "reference_f2", "reference_valid"}
    assert "outfit_label" in inspect.signature(B6ReferenceClassifierHardLookupAdapter.training_loss).parameters


def test_training_batch_has_exact_five_outfits() -> None:
    device = torch.device("cpu")
    cache = _cache()
    model, _ = model_factory("Ours-v2", 0, cache, device)
    output, losses, _ = forward_training_batch(
        model, "Ours-v2", cache, "cond_000000", _targets(device), device
    )
    assert output.shape == (5, 4)
    assert losses["total"].ndim == 0


def test_nested_optimizer_hash_is_stable() -> None:
    model = OursV2CandidateAdapter(input_dim=8)
    optimizer, _ = optimizer_factory(model)
    assert nested_sha256(optimizer.state_dict()) == nested_sha256(optimizer.state_dict())


def test_exact_resume_does_not_repeat_optimizer_steps(tmp_path: Path) -> None:
    device = torch.device("cpu")
    cache = _cache()
    run = {
        "formal_run_id": "TEST-OURS-R0", "method": "Ours-v2",
        "seed": 0, "replicate_index": 0,
    }
    resumed_attempt = tmp_path / "resumed"
    resumed_attempt.mkdir()
    _, interrupted = train_candidate(
        attempt=resumed_attempt, run=run, cache=cache, targets=_targets(device),
        device=device, stop_after_step=5,
    )
    assert interrupted["status"] == "INTERRUPTED_RESUMABLE"
    resumed_model, resumed = train_candidate(
        attempt=resumed_attempt, run=run, cache=cache, targets=_targets(device),
        device=device,
    )
    control_attempt = tmp_path / "control"
    control_attempt.mkdir()
    control_model, control = train_candidate(
        attempt=control_attempt, run=run, cache=cache, targets=_targets(device),
        device=device,
    )
    assert resumed["optimizer_steps"] == control["optimizer_steps"] == 300
    assert resumed["optimizer_steps_repeated"] == 0
    assert [row["step"] for row in __import__("tools.paper.p0_formal_candidate_runtime", fromlist=["read_jsonl"]).read_jsonl(resumed_attempt / "logs/train.jsonl")] == list(range(1, 301))
    assert all(torch.equal(resumed_model.state_dict()[name], value) for name, value in control_model.state_dict().items())


def test_training_milestones_are_exact(tmp_path: Path) -> None:
    device = torch.device("cpu")
    cache = _cache()
    run = {"formal_run_id": "TEST-OURS-R0", "method": "Ours-v2", "seed": 0, "replicate_index": 0}
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    train_candidate(attempt=attempt, run=run, cache=cache, targets=_targets(device), device=device)
    assert {
        int(path.stem.rsplit("_", 1)[1])
        for path in (attempt / "metrics/milestones").glob("step_*.json")
    } == set(MILESTONES)


def test_candidate_code_fingerprint_is_frozen() -> None:
    assert candidate_code_aggregate_sha256() == "92eaf05af2f8661ab29354d671b63f467c0be111825a324422a0e81952c57da8"


def test_evaluator_contract_is_20_80_and_27() -> None:
    payload = load_registry()["global_contract"]
    assert payload["correct_episode_count"] == 20
    assert payload["swap_tuple_count"] == 80
    assert payload["evaluator_metric_count"] == 27


def test_no_best_seed_selection_is_encoded() -> None:
    assert load_registry()["global_contract"]["best_seed_selection"] is False


def test_target_inputs_are_absent_from_candidate_forward_signatures() -> None:
    forbidden = {"target_rgb", "target_mask", "target_pose", "target_camera", "outfit_id"}
    for function in (
        OursV2CandidateAdapter.forward,
        B6ReferenceClassifierHardLookupAdapter.forward,
    ):
        assert not forbidden.intersection(inspect.signature(function).parameters)


def test_frozen_context_precedes_candidate_determinism_guard() -> None:
    source = (ROOT / "tools/paper/run_p0_formal_candidate_runs.py").read_text(encoding="utf-8")
    context_position = source.index("context = prepare_context(canary)")
    guard_position = source.index("torch.use_deterministic_algorithms(True, warn_only=True)")
    assert context_position < guard_position
