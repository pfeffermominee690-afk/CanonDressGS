from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from scene.p0_candidate_initialization_protocol import (
    DETERMINISTIC_ZERO_INITIALIZATION,
    PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD,
    RANDOM_SEEDED_INITIALIZATION,
    build_b6_candidate,
    build_m3_m4_paired_candidates,
    build_ours_v2_candidate,
    selected_state_sha256,
    tensor_mapping_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "71dd86c1c01e00e44e8135a45ce29db788141fe0"
FRESH_PROCESS = ROOT / "tools/paper/p0_initialization_fresh_process.py"
HISTORICAL_LINEAR = ROOT / "scene/multi_outfit_linear_coefficient_control.py"
FORMAL_RUNTIME = ROOT / "tools/paper/formal_batch_runtime.py"
AUDIT_RUNTIME = ROOT / "tools/paper/run_deterministic_initialization_protocol_audit.py"
REVIEWER_RISK = ROOT / "paper_protocol/reviewer_risk"
INITIALIZATION_AUDIT = REVIEWER_RISK / "initialization_policy_aware_audit.json"
SEMANTICS_AUDIT = REVIEWER_RISK / "mean_garment_zero_residual_semantics_audit.json"
HISTORICAL_AUDIT = REVIEWER_RISK / "historical_three_seed_interpretation_audit.json"
OPTIMIZER_AUDIT = REVIEWER_RISK / "optimizer_provenance_audit.json"
NO_TRAINING_AUDIT = REVIEWER_RISK / "no_training_gate.json"
OLD_SEED_AUDIT = REVIEWER_RISK / "seed_propagation_audit.json"
REVIEWER_REGISTRY = REVIEWER_RISK / "reviewer_risk_experiment_registry.yaml"


def _state_sha(module: torch.nn.Module) -> str:
    return tensor_mapping_sha256(module.state_dict())


def _synthetic_cache(path: Path) -> None:
    generator = torch.Generator().manual_seed(91)
    episodes = {}
    for outfit_index, outfit in enumerate(("O01", "O02", "O03", "O04", "O08")):
        episodes[f"{outfit}/cond_000000"] = {
            "normal": {
                "f2": torch.rand(3, 256, generator=generator) + outfit_index,
                "rff": torch.rand(3, 16, generator=generator) + outfit_index,
                "valid": torch.ones(3, 1),
            }
        }
    torch.save({"episodes": episodes}, path)


def _fresh(family: str, seed: int) -> dict:
    with tempfile.TemporaryDirectory() as value:
        cache = Path(value) / "feature_rows.pt"
        _synthetic_cache(cache)
        output = subprocess.check_output(
            [
                sys.executable,
                str(FRESH_PROCESS),
                "--family", family,
                "--seed", str(seed),
                "--feature-cache", str(cache),
                "--device", "cpu",
            ],
            cwd=ROOT,
            text=True,
        )
    return json.loads(output)


def test_ours_v2_uses_deterministic_zero_initialization() -> None:
    build = build_ours_v2_candidate(input_dim=512, seed=0)
    module = build.module
    assert build.initialization_policy == DETERMINISTIC_ZERO_INITIALIZATION
    assert torch.equal(module.normalization.weight, torch.ones_like(module.normalization.weight))
    assert torch.equal(module.normalization.bias, torch.zeros_like(module.normalization.bias))
    assert torch.equal(module.linear.weight, torch.zeros_like(module.linear.weight))
    assert torch.equal(module.linear.bias, torch.zeros_like(module.linear.bias))
    assert build.manifest()["trainable_parameter_count"] == 3076


def test_ours_v2_cross_seed_identity_is_expected() -> None:
    hashes = {_state_sha(build_ours_v2_candidate(seed=seed).module) for seed in (0, 1, 2)}
    assert len(hashes) == 1


def test_ours_v2_same_seed_fresh_process_is_bitwise_exact() -> None:
    first, second = _fresh("ours_v2", 2), _fresh("ours_v2", 2)
    assert first["pid"] != second["pid"]
    assert first["methods"]["Ours-v2"]["trainable_state_sha256"] == second["methods"]["Ours-v2"]["trainable_state_sha256"]
    assert first["methods"]["Ours-v2"]["step0_forward_sha256"] == second["methods"]["Ours-v2"]["step0_forward_sha256"]


def test_ours_v2_cross_seed_fresh_process_is_bitwise_exact() -> None:
    rows = [_fresh("ours_v2", seed) for seed in (0, 1, 2)]
    assert len({row["pid"] for row in rows}) == 3
    assert len({row["methods"]["Ours-v2"]["trainable_state_sha256"] for row in rows}) == 1
    assert len({row["methods"]["Ours-v2"]["step0_forward_sha256"] for row in rows}) == 1


def test_historical_linear_builder_remains_unchanged() -> None:
    historical = subprocess.check_output(
        ["git", "show", f"{SOURCE_HEAD}:scene/multi_outfit_linear_coefficient_control.py"],
        cwd=ROOT,
    )
    assert hashlib.sha256(historical).hexdigest() == "87347845019426bb44dd0044b9b52c1a264c6edbafcdeaea8d864de79ca488d2"
    assert hashlib.sha256(HISTORICAL_LINEAR.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == hashlib.sha256(historical).hexdigest()


def test_a6_builder_contract_is_not_modified() -> None:
    historical = subprocess.check_output(
        ["git", "show", f"{SOURCE_HEAD}:tools/paper/formal_batch_runtime.py"], cwd=ROOT
    )
    current = FORMAL_RUNTIME.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(current).hexdigest() == hashlib.sha256(historical).hexdigest()
    source = current.decode("utf-8")
    assert 'weight = 0.0 if method.startswith("A6_") else 0.10' in source
    assert "MultiOutfitLinearCoefficientControl(input_dim, rank)" in source


def test_b6_cross_seed_initialization_is_unique() -> None:
    builds = [build_b6_candidate(seed=seed) for seed in (0, 1, 2)]
    assert all(build.initialization_policy == RANDOM_SEEDED_INITIALIZATION for build in builds)
    assert len({_state_sha(build.module) for build in builds}) == 3


def test_m3_m4_share_same_seed_trunk_initialization() -> None:
    for seed in (0, 1, 2):
        m3, m4 = build_m3_m4_paired_candidates(raw_dim=16, seed=seed)
        assert m3.initialization_policy == m4.initialization_policy == PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD
        assert selected_state_sha256(m3.module, "trunk") == selected_state_sha256(m4.module, "trunk")
    assert len({selected_state_sha256(build_m3_m4_paired_candidates(raw_dim=16, seed=seed)[0].module, "trunk") for seed in (0, 1, 2)}) == 3


def test_m3_m4_output_heads_are_zero_initialized() -> None:
    m3, m4 = build_m3_m4_paired_candidates(raw_dim=16, seed=1)
    for build in (m3, m4):
        assert torch.count_nonzero(build.module.output_head.weight) == 0
        assert torch.count_nonzero(build.module.output_head.bias) == 0


def test_m1_m3_initial_output_semantics_match() -> None:
    m1 = build_ours_v2_candidate(input_dim=32, seed=0).module
    m3 = build_m3_m4_paired_candidates(raw_dim=16, seed=0)[0].module
    packed = torch.cat((torch.randn(3 * 16), torch.ones(3)))
    with torch.no_grad():
        m1_output = m1(torch.randn(1, 32)).standardized_coefficients
        m3_output = m3(packed).standardized_coefficients
    assert torch.equal(m1_output, m3_output)
    assert torch.count_nonzero(m1_output) == 0


def test_no_candidate_optimizer_is_created() -> None:
    for build in (
        build_ours_v2_candidate(seed=0),
        build_b6_candidate(seed=0),
        *build_m3_m4_paired_candidates(raw_dim=16, seed=0),
    ):
        assert build.candidate_optimizer_created is False
        assert build.candidate_optimizer_step_count == 0


def test_no_optimizer_step_occurs_in_protocol_implementation() -> None:
    sources = [
        (ROOT / "scene/p0_candidate_initialization_protocol.py").read_text(encoding="utf-8"),
        FRESH_PROCESS.read_text(encoding="utf-8"),
    ]
    joined = "\n".join(sources)
    assert "torch.optim" not in joined
    assert ".backward(" not in joined
    assert ".step(" not in joined
    assert "torch.save(" not in joined


def test_standardized_zero_semantics_are_explicit() -> None:
    audit = json.loads(SEMANTICS_AUDIT.read_text(encoding="utf-8"))
    assert audit["classifications"]["STANDARDIZED_ZERO_SEMANTICS"] == "MEAN_COEFFICIENT"
    assert audit["coefficient_normalization"]["standardized_zero_de_standardized"] == audit["coefficient_normalization"]["train_mean"]


def test_raw_coefficient_zero_semantics_are_explicit() -> None:
    audit = json.loads(SEMANTICS_AUDIT.read_text(encoding="utf-8"))
    assert audit["classifications"]["RAW_COEFFICIENT_ZERO_SEMANTICS"] == "MEAN_GARMENT_RESIDUAL"
    difference = audit["objects"]["raw_coefficient_zero"]["difference_to_basis_mean_residual"]
    assert difference["bitwise_equal"] is True
    assert difference["l1_norm"] == difference["l2_norm"] == difference["linf_norm"] == 0.0


def test_physical_zero_residual_means_base_avatar() -> None:
    audit = json.loads(SEMANTICS_AUDIT.read_text(encoding="utf-8"))
    assert audit["classifications"]["PHYSICAL_ZERO_RESIDUAL_SEMANTICS"] == "BASE_AVATAR"
    physical = audit["objects"]["physical_gaussian_residual_zero"]
    assert physical["l1_norm"] == physical["l2_norm"] == physical["linf_norm"] == 0.0


def test_zero_and_base_replacement_are_not_conflated() -> None:
    audit = json.loads(SEMANTICS_AUDIT.read_text(encoding="utf-8"))
    replacements = audit["evaluator_replacements"]
    assert audit["classifications"]["ZERO_REPLACEMENT_EQUALS_BASE_REPLACEMENT"] is False
    assert replacements["episode_count"] == 20
    assert replacements["bitwise_equal_feature_episode_count"] == 0


def test_historical_three_seed_language_is_corrected() -> None:
    audit = json.loads(HISTORICAL_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "CORRECTED"
    assert audit["recommended_language"]["deterministic"] == "three runs under a deterministic zero-initialization contract"
    assert "Ours_Seen_Outfit_Explicit_Basis_V1" in audit["methods_that_must_not_be_called_independent_random_seeds"]
    b5 = next(row for row in audit["method_groups"] if row["method"].startswith("B5_"))
    assert b5["classification"] == "INDEPENDENT_RANDOM_INITIALIZATIONS"


def test_legacy_context_optimizer_is_recorded_separately() -> None:
    audit = json.loads(OPTIMIZER_AUDIT.read_text(encoding="utf-8"))
    candidate, legacy = audit["candidate_optimizer"], audit["legacy_context_optimizer"]
    assert candidate["created"] is False and candidate["parameter_count"] == 0
    assert legacy["created"] is True and legacy["class"] == "torch.optim.adam.Adam"
    assert legacy["parameter_count"] == 128629 and len(legacy["parameter_groups"]) == 5
    assert legacy["discarded_by_caller"] is True


def test_no_optimizer_step_occurs() -> None:
    audit = json.loads(OPTIMIZER_AUDIT.read_text(encoding="utf-8"))
    assert audit["candidate_optimizer"]["step_count"] == 0
    assert audit["legacy_context_optimizer"]["step_count"] == 0
    assert audit["candidate_optimizer"]["zero_grad_count"] == 0
    assert audit["legacy_context_optimizer"]["zero_grad_count"] == 0


def test_no_candidate_or_legacy_optimizer_step_occurs() -> None:
    test_no_optimizer_step_occurs()


def test_candidate_parameters_are_not_in_legacy_optimizer() -> None:
    legacy = json.loads(OPTIMIZER_AUDIT.read_text(encoding="utf-8"))["legacy_context_optimizer"]
    assert legacy["contains_candidate_parameters"] is False
    assert legacy["candidate_parameter_overlap_count"] == 0


def test_optimizer_report_uses_separate_namespaces() -> None:
    audit = json.loads(OPTIMIZER_AUDIT.read_text(encoding="utf-8"))
    assert "candidate_optimizer" in audit and "legacy_context_optimizer" in audit
    assert "optimizer" not in audit


def test_formal_registry_is_unchanged() -> None:
    audit = json.loads(NO_TRAINING_AUDIT.read_text(encoding="utf-8"))
    assert audit["formal_registry_unchanged"] is True
    assert audit["before"]["formal_registry_sha256"] == audit["after"]["formal_registry_sha256"] == "1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e"


def test_formal_outputs_are_immutable() -> None:
    audit = json.loads(NO_TRAINING_AUDIT.read_text(encoding="utf-8"))
    assert audit["formal_outputs_unchanged"] is True
    assert audit["before"]["formal_output_metadata_sha256"] == audit["after"]["formal_output_metadata_sha256"] == "7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc"
    assert audit["after"]["formal_output_file_count"] == 4127
    assert audit["after"]["formal_output_total_bytes"] == 964043888


def test_old_seed_failure_audit_is_unchanged() -> None:
    audit = json.loads(OLD_SEED_AUDIT.read_text(encoding="utf-8"))
    canonical = OLD_SEED_AUDIT.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical).hexdigest() == "0b2e068203031b34e0004725923d452e185f319f3e9088843ecbc42ba5307f14"
    assert audit["status"] == "SEED-PROPAGATION-FAIL"
    assert audit["failed_families"] == ["ours_v2"]


def test_no_paper_final_is_created() -> None:
    no_training = json.loads(NO_TRAINING_AUDIT.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REVIEWER_REGISTRY.read_text(encoding="utf-8"))
    rows = {row["experiment_id"]: row for row in registry["experiments"]}
    assert no_training["paper_final_count"] == 0
    assert all(rows[f"RR-OURS-V2-S{seed}"]["status"] == "FAILED" for seed in (0, 1, 2))
    assert registry["paper_final_transition_allowed"] is False


def test_historical_audit_selects_the_complete_sealed_attempt() -> None:
    source = AUDIT_RUNTIME.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_find_attempt"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"Path": Path}
    exec(compile(module, str(AUDIT_RUNTIME), "exec"), namespace)

    with tempfile.TemporaryDirectory() as value:
        formal_root = Path(value)
        seed_root = formal_root / "PAPER-A1-K1-S0" / "seed_0"
        (seed_root / "attempt_001").mkdir(parents=True)
        sealed = seed_root / "attempt_002"
        for relative in (
            "checkpoints/checkpoint_step_000000.pth",
            "checkpoints/checkpoint_step_000300.pth",
            "evaluated_metrics/evaluated_metrics.json",
        ):
            path = sealed / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        assert namespace["_find_attempt"](formal_root, "PAPER-A1-K1-S0", 0) == sealed
