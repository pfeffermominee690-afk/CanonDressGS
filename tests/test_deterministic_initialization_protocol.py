from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

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
