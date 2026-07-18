from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.representation_capacity_oracle import (  # noqa: E402
    GARMENT_INITIALIZATION,
    GARMENT_LAYER_POINT_COUNT,
    AugmentedGarmentCapacityOracle,
    TemporaryGarmentGaussianLayer,
    UnboundedGaussianDeltaField,
    capacity_oracle_loss_v1,
    decide_representation_case,
)


def fake_base(count: int) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(17)
    weights = torch.zeros(count, 55); weights[:, 0] = 1
    return SimpleNamespace(
        _xyz=torch.randn(count, 3, generator=generator) * 0.1,
        _scaling=torch.full((count, 3), -4.0),
        _rotation=torch.nn.functional.normalize(torch.cat((torch.ones(count, 1), torch.zeros(count, 3)), 1), dim=1),
        _opacity=torch.zeros(count),
        _sh0=torch.zeros(count, 1, 3),
        _shN=torch.zeros(count, 3, 3),
        get_weights=weights,
    )


def sample() -> dict:
    return {
        "target_pose": torch.zeros(165), "target_Rh": torch.eye(3), "target_Th": torch.zeros(3),
        "target_camera": {"K": torch.eye(3), "w2c": torch.eye(4), "width": 2, "height": 2},
        "target_edit_rgb": torch.ones(3, 2, 2), "target_base_rgb": torch.zeros(3, 2, 2),
        "target_foreground_mask": torch.ones(1, 2, 2), "target_base_foreground_mask": torch.ones(1, 2, 2),
        "target_edit_mask": torch.ones(1, 2, 2), "target_clothing_mask": torch.ones(1, 2, 2),
        "target_old_clothing_mask": torch.ones(1, 2, 2), "target_protected_mask": torch.zeros(1, 2, 2),
        "target_transition_mask": torch.zeros(1, 2, 2),
    }


def loss(prediction: torch.Tensor, data: dict) -> dict[str, torch.Tensor]:
    return capacity_oracle_loss_v1(
        prediction, torch.ones(1, 2, 2), target_rgb=data["target_edit_rgb"], base_rgb=data["target_base_rgb"],
        target_foreground=data["target_foreground_mask"], base_foreground=data["target_base_foreground_mask"],
        edit_mask=data["target_edit_mask"], clothing_mask=data["target_clothing_mask"],
        old_clothing_mask=data["target_old_clothing_mask"], protected_mask=data["target_protected_mask"],
        transition_mask=data["target_transition_mask"],
        weights={"garment_rgb": 1, "alpha_foreground": 1, "new_silhouette_alpha": 1, "boundary_rgb": 1, "protected_rgb": 10, "protected_alpha": 5, "stability": 0},
        stability=prediction.sum() * 0,
    )


def run() -> dict:
    checks: dict[str, bool] = {}

    data = sample(); prediction = torch.zeros(3, 2, 2, requires_grad=True)
    parts = loss(prediction, data); parts["total"].backward()
    checks["test_capacity_loss_does_not_preserve_old_garment"] = bool(parts["garment_rgb"] > 0 and prediction.grad is not None and torch.count_nonzero(prediction.grad))

    protected = sample(); protected["target_edit_mask"].zero_(); protected["target_clothing_mask"].zero_(); protected["target_old_clothing_mask"].zero_(); protected["target_protected_mask"].fill_(1)
    protected_prediction = torch.ones(3, 2, 2)
    checks["test_capacity_loss_protects_identity_regions"] = bool(loss(protected_prediction, protected)["protected_rgb"] > 0)

    runner_source = (PROJECT_ROOT / "tools/run_representation_triage_ladder.py").read_text(encoding="utf-8")
    target_free_block = runner_source.split("def target_free_state", 1)[1].split("def chw", 1)[0]
    render_block = runner_source.split("def render_direct", 1)[1].split("def render_augmented", 1)[0]
    checks["test_capacity_target_never_enters_forward"] = all(name in target_free_block for name in ('"pose"', '"Rh"', '"Th"', '"camera"')) and "target_edit_rgb" not in render_block and "target_foreground" not in render_block

    base = fake_base(200_000); first = UnboundedGaussianDeltaField(base); second = UnboundedGaussianDeltaField(base)
    checks["test_single_view_runs_are_independent"] = first.raw_xyz is not second.raw_xyz and first.raw_xyz.data_ptr() != second.raw_xyz.data_ptr()

    checks["test_shared_oracle_uses_one_canonical_field"] = "model: Any = UnboundedGaussianDeltaField(base)" in runner_source and '"shared_canonical_field": True' in runner_source

    fingerprint = torch.cat((base._xyz.flatten(), base._scaling.flatten(), base._opacity.flatten())).clone()
    output = first(base); output.xyz.sum().backward()
    after = torch.cat((base._xyz.flatten(), base._scaling.flatten(), base._opacity.flatten()))
    checks["test_unbounded_oracle_does_not_modify_base"] = torch.equal(fingerprint, after) and all(value.grad is None for value in (base._xyz, base._scaling, base._opacity))

    layer_base = fake_base(GARMENT_LAYER_POINT_COUNT)
    layer = TemporaryGarmentGaussianLayer(layer_base, seed=3, point_count=GARMENT_LAYER_POINT_COUNT)
    checks["test_garment_layer_uses_fixed_30k_points"] = layer.point_count == 30_000 and layer.xyz.shape == (30_000, 3)

    init_source = inspect.getsource(TemporaryGarmentGaussianLayer.__init__)
    config_text = (PROJECT_ROOT / "configs/research/subject02_representation_triage_v1.yaml").read_text(encoding="utf-8")
    checks["test_garment_layer_does_not_initialize_from_target_rgb"] = "target" not in inspect.signature(TemporaryGarmentGaussianLayer.__init__).parameters and "target_rgb" not in init_source and GARMENT_INITIALIZATION in config_text

    augmented = AugmentedGarmentCapacityOracle(layer_base, seed=3)
    checks["test_garment_layer_uses_formal_lbs"] = augmented.garment.formal_lbs_weights.shape == (30_000, 55) and torch.allclose(augmented.garment.formal_lbs_weights.sum(1), torch.ones(30_000))

    expected = {
        (4, "RUNG2_SHARED_SUPPORT_PASS", None): "A",
        (4, "RUNG2_SHARED_SUPPORT_FAIL", "RUNG3_GARMENT_LAYER_PASS"): "B",
        (0, None, "RUNG3_GARMENT_LAYER_PASS"): "C",
        (4, "RUNG2_SHARED_SUPPORT_FAIL", "RUNG3_GARMENT_LAYER_FAIL"): "D",
        (0, None, "RUNG3_GARMENT_LAYER_FAIL"): "E",
    }
    checks["test_decision_matrix_is_deterministic"] = all(decide_representation_case(values[0], values[1], values[2]) == result for values, result in expected.items())

    writes_to_source = any(token in runner_source for token in ("atomic_json(args.source", "atomic_text(args.source", "write_csv(args.source", "torch.save(args.source"))
    checks["test_aaai_no_go_outputs_are_unchanged"] = not writes_to_source and "Rung 0 evidence incomplete" in runner_source

    config = yaml.safe_load((PROJECT_ROOT / "configs/research/subject02_representation_triage_v1.yaml").read_text(encoding="utf-8"))
    long_head = subprocess.run(
        ["git", "rev-parse", f"cloud/{config['long_term_branch']}"], cwd=PROJECT_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    checks["test_long_term_branch_is_unchanged"] = long_head == config["long_term_head"]

    failed = [name for name, passed in checks.items() if not passed]
    return {"status": "PASS" if not failed else "FAIL", "passed": len(checks) - len(failed), "total": len(checks), "failed": failed, "checks": checks}


if __name__ == "__main__":
    result = run(); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["status"] == "PASS" else 1)
