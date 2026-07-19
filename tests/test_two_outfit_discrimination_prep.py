from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.two_outfit_debug_export import (  # noqa: E402
    collect_reference_conditioning_debug,
    run_with_optional_debug,
)
from scene.two_outfit_episode_scheduler import (  # noqa: E402
    OutfitBalancedEpisodeScheduler,
    scheduler_from_manifest,
)
from tools.sprint.build_o01_o08_two_outfit_manifest import (  # noqa: E402
    REFERENCE_ASSET_FIELDS,
    TARGET_ASSET_FIELDS,
    build_multi_outfit_manifest,
)
from tools.sprint.build_two_outfit_discrimination_contact_sheet import (  # noqa: E402
    LAYOUT_SCHEMA_VERSION,
    PANEL_KEYS,
    REQUIRED_CROPS,
    build_contact_sheet,
    validate_contact_sheet_layout,
)
from tools.sprint.evaluate_reference_swap_discrimination import (  # noqa: E402
    TARGET_IMAGE_FIELDS,
    build_forward_payload,
    build_reference_swap_plan,
)
from tools.sprint.measure_two_outfit_conditioning_collapse import (  # noqa: E402
    feature_distance,
    gate_distance,
    mean_outfit_collapse_metrics,
    residual_channel_distance,
)
from utils.two_outfit_checkpoint_utils import (  # noqa: E402
    build_two_outfit_checkpoint_states,
    load_two_outfit_checkpoint,
    required_two_outfit_checkpoint_fields,
    save_two_outfit_checkpoint,
)


CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = dict(zip(CONDITIONS, ("front", "back", "left", "right")))


def _condition(condition_id: str, index: int) -> dict:
    return {
        "condition_id": condition_id,
        "pose": [float(index)] * 165,
        "Rh_raw": [0.0, 0.0, 0.0],
        "R_global": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "Th": [float(index), 0.0, 0.0],
        "K": [[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
        "w2c": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, float(index)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "c2w": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -float(index)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 32,
        "height": 32,
    }


def _write_source_manifest(root: Path, outfits: tuple[str, ...] = ("O01", "O08")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source_outfits = []
    fields = (*REFERENCE_ASSET_FIELDS, *TARGET_ASSET_FIELDS)
    for outfit in outfits:
        observations = []
        for condition in CONDITIONS:
            record = {"condition_id": condition}
            for field in fields:
                asset = root / f"{outfit}__{condition}__{field}.bin"
                asset.write_bytes(f"{outfit}|{condition}|{field}".encode())
                record[field] = asset.name
            observations.append(record)
        source_outfits.append({"outfit_id": outfit, "observations": observations})
    source = {
        "schema_version": "canondressgs.full_dataset.v1",
        "supervision_mode": "dual_target_region_aware_v1",
        "conditions": [_condition(value, index) for index, value in enumerate(CONDITIONS)],
        "splits": {"train": list(outfits), "val": [], "test": []},
        "outfits": source_outfits,
    }
    path = root / "source_manifest.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


@pytest.fixture()
def two_outfit_manifest(tmp_path: Path) -> dict:
    source = _write_source_manifest(tmp_path / "data")
    return build_multi_outfit_manifest(
        source,
        outfit_ids=("O01", "O08"),
        condition_ids=CONDITIONS,
        views=VIEWS,
        reference_counts={"O01": 3, "O08": 3},
        splits={"seen": ["O01", "O08"], "validation": [], "unseen": [], "held_out": []},
    )


def test_two_outfit_manifest_has_eight_episodes(two_outfit_manifest: dict) -> None:
    assert len(two_outfit_manifest["episodes"]) == 8
    assert two_outfit_manifest["validation"]["episode_count"] == 8


def test_each_outfit_has_four_targets(two_outfit_manifest: dict) -> None:
    counts = two_outfit_manifest["validation"]["per_outfit_episode_counts"]
    assert counts == {"O01": 4, "O08": 4}
    targets = {(item["outfit_id"], item["target_condition_id"]) for item in two_outfit_manifest["episodes"]}
    assert len(targets) == 8


def test_reference_target_overlap_is_zero(two_outfit_manifest: dict) -> None:
    for episode in two_outfit_manifest["episodes"]:
        assert episode["target_condition_id"] not in episode["reference_condition_ids"]
    assert two_outfit_manifest["validation"]["reference_target_overlap"] == 0


def test_episode_schema_is_identical_across_outfits(two_outfit_manifest: dict) -> None:
    def signature(value):
        if isinstance(value, dict):
            return {key: signature(item) for key, item in sorted(value.items())}
        if isinstance(value, list):
            return [signature(value[0])] if value else []
        return type(value).__name__

    signatures = {}
    for episode in two_outfit_manifest["episodes"]:
        signatures.setdefault(episode["outfit_id"], signature(episode))
        assert signatures[episode["outfit_id"]] == signature(episode)
    assert signatures["O01"] == signatures["O08"]


def test_manifest_contains_no_absolute_asset_paths(two_outfit_manifest: dict, tmp_path: Path) -> None:
    serialized = json.dumps(two_outfit_manifest)
    assert str(tmp_path) not in serialized
    assert two_outfit_manifest["source"]["absolute_paths_embedded"] is False
    assert all("sha256" in value for value in two_outfit_manifest["asset_registry"].values())


def test_balanced_scheduler_equal_outfit_counts(two_outfit_manifest: dict) -> None:
    scheduler = scheduler_from_manifest(two_outfit_manifest, seed=7)
    scheduler.take(scheduler.cycle_length * 3)
    assert set(scheduler.outfit_update_counts.values()) == {12}
    assert scheduler.balance_report()["outfit_count_spread"] == 0


def test_balanced_scheduler_equal_view_counts(two_outfit_manifest: dict) -> None:
    scheduler = scheduler_from_manifest(two_outfit_manifest, seed=7)
    scheduler.take(scheduler.cycle_length * 2)
    assert set(scheduler.view_update_counts.values()) == {4}
    assert scheduler.balance_report()["view_count_spread"] == 0


def test_scheduler_resume_exact(two_outfit_manifest: dict) -> None:
    uninterrupted = scheduler_from_manifest(two_outfit_manifest, seed=31)
    prefix = uninterrupted.take(5)
    state = uninterrupted.state_dict()
    expected_next = uninterrupted.take(11)
    resumed = scheduler_from_manifest(two_outfit_manifest, seed=31)
    resumed.load_state_dict(state)
    assert resumed.state_dict()["rng_state"] == state["rng_state"]
    assert resumed.state_dict()["current_cycle_position"] == 5
    assert resumed.take(11) == expected_next
    assert resumed.outfit_update_counts == uninterrupted.outfit_update_counts
    assert resumed.view_update_counts == uninterrupted.view_update_counts
    assert prefix[-1]["target_condition_id"] == CONDITIONS[2]


def test_checkpoint_state_contains_exact_resume_fields(two_outfit_manifest: dict) -> None:
    scheduler = scheduler_from_manifest(two_outfit_manifest, seed=9)
    scheduler.take(5)
    training, data, method = build_two_outfit_checkpoint_states(
        balanced_scheduler=scheduler,
        global_step=5,
        config={"seed": 9},
        manifest_sha256="manifest",
        base_fingerprint="base",
        trainable_fingerprint="trainable",
    )
    contract = required_two_outfit_checkpoint_fields()
    assert set(contract["training_state"]).issubset(training)
    assert set(contract["data_state"]).issubset(data)
    assert set(contract["method_state"]).issubset(method)


class _StateCarrier:
    """Optimizer-state protocol fixture; deliberately performs no optimization."""

    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.param_groups = [{"params": [parameter], "lr": 0.001, "weight_decay": 0.0}]
        self.loaded = None

    def state_dict(self) -> dict:
        return {"state": {}, "param_groups": [{"lr": 0.001, "weight_decay": 0.0}]}

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


def test_two_outfit_checkpoint_resume_restores_rng_and_next_episode(
    two_outfit_manifest: dict, tmp_path: Path
) -> None:
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    model = torch.nn.Linear(2, 1)
    state_carrier = _StateCarrier(next(model.parameters()))
    scheduler = scheduler_from_manifest(two_outfit_manifest, seed=17)
    scheduler.take(5)
    expected_next = scheduler.peek()
    checkpoint = tmp_path / "resume.pth"
    save_two_outfit_checkpoint(
        checkpoint,
        model=model,
        optimizer=state_carrier,
        optimizer_group_names=["trainable"],
        lr_scheduler=None,
        scaler=None,
        balanced_scheduler=scheduler,
        global_step=5,
        config={"seed": 17},
        manifest_sha256="manifest",
        base_fingerprint="base",
        trainable_fingerprint="trainable",
    )
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.random(); np.random.rand(); torch.rand(())
    restored_model = torch.nn.Linear(2, 1)
    restored_carrier = _StateCarrier(next(restored_model.parameters()))
    restored_scheduler = scheduler_from_manifest(two_outfit_manifest, seed=17)
    load_two_outfit_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_carrier,
        optimizer_group_names=["trainable"],
        lr_scheduler=None,
        scaler=None,
        balanced_scheduler=restored_scheduler,
        config={"seed": 17},
        manifest_sha256="manifest",
        base_fingerprint="base",
        trainable_fingerprint="trainable",
    )
    actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert all(actual == pytest.approx(expected) for actual, expected in zip(actual_rng, expected_rng))
    assert restored_scheduler.peek() == expected_next
    assert restored_scheduler.index == 5
    assert restored_carrier.loaded is not None


def test_no_outfit_id_enters_ours_forward(two_outfit_manifest: dict) -> None:
    plan = build_reference_swap_plan(two_outfit_manifest)
    for case in plan["cases"]:
        payload = build_forward_payload(case)
        assert "outfit_id" not in payload
        assert "cloth_id" not in payload


def test_target_image_fields_do_not_enter_forward(two_outfit_manifest: dict) -> None:
    plan = build_reference_swap_plan(two_outfit_manifest)
    for case in plan["cases"]:
        assert TARGET_IMAGE_FIELDS.isdisjoint(build_forward_payload(case))
    assert plan["target_rgb_is_forward_input"] is False


def test_reference_swap_uses_same_target_pose(two_outfit_manifest: dict) -> None:
    plan = build_reference_swap_plan(two_outfit_manifest)
    correct = {
        (item["target_outfit_id"], item["target_condition_id"]): item
        for item in plan["cases"] if item["kind"] == "correct_reference"
    }
    for case in plan["cases"]:
        expected = correct[(case["target_outfit_id"], case["target_condition_id"])]
        assert case["target_pose_camera_fingerprint"] == expected["target_pose_camera_fingerprint"]


def test_reference_swap_changes_only_references(two_outfit_manifest: dict) -> None:
    plan = build_reference_swap_plan(two_outfit_manifest)
    correct = {
        (item["target_outfit_id"], item["target_condition_id"]): item
        for item in plan["cases"] if item["kind"] == "correct_reference"
    }
    for case in (item for item in plan["cases"] if item["kind"] == "swapped_reference"):
        expected = correct[(case["target_outfit_id"], case["target_condition_id"])]
        assert case["target_geometry"] == expected["target_geometry"]
        assert case["reference_source_outfit_id"] != expected["reference_source_outfit_id"]


def test_feature_export_debug_off_is_bitwise_exact() -> None:
    value = torch.randn(4, requires_grad=True)
    calls = 0

    def forward():
        nonlocal calls
        calls += 1
        return {"value": value * 3.0}

    output, record, tensors = run_with_optional_debug(forward, debug_enabled=False)
    assert calls == 1 and record is None and tensors == {}
    assert torch.equal(output["value"], value * 3.0)
    assert output["value"].grad_fn is not None


def test_feature_export_contains_no_target_features() -> None:
    completion = SimpleNamespace(
        observed_anchor_features=torch.ones(3, 2),
        completed_anchor_features=torch.full((3, 2), 2.0),
        observed_clothing_probability=torch.ones(3, 1),
        confidence=torch.ones(3, 1),
        geometry_gate=torch.ones(3, 1),
        appearance_gate=torch.ones(3, 1),
    )
    record, tensors = collect_reference_conditioning_debug(
        {
            "global_clothing_embedding": torch.ones(1, 2),
            "completion": completion,
            "gated_anchor_residuals": {"delta_xyz": torch.zeros(3, 3)},
            "gaussian_residuals": {"delta_xyz": torch.zeros(5, 3)},
            "final_rgb": torch.zeros(3, 4, 4),
            "final_alpha": torch.zeros(1, 4, 4),
            "target_image_features": torch.full((2,), 99.0),
        },
        outfit_id="fixture-a",
        episode_id="episode",
        step=1,
        commit="commit",
        config_hash="config",
        include_full_tensors=True,
    )
    serialized_summaries = json.dumps(record["summaries"]).lower()
    assert "target_image_features" not in serialized_summaries
    assert all("target" not in key for key in tensors)
    assert record["target_image_features_exported"] is False


def test_conditioning_metrics_are_symmetric() -> None:
    first = torch.tensor([1.0, 2.0, 3.0])
    second = torch.tensor([3.0, 1.0, 0.0])
    assert feature_distance(first, second) == feature_distance(second, first)
    a = gate_distance(first.sigmoid(), second.sigmoid())
    b = gate_distance(second.sigmoid(), first.sigmoid())
    for key in ("cosine_distance", "normalized_l2", "mean_absolute_distance", "active_iou", "correlation"):
        assert a[key] == pytest.approx(b[key])


def test_residual_distance_is_bound_normalized() -> None:
    result = residual_channel_distance(torch.zeros(4, 3), torch.ones(4, 3), bound=2.0)
    assert result["bound_normalized_l1"] == pytest.approx(0.5)
    assert result["bound_normalized_l2"] == pytest.approx(0.5)


def test_mean_outfit_collapse_detects_average_prediction() -> None:
    first_target = torch.zeros(3, 4, 4)
    second_target = torch.ones(3, 4, 4)
    average_prediction = torch.full((3, 4, 4), 0.5)
    result = mean_outfit_collapse_metrics(
        average_prediction,
        average_prediction,
        first_target,
        second_target,
        garment_mask=torch.ones(1, 4, 4),
    )
    assert result["both_predictions_closer_to_mean_than_own_target"] is True
    assert result["mean_outfit_margin"] == pytest.approx(-0.5)


def _contact_layout(tmp_path: Path) -> tuple[dict, Path]:
    panels = {}
    for index, key in enumerate(PANEL_KEYS):
        path = tmp_path / f"{key}.png"
        Image.new("RGB", (32, 32), (index, index, index)).save(path)
        panels[key] = path.name
    crops = {
        "torso_garment": [4, 4, 20, 24],
        "sleeves_arms": [0, 4, 32, 24],
        "shoes_protected": [6, 24, 26, 32],
        "garment_boundary": [2, 2, 30, 30],
    }
    layout = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "outfits": ["fixture-a", "fixture-b"],
        "rows": [{
            "target_view": "front",
            "target_condition_id": "condition-a",
            "target_pose_camera_fingerprint": "same-camera",
            "panel_camera_fingerprints": {key: "same-camera" for key in PANEL_KEYS[2:]},
            "panels": panels,
            "crop_boxes_xyxy": crops,
        }],
    }
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(layout), encoding="utf-8")
    return layout, path


def test_contact_sheet_uses_same_camera_and_crop(tmp_path: Path) -> None:
    layout, path = _contact_layout(tmp_path)
    validation = validate_contact_sheet_layout(layout, tmp_path)
    assert validation["same_camera"] and validation["same_crop"] and validation["same_scaling"]
    output = build_contact_sheet(path, tmp_path / "contact")
    assert Path(output["main_png"]).is_file()
    assert set(output["crop_pngs"]) == set(REQUIRED_CROPS)
    assert Path(output["source_data"]).is_file()


def _prep_python_files() -> list[Path]:
    return [
        PROJECT_ROOT / "scene" / "two_outfit_episode_scheduler.py",
        PROJECT_ROOT / "scene" / "two_outfit_debug_export.py",
        PROJECT_ROOT / "utils" / "two_outfit_checkpoint_utils.py",
        *(PROJECT_ROOT / "tools" / "sprint").glob("*.py"),
    ]


def test_multi_outfit_code_has_no_o01_o08_branching() -> None:
    forbidden = re.compile(r"\b(?:if|elif)\b[^\n]*(?:O01|O08)|\bmatch\b[^\n]*(?:O01|O08)")
    for path in _prep_python_files():
        source = path.read_text(encoding="utf-8")
        assert forbidden.search(source) is None, path


def test_multi_outfit_scheduler_supports_dynamic_outfit_count(tmp_path: Path) -> None:
    source = _write_source_manifest(tmp_path / "dynamic", ("fixture-a", "fixture-b", "fixture-c"))
    manifest = build_multi_outfit_manifest(
        source,
        outfit_ids=("fixture-a", "fixture-b", "fixture-c"),
        condition_ids=CONDITIONS,
        views=VIEWS,
    )
    scheduler = scheduler_from_manifest(manifest, seed=1)
    scheduler.take(scheduler.cycle_length)
    assert set(scheduler.outfit_update_counts.values()) == {4}
    assert len(scheduler.episode_order) == 12


def test_no_optimizer_is_created_in_prep_task() -> None:
    forbidden_calls = {"Adam", "AdamW", "SGD", "RMSprop", "build_image_conditioned_optimizer"}
    for path in _prep_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            assert name not in forbidden_calls, f"{path}: constructs {name}"


def _tree_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_current_o01_output_is_unchanged(two_outfit_manifest: dict, tmp_path: Path) -> None:
    protected_output = tmp_path / "attempt_004"
    protected_output.mkdir()
    (protected_output / "checkpoint.pth").write_bytes(b"immutable")
    (protected_output / "training.jsonl").write_text('{"step": 1}\n', encoding="utf-8")
    before = _tree_fingerprint(protected_output)
    build_reference_swap_plan(two_outfit_manifest)
    scheduler_from_manifest(two_outfit_manifest, seed=5).take(17)
    assert _tree_fingerprint(protected_output) == before


def test_current_o01_process_is_untouched(two_outfit_manifest: dict) -> None:
    before_pid = os.getpid()
    build_reference_swap_plan(two_outfit_manifest)
    assert os.getpid() == before_pid
    forbidden_process_calls = {"kill", "terminate", "send_signal", "Popen"}
    for path in _prep_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert called.isdisjoint(forbidden_process_calls), path


def test_prep_permissions_forbid_training_gpu_and_current_attempt_mutation() -> None:
    import yaml

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "sprint" / "subject02_o01_o08_two_outfit_discrimination_v1.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["training"]["enabled"] is False
    assert config["permissions"]["start_training"] is False
    assert config["permissions"]["use_gpu"] is False
    assert config["permissions"]["create_optimizer_during_prep"] is False
    assert config["permissions"]["modify_current_o01_attempt"] is False
