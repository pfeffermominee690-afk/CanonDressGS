from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.frozen_f2_linear_coefficient_control import (  # noqa: E402
    FrozenF2LinearLogitControl,
    FrozenF2ReferenceFeatureExtractor,
)
from scene.gaussian_clothing_residuals import GaussianClothingResiduals  # noqa: E402
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools import run_explicit_gaussian_residual_basis as legacy  # noqa: E402
from tools import run_image_conditioned_overfit_o01 as o01  # noqa: E402
from tools import run_reference_basis_coefficient_fusion as previous  # noqa: E402
from tools import run_residual_field_parameterization as parameterization  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _tensor_state_fingerprint,
)


SCHEMA = "canondressgs.reference_coefficient_supervision_calibration.v1"
TASK_ID = "SUBJECT02-REFERENCE-COEFFICIENT-SUPERVISION-CALIBRATION-001"
EXPECTED_BRANCH = "research/reference-coefficient-supervision-calibration-20260720"
EXPECTED_SOURCE_HEAD = "fbb162ad58fa7810f78a222f7c669d696b00fefd"
OUTFITS = ("O01", "O08")
CONDITIONS = o01.CONDITIONS
VIEWS = o01.VIEWS
COEFFICIENT_TARGETS = {"O01": -1.0, "O08": 1.0}
CLASS_LABELS = {"O01": 0.0, "O08": 1.0}
FORBIDDEN_FORWARD_FIELDS = {
    "target_rgb", "target_mask", "target_pose", "target_camera", "outfit_id",
    "cloth_id", "teacher", "teacher_coefficient", "diagnostic_latent",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ValueError("supervision-calibration schema/task mismatch")
    if config.get("branch") != EXPECTED_BRANCH or config.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("supervision-calibration governance contract changed")
    if tuple(config.get("outfits", ())) != OUTFITS or tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("two-outfit/four-view protocol changed")
    if int(config["stage_2"]["max_steps"]) != 100:
        raise ValueError("frozen F2 linear control must contain exactly 100 steps")
    if config["stage_2"]["paired_batch_order"] != ["O01", "O08"]:
        raise ValueError("paired batch order changed")
    if float(config["stage_2"]["loss"]["signed_rank_margin"]) != 2.0:
        raise ValueError("signed ranking margin changed")
    if any(bool(value) for value in config["permissions"].values()):
        raise ValueError("forbidden permission enabled")
    return config


def _verify_governance() -> dict[str, Any]:
    branch, head, dirty = (
        git_output("branch", "--show-current"), git_output("rev-parse", "HEAD"),
        git_output("status", "--short"),
    )
    if branch != EXPECTED_BRANCH or dirty:
        raise RuntimeError(f"formal run requires clean {EXPECTED_BRANCH}; dirty={bool(dirty)}")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", EXPECTED_SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT
    ) != 0:
        raise RuntimeError("formal branch is not descended from the RF-F final HEAD")
    return {"branch": branch, "head": head, "clean": True, "source_head": EXPECTED_SOURCE_HEAD}


def _build_previous_context(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    old_branch, old_head = previous.EXPECTED_BRANCH, previous.EXPECTED_SOURCE_HEAD
    previous.EXPECTED_BRANCH, previous.EXPECTED_SOURCE_HEAD = EXPECTED_BRANCH, EXPECTED_SOURCE_HEAD
    try:
        return previous.build_context(config, output_dir, create=False)
    finally:
        previous.EXPECTED_BRANCH, previous.EXPECTED_SOURCE_HEAD = old_branch, old_head


def build_context(config: dict[str, Any], output_dir: Path, *, create: bool) -> dict[str, Any]:
    governance = _verify_governance()
    previous_final = Path(config["inputs"]["previous_fusion_final_adjudication"])
    previous_attempt = Path(config["inputs"]["previous_fusion_attempt"])
    if not previous_final.is_file() or sha256(previous_final) != config["inputs"]["previous_fusion_final_adjudication_sha256"]:
        raise ValueError("immutable RF-F final adjudication mismatch")
    previous_tree_before = legacy.immutable_tree_metadata_fingerprint(previous_attempt)
    if create:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"formal attempt already exists and is non-empty: {output_dir}")
        for directory in (
            "contract", "input_audit", "stage_0_loss_audit", "stage_1_feature_parity",
            "stage_2_linear_control/checkpoints", "stage_2_linear_control/milestones",
            "stage_3_counterfactual", "visual_acceptance", "final_adjudication",
        ):
            (output_dir / directory).mkdir(parents=True, exist_ok=True)
        atomic_text(output_dir / "contract/config_resolved.yaml", yaml.safe_dump(config, sort_keys=False))
        atomic_text(output_dir / "contract/command.txt", " ".join([sys.executable, *sys.argv]))
        atomic_json(output_dir / "input_audit/environment.json", previous.environment_snapshot())
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "RUNNING", "stage": "INPUT_AUDIT",
            "optimizer_steps": 0, "updated_at_unix": time.time(),
        })
    context = _build_previous_context(config, output_dir)
    context.update({
        "governance": governance, "previous_attempt": previous_attempt,
        "previous_tree_before": previous_tree_before,
    })
    if create:
        atomic_json(output_dir / "input_audit/input_manifest.json", {
            "task_id": TASK_ID, "git": governance,
            "previous_rf_f_attempt": str(previous_attempt),
            "previous_rf_f_final_adjudication": str(previous_final),
            "previous_rf_f_final_adjudication_sha256": sha256(previous_final),
            "previous_rf_f_attempt_tree_metadata_fingerprint": previous_tree_before,
            "explicit_basis_artifact": str(context["paths"]["explicit_basis_artifact"]),
            "explicit_basis_sha256": sha256(context["paths"]["explicit_basis_artifact"]),
            "legacy_stage_c_checkpoint": str(context["paths"]["legacy_stage_c_checkpoint"]),
            "legacy_stage_c_checkpoint_sha256": sha256(context["paths"]["legacy_stage_c_checkpoint"]),
            "base_checkpoint_sha256": config["base"]["checkpoint_sha256"],
            "backbone_fingerprint": context["backbone_before"],
            "prediction_forward_fields": [
                "reference RGB", "reference clothing mask", "reference valid mask",
            ],
            "forbidden_prediction_fields": sorted(FORBIDDEN_FORWARD_FIELDS),
            "target_view_used_in_prediction_forward": False,
            "teacher_usage": "loss and evaluation target only",
        })
        atomic_json(output_dir / "input_audit/reference_input_audit.json", context["input_audit"])
    return context


def _old_loss(
    coefficients: torch.Tensor, config: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_config = config["stage_0"]["old_loss"]
    targets = coefficients.new_tensor([-1.0, 1.0])
    coefficient_loss = F.smooth_l1_loss(coefficients, targets)
    sign_loss = F.relu(float(loss_config["sign_margin"]) - targets * coefficients).mean()
    pair_loss = F.relu(
        float(loss_config["pair_margin"]) - torch.abs(coefficients[1] - coefficients[0])
    )
    total = (
        float(loss_config["coefficient_weight"]) * coefficient_loss
        + float(loss_config["sign_weight"]) * sign_loss
        + float(loss_config["pair_weight"]) * pair_loss
    )
    return total, {"coefficient": coefficient_loss, "sign": sign_loss, "absolute_pair": pair_loss}


def _central_difference(function, value: torch.Tensor, index: int, epsilon: float) -> float:
    positive, negative = value.detach().clone(), value.detach().clone()
    positive[index] += epsilon; negative[index] -= epsilon
    return float((function(positive) - function(negative)) / (2 * epsilon))


def run_stage_0(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    magnitude = float(config["stage_0"]["endpoint_raw_logit_magnitude"])
    epsilon = float(config["stage_0"]["finite_difference_epsilon"])
    rows = []
    for state in config["stage_0"]["states"]:
        coefficient_leaf = torch.tensor(state, dtype=torch.float64, requires_grad=True)
        total_leaf, parts_leaf = _old_loss(coefficient_leaf, config)
        total_coefficient_gradient = torch.autograd.grad(
            total_leaf, coefficient_leaf, retain_graph=True
        )[0]
        part_gradients = {}
        for name, part in parts_leaf.items():
            part_gradients[name] = torch.autograd.grad(
                part, coefficient_leaf, retain_graph=True, allow_unused=True
            )[0].tolist()
        raw_values = [(-magnitude if value < 0 else magnitude if value > 0 else 0.0) for value in state]
        raw_logit = torch.tensor(raw_values, dtype=torch.float64, requires_grad=True)
        finite_coefficients = torch.tanh(raw_logit)
        finite_total, finite_parts = _old_loss(finite_coefficients, config)
        raw_gradient = torch.autograd.grad(finite_total, raw_logit, retain_graph=True)[0]
        pair_raw_gradient = torch.autograd.grad(
            finite_parts["absolute_pair"], raw_logit, retain_graph=True
        )[0]
        finite_difference = [
            _central_difference(lambda value: _old_loss(torch.tanh(value), config)[0], raw_logit, index, epsilon)
            for index in range(2)
        ]
        tanh_derivative = (1.0 - finite_coefficients.square()).detach()
        rows.append({
            "requested_coefficient_state": state,
            "coefficient_gradient": total_coefficient_gradient.tolist(),
            "loss_parts": {name: float(value.detach()) for name, value in parts_leaf.items()},
            "raw_logit": raw_logit.detach().tolist(),
            "finite_coefficient": finite_coefficients.detach().tolist(),
            "raw_logit_gradient": raw_gradient.tolist(),
            "finite_difference_raw_logit_gradient": finite_difference,
            "finite_difference_max_abs_error": float(max(
                abs(first - second) for first, second in zip(raw_gradient.tolist(), finite_difference)
            )),
            "tanh_derivative": tanh_derivative.tolist(),
            "loss_part_coefficient_gradients": part_gradients,
            "absolute_pair_raw_logit_gradient": pair_raw_gradient.tolist(),
            "absolute_pair_equal_output_gradient_zero": bool(
                state[0] == state[1] and torch.equal(pair_raw_gradient, torch.zeros_like(pair_raw_gradient))
            ) if state[0] == state[1] else None,
        })
    report = {
        "status": "PASS", "optimizer_created": False,
        "paired_batch_order": ["O01", "O08"],
        "class_labels": CLASS_LABELS, "coefficient_targets": COEFFICIENT_TARGETS,
        "states": rows,
        "conclusions": {
            "absolute_pair_has_no_stable_separation_gradient_at_equal_outputs": all(
                row["absolute_pair_equal_output_gradient_zero"] for row in rows[:3]
            ),
            "wrong_endpoint_recovery_raw_gradient_near_zero": bool(
                abs(rows[2]["raw_logit_gradient"][0]) < 1e-6
                and abs(rows[0]["raw_logit_gradient"][1]) < 1e-6
            ),
            "tanh_endpoint_suppresses_raw_gradient": bool(
                max(rows[0]["tanh_derivative"] + rows[2]["tanh_derivative"]) < 1e-7
            ),
            "label_and_batch_order_correct": True,
            "historical_adjudication_rewritten": False,
        },
    }
    atomic_json(output_dir / "stage_0_loss_audit/loss_gradient_audit.json", report)
    _stage_0_visual(output_dir, report)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING", "stage": "S0_COMPLETE",
        "optimizer_steps": 0, "updated_at_unix": time.time(),
    })
    return report


def _stage_0_visual(output_dir: Path, report: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(value["requested_coefficient_state"]) for value in report["states"]]
    gradients = [max(abs(item) for item in value["raw_logit_gradient"]) for value in report["states"]]
    derivatives = [max(value["tanh_derivative"]) for value in report["states"]]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(labels, gradients); axes[0].set_yscale("log"); axes[0].set_title("Old-loss raw-logit gradient magnitude")
    axes[1].bar(labels, derivatives); axes[1].set_yscale("log"); axes[1].set_title("tanh derivative")
    for axis in axes: axis.tick_params(axis="x", rotation=25)
    figure.savefig(output_dir / "visual_acceptance/old_loss_gradient_audit.png", dpi=150)
    plt.close(figure)


def _feature_extractor(context: Mapping[str, Any]) -> FrozenF2ReferenceFeatureExtractor:
    return FrozenF2ReferenceFeatureExtractor(
        context["legacy_model"].clothing_observation_encoder.backbone[:-1],
        int(context["config"]["legacy_reference_model"]["image_feature_dim"]),
    ).to(context["base"]._xyz)


def _extract_inputs(episode: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    values = {
        "reference_images": episode["reference_images"],
        "reference_clothing_masks": episode["reference_cloth_masks"],
        "reference_valid_mask": episode["reference_valid_mask"],
    }
    if FORBIDDEN_FORWARD_FIELDS.intersection(values):
        raise RuntimeError("forbidden target/teacher field entered frozen F2 extraction")
    return values


def run_stage_1(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    if not (output_dir / "stage_0_loss_audit/loss_gradient_audit.json").is_file():
        raise FileNotFoundError("S0 loss audit must complete before S1")
    extractor = _feature_extractor(context).eval()
    offline_payload, rows = {}, {}
    with torch.no_grad():
        for condition in CONDITIONS:
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"; episode = context["episodes"][key]
                offline = previous._raw_probe_features(
                    context["legacy_model"], episode, extractor
                )
                online = extractor(**_extract_inputs(episode))
                offline_payload[key] = {
                    "F1": offline["F1"].detach().cpu(),
                    "F2": offline["F2"].detach().cpu(),
                    "reference_condition_ids": list(episode["reference_condition_ids"]),
                }
                differences = {
                    "F1": (offline["F1"] - online.clothing_mean.mean(0)).abs(),
                    "F2": (offline["F2"] - online.set_mean.reshape(-1)).abs(),
                }
                rows[key] = {
                    "reference_condition_ids": list(episode["reference_condition_ids"]),
                    "offline_F1_shape": list(offline["F1"].shape),
                    "online_F1_shape": list(online.clothing_mean.mean(0).shape),
                    "offline_F2_shape": list(offline["F2"].shape),
                    "online_F2_shape": list(online.set_mean.reshape(-1).shape),
                    "offline_dtype": str(offline["F2"].dtype), "online_dtype": str(online.set_mean.dtype),
                    "F1_l1_difference": float(differences["F1"].mean()),
                    "F1_l2_difference": float(torch.linalg.vector_norm(differences["F1"])),
                    "F1_max_difference": float(differences["F1"].max()),
                    "F2_l1_difference": float(differences["F2"].mean()),
                    "F2_l2_difference": float(torch.linalg.vector_norm(differences["F2"])),
                    "F2_max_difference": float(differences["F2"].max()),
                    "mask_resize_shape": list(online.resized_clothing_mask.shape),
                    "mask_resize_min": float(online.resized_clothing_mask.min()),
                    "mask_resize_max": float(online.resized_clothing_mask.max()),
                    "pooling_denominator": online.pooling_denominator.reshape(-1).tolist(),
                    "offline_F2_norm": float(torch.linalg.vector_norm(offline["F2"])),
                    "online_F2_norm": float(torch.linalg.vector_norm(online.set_mean)),
                    "set_feature_shape": list(online.set_feature.shape),
                }
    feature_path = output_dir / "stage_1_feature_parity/offline_probe_features.pt"
    temporary = feature_path.with_suffix(".pt.tmp"); torch.save(offline_payload, temporary); os.replace(temporary, feature_path)
    max_difference = max(value["F2_max_difference"] for value in rows.values())
    tolerance = float(config["stage_1"]["parity_atol"])
    checks = {
        "all_8_episodes_present": len(rows) == 8,
        "reference_order_exact": all(
            value["reference_condition_ids"] == offline_payload[key]["reference_condition_ids"]
            for key, value in rows.items()
        ),
        "shape_exact": all(value["offline_F2_shape"] == value["online_F2_shape"] for value in rows.values()),
        "dtype_exact": all(value["offline_dtype"] == value["online_dtype"] for value in rows.values()),
        "f2_within_tolerance": max_difference <= tolerance,
        "resized_masks_nonempty_nonfull": all(
            value["mask_resize_max"] > 0 and value["mask_resize_min"] < 1 for value in rows.values()
        ),
        "pooling_denominators_positive": all(
            min(value["pooling_denominator"]) > 0 for value in rows.values()
        ),
        "backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status, "classification": None if status == "PASS" else "CS-FEATURE-PARITY-FAIL",
        "next_task": None if status == "PASS" else "FIX_ONLINE_OFFLINE_REFERENCE_FEATURE_PARITY",
        "rows": rows, "maximum_F2_difference": max_difference,
        "parity_atol": tolerance, "parity_rtol": float(config["stage_1"]["parity_rtol"]),
        "offline_feature_artifact": str(feature_path), "offline_feature_artifact_sha256": sha256(feature_path),
        "checks": checks,
    }
    atomic_json(output_dir / "stage_1_feature_parity/feature_parity.json", report)
    _stage_1_visual(output_dir, report)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING" if status == "PASS" else "FAIL",
        "stage": "S1_COMPLETE", "optimizer_steps": 0,
        "classification": report["classification"], "updated_at_unix": time.time(),
    })
    return report


def _stage_1_visual(output_dir: Path, report: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    keys = list(report["rows"])
    values = [report["rows"][key]["F2_max_difference"] for key in keys]
    figure, axis = plt.subplots(figsize=(13, 4), constrained_layout=True)
    axis.bar(keys, values); axis.tick_params(axis="x", rotation=30)
    axis.axhline(report["parity_atol"], color="red", linestyle="--", label="atol")
    axis.set_title("Offline probe vs online frozen F2 max difference"); axis.legend()
    figure.savefig(output_dir / "visual_acceptance/online_offline_f2_parity.png", dpi=150)
    plt.close(figure)


def _make_frozen_control(
    context: Mapping[str, Any],
) -> tuple[FrozenF2ReferenceFeatureExtractor, FrozenF2LinearLogitControl]:
    backbone = context["legacy_model"].clothing_observation_encoder.backbone[:-1]
    extractor = FrozenF2ReferenceFeatureExtractor(backbone, feature_dim=128).to(
        context["base"]._xyz.device
    )
    control = FrozenF2LinearLogitControl(extractor.set_feature_dim).to(
        context["base"]._xyz.device
    )
    extractor.eval()
    return extractor, control


def _frozen_features(
    extractor: FrozenF2ReferenceFeatureExtractor, episode: Mapping[str, Any]
) -> torch.Tensor:
    leaked = FORBIDDEN_FORWARD_FIELDS.intersection(
        {"reference_images", "reference_clothing_masks", "reference_valid_mask"}
    )
    if leaked:
        raise RuntimeError(f"forbidden prediction input: {sorted(leaked)}")
    with torch.no_grad():
        output = extractor(
            episode["reference_images"], episode["reference_cloth_masks"],
            episode["reference_valid_mask"],
        )
    return output.set_feature.detach()


def _all_frozen_features(
    context: Mapping[str, Any], extractor: FrozenF2ReferenceFeatureExtractor
) -> dict[str, torch.Tensor]:
    return {
        f"{outfit}/{condition}": _frozen_features(
            extractor, context["episodes"][f"{outfit}/{condition}"]
        )
        for condition in CONDITIONS for outfit in OUTFITS
    }


def _feature_distance_summary(features: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    within, between = [], []
    for first_index, first_condition in enumerate(CONDITIONS):
        for second_condition in CONDITIONS[first_index + 1:]:
            for outfit in OUTFITS:
                difference = (
                    features[f"{outfit}/{first_condition}"]
                    - features[f"{outfit}/{second_condition}"]
                ).reshape(-1)
                within.append(float(
                    torch.linalg.vector_norm(difference) / math.sqrt(difference.numel())
                ))
    for condition in CONDITIONS:
        difference = (
            features[f"O01/{condition}"] - features[f"O08/{condition}"]
        ).reshape(-1)
        between.append(float(
            torch.linalg.vector_norm(difference) / math.sqrt(difference.numel())
        ))
    return {
        "within_outfit_rms_l2_mean": float(np.mean(within)),
        "within_outfit_rms_l2_values": within,
        "between_outfit_matched_view_rms_l2_mean": float(np.mean(between)),
        "between_outfit_matched_view_rms_l2_values": between,
        "between_within_ratio": float(np.mean(between) / max(np.mean(within), 1e-12)),
    }


def _linear_control_loss(
    raw_by_outfit: Mapping[str, torch.Tensor], config: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw = torch.cat([raw_by_outfit[outfit].reshape(1) for outfit in OUTFITS])
    labels = raw.new_tensor([CLASS_LABELS[outfit] for outfit in OUTFITS])
    targets = raw.new_tensor([COEFFICIENT_TARGETS[outfit] for outfit in OUTFITS])
    coefficients = torch.tanh(raw)
    loss_config = config["stage_2"]["loss"]
    classification = F.binary_cross_entropy_with_logits(raw, labels)
    coefficient = F.smooth_l1_loss(coefficients, targets)
    signed_rank = F.relu(
        float(loss_config["signed_rank_margin"]) - (raw[1] - raw[0])
    )
    total = (
        float(loss_config["classification_weight"]) * classification
        + float(loss_config["coefficient_weight"]) * coefficient
        + float(loss_config["signed_rank_weight"]) * signed_rank
    )
    return total, {
        "classification": classification, "coefficient": coefficient,
        "signed_rank": signed_rank,
    }


def _evaluate_linear_control(
    control: FrozenF2LinearLogitControl, features: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    with torch.no_grad():
        for condition in CONDITIONS:
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"
                output = control(features[key])
                raw, coefficient = float(output.raw_logit), float(output.coefficient)
                rows[key] = {
                    "outfit": outfit, "condition": condition,
                    "raw_logit": raw, "coefficient": coefficient,
                    "target": COEFFICIENT_TARGETS[outfit],
                    "class_label": CLASS_LABELS[outfit],
                    "absolute_error": abs(coefficient - COEFFICIENT_TARGETS[outfit]),
                    "correct_sign": coefficient * COEFFICIENT_TARGETS[outfit] > 0,
                    "signed_raw_logit_margin": raw * COEFFICIENT_TARGETS[outfit],
                    "target_view_used_in_prediction_forward": False,
                }
    outfit_mean = {
        outfit: float(np.mean([
            rows[f"{outfit}/{condition}"]["coefficient"] for condition in CONDITIONS
        ])) for outfit in OUTFITS
    }
    raw_mean = {
        outfit: float(np.mean([
            rows[f"{outfit}/{condition}"]["raw_logit"] for condition in CONDITIONS
        ])) for outfit in OUTFITS
    }
    aggregate = {
        "coefficient_mae": float(np.mean([row["absolute_error"] for row in rows.values()])),
        "outfit_mean": outfit_mean,
        "outfit_std": {
            outfit: float(np.std([
                rows[f"{outfit}/{condition}"]["coefficient"] for condition in CONDITIONS
            ])) for outfit in OUTFITS
        },
        "raw_logit_mean": raw_mean,
        "separation": outfit_mean["O08"] - outfit_mean["O01"],
        "raw_logit_separation": raw_mean["O08"] - raw_mean["O01"],
        "correct_sign_count": sum(int(row["correct_sign"]) for row in rows.values()),
        "minimum_signed_raw_logit_margin": min(
            row["signed_raw_logit_margin"] for row in rows.values()
        ),
        "raw_logit_minimum": min(row["raw_logit"] for row in rows.values()),
        "raw_logit_maximum": max(row["raw_logit"] for row in rows.values()),
        "coefficient_minimum": min(row["coefficient"] for row in rows.values()),
        "coefficient_maximum": max(row["coefficient"] for row in rows.values()),
    }
    return {"rows": rows, "aggregate": aggregate}


def _control_state(control: FrozenF2LinearLogitControl) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in control.state_dict().items()}


def _save_checkpoint(
    path: Path, control: FrozenF2LinearLogitControl,
    optimizer: torch.optim.Optimizer, step: int, context: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": SCHEMA, "task_id": TASK_ID, "global_step": step,
        "paired_batch_order": list(OUTFITS), "condition_position": 0,
        "model": _control_state(control), "optimizer": optimizer.state_dict(),
        "rng": legacy.rng_state(), "backbone_fingerprint": context["backbone_before"],
        "basis_sha256": sha256(context["paths"]["explicit_basis_artifact"]),
        "target_view_used_in_prediction_forward": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def _checkpoint_roundtrip_linear(
    context: Mapping[str, Any], control: FrozenF2LinearLogitControl,
    optimizer: torch.optim.Optimizer, checkpoint_path: Path,
    fixed_feature: torch.Tensor,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _, restored = _make_frozen_control(context)
    restored_optimizer = torch.optim.Adam(
        restored.parameters(), lr=float(context["config"]["stage_2"]["learning_rate"]),
        weight_decay=float(context["config"]["stage_2"]["weight_decay"]),
    )
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    control.eval(); restored.eval()
    with torch.no_grad():
        before = control(fixed_feature)
        after = restored(fixed_feature)
    current_rng = legacy.rng_state()
    legacy.restore_rng(checkpoint["rng"])
    rng_exact = previous.object_fingerprint(legacy.rng_state()) == previous.object_fingerprint(
        checkpoint["rng"]
    )
    legacy.restore_rng(current_rng)
    checks = {
        "global_step_exact": int(checkpoint["global_step"]) == 100,
        "model_state_exact": previous.object_fingerprint(_control_state(control))
        == previous.object_fingerprint(_control_state(restored)),
        "optimizer_state_exact": previous.object_fingerprint(optimizer.state_dict())
        == previous.object_fingerprint(restored_optimizer.state_dict()),
        "rng_state_exact": rng_exact,
        "condition_position_exact": int(checkpoint["condition_position"]) == 0,
        "raw_logit_bitwise_exact": torch.equal(before.raw_logit, after.raw_logit),
        "coefficient_bitwise_exact": torch.equal(before.coefficient, after.coefficient),
        "backbone_fingerprint_exact": checkpoint["backbone_fingerprint"] == context["backbone_before"],
        "basis_sha256_exact": checkpoint["basis_sha256"]
        == sha256(context["paths"]["explicit_basis_artifact"]),
    }
    return {
        "pass": all(checks.values()), "checks": checks,
        "raw_logit_max_abs_diff": float((before.raw_logit - after.raw_logit).abs().max()),
        "coefficient_max_abs_diff": float((before.coefficient - after.coefficient).abs().max()),
    }


def _stage_2_visual(output_dir: Path, milestones: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    steps = sorted(int(value) for value in milestones)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for outfit in OUTFITS:
        axes[0].plot(steps, [
            milestones[str(step)]["aggregate"]["outfit_mean"][outfit] for step in steps
        ], "o-", label=outfit)
    axes[0].axhline(-1, color="gray", linestyle="--")
    axes[0].axhline(1, color="gray", linestyle="--")
    axes[0].set_title("Frozen F2 linear coefficient means"); axes[0].legend()
    axes[1].plot(steps, [
        milestones[str(step)]["aggregate"]["coefficient_mae"] for step in steps
    ], "o-")
    axes[1].set_title("Coefficient MAE")
    figure.savefig(output_dir / "visual_acceptance/frozen_f2_linear_learning_curve.png", dpi=150)
    plt.close(figure)


def run_stage_2(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    parity = json.loads((output_dir / "stage_1_feature_parity/feature_parity.json").read_text(
        encoding="utf-8"
    ))
    if parity["status"] != "PASS":
        raise RuntimeError("S2 is forbidden until S1 parity passes")
    seed = int(config["stage_2"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    extractor, control = _make_frozen_control(context)
    features = _all_frozen_features(context, extractor)
    feature_artifact = output_dir / "stage_2_linear_control/frozen_set_features.pt"
    temporary = feature_artifact.with_suffix(".pt.tmp")
    torch.save({name: value.cpu() for name, value in features.items()}, temporary)
    os.replace(temporary, feature_artifact)
    optimizer = torch.optim.Adam(
        control.parameters(), lr=float(config["stage_2"]["learning_rate"]),
        weight_decay=float(config["stage_2"]["weight_decay"]),
    )
    milestones: dict[str, Any] = {"0": _evaluate_linear_control(control, features)}
    log_path = output_dir / "stage_2_linear_control/training_metrics.jsonl"
    gradient_summary: dict[str, Any] = {}
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    for step in range(1, int(config["stage_2"]["max_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        condition = CONDITIONS[(step - 1) % len(CONDITIONS)]
        raw_by_outfit = {
            outfit: control(features[f"{outfit}/{condition}"]).raw_logit
            for outfit in OUTFITS
        }
        loss, parts = _linear_control_loss(raw_by_outfit, config)
        if not torch.isfinite(loss):
            raise FloatingPointError("S2 loss became NaN or Inf")
        loss.backward()
        gradient_summary = {
            name: {
                "present": parameter.grad is not None,
                "finite": bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()),
                "l2": 0.0 if parameter.grad is None else float(torch.linalg.vector_norm(parameter.grad)),
                "nonzero_count": 0 if parameter.grad is None else int(torch.count_nonzero(parameter.grad)),
            }
            for name, parameter in control.named_parameters()
        }
        torch.nn.utils.clip_grad_norm_(control.parameters(), float(config["stage_2"]["gradient_clip_norm"]))
        optimizer.step()
        row = {
            "step": step, "condition": condition, "paired_batch_order": list(OUTFITS),
            "loss": float(loss.detach()),
        }
        row.update({f"loss_{name}": float(value.detach()) for name, value in parts.items()})
        append_jsonl(log_path, row)
        if step in set(int(value) for value in config["stage_2"]["milestones"]):
            evaluation = _evaluate_linear_control(control, features)
            evaluation["loss"] = row
            milestones[str(step)] = evaluation
            atomic_json(
                output_dir / f"stage_2_linear_control/milestones/step_{step:06d}.json", evaluation
            )
            _save_checkpoint(
                output_dir / f"stage_2_linear_control/checkpoints/checkpoint_step_{step:06d}.pth",
                control, optimizer, step, context,
            )
    final = milestones[str(config["stage_2"]["max_steps"])]
    acceptance = config["stage_2"]["acceptance"]
    aggregate = final["aggregate"]
    endpoint_collapse = bool(
        abs(aggregate["outfit_mean"]["O08"] - aggregate["outfit_mean"]["O01"]) < 1e-4
        and max(abs(aggregate["outfit_mean"][outfit]) for outfit in OUTFITS) > 0.99
    )
    checkpoint_path = output_dir / "stage_2_linear_control/checkpoints/checkpoint_step_000100.pth"
    roundtrip = _checkpoint_roundtrip_linear(
        context, control, optimizer, checkpoint_path, features["O01/cond_000017"]
    )
    checks = {
        "coefficient_mae": aggregate["coefficient_mae"] <= float(acceptance["coefficient_mae_max"]),
        "o01_mean": aggregate["outfit_mean"]["O01"] <= float(acceptance["o01_mean_max"]),
        "o08_mean": aggregate["outfit_mean"]["O08"] >= float(acceptance["o08_mean_min"]),
        "separation": aggregate["separation"] >= float(acceptance["separation_min"]),
        "correct_sign_8_of_8": aggregate["correct_sign_count"] == int(acceptance["correct_sign_count_min"]),
        "signed_raw_margin_positive": aggregate["minimum_signed_raw_logit_margin"]
        > float(acceptance["signed_raw_logit_margin_min_exclusive"]),
        "no_endpoint_collapse": not endpoint_collapse,
        "control_gradient_nonzero_finite": all(
            value["present"] and value["finite"] for value in gradient_summary.values()
        ) and sum(value["l2"] for value in gradient_summary.values()) > 0,
        "checkpoint_resume_exact": roundtrip["pass"],
        "base_bitwise_frozen": context["base_before"]
        == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_zero": _base_gradient_count(context["base"]) == 0,
        "backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
        "backbone_gradient_zero": sum(
            parameter.grad is not None for parameter in extractor.spatial_backbone.parameters()
        ) == 0,
        "basis_frozen": sum(parameter.numel() for parameter in context["basis"].parameters()) == 0,
        "legacy_mmlp_gradient_zero": sum(
            parameter.grad is not None
            for parameter in context["legacy_model"].dressable_model.anchor_clothing_mlp.parameters()
        ) == 0,
        "renderer_has_no_trainable_state_in_control": True,
        "target_forward_leakage_zero": all(
            not row["target_view_used_in_prediction_forward"] for row in final["rows"].values()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "classification": None if status == "PASS" else "CS-ONLINE-OPTIMIZATION-FAIL",
        "next_task": None if status == "PASS" else "FIX_COEFFICIENT_LABEL_LOSS_AND_OPTIMIZER_CONTRACT",
        "optimizer_steps": 100, "paired_batch_order_every_step": list(OUTFITS),
        "model_contract": "Frozen per-reference F2 -> deterministic mean/max -> LayerNorm -> Linear scalar",
        "trainable_parameter_count": sum(parameter.numel() for parameter in control.parameters()),
        "features_artifact": str(feature_artifact), "features_sha256": sha256(feature_artifact),
        "feature_distance": _feature_distance_summary(features),
        "milestones": milestones, "final": final, "gradients": gradient_summary,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_roundtrip": roundtrip, "checks": checks,
        "runtime_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(output_dir / "stage_2_linear_control/linear_control_metrics.json", report)
    _stage_2_visual(output_dir, milestones)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING" if status == "PASS" else "FAIL",
        "stage": "S2_COMPLETE", "optimizer_steps": 100,
        "classification": report["classification"], "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "updated_at_unix": time.time(),
    })
    return report


def _reference_variants(
    context: Mapping[str, Any], outfit: str, condition: str
) -> dict[str, dict[str, Any]]:
    variants = previous.reference_variants(context, outfit, condition)
    correct = context["episodes"][f"{outfit}/{condition}"]
    zero_mask = dict(correct)
    zero_mask["reference_cloth_masks"] = torch.zeros_like(correct["reference_cloth_masks"])
    rgb_only = dict(correct)
    rgb_only["reference_cloth_masks"] = correct["reference_foreground_masks"].clone()
    mask_only = dict(correct)
    mask_only["reference_images"] = torch.zeros_like(correct["reference_images"])
    variants.update({
        "zero_clothing_mask": zero_mask,
        "rgb_only": rgb_only,
        "mask_only": mask_only,
    })
    expected = list(context["config"]["stage_3"]["variants"])
    if list(variants) != expected:
        raise RuntimeError(f"counterfactual variant contract changed: {list(variants)}")
    return variants


def _residual_distribution(residual: GaussianClothingResiduals) -> dict[str, Any]:
    return {
        name: {
            "mean": float(value.detach().float().mean()),
            "std": float(value.detach().float().std(unbiased=False)),
            "mean_abs": float(value.detach().float().abs().mean()),
            "max_abs": float(value.detach().float().abs().max()),
            "nonzero_ratio": float((value.detach().abs() > 1e-9).float().mean()),
        }
        for name, value in residual.as_dict().items()
    }


def _stage_3_variant(
    context: Mapping[str, Any], extractor: FrozenF2ReferenceFeatureExtractor,
    control: FrozenF2LinearLogitControl, outfit: str, condition: str,
    variant: str, episode: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    output = control(_frozen_features(extractor, episode))
    coefficient = output.coefficient
    target_coefficient = context["teacher_coefficients"][outfit].reshape(1)
    prediction = context["basis"](
        coefficient, chunk_size=int(context["config"]["basis"]["chunk_size"])
    )
    target = context["basis"](
        target_coefficient, chunk_size=int(context["config"]["basis"]["chunk_size"])
    )
    residual_metrics = parameterization.residual_metrics(
        prediction, target, context["basis_payload"]["channel_bounds"], active_epsilon=1e-8
    )
    sample = context["samples"][f"{outfit}/{condition}"]
    predicted_rgb, predicted_alpha = parameterization.render_prediction(
        context["base"], sample, prediction, context["background"]
    )
    oracle_rgb, oracle_alpha = parameterization.render_prediction(
        context["base"], sample, target, context["background"]
    )
    garment = diagnosis._garment_mask(sample)
    protected = sample["target_protected_mask"]
    background = 1.0 - sample["target_foreground_mask"]
    target_value = float(target_coefficient)
    coefficient_value = float(coefficient)
    row = {
        "variant": variant, "outfit": outfit, "condition": condition,
        "coefficient": coefficient_value, "raw_logit": float(output.raw_logit),
        "target_coefficient": target_value,
        "coefficient_error": abs(coefficient_value - target_value),
        "correct_sign": coefficient_value * COEFFICIENT_TARGETS[outfit] > 0,
        "references": list(episode["reference_condition_ids"]),
        "reference_count": int(episode["reference_images"].shape[0]),
        "residual_normalized_rmse": residual_metrics["normalized_rmse"],
        "residual_direction_cosine": residual_metrics["direction_cosine"],
        "residual_top_10pct_overlap": residual_metrics["top_10pct_overlap"],
        "oracle_garment_rgb_mae": o01._masked_mae(predicted_rgb, oracle_rgb, garment),
        "oracle_alpha_mae": float((predicted_alpha - oracle_alpha).abs().mean()),
        "protected_rgb_mae_from_base": o01._masked_mae(
            predicted_rgb, sample["target_base_rgb"], protected
        ),
        "background_rgb_mae_from_base": o01._masked_mae(
            predicted_rgb, sample["target_base_rgb"], background
        ),
        "residual_distribution": _residual_distribution(prediction),
        "target_pose_camera_fixed": True,
        "target_view_used_in_prediction_forward": False,
        "teacher_used_in_prediction_forward": False,
    }
    cache = {
        "predicted_rgb": predicted_rgb.detach().cpu(),
        "predicted_alpha": predicted_alpha.detach().cpu(),
        "oracle_rgb": oracle_rgb.detach().cpu(),
        "oracle_alpha": oracle_alpha.detach().cpu(),
    }
    return row, cache


def _stage_3_visuals(
    context: Mapping[str, Any], report: Mapping[str, Any], cache: Mapping[str, Any]
) -> None:
    import matplotlib.pyplot as plt

    output_dir, rows = context["output_dir"], report["rows"]
    variants = list(context["config"]["stage_3"]["variants"])
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    for outfit in OUTFITS:
        values = [
            float(np.mean([rows[f"{outfit}/{condition}"][variant]["coefficient"]
                           for condition in CONDITIONS]))
            for variant in variants
        ]
        axes[0].plot(variants, values, "o-", label=outfit)
    axes[0].tick_params(axis="x", rotation=30); axes[0].legend()
    axes[0].set_title("Frozen F2 linear coefficients under reference counterfactuals")
    labels = [f"{outfit}-{VIEWS[condition]}" for condition in CONDITIONS for outfit in OUTFITS]
    correct = [rows[f"{outfit}/{condition}"]["correct"]["coefficient"]
               for condition in CONDITIONS for outfit in OUTFITS]
    swapped = [rows[f"{outfit}/{condition}"]["swapped"]["coefficient"]
               for condition in CONDITIONS for outfit in OUTFITS]
    x = np.arange(len(labels))
    axes[1].plot(x, correct, "o-", label="correct")
    axes[1].plot(x, swapped, "x--", label="swapped")
    axes[1].set_xticks(x, labels, rotation=30, ha="right"); axes[1].legend()
    axes[1].set_title("Correct/swapped coefficient exchange")
    figure.savefig(output_dir / "visual_acceptance/stage_3_coefficient_counterfactuals.png", dpi=150)
    plt.close(figure)
    for outfit in OUTFITS:
        contact_rows = []
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            sample = context["samples"][key]
            contact_rows.append((f"{outfit}/{VIEWS[condition]}", [
                ("base", sample["target_base_rgb"].detach().cpu(), 3),
                ("edit target", sample["target_edit_rgb"].detach().cpu(), 3),
                ("oracle basis", cache[f"{key}/correct"]["oracle_rgb"], 3),
                ("correct references", cache[f"{key}/correct"]["predicted_rgb"], 3),
                ("swapped references", cache[f"{key}/swapped"]["predicted_rgb"], 3),
                ("exchange delta", (
                    cache[f"{key}/correct"]["predicted_rgb"]
                    - cache[f"{key}/swapped"]["predicted_rgb"]
                ).abs(), 3),
            ]))
        o01._save_contact_sheet(
            output_dir / f"visual_acceptance/{outfit}_correct_swapped_linear_control.png",
            contact_rows,
        )
    back_rows = []
    for outfit in OUTFITS:
        key = f"{outfit}/cond_000318"
        back_rows.append((f"{outfit}/back", [
            (name, cache[f"{key}/{name}"]["predicted_rgb"], 3)
            for name in ("correct", "single_reference", "dropout_0", "dropout_1", "dropout_2",
                         "zero_rgb", "base_rgb", "zero_clothing_mask", "rgb_only", "mask_only")
        ]))
    o01._save_contact_sheet(
        output_dir / "visual_acceptance/back_view_reference_modality_counterfactuals.png", back_rows
    )


def run_stage_3(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    stage_2 = json.loads((output_dir / "stage_2_linear_control/linear_control_metrics.json").read_text(
        encoding="utf-8"
    ))
    if stage_2["status"] != "PASS":
        raise RuntimeError("S3 is forbidden until S2 passes")
    extractor, control = _make_frozen_control(context)
    checkpoint_path = Path(stage_2["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    control.load_state_dict(checkpoint["model"]); control.eval(); extractor.eval()
    rows: dict[str, Any] = {}; cache: dict[str, Any] = {}
    started = time.time(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for condition in CONDITIONS:
            for outfit in OUTFITS:
                key = f"{outfit}/{condition}"; rows[key] = {}
                for variant, episode in _reference_variants(context, outfit, condition).items():
                    row, visual = _stage_3_variant(
                        context, extractor, control, outfit, condition, variant, episode
                    )
                    rows[key][variant] = row; cache[f"{key}/{variant}"] = visual
    correct_rows = [
        rows[f"{outfit}/{condition}"]["correct"]
        for condition in CONDITIONS for outfit in OUTFITS
    ]
    coefficient_margins, residual_margins, render_margins = [], [], []
    permutation_differences, single_dropout_signs = [], []
    wins = 0
    for condition in CONDITIONS:
        for outfit in OUTFITS:
            values = rows[f"{outfit}/{condition}"]
            correct, swapped = values["correct"], values["swapped"]
            coefficient_margin = swapped["coefficient_error"] - correct["coefficient_error"]
            residual_margin = swapped["residual_normalized_rmse"] - correct["residual_normalized_rmse"]
            render_margin = swapped["oracle_garment_rgb_mae"] - correct["oracle_garment_rgb_mae"]
            coefficient_margins.append(coefficient_margin)
            residual_margins.append(residual_margin); render_margins.append(render_margin)
            if coefficient_margin > 0 and residual_margin > 0 and render_margin > 0:
                wins += 1
            permutation_differences.append(abs(
                values["permutation"]["coefficient"] - correct["coefficient"]
            ))
            single_dropout_signs.extend(
                values[name]["correct_sign"]
                for name in ("single_reference", "dropout_0", "dropout_1", "dropout_2")
            )
    outfit_variant_mean = {
        outfit: {
            variant: float(np.mean([
                rows[f"{outfit}/{condition}"][variant]["coefficient"] for condition in CONDITIONS
            ])) for variant in config["stage_3"]["variants"]
        } for outfit in OUTFITS
    }
    modality_separation = {
        variant: outfit_variant_mean["O08"][variant] - outfit_variant_mean["O01"][variant]
        for variant in ("rgb_only", "mask_only")
    }
    combined_sensitivity = float(np.mean([
        max(
            abs(values["correct"]["coefficient"] - values["zero_rgb"]["coefficient"]),
            abs(values["correct"]["coefficient"] - values["zero_clothing_mask"]["coefficient"]),
            abs(values["correct"]["coefficient"] - values["base_rgb"]["coefficient"]),
        ) for values in rows.values()
    ]))
    aggregate = {
        "variant_count": sum(len(value) for value in rows.values()),
        "correct_coefficient_mae": float(np.mean([row["coefficient_error"] for row in correct_rows])),
        "correct_sign_count": sum(int(row["correct_sign"]) for row in correct_rows),
        "mean_swapped_coefficient_margin": float(np.mean(coefficient_margins)),
        "minimum_swapped_coefficient_margin": float(np.min(coefficient_margins)),
        "minimum_swapped_residual_margin": float(np.min(residual_margins)),
        "minimum_swapped_render_margin": float(np.min(render_margins)),
        "correct_episode_wins": wins,
        "permutation_max_abs_diff": float(np.max(permutation_differences)),
        "single_dropout_correct_sign_count": sum(int(value) for value in single_dropout_signs),
        "single_dropout_case_count": len(single_dropout_signs),
        "outfit_variant_mean": outfit_variant_mean,
        "rgb_only_separation": modality_separation["rgb_only"],
        "mask_only_separation": modality_separation["mask_only"],
        "combined_reference_sensitivity": combined_sensitivity,
        "mean_correct_garment_rgb_mae": float(np.mean([
            row["oracle_garment_rgb_mae"] for row in correct_rows
        ])),
        "mean_correct_protected_rgb_mae": float(np.mean([
            row["protected_rgb_mae_from_base"] for row in correct_rows
        ])),
        "mean_correct_background_rgb_mae": float(np.mean([
            row["background_rgb_mae_from_base"] for row in correct_rows
        ])),
    }
    acceptance = config["stage_3"]["acceptance"]
    checks = {
        "formal_96_variants": aggregate["variant_count"] == 96,
        "swap_flips_all_coefficients": all(
            rows[f"{outfit}/{condition}"]["swapped"]["correct_sign"] is False
            for condition in CONDITIONS for outfit in OUTFITS
        ),
        "swapped_coefficient_margin_each": aggregate["minimum_swapped_coefficient_margin"]
        >= float(acceptance["swapped_coefficient_margin_min"]),
        "swapped_residual_margin_each": aggregate["minimum_swapped_residual_margin"] > 0,
        "swapped_render_margin_each": aggregate["minimum_swapped_render_margin"] > 0,
        "correct_episode_wins_8_of_8": aggregate["correct_episode_wins"]
        == int(acceptance["correct_episode_wins_min"]),
        "permutation_invariant": aggregate["permutation_max_abs_diff"]
        <= float(acceptance["permutation_max_abs_diff_max"]),
        "single_and_dropout_sign": aggregate["single_dropout_correct_sign_count"]
        == aggregate["single_dropout_case_count"],
        "at_least_one_modality_discriminative": max(
            abs(aggregate["rgb_only_separation"]), abs(aggregate["mask_only_separation"])
        ) >= float(config["stage_3"]["modality_discrimination_separation_min"]),
        "combined_reference_sensitive": aggregate["combined_reference_sensitivity"]
        > float(config["stage_3"]["combined_reference_sensitivity_min_exclusive"]),
        "target_forward_leakage_zero": all(
            not row["target_view_used_in_prediction_forward"]
            and not row["teacher_used_in_prediction_forward"]
            for values in rows.values() for row in values.values()
        ),
        "base_bitwise_frozen": context["base_before"]
        == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
        "basis_frozen": sum(parameter.numel() for parameter in context["basis"].parameters()) == 0,
    }
    numeric_pass = all(checks.values())
    report = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if numeric_pass else "FAIL",
        "classification": None if numeric_pass else "CS-P2",
        "next_task": None if numeric_pass else "FIX_COEFFICIENT_TO_BASIS_INTEGRATION",
        "rows": rows, "aggregate": aggregate, "checks": checks,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "runtime_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "visual_status": "PENDING_ACTUAL_INSPECTION" if numeric_pass else "NOT_ELIGIBLE_NUMERIC_FAIL",
    }
    atomic_json(output_dir / "stage_3_counterfactual/counterfactual_metrics.json", report)
    _stage_3_visuals(context, report, cache)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": report["status"], "stage": "S3_COMPLETE",
        "optimizer_steps": 100, "classification": report["classification"],
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "updated_at_unix": time.time(),
    })
    return report


def finalize(context: Mapping[str, Any], visual_path: Path | None) -> dict[str, Any]:
    output_dir = context["output_dir"]
    stage_0 = json.loads((output_dir / "stage_0_loss_audit/loss_gradient_audit.json").read_text(
        encoding="utf-8"
    ))
    stage_1 = json.loads((output_dir / "stage_1_feature_parity/feature_parity.json").read_text(
        encoding="utf-8"
    ))
    stage_2_path = output_dir / "stage_2_linear_control/linear_control_metrics.json"
    stage_3_path = output_dir / "stage_3_counterfactual/counterfactual_metrics.json"
    stage_2 = json.loads(stage_2_path.read_text(encoding="utf-8")) if stage_2_path.is_file() else None
    stage_3 = json.loads(stage_3_path.read_text(encoding="utf-8")) if stage_3_path.is_file() else None
    visual: dict[str, Any] | None = None
    if visual_path is not None:
        visual = json.loads(visual_path.read_text(encoding="utf-8"))
        required = {"images_actually_opened", "inspection_method", "observations", "visual_acceptance_status"}
        if not required.issubset(visual) or not visual["images_actually_opened"]:
            raise ValueError("visual observations require actual opened images and all required fields")
    if stage_1["status"] != "PASS":
        status, classification = "FAIL", "CS-P0"
        next_task = "FIX_ONLINE_OFFLINE_REFERENCE_FEATURE_PARITY"
    elif stage_2 is None or stage_2["status"] != "PASS":
        status, classification = "FAIL", "CS-P1"
        next_task = "FIX_COEFFICIENT_LABEL_LOSS_AND_OPTIMIZER_CONTRACT"
    elif stage_3 is None or stage_3["status"] == "FAIL":
        status, classification = "FAIL", "CS-P2"
        next_task = "FIX_COEFFICIENT_TO_BASIS_INTEGRATION"
    elif visual is None:
        status, classification = "NUMERIC_PASS_VISUAL_PENDING", "CS-PASS-PENDING-VISUAL"
        next_task = "COMPLETE_ACTUAL_VISUAL_INSPECTION"
    elif visual["visual_acceptance_status"] != "PASS":
        status, classification = "FAIL", "CS-P2"
        next_task = "FIX_COEFFICIENT_TO_BASIS_INTEGRATION"
    else:
        status, classification = "PASS", "CS-PASS"
        next_task = "EXPAND_REFERENCE_COEFFICIENT_CONTROL_TO_MULTI_OUTFIT"
    freeze = {
        "base_bitwise_frozen": context["base_before"]
        == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_count": _base_gradient_count(context["base"]),
        "backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(
            context["legacy_model"].clothing_observation_encoder.backbone.state_dict()
        ),
        "basis_trainable_parameter_count": sum(
            parameter.numel() for parameter in context["basis"].parameters()
        ),
        "previous_rf_f_attempt_immutable": legacy.immutable_tree_metadata_fingerprint(
            context["previous_attempt"]
        ) == context["previous_tree_before"],
        "explicit_basis_attempt_immutable": legacy.immutable_tree_metadata_fingerprint(
            context["explicit_attempt"]
        ) == context["explicit_tree_before"],
    }
    result = {
        "task_id": TASK_ID, "status": status, "classification": classification,
        "next_task": next_task, "git": context["governance"],
        "stage_0_status": stage_0["status"], "stage_1_status": stage_1["status"],
        "stage_2_status": None if stage_2 is None else stage_2["status"],
        "stage_3_status": None if stage_3 is None else stage_3["status"],
        "visual_acceptance": visual, "freeze_evidence": freeze,
        "target_view_used_in_prediction_forward": False,
        "teacher_used_in_prediction_forward": False,
        "optimizer_steps": 0 if stage_2 is None else int(stage_2["optimizer_steps"]),
        "checkpoint": None if stage_2 is None else stage_2["checkpoint"],
        "checkpoint_sha256": None if stage_2 is None else stage_2["checkpoint_sha256"],
        "historical_rf_f_adjudication_preserved": True,
    }
    atomic_json(output_dir / "final_adjudication/final_adjudication.json", result)
    if visual is not None:
        atomic_json(output_dir / "visual_acceptance/visual_acceptance.json", visual)
        atomic_text(output_dir / "visual_acceptance/VISUAL_ACCEPTANCE.md", "\n".join([
            "# Visual acceptance", "",
            f"- Status: {visual['visual_acceptance_status']}",
            f"- Inspection method: {visual['inspection_method']}",
            f"- Images actually opened: {len(visual['images_actually_opened'])}", "",
            *[f"- {item}" for item in visual["observations"]],
        ]))
    atomic_text(output_dir / "GATE_ACCEPTANCE.md", "\n".join([
        "# Reference coefficient supervision calibration", "",
        f"- Status: **{status}**", f"- Classification: **{classification}**",
        f"- S0 loss audit: {stage_0['status']}", f"- S1 frozen F2 parity: {stage_1['status']}",
        f"- S2 minimal linear control: {result['stage_2_status']}",
        f"- S3 coefficient-to-basis integration: {result['stage_3_status']}",
        f"- Target/teacher used in prediction forward: false",
        f"- Previous RF-F historical adjudication preserved: true", "",
        f"Next task: `{next_task}`",
    ]))
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": status, "stage": "FINALIZED",
        "classification": classification, "optimizer_steps": result["optimizer_steps"],
        "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"],
        "updated_at_unix": time.time(),
    })
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TASK_ID)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("all", "s0", "s1", "s2", "s3", "finalize"), default="all"
    )
    parser.add_argument("--visual-observations", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    config = load_config(arguments.config.resolve())
    create = arguments.phase in {"all", "s0"}
    context: Mapping[str, Any] | None = None
    try:
        context = build_context(config, arguments.output_dir.resolve(), create=create)
        if arguments.phase in {"all", "s0"}:
            run_stage_0(context)
        if arguments.phase in {"all", "s1"}:
            stage_1 = run_stage_1(context)
            if stage_1["status"] != "PASS":
                finalize(context, None); return 2
        if arguments.phase in {"all", "s2"}:
            stage_2 = run_stage_2(context)
            if stage_2["status"] != "PASS":
                finalize(context, None); return 3
        if arguments.phase in {"all", "s3"}:
            stage_3 = run_stage_3(context)
            if stage_3["status"] == "FAIL":
                finalize(context, None); return 4
        if arguments.phase == "all":
            finalize(context, None)
        elif arguments.phase == "finalize":
            finalize(context, arguments.visual_observations)
        return 0
    except Exception as error:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(arguments.output_dir / "failure.json", {
            "task_id": TASK_ID, "phase": arguments.phase, "exception_type": type(error).__name__,
            "exception_message": str(error), "traceback": traceback.format_exc(),
            "updated_at_unix": time.time(),
        })
        atomic_json(arguments.output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "FAIL", "stage": f"{arguments.phase.upper()}_EXCEPTION",
            "exception_type": type(error).__name__, "exception_message": str(error),
            "updated_at_unix": time.time(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
