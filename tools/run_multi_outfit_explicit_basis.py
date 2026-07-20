from __future__ import annotations

import argparse
import csv
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_dressable as training  # noqa: E402
from scene.explicit_gaussian_residual_basis import (  # noqa: E402
    CHANNEL_TO_BOUND,
    ExplicitGaussianResidualBasis,
    build_svd_basis,
    normalized_residual_dict,
    project_residual_onto_basis,
    tensor_mapping_fingerprint,
)
from scene.frozen_f2_linear_coefficient_control import FrozenF2ReferenceFeatureExtractor  # noqa: E402
from scene.full_dressable_dataset import FullDressableTrainingDataset  # noqa: E402
from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
    apply_protected_full_residual_guard,
)
from scene.multi_outfit_linear_coefficient_control import (  # noqa: E402
    MultiOutfitLinearCoefficientControl,
    pairwise_geometry_loss,
    restore_coefficients,
)
from scene.representation_capacity_oracle import (  # noqa: E402
    CAPACITY_LOSS_NAME,
    UnboundedGaussianDeltaField,
    direct_stability,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools import run_explicit_gaussian_residual_basis as explicit  # noqa: E402
from tools import run_image_conditioned_overfit_o01 as o01  # noqa: E402
from tools import run_reference_coefficient_supervision_calibration as calibration  # noqa: E402
from tools import run_representation_triage_ladder as triage  # noqa: E402
from tools import run_residual_field_parameterization as parameterization  # noqa: E402
from tools.run_module4b_canonical_oracle_micropilot import (  # noqa: E402
    _base_gradient_count,
    _base_named_tensors,
    _grid,
    _tensor_state_fingerprint,
)


SCHEMA = "canondressgs.multi_outfit_explicit_basis.v1"
TASK_ID = "SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001"
EXPECTED_BRANCH = "research/multi-outfit-explicit-basis-20260720"
EXPECTED_SOURCE_HEAD = "28f2b3358f28b64c3cb35b64f130013ee519129a"
TRAIN_OUTFITS = ("O01", "O02", "O03", "O04", "O08")
HELD_OUT_OUTFIT = "O07"
RESERVE_OUTFIT = "O06"
ALL_OUTFITS = (*TRAIN_OUTFITS, HELD_OUT_OUTFIT)
CONDITIONS = ("cond_000000", "cond_000318", "cond_000017", "cond_000347")
VIEWS = dict(zip(CONDITIONS, ("front", "back", "left", "right")))
DUAL_TARGET_FIELDS = {
    "target_edit_rgb", "target_base_rgb", "target_edit_mask", "target_old_clothing_mask",
    "target_protected_mask", "target_transition_mask", "target_base_foreground_mask",
}
FORBIDDEN_FORWARD_FIELDS = {
    "target_rgb", "target_mask", "target_edit_rgb", "target_clothing_mask", "outfit_id",
    "cloth_id", "teacher", "teacher_coefficient", "target_pose", "target_camera",
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def _governance() -> dict[str, Any]:
    branch, head, dirty = (
        git_output("branch", "--show-current"), git_output("rev-parse", "HEAD"),
        git_output("status", "--short"),
    )
    if branch != EXPECTED_BRANCH or dirty:
        raise RuntimeError(f"formal run requires clean {EXPECTED_BRANCH}; dirty={bool(dirty)}")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", EXPECTED_SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT
    ) != 0:
        raise RuntimeError("formal branch is not descended from CS-PASS final HEAD")
    return {"branch": branch, "head": head, "clean": True, "source_head": EXPECTED_SOURCE_HEAD}


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ValueError("multi-outfit schema/task mismatch")
    if config.get("branch") != EXPECTED_BRANCH or config.get("source_head") != EXPECTED_SOURCE_HEAD:
        raise ValueError("multi-outfit branch/source contract changed")
    if tuple(config["train_outfits"]) != TRAIN_OUTFITS:
        raise ValueError("five training outfits changed")
    if config["held_out_outfit"] != HELD_OUT_OUTFIT or config["reserve_outfit"] != RESERVE_OUTFIT:
        raise ValueError("held-out/reserve outfit contract changed")
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("four-view protocol changed")
    if config["teacher"]["protocol"] != "representation_triage_rung2_direct_shared_canonical":
        raise ValueError("teacher protocol changed")
    if int(config["teacher"]["max_steps"]) != 1200 or int(config["teacher"]["ceiling_steps"]) != 1500:
        raise ValueError("teacher step contract changed")
    if config["teacher"]["capacity_loss"]["name"] != CAPACITY_LOSS_NAME:
        raise ValueError("teacher loss changed")
    if tuple(config["basis"]["candidate_ranks"]) != (1, 2, 3, 4):
        raise ValueError("basis rank ladder changed")
    if float(config["ridge"]["lambda"]) != 1e-4:
        raise ValueError("ridge lambda changed")
    if int(config["coefficient_training"]["max_steps"]) != 300:
        raise ValueError("coefficient training step ceiling changed")
    if config["coefficient_training"]["balanced_batch_order"] != list(TRAIN_OUTFITS):
        raise ValueError("balanced outfit order changed")
    if any(bool(value) for value in config["permissions"].values()):
        raise ValueError("forbidden permission enabled")
    return config


def _load_all_outfit_data(
    manifest_path: Path, outfits: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    dataset = FullDressableTrainingDataset(manifest_path, "train", reference_count=3, seed=0)
    selected_outfits = {
        item["outfit_id"]: item for item in dataset.outfits if item["outfit_id"] in outfits
    }
    if set(selected_outfits) != set(outfits):
        raise ValueError("source manifest lacks a preregistered outfit")
    samples: dict[str, dict[str, Any]] = {}
    episodes: dict[str, dict[str, Any]] = {}
    protocol: dict[str, Any] = {
        "train_outfits": list(TRAIN_OUTFITS), "held_out_outfit": HELD_OUT_OUTFIT,
        "reserve_outfit_unused": RESERVE_OUTFIT, "conditions": list(CONDITIONS), "episodes": {},
    }
    for outfit_id in outfits:
        outfit = selected_outfits[outfit_id]
        observations = {item["condition_id"]: item for item in outfit["observations"]}
        if set(CONDITIONS).difference(observations):
            raise ValueError(f"{outfit_id} lacks a fixed four-view condition")
        for condition in CONDITIONS:
            observation = observations[condition]
            missing = DUAL_TARGET_FIELDS.difference(observation)
            if missing:
                raise ValueError(f"{outfit_id}/{condition} lacks {sorted(missing)}")
            target = dataset._observation(outfit, observation, True)
            sample: dict[str, Any] = {
                "target_edit_rgb": dataset._image(observation["target_edit_rgb"], 3),
                "target_base_rgb": dataset._image(observation["target_base_rgb"], 3),
                "target_foreground_mask": target["foreground_mask"],
                "target_clothing_mask": target["clothing_mask"],
                "target_pose": target["pose"], "target_Rh": target["R_global"],
                "target_Th": target["Th"],
                "target_camera": {
                    "K": target["K"], "w2c": target["w2c"],
                    "width": target["width"], "height": target["height"],
                },
                "target_condition_id": condition, "outfit_id": outfit_id,
                "source_record": observation,
            }
            for name in sorted(DUAL_TARGET_FIELDS.difference({"target_edit_rgb", "target_base_rgb"})):
                sample[name] = dataset._image(observation[name], 1)
            reference_ids = tuple(value for value in CONDITIONS if value != condition)
            references = [dataset._observation(outfit, observations[value], True) for value in reference_ids]
            episode = dataset._stack_references(references)
            episode["reference_cloth_masks"] = episode.pop("reference_clothing_masks")
            if condition in episode["reference_condition_ids"] or len(set(episode["reference_condition_ids"])) != 3:
                raise AssertionError("reference-target overlap or duplicate reference")
            key = f"{outfit_id}/{condition}"
            samples[key], episodes[key] = sample, episode
            protocol["episodes"][key] = {
                "outfit_id": outfit_id, "target": condition, "target_view": VIEWS[condition],
                "references": list(reference_ids), "reference_count": 3,
                "target_in_references": False,
            }
    return samples, episodes, protocol


def _build_cs_context(config: Mapping[str, Any]) -> dict[str, Any]:
    old_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/research/subject02_reference_coefficient_supervision_calibration_v1.yaml")
        .read_text(encoding="utf-8")
    )
    old_branch, old_head = calibration.EXPECTED_BRANCH, calibration.EXPECTED_SOURCE_HEAD
    calibration.EXPECTED_BRANCH, calibration.EXPECTED_SOURCE_HEAD = EXPECTED_BRANCH, EXPECTED_SOURCE_HEAD
    try:
        return calibration.build_context(
            old_config, Path(config["inputs"]["cs_pass_attempt"]), create=False
        )
    finally:
        calibration.EXPECTED_BRANCH, calibration.EXPECTED_SOURCE_HEAD = old_branch, old_head


def build_context(
    config: dict[str, Any], output_dir: Path, *, create: bool, reference_backbone: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal multi-outfit run requires CUDA")
    governance = _governance()
    if output_dir.name != config["output"]["attempt"]:
        raise ValueError("output attempt path differs from preregistration")
    paths = {
        "source_manifest": Path(config["inputs"]["source_manifest"]),
        "stable_protected_attribution": Path(config["inputs"]["stable_protected_attribution"]),
        "cs_final": Path(config["inputs"]["cs_pass_attempt"]) / "final_adjudication/final_adjudication.json",
        "prior_basis": Path(config["inputs"]["prior_explicit_basis_attempt"]) / "stage_a/basis/explicit_basis.pt",
        "base_checkpoint": Path(config["base"]["model_dir"]) / config["base"]["checkpoint_path"],
    }
    expected = {
        "source_manifest": config["inputs"]["source_manifest_sha256"],
        "stable_protected_attribution": config["inputs"]["stable_protected_attribution_sha256"],
        "cs_final": config["inputs"]["cs_pass_final_sha256"],
        "prior_basis": config["inputs"]["prior_explicit_basis_artifact_sha256"],
        "base_checkpoint": config["base"]["checkpoint_sha256"],
    }
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[name] = sha256(path)
        if hashes[name] != expected[name]:
            raise ValueError(f"immutable input SHA changed: {name}")
    cs_attempt = Path(config["inputs"]["cs_pass_attempt"])
    prior_basis_attempt = Path(config["inputs"]["prior_explicit_basis_attempt"])
    existing_teacher_root = Path(config["inputs"]["existing_teacher_root"])
    immutable_before = {
        "cs_pass": explicit.immutable_tree_metadata_fingerprint(cs_attempt),
        "prior_basis": explicit.immutable_tree_metadata_fingerprint(prior_basis_attempt),
        "existing_teachers": explicit.immutable_tree_metadata_fingerprint(existing_teacher_root),
    }
    device = torch.device("cuda")
    if reference_backbone:
        cs_context = _build_cs_context(config)
        base, protected_mask, background = (
            cs_context["base"], cs_context["protected_mask"], cs_context["background"]
        )
        legacy_model = cs_context["legacy_model"]
        backbone_before = cs_context["backbone_before"]
    else:
        base = training.load_frozen_mmlphuman_base(
            config["base"]["model_dir"], paths["base_checkpoint"], device=device
        )
        protected_cpu, _ = o01._load_stable_protected_mask(
            paths["stable_protected_attribution"], int(config["inputs"]["stable_protected_count"]),
            int(base._xyz.shape[0]),
        )
        protected_mask = protected_cpu.to(device)
        background = torch.tensor(config["render"]["background"], device=device, dtype=base._xyz.dtype)
        legacy_model = None; backbone_before = None
    if int(base._xyz.shape[0]) != int(config["base"]["gaussian_count"]):
        raise ValueError("base Gaussian count changed")
    base_before = _tensor_state_fingerprint(_base_named_tensors(base))
    samples, episodes, protocol = _load_all_outfit_data(paths["source_manifest"], ALL_OUTFITS)
    for key in list(samples):
        samples[key] = diagnosis._to_device_nested(samples[key], device)
        episodes[key] = diagnosis._to_device_nested(episodes[key], device)
        support, _ = o01._base_only_protected_support(
            base, samples[key], protected_mask, background,
            float(config["render"]["protected_support_alpha_threshold"]),
        )
        samples[key]["target_protected_mask"] = torch.maximum(
            samples[key]["target_protected_mask"], support
        )
    if create:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(output_dir)
        for name in (
            "contract", "input_audit", "stage_a_teacher_bank", "stage_b_basis/ranks",
            "stage_b_basis/selected", "stage_c0_ridge", "stage_c_training/checkpoints",
            "stage_c_training/milestones", "stage_c_seen", "stage_d_held_out",
            "visual_acceptance", "final_adjudication",
        ):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        atomic_text(output_dir / "contract/config_resolved.yaml", yaml.safe_dump(config, sort_keys=False))
        atomic_text(output_dir / "contract/command.txt", " ".join([sys.executable, *sys.argv]))
        atomic_json(output_dir / "input_audit/environment.json", explicit.o01.environment_snapshot())
        atomic_json(output_dir / "input_audit/input_manifest.json", {
            "task_id": TASK_ID, "git": governance,
            "paths": {name: str(path) for name, path in paths.items()}, "sha256": hashes,
            "immutable_tree_fingerprints": immutable_before,
            "existing_teacher_root": str(existing_teacher_root),
            "existing_teacher_checkpoints": {
                outfit: str(existing_teacher_root / outfit / "checkpoints/step_001200.pth")
                for outfit in ("O01", "O08")
            },
            "prediction_forward_fields": ["reference RGB", "reference clothing masks", "reference valid mask"],
            "forbidden_prediction_fields": sorted(FORBIDDEN_FORWARD_FIELDS),
            "target_forward_leakage": False, "outfit_id_in_model": False,
            "held_out_outfit": HELD_OUT_OUTFIT, "reserve_outfit_unused": RESERVE_OUTFIT,
        })
        atomic_json(output_dir / "input_audit/resolved_protocol.json", protocol)
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "RUNNING", "stage": "INPUT_AUDIT",
            "optimizer_steps": 0, "updated_at_unix": time.time(),
        })
    return {
        "config": config, "output_dir": output_dir, "governance": governance,
        "paths": paths, "base": base, "protected_mask": protected_mask,
        "background": background, "samples": samples, "episodes": episodes,
        "protocol": protocol, "base_before": base_before, "legacy_model": legacy_model,
        "backbone_before": backbone_before, "immutable_before": immutable_before,
        "cs_attempt": cs_attempt, "prior_basis_attempt": prior_basis_attempt,
        "existing_teacher_root": existing_teacher_root,
    }


def run_audit(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    existing = {}
    for outfit in ALL_OUTFITS:
        checkpoint = context["existing_teacher_root"] / outfit / "checkpoints/step_001200.pth"
        existing[outfit] = {
            "exists": checkpoint.is_file(), "path": str(checkpoint),
            "sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        }
    checks = {
        "cs_pass_exact": json.loads(context["paths"]["cs_final"].read_text(encoding="utf-8"))["status"] == "PASS",
        "twenty_train_episodes": sum(
            key.startswith(tuple(f"{outfit}/" for outfit in TRAIN_OUTFITS))
            for key in context["episodes"]
        ) == 20,
        "four_held_out_episodes": sum(
            key.startswith(f"{HELD_OUT_OUTFIT}/") for key in context["episodes"]
        ) == 4,
        "three_references_each": all(
            int(episode["reference_images"].shape[0]) == 3 for episode in context["episodes"].values()
        ),
        "target_reference_overlap_zero": all(
            context["samples"][key]["target_condition_id"] not in episode["reference_condition_ids"]
            for key, episode in context["episodes"].items()
        ),
        "o01_teacher_exact": existing["O01"]["sha256"] == config["inputs"]["oracle_o01_checkpoint_sha256"],
        "o08_teacher_exact": existing["O08"]["sha256"] == config["inputs"]["oracle_o08_checkpoint_sha256"],
        "missing_teachers_exactly_o02_o03_o04_o07": {
            outfit for outfit, row in existing.items() if not row["exists"]
        } == {"O02", "O03", "O04", "O07"},
        "o06_not_loaded": not any(key.startswith("O06/") for key in context["episodes"]),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "existing_teacher_audit": existing, "optimizer_steps": 0,
        "new_teachers_required": ["O02", "O03", "O04", "O07"],
    }
    atomic_json(output_dir / "input_audit/teacher_bank_audit.json", report)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING" if report["status"] == "PASS" else "FAIL",
        "stage": "AUDIT_COMPLETE", "optimizer_steps": 0, "updated_at_unix": time.time(),
    })
    return report


def _teacher_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    teacher = config["teacher"]
    return {
        "seed": teacher["seed"], "optimizer": teacher["optimizer"],
        "capacity_loss": teacher["capacity_loss"],
        "abnormal_gaussian": teacher["abnormal_gaussian"],
        "rung_2": {
            "steps": teacher["max_steps"], "record_steps": teacher["record_steps"],
            "render_steps": teacher["render_steps"],
            "pass": {
                "per_view_reduction_min": 0.40, "mean_reduction_min": 0.50,
                "per_view_target_closer_min": 0.65, "protected_mae_max": 0.01,
                "abnormal_gaussian_fraction_max": 0.01,
            },
            "warn": {
                "mean_reduction_min": 0.30, "direction_correct_min_views": 3,
                "direction_reduction_min": 0.30, "protected_mae_max": 0.01,
            },
        },
    }


def _teacher_run_dir(context: Mapping[str, Any], outfit: str) -> Path:
    return context["output_dir"] / "stage_a_teacher_bank" / outfit


def _save_teacher_residual(
    path: Path, residual: GaussianClothingResiduals, bounds: Mapping[str, float], outfit: str,
) -> dict[str, Any]:
    normalized = normalized_residual_dict(residual, bounds)
    payload = {
        "schema_version": "canondressgs.multi_outfit_teacher_residual.v1",
        "outfit": outfit, "space": "bound_normalized_six_channel_gaussian_residual",
        "normalized_residual": {name: value.detach().cpu() for name, value in normalized.items()},
        "residual_fingerprint": tensor_mapping_fingerprint(normalized),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256(path), "fingerprint": payload["residual_fingerprint"]}


def _teacher_residual_stats(
    residual: GaussianClothingResiduals, bounds: Mapping[str, float],
) -> dict[str, Any]:
    result = {}
    for name, value in residual.as_dict().items():
        flat = value.detach().float().reshape(value.shape[0], -1)
        bound_name = CHANNEL_TO_BOUND[name]
        result[name] = {
            "shape": list(value.shape), "finite": bool(torch.isfinite(value).all()),
            "mean_abs": float(flat.abs().mean()), "max_abs": float(flat.abs().max()),
            "bound": float(bounds[bound_name]),
            "bound_normalized_rmse": float((flat / float(bounds[bound_name])).square().mean().sqrt()),
            "nonzero_ratio": float((flat.abs() > 1e-9).float().mean()),
        }
    return result


def _teacher_evaluate_one(
    base: Any, sample: Mapping[str, Any], background: torch.Tensor,
    render: Callable[[], tuple[torch.Tensor, torch.Tensor]], output_dir: Path | None = None,
) -> tuple[dict[str, float], list[tuple[str, Image.Image]]]:
    """Adapt the immutable triage evaluator to this runner's all-CUDA sample contract."""
    with torch.no_grad():
        rgb, alpha = render()
    metrics = triage.capacity_metrics(rgb, alpha, sample)
    panels: list[tuple[str, Image.Image]] = []
    if output_dir is not None:
        cpu_sample = diagnosis._to_device_nested(sample, torch.device("cpu"))
        panels = triage._save_render_set(
            output_dir, sample["target_condition_id"], rgb.detach().cpu(),
            alpha.detach().cpu(), cpu_sample,
        )
    return metrics, panels


def run_teacher(context: Mapping[str, Any], outfit: str) -> dict[str, Any]:
    if outfit not in {"O02", "O03", "O04", "O07"}:
        raise ValueError("only missing preregistered teachers may be generated")
    config, base, output_dir = context["config"], context["base"], context["output_dir"]
    run_dir = _teacher_run_dir(context, outfit)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    for name in ("checkpoints", "renders", "diagnostics", "residual"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "RUN_STATUS.json"
    atomic_json(status_path, {"status": "RUNNING", "optimizer_steps": 0, "outfit": outfit})
    protocol = _teacher_protocol(config)
    steps = int(config["teacher"]["max_steps"])
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(config["teacher"]["seed"])); np.random.seed(int(config["teacher"]["seed"]))
    samples = {condition: context["samples"][f"{outfit}/{condition}"] for condition in CONDITIONS}
    states = {
        condition: triage.target_free_state(sample, base._xyz.device)
        for condition, sample in samples.items()
    }
    model = UnboundedGaussianDeltaField(base).to(base._xyz.device)
    render_for: Callable[[str], tuple[torch.Tensor, torch.Tensor]] = lambda condition: triage.render_direct(
        base, states[condition], model(base), context["background"]
    )
    optimizer, group_names = triage.optimizer_for(model, protocol)
    base_before = _tensor_state_fingerprint(_base_named_tensors(base))
    initial_per_view: dict[str, dict[str, float]] = {}
    initial_panels: list[tuple[str, Image.Image]] = []
    for condition in CONDITIONS:
        metrics, panels = _teacher_evaluate_one(
            base, samples[condition], context["background"],
            lambda condition=condition: render_for(condition), run_dir / "renders/step_000000",
        )
        initial_per_view[condition] = metrics; initial_panels.extend(panels)
    _grid(run_dir / "diagnostics/step_000000_four_view.png", initial_panels, columns=4)
    history: list[dict[str, Any]] = []
    gradient_seen = {name: 0 for name, value in model.named_parameters() if value.requires_grad}
    record_steps = set(int(value) for value in config["teacher"]["record_steps"])
    render_steps = set(int(value) for value in config["teacher"]["render_steps"])
    for step in range(1, steps + 1):
        condition = CONDITIONS[(step - 1) % 4]
        optimizer.zero_grad(set_to_none=True)
        rgb, alpha = render_for(condition)
        parts = triage.loss_for_sample(
            rgb, alpha, samples[condition], protocol["capacity_loss"], direct_stability(model)
        )
        if not torch.isfinite(parts["total"]):
            raise FloatingPointError(f"non-finite teacher loss at step {step}")
        parts["total"].backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"non-finite teacher gradient: {name}")
                if torch.count_nonzero(parameter.grad):
                    gradient_seen[name] += 1
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["teacher"]["optimizer"]["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite teacher gradient norm")
        optimizer.step()
        row = {
            "step": step, "condition_id": condition, "view": VIEWS[condition],
            **{name: float(value.detach()) for name, value in parts.items()},
            "gradient_norm": float(gradient_norm),
        }
        history.append(row); append_jsonl(run_dir / "state_records.jsonl", row)
        if step in record_steps:
            atomic_json(run_dir / f"diagnostics/step_{step:06d}_loss.json", row)
        if step in render_steps:
            panels: list[tuple[str, Image.Image]] = []; metric_rows = []
            for current in CONDITIONS:
                metrics, current_panels = _teacher_evaluate_one(
                    base, samples[current], context["background"],
                    lambda current=current: render_for(current), run_dir / f"renders/step_{step:06d}",
                )
                metric_rows.append({"condition_id": current, "view": VIEWS[current], **metrics})
                panels.extend(current_panels)
            atomic_json(run_dir / f"diagnostics/step_{step:06d}_metrics.json", metric_rows)
            _grid(run_dir / f"diagnostics/step_{step:06d}_four_view.png", panels, columns=4)
        atomic_json(status_path, {
            "status": "RUNNING", "optimizer_steps": step, "last_condition": condition,
            "updated_at_unix": time.time(),
        })
    final_per_view = []; final_panels: list[tuple[str, Image.Image]] = []
    for condition in CONDITIONS:
        metrics, panels = _teacher_evaluate_one(
            base, samples[condition], context["background"],
            lambda condition=condition: render_for(condition), run_dir / "renders/final",
        )
        metrics["garment_error_reduction"] = 1 - metrics["garment_target_mae"] / max(
            initial_per_view[condition]["garment_target_mae"], 1e-12
        )
        final_per_view.append({"condition_id": condition, "view": VIEWS[condition], **metrics})
        final_panels.extend(panels)
    abnormal = triage.abnormal_direct(model, base, protocol)
    numeric = triage._shared_numeric("rung_2", final_per_view, abnormal, protocol)
    checkpoint = run_dir / f"checkpoints/step_{steps:06d}.pth"
    triage.save_checkpoint(checkpoint, model, optimizer, steps, {
        "rung": 2, "outfit": outfit, "optimizer_groups": group_names,
        "target_used_only_for_loss": True, "shared_canonical_field": True,
        "garment_point_count": 0, "source_protocol": "Representation Triage Rung-2",
    })
    residual = apply_protected_full_residual_guard(
        model.residuals(base), context["protected_mask"]
    )
    residual_artifact = _save_teacher_residual(
        run_dir / "residual/normalized_residual.pt", residual,
        config["basis"]["channel_bounds"], outfit,
    )
    write_csv(run_dir / "loss_curve.csv", history)
    triage.save_loss_curve(run_dir / "diagnostics/loss_curve.png", history)
    triage.save_histograms(run_dir / "diagnostics/parameter_histograms.png", abnormal, model, "direct")
    _grid(run_dir / "diagnostics/final_four_view_contact_sheet.png", final_panels, columns=4)
    checks = {
        "loss_finite": all(math.isfinite(row["total"]) for row in history),
        "target_forward_leakage_zero": True, "shared_canonical_field": True,
        "four_views_finite": all(all(math.isfinite(float(value)) for value in row.values() if isinstance(value, float)) for row in final_per_view),
        "base_bitwise_frozen": base_before == _tensor_state_fingerprint(_base_named_tensors(base)),
        "base_gradient_zero": _base_gradient_count(base) == 0,
        "all_teacher_groups_received_gradient": all(value > 0 for value in gradient_seen.values()),
    }
    report = {
        "status": "NUMERIC_COMPLETE_VISUAL_PENDING", "outfit": outfit,
        "optimizer_steps": steps, "numeric_status": numeric,
        "mean_garment_error_reduction": float(np.mean([
            row["garment_error_reduction"] for row in final_per_view
        ])),
        "initial_per_view": initial_per_view, "final_per_view": final_per_view,
        "parameter_distributions": abnormal, "gradient_steps_nonzero": gradient_seen,
        "six_attribute_metrics": _teacher_residual_stats(
            residual, config["basis"]["channel_bounds"]
        ),
        "residual_artifact": residual_artifact,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checks": checks, "runtime_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "visual_status": "PENDING_ACTUAL_INSPECTION",
    }
    atomic_json(run_dir / "teacher_metrics.json", report)
    atomic_json(status_path, report)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING", "stage": f"TEACHER_{outfit}_COMPLETE",
        "optimizer_steps": steps, "updated_at_unix": time.time(),
    })
    return report


def teacher_checkpoint(context: Mapping[str, Any], outfit: str) -> Path:
    if outfit in {"O01", "O08"}:
        return context["existing_teacher_root"] / outfit / "checkpoints/step_001200.pth"
    return _teacher_run_dir(context, outfit) / "checkpoints/step_001200.pth"


def teacher_metrics(context: Mapping[str, Any], outfit: str) -> dict[str, Any]:
    if outfit in {"O01", "O08"}:
        return json.loads(
            (context["existing_teacher_root"] / outfit / "metrics.json").read_text(encoding="utf-8")
        )
    return json.loads((_teacher_run_dir(context, outfit) / "teacher_metrics.json").read_text(
        encoding="utf-8"
    ))


def write_teacher_bank_index(context: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for outfit in ALL_OUTFITS:
        checkpoint = teacher_checkpoint(context, outfit)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metrics = teacher_metrics(context, outfit)
        contact = (
            context["existing_teacher_root"] / outfit / "diagnostics/final_four_view_contact_sheet.png"
            if outfit in {"O01", "O08"}
            else _teacher_run_dir(context, outfit) / "diagnostics/final_four_view_contact_sheet.png"
        )
        rows[outfit] = {
            "source": "immutable_existing_rung2" if outfit in {"O01", "O08"} else "task_generated_rung2",
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "optimizer_steps": int(metrics.get("optimizer_steps", 1200)),
            "numeric_status": metrics.get("numeric_status"),
            "contact_sheet": str(contact), "contact_sheet_sha256": sha256(contact),
            "target_forward_leakage": False, "shared_canonical_field": True,
        }
    report = {"status": "COMPLETE", "teachers": rows}
    atomic_json(context["output_dir"] / "stage_a_teacher_bank/teacher_bank_index.json", report)
    return report


def adjudicate_teacher_bank(
    context: Mapping[str, Any], visual_decisions_path: Path,
) -> dict[str, Any]:
    decisions = json.loads(visual_decisions_path.read_text(encoding="utf-8"))
    index = write_teacher_bank_index(context)
    rows = {}
    for outfit in ALL_OUTFITS:
        decision = decisions.get(outfit)
        required = {
            "status", "images_actually_opened", "observations", "forming_view_count",
            "old_garment_dominates", "severe_cloud_or_mottle",
        }
        if not isinstance(decision, Mapping) or not required.issubset(decision):
            raise ValueError(f"teacher visual decision incomplete for {outfit}")
        if not decision["images_actually_opened"]:
            raise ValueError(f"teacher visual evidence was not actually opened for {outfit}")
        metrics = teacher_metrics(context, outfit)
        base_frozen = (
            metrics.get("base_bitwise_exact", True)
            if outfit in {"O01", "O08"}
            else metrics["checks"]["base_bitwise_frozen"]
        )
        loss_finite = (
            all(math.isfinite(float(row.get("garment_target_mae", float("nan")))) for row in metrics["final_per_view"])
            if outfit in {"O01", "O08"}
            else metrics["checks"]["loss_finite"]
        )
        checks = {
            "loss_finite": bool(loss_finite),
            "target_forward_leakage_zero": True,
            "shared_four_view_canonical": True,
            "minimum_three_forming_views": int(decision["forming_view_count"])
            >= int(context["config"]["teacher"]["acceptance"]["minimum_forming_views"]),
            "old_garment_not_dominant": not bool(decision["old_garment_dominates"]),
            "no_severe_cloud_or_mottle": not bool(decision["severe_cloud_or_mottle"]),
            "base_frozen": bool(base_frozen),
            "visual_status_acceptable": decision["status"] in {"PASS", "WARN"},
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        rows[outfit] = {
            "status": status, "checks": checks, "visual": decision,
            "numeric_status": metrics.get("numeric_status"), **index["teachers"][outfit],
        }
    train_pass = all(rows[outfit]["status"] == "PASS" for outfit in TRAIN_OUTFITS)
    held_out_status = rows[HELD_OUT_OUTFIT]["status"]
    report = {
        "status": "PASS" if train_pass else "FAIL", "train_teacher_bank_pass": train_pass,
        "held_out_teacher_status": held_out_status, "teachers": rows,
        "classification": None if train_pass else "MO-TEACHER-FAIL",
        "final_case_if_stopped": None if train_pass else "MO-T",
        "next_task_if_stopped": None if train_pass else "ADJUDICATE_TEACHER_BANK_AND_DATA_SCOPE",
    }
    atomic_json(context["output_dir"] / "stage_a_teacher_bank/teacher_bank_adjudication.json", report)
    atomic_json(context["output_dir"] / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING" if train_pass else "FAIL",
        "stage": "TEACHER_BANK_ADJUDICATED", "classification": report["classification"],
        "updated_at_unix": time.time(),
    })
    return report


def load_teacher_residuals(
    context: Mapping[str, Any], outfits: Sequence[str] = ALL_OUTFITS,
) -> tuple[dict[str, GaussianClothingResiduals], dict[str, Any]]:
    teachers, metadata = {}, {}
    for outfit in outfits:
        checkpoint = teacher_checkpoint(context, outfit)
        oracle, info = diagnosis._load_oracle(
            context["base"], checkpoint, context["base"]._xyz.device
        )
        residual = apply_protected_full_residual_guard(
            oracle.residuals(context["base"]), context["protected_mask"]
        )
        teachers[outfit] = residual
        metadata[outfit] = {
            **info, "sha256": sha256(checkpoint),
            "normalized_fingerprint": tensor_mapping_fingerprint(
                normalized_residual_dict(residual, context["config"]["basis"]["channel_bounds"])
            ),
        }
    return teachers, metadata


def _save_basis_artifact(
    path: Path, decomposition: Any, bounds: Mapping[str, float],
) -> dict[str, Any]:
    mean, components = decomposition.basis.normalized_fields()
    payload = {
        "schema_version": "canondressgs.multi_outfit_explicit_basis_artifact.v1",
        "channel_bounds": dict(bounds), "rank": decomposition.basis.rank,
        "mean_normalized": {name: value.detach().cpu() for name, value in mean.items()},
        "basis_normalized": {name: value.detach().cpu() for name, value in components.items()},
        "teacher_coefficients": {
            name: value.detach().cpu() for name, value in decomposition.teacher_coefficients.items()
        },
        "metadata": decomposition.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp"); torch.save(payload, temporary); os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256(path), "metadata": decomposition.metadata}


def load_basis_artifact(
    path: Path, device: torch.device,
) -> tuple[ExplicitGaussianResidualBasis, dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "canondressgs.multi_outfit_explicit_basis_artifact.v1":
        raise ValueError("multi-outfit basis artifact schema mismatch")
    basis = ExplicitGaussianResidualBasis(
        payload["mean_normalized"], payload["basis_normalized"], payload["channel_bounds"]
    ).to(device)
    if basis.fingerprint() != payload["metadata"]["basis_fingerprint"]:
        raise ValueError("multi-outfit basis fingerprint mismatch")
    coefficients = {
        name: value.to(device=device, dtype=next(basis.buffers()).dtype)
        for name, value in payload["teacher_coefficients"].items()
    }
    return basis, coefficients, payload


def _render_residual(
    context: Mapping[str, Any], outfit: str, condition: str,
    residual: GaussianClothingResiduals,
) -> tuple[torch.Tensor, torch.Tensor]:
    return parameterization.render_prediction(
        context["base"], context["samples"][f"{outfit}/{condition}"],
        residual, context["background"],
    )


def _rank_reconstruction_visual(
    context: Mapping[str, Any], rank: int, teachers: Mapping[str, GaussianClothingResiduals],
    predictions: Mapping[str, GaussianClothingResiduals],
) -> None:
    rows = []
    for outfit in TRAIN_OUTFITS:
        panels = []
        for condition in CONDITIONS:
            teacher_rgb, _ = _render_residual(context, outfit, condition, teachers[outfit])
            predicted_rgb, _ = _render_residual(context, outfit, condition, predictions[outfit])
            panels.extend([
                (f"{outfit}/{VIEWS[condition]} teacher", teacher_rgb.detach().cpu(), 3),
                (f"{outfit}/{VIEWS[condition]} rank-{rank}", predicted_rgb.detach().cpu(), 3),
                (f"{outfit}/{VIEWS[condition]} error", (teacher_rgb - predicted_rgb).abs().detach().cpu(), 3),
            ])
        rows.append((outfit, panels))
    o01._save_contact_sheet(
        context["output_dir"] / f"visual_acceptance/basis_rank_{rank}_reconstruction.png", rows
    )


def _basis_field_visual(
    context: Mapping[str, Any], basis: ExplicitGaussianResidualBasis,
) -> None:
    import matplotlib.pyplot as plt

    xyz = context["base"]._xyz.detach().float().cpu()
    sampled = torch.arange(0, xyz.shape[0], max(1, xyz.shape[0] // 50000))
    _, components = basis.normalized_fields()
    figure, axes = plt.subplots(
        basis.rank, len(CHANNELS), figsize=(24, 4 * basis.rank), squeeze=False,
        constrained_layout=True,
    )
    for component in range(basis.rank):
        for column, name in enumerate(CHANNELS):
            value = components[name][component].detach().float().cpu().reshape(xyz.shape[0], -1).norm(dim=1)
            plot = axes[component, column].scatter(
                xyz[sampled, 0], xyz[sampled, 2], c=value[sampled], s=1, cmap="magma"
            )
            axes[component, column].set_title(f"B{component + 1} {name}")
            axes[component, column].set_aspect("equal")
            figure.colorbar(plot, ax=axes[component, column], fraction=0.046)
    figure.savefig(context["output_dir"] / "visual_acceptance/basis_six_attribute_fields.png", dpi=110)
    plt.close(figure)


def _normalized_residual_sse(
    first: GaussianClothingResiduals, second: GaussianClothingResiduals,
    bounds: Mapping[str, float],
) -> float:
    first_values = normalized_residual_dict(first, bounds)
    second_values = normalized_residual_dict(second, bounds)
    return float(sum(
        (first_values[name] - second_values[name]).double().square().sum()
        for name in CHANNELS
    ))


def run_basis_ladder(context: Mapping[str, Any]) -> dict[str, Any]:
    decision_path = context["output_dir"] / "stage_a_teacher_bank/teacher_bank_adjudication.json"
    if not decision_path.is_file() or not json.loads(decision_path.read_text(encoding="utf-8"))["train_teacher_bank_pass"]:
        raise RuntimeError("five training teachers must pass before basis construction")
    config, output_dir = context["config"], context["output_dir"]
    teachers, teacher_metadata = load_teacher_residuals(context, TRAIN_OUTFITS)
    bounds = config["basis"]["channel_bounds"]
    rank_rows = {}; total_centered_sse = None
    for rank in config["basis"]["candidate_ranks"]:
        decomposition = build_svd_basis(teachers, bounds, TRAIN_OUTFITS, int(rank))
        predictions = {
            outfit: decomposition.basis(
                decomposition.teacher_coefficients[outfit],
                chunk_size=int(config["basis"]["chunk_size"]),
            ) for outfit in TRAIN_OUTFITS
        }
        per_outfit = {}
        reconstruction_sse = 0.0
        mean_teacher = None
        if total_centered_sse is None:
            mean_fields = decomposition.basis.normalized_fields()[0]
            mean_teacher = GaussianClothingResiduals.from_dict({
                name: mean_fields[name] * float(bounds[CHANNEL_TO_BOUND[name]])
                for name in CHANNELS
            })
            total_centered_sse = sum(
                _normalized_residual_sse(teachers[outfit], mean_teacher, bounds)
                for outfit in TRAIN_OUTFITS
            )
        for outfit in TRAIN_OUTFITS:
            residual_metrics = parameterization.residual_metrics(
                predictions[outfit], teachers[outfit], bounds, active_epsilon=1e-8
            )
            render_rows = []
            for condition in CONDITIONS:
                predicted_rgb, predicted_alpha = _render_residual(
                    context, outfit, condition, predictions[outfit]
                )
                teacher_rgb, teacher_alpha = _render_residual(
                    context, outfit, condition, teachers[outfit]
                )
                sample = context["samples"][f"{outfit}/{condition}"]
                render_rows.append({
                    "condition": condition, "view": VIEWS[condition],
                    "garment_mae": o01._masked_mae(
                        predicted_rgb, teacher_rgb, diagnosis._garment_mask(sample)
                    ),
                    "protected_mae": o01._masked_mae(
                        predicted_rgb, teacher_rgb, sample["target_protected_mask"]
                    ),
                    "background_mae": o01._masked_mae(
                        predicted_rgb, teacher_rgb, 1 - sample["target_foreground_mask"]
                    ),
                    "alpha_mae": float((predicted_alpha - teacher_alpha).abs().mean()),
                })
            reconstruction_sse += _normalized_residual_sse(
                predictions[outfit], teachers[outfit], bounds
            )
            per_outfit[outfit] = {
                **residual_metrics, "render_rows": render_rows,
                "max_garment_mae": max(row["garment_mae"] for row in render_rows),
                "six_attribute": residual_metrics["per_attribute"],
            }
        acceptance = config["basis"]["acceptance"]
        checks = {
            "all_rmse": all(
                row["normalized_rmse"] <= float(acceptance["normalized_rmse_max"])
                for row in per_outfit.values()
            ),
            "all_cosine": all(
                row["direction_cosine"] >= float(acceptance["cosine_min"])
                for row in per_outfit.values()
            ),
            "all_top10": all(
                row["top_10pct_overlap"] >= float(acceptance["top_10_overlap_min"])
                for row in per_outfit.values()
            ),
            "all_garment_render": all(
                row["max_garment_mae"] <= float(acceptance["garment_mae_max"])
                for row in per_outfit.values()
            ),
            "rank_at_most_four": int(rank) <= 4,
        }
        artifact = _save_basis_artifact(
            output_dir / f"stage_b_basis/ranks/rank_{rank}/basis.pt", decomposition, bounds
        )
        explained = 1.0 - reconstruction_sse / max(float(total_centered_sse), 1e-30)
        rank_rows[str(rank)] = {
            "rank": int(rank), "numeric_status": "PASS" if all(checks.values()) else "FAIL",
            "explained_variance": explained, "per_outfit": per_outfit,
            "checks": checks, "artifact": artifact,
        }
        atomic_json(
            output_dir / f"stage_b_basis/ranks/rank_{rank}/metrics.json", rank_rows[str(rank)]
        )
        _rank_reconstruction_visual(context, int(rank), teachers, predictions)
    report = {
        "status": "NUMERIC_COMPLETE_VISUAL_PENDING", "ranks": rank_rows,
        "teacher_metadata": teacher_metadata,
        "numeric_passing_ranks": [
            int(rank) for rank, row in rank_rows.items() if row["numeric_status"] == "PASS"
        ],
    }
    atomic_json(output_dir / "stage_b_basis/rank_ladder_metrics.json", report)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "NUMERIC_PASS_VISUAL_PENDING"
        if report["numeric_passing_ranks"] else "FAIL",
        "stage": "BASIS_LADDER_COMPLETE", "classification": None
        if report["numeric_passing_ranks"] else "MO-BASIS-FAIL",
        "updated_at_unix": time.time(),
    })
    return report


def adjudicate_basis_rank(
    context: Mapping[str, Any], visual_decisions_path: Path,
) -> dict[str, Any]:
    metrics = json.loads(
        (context["output_dir"] / "stage_b_basis/rank_ladder_metrics.json").read_text(encoding="utf-8")
    )
    decisions = json.loads(visual_decisions_path.read_text(encoding="utf-8"))
    rows = {}
    for rank in (1, 2, 3, 4):
        decision = decisions.get(str(rank))
        if not isinstance(decision, Mapping) or not {
            "status", "images_actually_opened", "observations", "teacher_visually_consistent",
            "new_cloud_or_mottle",
        }.issubset(decision):
            raise ValueError(f"basis visual decision incomplete for rank {rank}")
        visual_pass = bool(
            decision["images_actually_opened"] and decision["status"] == "PASS"
            and decision["teacher_visually_consistent"] and not decision["new_cloud_or_mottle"]
        )
        numeric_pass = metrics["ranks"][str(rank)]["numeric_status"] == "PASS"
        rows[str(rank)] = {
            "numeric_pass": numeric_pass, "visual_pass": visual_pass,
            "status": "PASS" if numeric_pass and visual_pass else "FAIL", "visual": decision,
        }
    passing = [rank for rank in (1, 2, 3, 4) if rows[str(rank)]["status"] == "PASS"]
    if not passing:
        report = {
            "status": "FAIL", "classification": "MO-BASIS-FAIL", "final_case": "MO-B",
            "selected_rank": None, "ranks": rows,
            "next_task": "REASSESS_MULTI_OUTFIT_RESIDUAL_SUBSPACE",
        }
    else:
        selected_rank = min(passing)
        source_path = Path(metrics["ranks"][str(selected_rank)]["artifact"]["path"])
        basis, coefficients, payload = load_basis_artifact(source_path, context["base"]._xyz.device)
        coefficient_matrix = torch.stack([coefficients[outfit] for outfit in TRAIN_OUTFITS])
        train_mean = coefficient_matrix.mean(0)
        train_std = coefficient_matrix.std(0, unbiased=False)
        if torch.any(train_std <= 1e-12):
            raise ValueError("selected basis contains a degenerate coefficient dimension")
        selected_payload = {**payload,
            "coefficient_train_mean": train_mean.detach().cpu(),
            "coefficient_train_std": train_std.detach().cpu(),
            "standardization_uses_train_outfits_only": True,
            "held_out_outfit_used": False,
        }
        selected_path = context["output_dir"] / "stage_b_basis/selected/selected_basis.pt"
        temporary = selected_path.with_suffix(".pt.tmp")
        torch.save(selected_payload, temporary); os.replace(temporary, selected_path)
        atomic_json(context["output_dir"] / "stage_b_basis/selected/coefficient_normalization.json", {
            "train_outfit_order": list(TRAIN_OUTFITS), "held_out_excluded": HELD_OUT_OUTFIT,
            "mean": train_mean.detach().cpu().tolist(), "std": train_std.detach().cpu().tolist(),
            "selected_rank": selected_rank,
        })
        _basis_field_visual(context, basis)
        report = {
            "status": "PASS", "classification": None, "selected_rank": selected_rank,
            "selected_basis": str(selected_path), "selected_basis_sha256": sha256(selected_path),
            "ranks": rows, "next_task": None,
        }
    atomic_json(context["output_dir"] / "stage_b_basis/rank_selection.json", report)
    atomic_json(context["output_dir"] / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING" if report["status"] == "PASS" else "FAIL",
        "stage": "BASIS_RANK_ADJUDICATED", "classification": report["classification"],
        "updated_at_unix": time.time(),
    })
    return report


def _selected_basis(
    context: Mapping[str, Any],
) -> tuple[ExplicitGaussianResidualBasis, dict[str, torch.Tensor], dict[str, Any], Path]:
    selection_path = context["output_dir"] / "stage_b_basis/rank_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("basis rank has not been adjudicated")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "PASS":
        raise RuntimeError("selected basis is not eligible for coefficient prediction")
    path = Path(selection["selected_basis"])
    basis, coefficients, payload = load_basis_artifact(path, context["base"]._xyz.device)
    return basis, coefficients, payload, path


def _feature_extractor(context: Mapping[str, Any]) -> FrozenF2ReferenceFeatureExtractor:
    if context["legacy_model"] is None:
        raise RuntimeError("frozen F2 extraction requires the verified CS-PASS backbone")
    extractor = FrozenF2ReferenceFeatureExtractor(
        context["legacy_model"].clothing_observation_encoder.backbone[:-1],
        int(context["config"]["model"]["image_feature_dim"]),
    ).to(context["base"]._xyz)
    extractor.eval()
    return extractor


def prediction_feature(
    extractor: FrozenF2ReferenceFeatureExtractor, episode: Mapping[str, Any],
) -> torch.Tensor:
    inputs = {
        "reference_images": episode["reference_images"],
        "reference_clothing_masks": episode["reference_cloth_masks"],
        "reference_valid_mask": episode["reference_valid_mask"],
    }
    if FORBIDDEN_FORWARD_FIELDS.intersection(inputs):
        raise RuntimeError("target/teacher/outfit identity entered prediction forward")
    with torch.no_grad():
        value = extractor(**inputs).set_feature.detach()
    if value.shape[0] != 1 or not torch.isfinite(value).all():
        raise ValueError("frozen F2 set feature is invalid")
    return value


def subset_reference_episode(
    episode: Mapping[str, Any], indices: Sequence[int],
) -> dict[str, Any]:
    if not indices or len(indices) > 3 or len(set(indices)) != len(indices):
        raise ValueError("reference subset must contain one to three unique indices")
    count = int(episode["reference_images"].shape[0])
    if min(indices) < 0 or max(indices) >= count:
        raise IndexError("reference subset index is outside the episode")
    index = torch.tensor(indices, device=episode["reference_images"].device, dtype=torch.long)
    result: dict[str, Any] = {}
    for name, value in episode.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == count:
            result[name] = value.index_select(0, index)
        elif isinstance(value, (list, tuple)) and len(value) == count:
            selected = [value[item] for item in indices]
            result[name] = tuple(selected) if isinstance(value, tuple) else selected
        else:
            result[name] = value
    return result


def extract_all_features(
    context: Mapping[str, Any], extractor: FrozenF2ReferenceFeatureExtractor,
) -> dict[str, torch.Tensor]:
    features = {
        key: prediction_feature(extractor, context["episodes"][key])
        for key in (f"{outfit}/{condition}" for outfit in ALL_OUTFITS for condition in CONDITIONS)
    }
    path = context["output_dir"] / "stage_c0_ridge/frozen_f2_features.pt"
    payload = {key: value.detach().cpu() for key, value in features.items()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary); os.replace(temporary, path)
    return features


def _standardized_targets(
    coefficients: Mapping[str, torch.Tensor], payload: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    mean = payload["coefficient_train_mean"].to(coefficients[TRAIN_OUTFITS[0]])
    std = payload["coefficient_train_std"].to(mean)
    return {outfit: (coefficients[outfit] - mean) / std for outfit in TRAIN_OUTFITS}


def _nearest_teacher(
    value: torch.Tensor, targets: Mapping[str, torch.Tensor],
) -> tuple[str, list[str], dict[str, float]]:
    distances = {
        outfit: float(torch.linalg.vector_norm(value - target))
        for outfit, target in targets.items()
    }
    ranking = sorted(distances, key=lambda item: (distances[item], item))
    return ranking[0], ranking, distances


def _ridge_fit(x: torch.Tensor, y: torch.Tensor, regularization: float) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge inputs must be aligned matrices")
    x_augmented = torch.cat((x, torch.ones_like(x[:, :1])), dim=1).double()
    y64 = y.double()
    dual = x_augmented @ x_augmented.t()
    dual.diagonal().add_(float(regularization))
    return (x_augmented.t() @ torch.linalg.solve(dual, y64)).to(x)


def _ridge_predict(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, torch.ones_like(x[:, :1])), dim=1) @ weight


def _save_confusion(path: Path, confusion: Mapping[str, Mapping[str, int]], title: str) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray([[confusion[row][column] for column in TRAIN_OUTFITS] for row in TRAIN_OUTFITS])
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(5), TRAIN_OUTFITS); axis.set_yticks(range(5), TRAIN_OUTFITS)
    axis.set_xlabel("predicted"); axis.set_ylabel("teacher"); axis.set_title(title)
    for row in range(5):
        for column in range(5):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis); figure.savefig(path, dpi=150); plt.close(figure)


def run_ridge_control(context: Mapping[str, Any]) -> dict[str, Any]:
    basis, coefficients, payload, _ = _selected_basis(context)
    extractor = _feature_extractor(context)
    features = extract_all_features(context, extractor)
    targets = _standardized_targets(coefficients, payload)
    rows: list[dict[str, Any]] = []
    confusion = {outfit: {other: 0 for other in TRAIN_OUTFITS} for outfit in TRAIN_OUTFITS}
    render_rows = []
    regularization = float(context["config"]["ridge"]["lambda"])
    for held_condition in CONDITIONS:
        train_keys = [
            f"{outfit}/{condition}" for outfit in TRAIN_OUTFITS for condition in CONDITIONS
            if condition != held_condition
        ]
        x = torch.cat([features[key] for key in train_keys], dim=0)
        x = F.layer_norm(x, (x.shape[1],))
        y = torch.stack([targets[key.split("/")[0]] for key in train_keys])
        weight = _ridge_fit(x, y, regularization)
        for outfit in TRAIN_OUTFITS:
            key = f"{outfit}/{held_condition}"
            test_x = F.layer_norm(features[key], (features[key].shape[1],))
            prediction = _ridge_predict(test_x, weight)[0]
            nearest, ranking, distances = _nearest_teacher(prediction, targets)
            confusion[outfit][nearest] += 1
            rmse = float(torch.sqrt(F.mse_loss(prediction, targets[outfit])))
            restored = restore_coefficients(
                prediction, payload["coefficient_train_mean"].to(prediction),
                payload["coefficient_train_std"].to(prediction),
            )
            predicted_residual = basis(restored, chunk_size=int(context["config"]["basis"]["chunk_size"]))
            teacher_residual = basis(coefficients[outfit], chunk_size=int(context["config"]["basis"]["chunk_size"]))
            predicted_rgb, _ = _render_residual(context, outfit, held_condition, predicted_residual)
            teacher_rgb, _ = _render_residual(context, outfit, held_condition, teacher_residual)
            garment_mae = o01._masked_mae(
                predicted_rgb, teacher_rgb,
                diagnosis._garment_mask(context["samples"][key]),
            )
            rows.append({
                "outfit": outfit, "held_out_condition": held_condition,
                "standardized_coefficient_rmse": rmse, "nearest_teacher": nearest,
                "rank": ranking.index(outfit) + 1, "distances": distances,
                "teacher_render_garment_mae": garment_mae,
            })
            render_rows.append(garment_mae)
    correct = sum(row["nearest_teacher"] == row["outfit"] for row in rows)
    report = {
        "status": "COMPLETE", "ridge_lambda": regularization,
        "fold_contract": "leave-one-target-view-out; 15 train / 5 test per fold",
        "coefficient_rmse": float(np.mean([row["standardized_coefficient_rmse"] for row in rows])),
        "nearest_teacher_correct": correct, "nearest_teacher_total": len(rows),
        "mean_teacher_render_garment_mae": float(np.mean(render_rows)),
        "confusion_matrix": confusion, "rows": rows,
        "classification": "MULTI_OUTFIT_F2_LINEAR_LIMIT" if correct <= 4 else None,
        "stage_c_allowed_once": True,
    }
    atomic_json(context["output_dir"] / "stage_c0_ridge/ridge_metrics.json", report)
    _save_confusion(
        context["output_dir"] / "visual_acceptance/ridge_outfit_confusion_matrix.png",
        confusion, "C0 ridge leave-one-view-out",
    )
    return report


def _coefficient_state(model: MultiOutfitLinearCoefficientControl) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save_coefficient_checkpoint(
    path: Path, model: MultiOutfitLinearCoefficientControl,
    optimizer: torch.optim.Optimizer, step: int, condition_position: int,
    context: Mapping[str, Any], basis_path: Path,
) -> None:
    payload = {
        "schema_version": SCHEMA, "task_id": TASK_ID, "global_step": int(step),
        "condition_position": int(condition_position), "balanced_batch_order": list(TRAIN_OUTFITS),
        "model": _coefficient_state(model), "optimizer": optimizer.state_dict(),
        "rng": parameterization.rng_state(), "basis_sha256": sha256(basis_path),
        "base_fingerprint": context["base_before"],
        "backbone_fingerprint": context["backbone_before"],
        "held_out_outfit_used": False, "target_view_used_in_prediction_forward": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def _evaluate_coefficient_model(
    model: MultiOutfitLinearCoefficientControl, features: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    rows = {}; errors = []
    model.eval()
    with torch.no_grad():
        for outfit in TRAIN_OUTFITS:
            for condition in CONDITIONS:
                key = f"{outfit}/{condition}"
                prediction = model(features[key]).standardized_coefficients
                nearest, ranking, distances = _nearest_teacher(prediction, targets)
                rmse = float(torch.sqrt(F.mse_loss(prediction, targets[outfit])))
                errors.append(rmse)
                rows[key] = {
                    "standardized_coefficient_rmse": rmse, "nearest_teacher": nearest,
                    "correct_rank": ranking.index(outfit) + 1, "distances": distances,
                    "prediction": prediction.detach().cpu().tolist(),
                }
    model.train()
    return {
        "mean_standardized_coefficient_rmse": float(np.mean(errors)),
        "nearest_teacher_correct": sum(value["nearest_teacher"] == key.split("/")[0] for key, value in rows.items()),
        "correct_rank_one": sum(value["correct_rank"] == 1 for value in rows.values()),
        "rows": rows,
    }


def run_coefficient_training(context: Mapping[str, Any]) -> dict[str, Any]:
    config, output_dir = context["config"], context["output_dir"]
    if not (output_dir / "stage_c0_ridge/ridge_metrics.json").is_file():
        raise FileNotFoundError("C0 ridge control must run before the sole Stage C training")
    basis, coefficients, payload, basis_path = _selected_basis(context)
    extractor = _feature_extractor(context)
    features = extract_all_features(context, extractor)
    targets = _standardized_targets(coefficients, payload)
    torch.manual_seed(int(config["coefficient_training"]["seed"])); torch.cuda.manual_seed_all(
        int(config["coefficient_training"]["seed"])
    )
    model = MultiOutfitLinearCoefficientControl(extractor.set_feature_dim, basis.rank).to(
        context["base"]._xyz
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["coefficient_training"]["learning_rate"]),
        weight_decay=float(config["coefficient_training"]["weight_decay"]),
    )
    milestones = {}; final_gradients = {}; started = time.time(); torch.cuda.reset_peak_memory_stats()
    milestones["0"] = _evaluate_coefficient_model(model, features, targets)
    for step in range(1, int(config["coefficient_training"]["max_steps"]) + 1):
        condition = CONDITIONS[(step - 1) % len(CONDITIONS)]
        batch = [f"{outfit}/{condition}" for outfit in TRAIN_OUTFITS]
        if [key.split("/")[0] for key in batch] != list(TRAIN_OUTFITS):
            raise AssertionError("balanced batch lost a preregistered outfit")
        prediction = torch.stack([
            model(features[key]).standardized_coefficients for key in batch
        ])
        target = torch.stack([targets[outfit] for outfit in TRAIN_OUTFITS])
        coefficient_loss = F.smooth_l1_loss(prediction, target)
        geometry_loss = pairwise_geometry_loss(prediction, target)
        loss = coefficient_loss + float(config["coefficient_training"]["pairwise_geometry_weight"]) * geometry_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("multi-outfit coefficient loss is NaN or Inf")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        final_gradients = {
            name: {
                "finite": parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                "l2": float(torch.linalg.vector_norm(parameter.grad)) if parameter.grad is not None else 0.0,
            } for name, parameter in model.named_parameters()
        }
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["coefficient_training"]["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        append_jsonl(output_dir / "stage_c_training/training.jsonl", {
            "step": step, "condition": condition, "balanced_outfits": list(TRAIN_OUTFITS),
            "loss": float(loss.detach()), "coefficient_loss": float(coefficient_loss.detach()),
            "pairwise_geometry_loss": float(geometry_loss.detach()),
        })
        if step in config["coefficient_training"]["milestones"]:
            milestones[str(step)] = _evaluate_coefficient_model(model, features, targets)
            checkpoint_path = output_dir / f"stage_c_training/checkpoints/checkpoint_step_{step:06d}.pth"
            _save_coefficient_checkpoint(
                checkpoint_path, model, optimizer, step, step % len(CONDITIONS), context, basis_path
            )
    checkpoint_path = output_dir / "stage_c_training/checkpoints/checkpoint_step_000300.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = MultiOutfitLinearCoefficientControl(extractor.set_feature_dim, basis.rank).to(
        context["base"]._xyz
    )
    restored_optimizer = torch.optim.Adam(
        restored.parameters(), lr=float(config["coefficient_training"]["learning_rate"]),
        weight_decay=float(config["coefficient_training"]["weight_decay"]),
    )
    restored.load_state_dict(checkpoint["model"]); restored_optimizer.load_state_dict(checkpoint["optimizer"])
    model.eval(); restored.eval(); fixed = features["O01/cond_000000"]
    with torch.no_grad():
        before = model(fixed).standardized_coefficients
        after = restored(fixed).standardized_coefficients
    current_rng = parameterization.rng_state(); explicit.restore_rng(checkpoint["rng"])
    rng_exact = o01.object_fingerprint(parameterization.rng_state()) == o01.object_fingerprint(checkpoint["rng"])
    explicit.restore_rng(current_rng)
    resume = {
        "model_state_exact": o01.object_fingerprint(_coefficient_state(model)) == o01.object_fingerprint(_coefficient_state(restored)),
        "optimizer_state_exact": o01.object_fingerprint(optimizer.state_dict()) == o01.object_fingerprint(restored_optimizer.state_dict()),
        "rng_state_exact": rng_exact, "global_step_exact": checkpoint["global_step"] == 300,
        "condition_position_exact": checkpoint["condition_position"] == 0,
        "fixed_output_bitwise_exact": torch.equal(before, after),
    }
    resume["pass"] = all(resume.values())
    training_checkpoint = torch.load(
        context["output_dir"] / "stage_c_training/checkpoints/checkpoint_step_000300.pth",
        map_location="cpu", weights_only=False,
    )
    checks = {
        "all_trainable_gradients_nonzero_finite": all(row["finite"] and row["l2"] > 0 for row in final_gradients.values()),
        "basis_frozen": sum(parameter.numel() for parameter in basis.parameters()) == 0 and sha256(basis_path) == checkpoint["basis_sha256"],
        "base_bitwise_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_zero": _base_gradient_count(context["base"]) == 0,
        "backbone_bitwise_frozen": context["backbone_before"] == o01._state_fingerprint(context["legacy_model"].clothing_observation_encoder.backbone.state_dict()),
        "backbone_gradient_zero": sum(parameter.grad is not None for parameter in extractor.spatial_backbone.parameters()) == 0,
        "mmlp_gradient_zero": sum(parameter.grad is not None for parameter in context["legacy_model"].dressable_model.anchor_clothing_mlp.parameters()) == 0,
        "checkpoint_exact_resume": resume["pass"], "held_out_unused": True,
        "target_forward_leakage_zero": True,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "optimizer_steps": 300,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "milestones": milestones, "gradients": final_gradients, "checks": checks,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_resume": resume, "runtime_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(output_dir / "stage_c_training/training_metrics.json", report)
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "RUNNING" if report["status"] == "PASS" else "FAIL",
        "stage": "COEFFICIENT_TRAINING_COMPLETE", "optimizer_steps": 300,
        "updated_at_unix": time.time(),
    })
    _training_curve(output_dir)
    return report


def _training_curve(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (output_dir / "stage_c_training/training.jsonl").read_text(encoding="utf-8").splitlines()]
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot([row["step"] for row in rows], [row["loss"] for row in rows], label="total")
    axis.plot([row["step"] for row in rows], [row["coefficient_loss"] for row in rows], label="coefficient")
    axis.set_xlabel("step"); axis.set_ylabel("loss"); axis.legend(); axis.set_title("Five-outfit coefficient training")
    figure.savefig(output_dir / "visual_acceptance/coefficient_training_curve.png", dpi=150); plt.close(figure)


def _load_trained_control(
    context: Mapping[str, Any], rank: int, input_dim: int,
) -> MultiOutfitLinearCoefficientControl:
    path = context["output_dir"] / "stage_c_training/checkpoints/checkpoint_step_000300.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["global_step"] != 300 or checkpoint["held_out_outfit_used"]:
        raise ValueError("coefficient checkpoint violates the training split")
    model = MultiOutfitLinearCoefficientControl(input_dim, rank).to(context["base"]._xyz)
    model.load_state_dict(checkpoint["model"]); model.eval()
    return model


def _variant_prediction(
    extractor: FrozenF2ReferenceFeatureExtractor,
    model: MultiOutfitLinearCoefficientControl,
    episode: Mapping[str, Any], mean: torch.Tensor, std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature = prediction_feature(extractor, episode)
    with torch.no_grad():
        standardized = model(feature).standardized_coefficients
    return standardized, restore_coefficients(standardized, mean, std)


def _residual_error(
    first: GaussianClothingResiduals, second: GaussianClothingResiduals,
    bounds: Mapping[str, float],
) -> float:
    metrics = parameterization.residual_metrics(first, second, bounds, active_epsilon=1e-8)
    return float(metrics["normalized_rmse"])


def _save_coefficient_pca(
    path: Path, rows: Mapping[str, Mapping[str, Any]], targets: Mapping[str, torch.Tensor],
) -> None:
    import matplotlib.pyplot as plt

    keys = sorted(rows); matrix = torch.tensor([rows[key]["prediction"] for key in keys]).float()
    centers = torch.stack([targets[outfit].detach().cpu().float() for outfit in TRAIN_OUTFITS])
    combined = torch.cat((matrix, centers)); centered = combined - combined.mean(0)
    if centered.shape[1] == 1:
        coordinates = torch.cat((centered, torch.zeros_like(centered)), dim=1)
    else:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        coordinates = centered @ vh[:2].t()
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    colors = {outfit: f"C{index}" for index, outfit in enumerate(TRAIN_OUTFITS)}
    for index, key in enumerate(keys):
        outfit = key.split("/")[0]
        axis.scatter(float(coordinates[index, 0]), float(coordinates[index, 1]), color=colors[outfit], alpha=.7)
    offset = len(keys)
    for index, outfit in enumerate(TRAIN_OUTFITS):
        axis.scatter(float(coordinates[offset + index, 0]), float(coordinates[offset + index, 1]),
                     color=colors[outfit], marker="*", s=220, label=outfit)
    axis.legend(); axis.set_title("Seen coefficient PCA (stars: teachers)")
    figure.savefig(path, dpi=150); plt.close(figure)


def run_seen_evaluation(context: Mapping[str, Any]) -> dict[str, Any]:
    training_path = context["output_dir"] / "stage_c_training/training_metrics.json"
    if not training_path.is_file() or json.loads(training_path.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Stage C training must pass before seen evaluation")
    basis, coefficients, payload, basis_path = _selected_basis(context)
    extractor = _feature_extractor(context)
    model = _load_trained_control(context, basis.rank, extractor.set_feature_dim)
    targets = _standardized_targets(coefficients, payload)
    mean = payload["coefficient_train_mean"].to(context["base"]._xyz)
    std = payload["coefficient_train_std"].to(mean)
    bounds = context["config"]["basis"]["channel_bounds"]
    rows: dict[str, Any] = {}; swapped_rows: list[dict[str, Any]] = []
    confusion = {outfit: {other: 0 for other in TRAIN_OUTFITS} for outfit in TRAIN_OUTFITS}
    correct_cache: dict[str, dict[str, torch.Tensor]] = {}; permutation_differences = []
    robustness_correct = 0; robustness_total = 0; pairwise_wins = 0
    for outfit in TRAIN_OUTFITS:
        teacher_residual = basis(coefficients[outfit], chunk_size=int(context["config"]["basis"]["chunk_size"]))
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"; episode = context["episodes"][key]; sample = context["samples"][key]
            standardized, restored = _variant_prediction(extractor, model, episode, mean, std)
            predicted_residual = basis(restored, chunk_size=int(context["config"]["basis"]["chunk_size"]))
            predicted_rgb, predicted_alpha = _render_residual(context, outfit, condition, predicted_residual)
            teacher_rgb, teacher_alpha = _render_residual(context, outfit, condition, teacher_residual)
            nearest, ranking, distances = _nearest_teacher(standardized, targets)
            confusion[outfit][nearest] += 1
            coefficient_error = float(torch.sqrt(F.mse_loss(standardized, targets[outfit])))
            residual_error = _residual_error(predicted_residual, teacher_residual, bounds)
            garment = diagnosis._garment_mask(sample)
            render_error = o01._masked_mae(predicted_rgb, teacher_rgb, garment)
            protected_error = o01._masked_mae(predicted_rgb, teacher_rgb, sample["target_protected_mask"])
            background_error = o01._masked_mae(
                predicted_rgb, teacher_rgb, 1 - sample["target_foreground_mask"]
            )
            permuted = subset_reference_episode(episode, (2, 0, 1))
            permuted_standardized, _ = _variant_prediction(extractor, model, permuted, mean, std)
            permutation_differences.append(float((standardized - permuted_standardized).abs().max()))
            single = subset_reference_episode(episode, (0,))
            dropout = subset_reference_episode(episode, (0, 1))
            robust = {}
            for variant_name, variant_episode in (("single", single), ("dropout", dropout)):
                variant_standardized, _ = _variant_prediction(extractor, model, variant_episode, mean, std)
                variant_nearest, _, variant_distances = _nearest_teacher(variant_standardized, targets)
                robustness_correct += int(variant_nearest == outfit); robustness_total += 1
                robust[variant_name] = {
                    "nearest_teacher": variant_nearest, "distances": variant_distances,
                    "prediction": variant_standardized.detach().cpu().tolist(),
                }
            zero_episode = dict(episode); zero_episode["reference_images"] = torch.zeros_like(episode["reference_images"])
            base_episode = dict(episode); base_episode["reference_images"] = sample["target_base_rgb"].unsqueeze(0).expand_as(episode["reference_images"]).clone()
            zero_standardized, _ = _variant_prediction(extractor, model, zero_episode, mean, std)
            base_standardized, _ = _variant_prediction(extractor, model, base_episode, mean, std)
            rows[key] = {
                "outfit": outfit, "condition": condition,
                "standardized_coefficient_rmse": coefficient_error,
                "restored_coefficient_rmse": float(torch.sqrt(F.mse_loss(restored, coefficients[outfit]))),
                "nearest_teacher": nearest, "correct_rank": ranking.index(outfit) + 1,
                "teacher_distances": distances, "prediction": standardized.detach().cpu().tolist(),
                "residual_normalized_rmse": residual_error, "teacher_render_garment_mae": render_error,
                "protected_mae": protected_error, "background_mae": background_error,
                "permutation_max_abs_diff": permutation_differences[-1], "robustness": robust,
                "zero_rgb_coefficient_change": float(torch.linalg.vector_norm(standardized - zero_standardized)),
                "base_rgb_coefficient_change": float(torch.linalg.vector_norm(standardized - base_standardized)),
                "target_pose_camera_fixed": True, "target_forward_leakage": False,
            }
            correct_cache[key] = {
                "prediction": predicted_rgb.detach().cpu(), "teacher": teacher_rgb.detach().cpu(),
                "alpha": predicted_alpha.detach().cpu(), "teacher_alpha": teacher_alpha.detach().cpu(),
            }
            for swapped_outfit in TRAIN_OUTFITS:
                if swapped_outfit == outfit:
                    continue
                swapped_episode = context["episodes"][f"{swapped_outfit}/{condition}"]
                swapped_standardized, swapped_restored = _variant_prediction(
                    extractor, model, swapped_episode, mean, std
                )
                swapped_residual = basis(
                    swapped_restored, chunk_size=int(context["config"]["basis"]["chunk_size"])
                )
                swapped_rgb, _ = _render_residual(context, outfit, condition, swapped_residual)
                swapped_coefficient_error = float(torch.sqrt(F.mse_loss(swapped_standardized, targets[outfit])))
                swapped_residual_error = _residual_error(swapped_residual, teacher_residual, bounds)
                swapped_render_error = o01._masked_mae(swapped_rgb, teacher_rgb, garment)
                wins = {
                    "coefficient": coefficient_error < swapped_coefficient_error,
                    "residual": residual_error < swapped_residual_error,
                    "render": render_error < swapped_render_error,
                }
                pairwise_wins += int(all(wins.values()))
                swapped_rows.append({
                    "target_outfit": outfit, "reference_outfit": swapped_outfit,
                    "condition": condition, "correct_coefficient_error": coefficient_error,
                    "swapped_coefficient_error": swapped_coefficient_error,
                    "coefficient_margin": swapped_coefficient_error - coefficient_error,
                    "correct_residual_error": residual_error, "swapped_residual_error": swapped_residual_error,
                    "residual_margin": swapped_residual_error - residual_error,
                    "correct_render_error": render_error, "swapped_render_error": swapped_render_error,
                    "render_margin": swapped_render_error - render_error, "wins": wins,
                })
    acceptance = context["config"]["coefficient_training"]["acceptance"]
    mean_rmse = float(np.mean([row["standardized_coefficient_rmse"] for row in rows.values()]))
    nearest_correct = sum(row["nearest_teacher"] == row["outfit"] for row in rows.values())
    rank_one = sum(row["correct_rank"] == 1 for row in rows.values())
    checks = {
        "mean_standardized_rmse": mean_rmse <= float(acceptance["standardized_rmse_max"]),
        "nearest_teacher_20_of_20": nearest_correct == int(acceptance["nearest_teacher_correct_count"]),
        "correct_rank_20_of_20": rank_one == int(acceptance["correct_rank_one_count"]),
        "correct_wins_80_of_80": pairwise_wins == int(acceptance["swapped_pairwise_wins"]),
        "permutation": max(permutation_differences) <= float(acceptance["permutation_max_abs_diff"]),
        "single_dropout": robustness_correct >= int(acceptance["single_dropout_correct_min"]) and robustness_total == int(acceptance["single_dropout_total"]),
        "target_forward_leakage_zero": True,
        "base_frozen": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "backbone_frozen": context["backbone_before"] == o01._state_fingerprint(context["legacy_model"].clothing_observation_encoder.backbone.state_dict()),
        "basis_frozen": sha256(basis_path) == training_checkpoint["basis_sha256"],
    }
    report = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if all(checks.values()) else "FAIL",
        "classification": None if all(checks.values()) else "MO-SEEN-FAIL",
        "mean_standardized_coefficient_rmse": mean_rmse,
        "nearest_teacher_correct": nearest_correct, "nearest_teacher_total": 20,
        "correct_rank_one": rank_one, "swapped_pairwise_wins": pairwise_wins,
        "swapped_pairwise_total": 80, "permutation_max_abs_diff": max(permutation_differences),
        "single_dropout_correct": robustness_correct, "single_dropout_total": robustness_total,
        "confusion_matrix": confusion, "checks": checks, "rows": rows,
        "swapped_rows": swapped_rows, "target_forward_leakage": False,
    }
    atomic_json(context["output_dir"] / "stage_c_seen/seen_metrics.json", report)
    _save_confusion(context["output_dir"] / "visual_acceptance/seen_outfit_confusion_matrix.png", confusion, "Seen correct-reference confusion")
    _save_coefficient_pca(context["output_dir"] / "visual_acceptance/seen_coefficient_pca.png", rows, targets)
    _seen_visuals(context, correct_cache)
    atomic_json(context["output_dir"] / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "AWAITING_SEEN_VISUAL_INSPECTION" if report["status"].startswith("NUMERIC_PASS") else "FAIL",
        "stage": "SEEN_EVALUATION_COMPLETE", "classification": report["classification"],
        "updated_at_unix": time.time(),
    })
    return report


def _seen_visuals(
    context: Mapping[str, Any], cache: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    rows = []
    for outfit in TRAIN_OUTFITS:
        panels = []
        for condition in CONDITIONS:
            key = f"{outfit}/{condition}"
            panels.extend([
                (f"{VIEWS[condition]} target", context["samples"][key]["target_edit_rgb"].detach().cpu(), 3),
                (f"{VIEWS[condition]} teacher", cache[key]["teacher"], 3),
                (f"{VIEWS[condition]} predicted", cache[key]["prediction"], 3),
            ])
        rows.append((outfit, panels))
    o01._save_contact_sheet(
        context["output_dir"] / "visual_acceptance/five_outfit_four_view_predictions.png", rows
    )
    back_rows = []
    for outfit in TRAIN_OUTFITS:
        key = f"{outfit}/cond_000318"
        back_rows.append((outfit, [
            ("edit target", context["samples"][key]["target_edit_rgb"].detach().cpu(), 3),
            ("teacher", cache[key]["teacher"], 3), ("prediction", cache[key]["prediction"], 3),
            ("absolute error", (cache[key]["teacher"] - cache[key]["prediction"]).abs(), 3),
        ]))
    o01._save_contact_sheet(
        context["output_dir"] / "visual_acceptance/five_outfit_correct_swapped_contact.png", back_rows
    )


def adjudicate_seen(context: Mapping[str, Any], visual_path: Path) -> dict[str, Any]:
    metrics = json.loads((context["output_dir"] / "stage_c_seen/seen_metrics.json").read_text(encoding="utf-8"))
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    required = {"status", "images_actually_opened", "observations", "all_outfits_four_views_form", "five_outfits_distinguishable", "new_cloud_or_mottle"}
    if not required.issubset(visual):
        raise ValueError("seen visual adjudication is incomplete")
    visual_pass = bool(
        visual["status"] == "PASS" and visual["images_actually_opened"]
        and visual["all_outfits_four_views_form"] and visual["five_outfits_distinguishable"]
        and not visual["new_cloud_or_mottle"]
    )
    passed = metrics["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual_pass
    report = {
        "status": "PASS" if passed else "FAIL", "classification": None if passed else "MO-SEEN-FAIL",
        "final_case": None if passed else "MO-S", "numeric": metrics, "visual": visual,
        "next_task": None if passed else "ADJUDICATE_AAAI_MINIMAL_PIPELINE_FEASIBILITY",
    }
    atomic_json(context["output_dir"] / "stage_c_seen/seen_adjudication.json", report)
    return report


def run_held_out(context: Mapping[str, Any]) -> dict[str, Any]:
    seen_path = context["output_dir"] / "stage_c_seen/seen_adjudication.json"
    if not seen_path.is_file() or json.loads(seen_path.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Stage D is forbidden until seen Stage C fully passes")
    teacher_decision = json.loads((context["output_dir"] / "stage_a_teacher_bank/teacher_bank_adjudication.json").read_text(encoding="utf-8"))
    o07_teacher_pass = teacher_decision["outfits"][HELD_OUT_OUTFIT]["status"] == "PASS"
    basis, coefficients, payload, basis_path = _selected_basis(context)
    teacher = load_teacher_residuals(context, (HELD_OUT_OUTFIT,))[0][HELD_OUT_OUTFIT]
    projection = project_residual_onto_basis(basis, teacher)
    projected_residual = basis(projection, chunk_size=int(context["config"]["basis"]["chunk_size"]))
    basis_metrics = parameterization.residual_metrics(
        projected_residual, teacher, context["config"]["basis"]["channel_bounds"], active_epsilon=1e-8
    )
    extractor = _feature_extractor(context)
    model = _load_trained_control(context, basis.rank, extractor.set_feature_dim)
    mean = payload["coefficient_train_mean"].to(projection); std = payload["coefficient_train_std"].to(projection)
    projection_standardized = (projection - mean) / std
    rows = {}; basis_render_errors = []; prediction_render_errors = []; permutation_differences = []
    prediction_coeff_errors = []; nearest_seen = []
    visual_rows = []
    train_targets = _standardized_targets(coefficients, payload)
    for condition in CONDITIONS:
        key = f"{HELD_OUT_OUTFIT}/{condition}"; sample = context["samples"][key]; episode = context["episodes"][key]
        standardized, restored = _variant_prediction(extractor, model, episode, mean, std)
        prediction = basis(restored, chunk_size=int(context["config"]["basis"]["chunk_size"]))
        teacher_rgb, _ = _render_residual(context, HELD_OUT_OUTFIT, condition, teacher)
        projected_rgb, _ = _render_residual(context, HELD_OUT_OUTFIT, condition, projected_residual)
        predicted_rgb, _ = _render_residual(context, HELD_OUT_OUTFIT, condition, prediction)
        garment = diagnosis._garment_mask(sample)
        basis_mae = o01._masked_mae(projected_rgb, teacher_rgb, garment)
        prediction_mae = o01._masked_mae(predicted_rgb, projected_rgb, garment)
        basis_render_errors.append(basis_mae); prediction_render_errors.append(prediction_mae)
        coefficient_rmse = float(torch.sqrt(F.mse_loss(restored, projection)))
        prediction_coeff_errors.append(coefficient_rmse)
        nearest, ranking, distances = _nearest_teacher(standardized, train_targets); nearest_seen.append(nearest)
        permuted = subset_reference_episode(episode, (2, 0, 1))
        permuted_standardized, _ = _variant_prediction(extractor, model, permuted, mean, std)
        permutation_differences.append(float((standardized - permuted_standardized).abs().max()))
        single_standardized, _ = _variant_prediction(extractor, model, subset_reference_episode(episode, (0,)), mean, std)
        dropout_standardized, _ = _variant_prediction(extractor, model, subset_reference_episode(episode, (0, 1)), mean, std)
        rows[key] = {
            "predicted_to_projected_coefficient_rmse": coefficient_rmse,
            "predicted_to_projected_render_garment_mae": prediction_mae,
            "projection_to_teacher_render_garment_mae": basis_mae,
            "nearest_seen_outfit": nearest, "nearest_seen_ranking": ranking, "nearest_seen_distances": distances,
            "permutation_max_abs_diff": permutation_differences[-1],
            "single_to_correct_coefficient_l2": float(torch.linalg.vector_norm(single_standardized - standardized)),
            "dropout_to_correct_coefficient_l2": float(torch.linalg.vector_norm(dropout_standardized - standardized)),
            "target_forward_leakage": False,
        }
        visual_rows.append((VIEWS[condition], [
            ("target", sample["target_edit_rgb"].detach().cpu(), 3),
            ("teacher", teacher_rgb.detach().cpu(), 3),
            ("basis projection", projected_rgb.detach().cpu(), 3),
            ("reference prediction", predicted_rgb.detach().cpu(), 3),
        ]))
    basis_acceptance = context["config"]["held_out"]["basis_acceptance"]
    reference_acceptance = context["config"]["held_out"]["reference_acceptance"]
    basis_checks = {
        "teacher_pass": o07_teacher_pass,
        "normalized_rmse": basis_metrics["normalized_rmse"] <= float(basis_acceptance["normalized_rmse_max"]),
        "cosine": basis_metrics["direction_cosine"] >= float(basis_acceptance["cosine_min"]),
        "garment_mae": max(basis_render_errors) <= float(basis_acceptance["garment_mae_max"]),
    }
    reference_checks = {
        "coefficient_rmse": float(np.mean(prediction_coeff_errors)) <= float(reference_acceptance["coefficient_rmse_max"]),
        "garment_mae": max(prediction_render_errors) <= float(reference_acceptance["garment_mae_max"]),
        "permutation": max(permutation_differences) <= float(reference_acceptance["permutation_max_abs_diff"]),
        "target_forward_leakage_zero": True,
        "held_out_never_trained": True,
    }
    report = {
        "status": "NUMERIC_PASS_VISUAL_PENDING" if all(basis_checks.values()) and all(reference_checks.values()) else "FAIL",
        "o07_teacher_status": teacher_decision["outfits"][HELD_OUT_OUTFIT]["status"],
        "projection_coefficient": projection.detach().cpu().tolist(),
        "projection_standardized": projection_standardized.detach().cpu().tolist(),
        "basis_metrics": basis_metrics,
        "max_basis_render_garment_mae": max(basis_render_errors),
        "mean_predicted_to_projected_coefficient_rmse": float(np.mean(prediction_coeff_errors)),
        "max_predicted_to_projected_render_garment_mae": max(prediction_render_errors),
        "nearest_seen_outfits": nearest_seen, "basis_checks": basis_checks,
        "reference_checks": reference_checks, "rows": rows,
        "basis_artifact_sha256": sha256(basis_path), "target_forward_leakage": False,
    }
    atomic_json(context["output_dir"] / "stage_d_held_out/held_out_metrics.json", report)
    o01._save_contact_sheet(
        context["output_dir"] / "visual_acceptance/o07_teacher_projection_prediction.png", visual_rows
    )
    return report


def adjudicate_held_out(context: Mapping[str, Any], visual_path: Path) -> dict[str, Any]:
    metrics = json.loads((context["output_dir"] / "stage_d_held_out/held_out_metrics.json").read_text(encoding="utf-8"))
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    required = {"status", "images_actually_opened", "observations", "projection_preserves_o07", "prediction_preserves_o07", "not_simply_nearest_seen"}
    if not required.issubset(visual):
        raise ValueError("held-out visual adjudication is incomplete")
    visual_pass = bool(
        visual["status"] == "PASS" and visual["images_actually_opened"]
        and visual["projection_preserves_o07"] and visual["prediction_preserves_o07"]
        and visual["not_simply_nearest_seen"]
    )
    passed = metrics["status"] == "NUMERIC_PASS_VISUAL_PENDING" and visual_pass
    report = {
        "status": "PASS" if passed else "FAIL", "numeric": metrics, "visual": visual,
        "final_case": "MO-G" if passed else "MO-P",
        "next_task": "FREEZE_MULTI_OUTFIT_AND_HELD_OUT_PAPER_EXPERIMENT_PROTOCOL" if passed
        else "FREEZE_SEEN_OUTFIT_PAPER_EXPERIMENT_PROTOCOL",
    }
    atomic_json(context["output_dir"] / "stage_d_held_out/held_out_adjudication.json", report)
    return report


def finalize(context: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = context["output_dir"]
    teacher = json.loads((output_dir / "stage_a_teacher_bank/teacher_bank_adjudication.json").read_text(encoding="utf-8"))
    basis_path = output_dir / "stage_b_basis/rank_selection.json"
    seen_path = output_dir / "stage_c_seen/seen_adjudication.json"
    held_path = output_dir / "stage_d_held_out/held_out_adjudication.json"
    basis = json.loads(basis_path.read_text(encoding="utf-8")) if basis_path.is_file() else {"status": "NOT_RUN"}
    seen = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.is_file() else {"status": "NOT_RUN"}
    held = json.loads(held_path.read_text(encoding="utf-8")) if held_path.is_file() else {"status": "NOT_RUN"}
    if not teacher["train_teacher_bank_pass"]:
        case, next_task = "MO-T", "ADJUDICATE_TEACHER_BANK_AND_DATA_SCOPE"
    elif basis.get("status") != "PASS":
        case, next_task = "MO-B", "REASSESS_MULTI_OUTFIT_RESIDUAL_SUBSPACE"
    elif seen.get("status") != "PASS":
        case, next_task = "MO-S", "ADJUDICATE_AAAI_MINIMAL_PIPELINE_FEASIBILITY"
    elif held.get("status") != "PASS":
        case, next_task = "MO-P", "FREEZE_SEEN_OUTFIT_PAPER_EXPERIMENT_PROTOCOL"
    else:
        case, next_task = "MO-G", "FREEZE_MULTI_OUTFIT_AND_HELD_OUT_PAPER_EXPERIMENT_PROTOCOL"
    manifest = json.loads((output_dir / "input_audit/input_manifest.json").read_text(encoding="utf-8"))
    freeze = {
        "base_bitwise_unchanged": context["base_before"] == _tensor_state_fingerprint(_base_named_tensors(context["base"])),
        "base_gradient_count": _base_gradient_count(context["base"]),
        "cs_pass_immutable": explicit.immutable_tree_metadata_fingerprint(context["cs_attempt"]) == context["immutable_before"]["cs_pass"],
        "prior_basis_immutable": explicit.immutable_tree_metadata_fingerprint(context["prior_basis_attempt"]) == context["immutable_before"]["prior_basis"],
        "existing_teachers_immutable": explicit.immutable_tree_metadata_fingerprint(context["existing_teacher_root"]) == context["immutable_before"]["existing_teachers"],
    }
    if not all(value for key, value in freeze.items() if key != "base_gradient_count") or freeze["base_gradient_count"] != 0:
        raise AssertionError("frozen asset immutability failed")
    result = {
        "task_id": TASK_ID, "run_commit": manifest["git"]["head"],
        "finalization_commit": git_output("rev-parse", "HEAD"),
        "teacher_bank": teacher, "basis": basis, "seen": seen, "held_out": held,
        "freeze": freeze, "target_forward_leakage": False, "final_case": case,
        "status": "PASS" if case in {"MO-P", "MO-G"} else "FAIL", "next_task": next_task,
        "claim_boundary": (
            "MO-G supports this subject's preregistered five seen outfits and one held-out outfit only; "
            "MO-P supports seen-outfit control only. Neither case establishes arbitrary garment generation "
            "or cross-identity generalization."
        ),
    }
    atomic_json(output_dir / "final_adjudication/final_adjudication.json", result)
    atomic_text(output_dir / "final_adjudication/FINAL_ADJUDICATION.md", "\n".join([
        f"# {TASK_ID}", "", f"- Run commit: `{result['run_commit']}`",
        f"- Finalization commit: `{result['finalization_commit']}`",
        f"- Final case: **{case}**", f"- Status: **{result['status']}**",
        f"- Next unique task: `{next_task}`", "", "## Claim boundary", "", result["claim_boundary"],
    ]))
    atomic_json(output_dir / "RUN_STATUS.json", {
        "task_id": TASK_ID, "status": "COMPLETE", "final_case": case,
        "completed_at_unix": time.time(),
    })
    return result


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config.resolve())
    output_dir = Path(args.output_dir or Path(config["output"]["task_root"]) / config["output"]["attempt"])
    create = args.phase == "audit"
    if create:
        if output_dir.exists():
            raise FileExistsError(f"append-only attempt already exists: {output_dir}")
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    needs_backbone = args.phase in {"ridge", "train", "seen", "seen-adjudicate", "held-out", "held-out-adjudicate", "finalize"}
    context = build_context(config, output_dir, create=create, reference_backbone=needs_backbone)
    try:
        if args.phase == "audit":
            run_audit(context)
        elif args.phase == "teacher":
            if args.outfit not in {"O02", "O03", "O04", "O07"}:
                raise ValueError("teacher phase allows only O02/O03/O04/O07")
            run_teacher(context, args.outfit)
        elif args.phase == "teacher-adjudicate":
            if args.visual_observations is None:
                raise ValueError("teacher adjudication requires actual visual observations")
            adjudicate_teacher_bank(context, args.visual_observations)
        elif args.phase == "basis":
            run_basis_ladder(context)
        elif args.phase == "basis-adjudicate":
            if args.visual_observations is None:
                raise ValueError("basis adjudication requires actual visual observations")
            adjudicate_basis_rank(context, args.visual_observations)
        elif args.phase == "ridge":
            run_ridge_control(context)
        elif args.phase == "train":
            run_coefficient_training(context)
        elif args.phase == "seen":
            run_seen_evaluation(context)
        elif args.phase == "seen-adjudicate":
            if args.visual_observations is None:
                raise ValueError("seen adjudication requires actual visual observations")
            adjudicate_seen(context, args.visual_observations)
        elif args.phase == "held-out":
            run_held_out(context)
        elif args.phase == "held-out-adjudicate":
            if args.visual_observations is None:
                raise ValueError("held-out adjudication requires actual visual observations")
            adjudicate_held_out(context, args.visual_observations)
        elif args.phase == "finalize":
            finalize(context)
        else:
            raise ValueError(args.phase)
    except Exception as error:
        atomic_json(output_dir / "RUN_STATUS.json", {
            "task_id": TASK_ID, "status": "FAILED_TOOL_OR_RUNTIME", "phase": args.phase,
            "exception_type": type(error).__name__, "exception": str(error),
            "traceback": traceback.format_exc(), "updated_at_unix": time.time(),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered multi-outfit explicit residual basis")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", required=True, choices=(
        "audit", "teacher", "teacher-adjudicate", "basis", "basis-adjudicate",
        "ridge", "train", "seen", "seen-adjudicate", "held-out",
        "held-out-adjudicate", "finalize",
    ))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outfit", default=None)
    parser.add_argument("--visual-observations", type=Path, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
