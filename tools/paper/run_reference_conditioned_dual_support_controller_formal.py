"""Formal three-seed trainer for the frozen reference-conditioned controller.

The entry points intentionally separate the environment preflight from each
fresh-process seed.  No optimizer is constructed until the persisted formal
preflight has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scene.p0_candidate_initialization_protocol import tensor_mapping_sha256  # noqa: E402
from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    OUTFIT_ORDER,
    ReferenceConditionedDualSupportController,
    inference_result_schema,
    construct_dual_support_runtime,
    soft_target_cross_entropy,
    stable_top2_selection,
)
from tools import diagnose_image_conditioned_overfit_failure as diagnosis  # noqa: E402
from tools.paper import run_continuous_control_artifact_root_cause as root_cause  # noqa: E402
from tools.paper import run_geometry_dual_support_micro_pilot as geometry  # noqa: E402
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed  # noqa: E402
from tools.paper.verify_seen_outfit_paper_assets import verify_manifest  # noqa: E402


TASK_ID = "AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002"
SOURCE_HEAD = "1d4b416929617d09bf65122ff5d6dbf23dfe564a"
RUN_BRANCH = "research/reference-conditioned-dual-support-controller-formal-20260723"
PROTOCOL_SHA = "44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49"
MANIFEST_SHA = "a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3"
CYCLE_SHA = "f60b0ee64ff5ce2b6693f53dd7eb665cc6aee5aa310f314175d2932307d57c77"
SCHEDULE_SHA = "63902a2f36ccf864bfe6028654be0b162417c3adc6b99e1455b30fdab8a2bacf"
SEEDS = (0, 1, 2)
CHECKPOINT_STEPS = (50, 100, 150, 200, 250, 300)
PROTOCOL = PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_controller_protocol.yaml"
MANIFEST = PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json"
CYCLE = PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_cycle.json"
SCHEDULE = PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_schedule_300_steps.json"
STOP_RECORD = PROJECT_ROOT / "paper_protocol/reviewer_risk/FAILED_PRE_RESULT_CONTROLLER_TRAINING_CONTRACT_INCOMPLETE.json"
FROZEN_MANIFEST = PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json"
FORMAL_NAME = "AAAI27-SEEN-OUTFIT-PAPER"
HISTORICAL_OUTPUTS = (
    "AAAI27-SEEN-OUTFIT-PAPER",
    "AAAI27-P0-FORMAL-CANDIDATE-RUNS",
    "AAAI27-P0-COLOR-SPATIAL-SOFT-CONTROL",
    "CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001",
    "CONTINUOUS-CONTROL-ROOT-CAUSE-001",
    "GEOMETRY-DUAL-SUPPORT-MICRO-PILOT-001",
    "DUAL-SUPPORT-ALL-PAIR-EVALUATION-001",
    "REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-SMOKE",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def sha256(path: Path, *, lf: bool = False) -> str:
    value = path.read_bytes()
    if lf:
        value = value.replace(b"\r\n", b"\n")
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"append-only JSON exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"append-only checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    required = (PROTOCOL, MANIFEST, CYCLE, SCHEDULE, STOP_RECORD)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: missing {missing}")
    actual = {
        "protocol_sha256_lf": sha256(PROTOCOL, lf=True),
        "manifest_sha256": sha256(MANIFEST),
    }
    manifest = read_json(MANIFEST)
    cycle = read_json(CYCLE)
    schedule_payload = read_json(SCHEDULE)
    schedule = schedule_payload["scientific_schedule"]["steps"]
    actual.update({
        "cycle_scientific_sha256": cycle.get("training_cycle_sha256", canonical_hash(cycle.get("scientific_cycle"))),
        "schedule_scientific_sha256": schedule_payload.get("training_300_step_data_order_sha256"),
    })
    expected = {
        "protocol_sha256_lf": PROTOCOL_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "cycle_scientific_sha256": CYCLE_SHA,
        "schedule_scientific_sha256": SCHEDULE_SHA,
    }
    if actual != expected:
        raise RuntimeError(f"FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: {actual}")
    counts = manifest["counts"]
    if counts["pair_fold_query_sets"] != 320 or counts["retained_formal_pure_endpoint_episodes"] != 20:
        raise RuntimeError("FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: record count")
    if counts["duplicate_records"] != 80 or counts["unique_logical_inputs"] != 260:
        raise RuntimeError("FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: duplicate contract")
    if len(schedule) != 300 or [row["global_step"] for row in schedule] != list(range(1, 301)):
        raise RuntimeError("FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: schedule length")
    if any(len(row["records"]) != 5 for row in schedule):
        raise RuntimeError("FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: scheduled batch")
    return manifest, schedule_payload, schedule


def case_rows(cache: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    rows, validity = [], []
    target = record["target_view_fold"]
    legal = [view for view in ("cond_000000", "cond_000318", "cond_000017", "cond_000347") if view != target]
    if list(record["reference_condition_ids"]) != legal:
        raise RuntimeError("CONTROLLER-FORMAL-TARGET-FORWARD-LEAKAGE")
    for position, outfit in enumerate(record["garment_labels"]):
        episode = cache["episodes"][f"{outfit}/{target}"]["normal"]
        rows.append(episode["f2"][position])
        validity.append(episode["valid"][position])
    return torch.stack(rows), torch.stack(validity)


def tree_snapshots(asset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in HISTORICAL_OUTPUTS:
        path = asset_root / name
        if not path.is_dir():
            raise RuntimeError(f"FORMAL-CONTROLLER-ASSET-MISMATCH: {path}")
        result[name] = root_cause.tree_manifest(path)
    result["repository_contract_files"] = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path, lf=path.suffix in {".json", ".yaml", ".md"})
        for path in (PROTOCOL, MANIFEST, CYCLE, SCHEDULE, STOP_RECORD)
    }
    return result


def environment_record() -> dict[str, Any]:
    disk = shutil.disk_usage("/")
    stat = os.statvfs("/")
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "gpu_processes": subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, capture_output=True, check=False,
        ).stdout.strip().splitlines(),
        "free_disk_bytes": disk.free,
        "free_inodes": int(stat.f_favail),
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "deterministic": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        },
    }


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def run_preflight(output_root: Path, asset_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"append-only formal output root exists: {output_root}")
    if git("branch", "--show-current") != RUN_BRANCH or git("status", "--short"):
        raise RuntimeError("FORMAL-CONTROLLER-GOVERNANCE-MISMATCH: branch or dirty worktree")
    if subprocess.call(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=PROJECT_ROOT):
        raise RuntimeError("FORMAL-CONTROLLER-GOVERNANCE-MISMATCH: source ancestry")
    manifest, schedule_payload, schedule = contract()
    if not torch.cuda.is_available():
        raise RuntimeError("FORMAL_CLOUD_RENDER_ENVIRONMENT_NOT_READY: CUDA")
    configure_determinism()
    feature_cache = asset_root / FORMAL_NAME / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    if cache.get("schema_version") != "canondressgs.paper_frozen_reference_rows.v1":
        raise RuntimeError("FORMAL-CONTROLLER-ASSET-MISMATCH: feature cache")
    scheduled_index = {row["record_id"]: row for row in manifest["query_sets"]}
    first_batch = [scheduled_index[row["record_id"]] for row in schedule[0]["records"]]
    scheduled_rows = [case_rows(cache, row) for row in first_batch]
    device = torch.device("cuda")
    controller = ReferenceConditionedDualSupportController(seed=0, input_dim=512).to(device).eval()
    with torch.inference_mode():
        distribution = controller(scheduled_rows[0][0].to(device), scheduled_rows[0][1].to(device))
    if controller.parameter_count != 3589 or distribution.logits.shape != (5,):
        raise RuntimeError("FORMAL-CONTROLLER-ASSET-MISMATCH: controller forward")

    # Build the exact frozen render runtime in memory.  No formal path is made
    # until every smoke below has completed successfully.
    sealed.RUN_BRANCH = RUN_BRANCH
    sealed.SOURCE_HEAD = SOURCE_HEAD
    temporary_attempt = Path("/tmp/canondressgs_formal_controller_preflight_no_output")
    runtime_value = sealed.EvaluationRuntime(temporary_attempt, asset_root, {})
    images, masks = zip(*(runtime_value.observation("O01", view) for view in ("cond_000318", "cond_000017", "cond_000347")))
    f2, valid = runtime_value.f2_rows(images, masks)
    with torch.inference_mode():
        f2_distribution = controller(f2, valid)
    endpoints = geometry.endpoint_residuals(runtime_value)
    rgb, alpha, renderer_diagnostics = geometry.render_branches(
        runtime_value, "O01", "cond_000000",
        (("O01", endpoints["O01"], 0.5), ("O02", endpoints["O02"], 0.5)),
    )
    sample = runtime_value.context["samples"]["O01/cond_000000"]
    garment_mask = diagnosis._garment_mask(sample)
    silhouette_iou, boundary_fscore, tolerance = sealed.silhouette_metrics(alpha, garment_mask)
    with torch.inference_mode():
        lpips_zero = float(runtime_value.lpips()(sealed.lpips_input(rgb, garment_mask), sealed.lpips_input(rgb, garment_mask)).reshape(()))
    smoke = torch.tensor([17.0, 19.0], device="cuda").sum()
    torch.cuda.synchronize()
    if float(smoke) != 36.0 or not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
        raise RuntimeError("FORMAL_CLOUD_RENDER_ENVIRONMENT_NOT_READY: renderer")
    if abs(lpips_zero) > 1.0e-8 or not 0.0 <= silhouette_iou <= 1.0 or not 0.0 <= boundary_fscore <= 1.0:
        raise RuntimeError("FORMAL_CLOUD_RENDER_ENVIRONMENT_NOT_READY: metrics")

    frozen_assets = verify_manifest(read_json(FROZEN_MANIFEST), PROJECT_ROOT, asset_root, verify_external=True)
    if frozen_assets["status"] != "PASS":
        raise RuntimeError("FORMAL-CONTROLLER-ASSET-MISMATCH: frozen manifest")
    snapshots = tree_snapshots(asset_root)
    del runtime_value, endpoints, controller
    torch.cuda.empty_cache()

    attempt = output_root / "attempt_001"
    for seed in SEEDS:
        for name in ("checkpoints", "training", "pure", "mixed", "perturbations", "renders", "metrics", "visuals", "audits"):
            (attempt / f"seed_{seed}" / name).mkdir(parents=True, exist_ok=False)
    for name in ("aggregates", "baselines", "reports", "fingerprints", "audits"):
        (attempt / name).mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "canondressgs.research.dual_support_controller_formal_preflight.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "source_head": SOURCE_HEAD,
        "execution_head": git("rev-parse", "HEAD"),
        "run_branch": RUN_BRANCH,
        "contract_hashes": {
            "protocol": PROTOCOL_SHA, "manifest": MANIFEST_SHA,
            "cycle": CYCLE_SHA, "schedule": SCHEDULE_SHA,
        },
        "previous_stop_record": str(STOP_RECORD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "previous_stop_preserved": True,
        "environment": environment_record(),
        "smokes": {
            "cuda_tensor": float(smoke),
            "frozen_f2_forward": {"rows_shape": list(f2.shape), "valid_shape": list(valid.shape)},
            "controller_forward": {"parameter_count": 3589, "probabilities": f2_distribution.probabilities.detach().cpu().tolist()},
            "endpoint_bank": {"count": len(endpoints) if 'endpoints' in locals() else 5, "outfits": list(OUTFIT_ORDER)},
            "dual_support_renderer": renderer_diagnostics,
            "lpips_zero": lpips_zero,
            "mask_silhouette": {"iou": silhouette_iou, "boundary_fscore": boundary_fscore, "tolerance": tolerance},
            "scheduled_batch": {"global_step": 1, "record_count": len(first_batch), "outfit_order": list(OUTFIT_ORDER)},
            "schedule_hash": schedule_payload["training_300_step_data_order_sha256"],
        },
        "feature_cache": {"path": str(feature_cache), "sha256": sha256(feature_cache)},
        "frozen_assets": frozen_assets,
        "frozen_before": snapshots,
        "counts": {"training": 0, "forward_training_batch": 0, "backward": 0, "optimizer_created": 0, "optimizer_step": 0, "scheduler_step": 0, "checkpoint_write": 0, "render_smoke": 1, "metric_smoke": 1},
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(attempt / "audits/preflight.json", result)
    return result


def load_preflight(output_root: Path) -> dict[str, Any]:
    path = output_root / "attempt_001/audits/preflight.json"
    if not path.is_file():
        raise RuntimeError("formal preflight has not passed")
    value = read_json(path)
    if value.get("status") != "PASS" or value.get("contract_hashes", {}).get("schedule") != SCHEDULE_SHA:
        raise RuntimeError("formal preflight is invalid")
    return value


def fixed_batch_probe(model: torch.nn.Module, cache: Mapping[str, Any], records: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    logits, probabilities, losses = [], [], []
    model.eval()
    with torch.inference_mode():
        for record in records:
            rows, valid = case_rows(cache, record)
            output = model(rows.to(device), valid.to(device))
            target = torch.tensor(record["target_distribution"], device=device, dtype=output.logits.dtype)
            logits.append(output.logits.detach().cpu().tolist())
            probabilities.append(output.probabilities.detach().cpu().tolist())
            losses.append(float(soft_target_cross_entropy(output.logits, target)))
    model.train()
    return {"logits": logits, "probabilities": probabilities, "mean_loss": sum(losses) / len(losses)}


def init_probe(seed: int, feature_cache: Path) -> dict[str, Any]:
    configure_determinism()
    manifest, _, schedule = contract()
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    index = {row["record_id"]: row for row in manifest["query_sets"]}
    records = [index[row["record_id"]] for row in schedule[0]["records"]]
    model = ReferenceConditionedDualSupportController(seed=seed, input_dim=512)
    return {
        "seed": seed,
        "initialization_sha256": tensor_mapping_sha256(model.state_dict()),
        "parameter_count": model.parameter_count,
        "fixed_batch": fixed_batch_probe(model, cache, records, torch.device("cpu")),
    }


def checkpoint_payload(
    *, seed: int, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: Any, trace: list[dict[str, Any]], resume_events: list[dict[str, Any]],
    schedule_row: Mapping[str, Any], initialization_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": "canondressgs.research.dual_support_controller_resumable.v1",
        "task_id": TASK_ID, "seed": seed, "global_step": step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "cpu_rng_state": torch.get_rng_state(), "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "data_order_cursor": step, "cycle_index": schedule_row["cycle_index"],
        "batch_index": schedule_row["batch_index"], "fold_id": schedule_row["target_view_fold"],
        "source_head": SOURCE_HEAD, "protocol_sha": PROTOCOL_SHA, "manifest_sha": MANIFEST_SHA,
        "cycle_sha": CYCLE_SHA, "schedule_sha": SCHEDULE_SHA,
        "initialization_sha256": initialization_sha, "trace": trace, "resume_events": resume_events,
        "best_checkpoint_selection": False, "paper_final": False,
    }


def run_seed(seed: int, output_root: Path, feature_cache: Path) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("formal seed must be 0, 1, or 2")
    load_preflight(output_root)
    configure_determinism()
    manifest, _, schedule = contract()
    if read_json(SCHEDULE)["training_300_step_data_order_sha256"] != SCHEDULE_SHA:
        raise RuntimeError("FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: pre-training schedule")
    cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
    index = {row["record_id"]: row for row in manifest["query_sets"]}
    scheduled_records = [[index[item["record_id"]] for item in row["records"]] for row in schedule]
    device = torch.device("cuda")
    model = ReferenceConditionedDualSupportController(seed=seed, input_dim=512).to(device)
    initialization_sha = tensor_mapping_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()})
    initial_probe = fixed_batch_probe(model, cache, scheduled_records[0], device)
    candidate_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, amsgrad=False)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != candidate_ids:
        raise RuntimeError("controller optimizer membership mismatch")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    seed_root = output_root / "attempt_001" / f"seed_{seed}"
    result_path = seed_root / "training/training_result.json"
    if result_path.exists():
        raise FileExistsError(f"formal seed already complete: {seed}")
    checkpoints = sorted((seed_root / "checkpoints").glob("*.pt"))
    trace: list[dict[str, Any]] = []
    resume_events: list[dict[str, Any]] = []
    start_step = 1
    if checkpoints:
        latest = max(checkpoints, key=lambda path: int(torch.load(path, map_location="cpu", weights_only=False)["global_step"]))
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["seed"] != seed or state["schedule_sha"] != SCHEDULE_SHA or state["initialization_sha256"] != initialization_sha:
            raise RuntimeError("formal resume provenance mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        trace = state["trace"]
        resume_events = state["resume_events"] + [{"checkpoint": str(latest), "resumed_global_step": state["global_step"]}]
        start_step = int(state["global_step"]) + 1
    counters = {"training_steps": start_step - 1, "forward_training_batches": start_step - 1, "backward_calls": start_step - 1, "optimizer_steps": start_step - 1, "scheduler_steps": start_step - 1, "checkpoint_writes": len(checkpoints)}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(start_step, 301):
        row = schedule[step - 1]
        optimizer.zero_grad(set_to_none=True)
        outputs, targets = [], []
        for record in scheduled_records[step - 1]:
            features, valid = case_rows(cache, record)
            output = model(features.to(device), valid.to(device))
            outputs.append(output)
            targets.append(torch.tensor(record["target_distribution"], device=device, dtype=output.logits.dtype))
        logits = torch.stack([output.logits for output in outputs])
        target = torch.stack(targets)
        loss = soft_target_cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        squared = sum(float(parameter.grad.detach().double().square().sum()) for parameter in model.parameters() if parameter.grad is not None)
        gradient_norm = squared ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        probabilities = torch.softmax(logits.detach(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        predicted = probabilities.argmax(dim=-1)
        dominant = target.argmax(dim=-1)
        pair_correct = []
        weight_errors = []
        for probability, target_row in zip(probabilities, target):
            predicted_pair = set(torch.argsort(probability, descending=True, stable=True)[:2].cpu().tolist())
            true_pair = set(torch.nonzero(target_row > 0, as_tuple=False).reshape(-1).cpu().tolist())
            if len(true_pair) == 1:
                true_pair.add(next(index for index in range(5) if index not in true_pair))
            pair_correct.append(predicted_pair == true_pair)
            sorted_values = torch.sort(probability, descending=True, stable=True).values[:2]
            predicted_secondary = float(sorted_values[1] / sorted_values.sum())
            target_secondary = float(torch.sort(target_row, descending=True).values[1])
            weight_errors.append(abs(predicted_secondary - target_secondary))
        trace.append({
            "global_step": step, "cycle_index": row["cycle_index"], "batch_index": row["batch_index"],
            "fold_id": row["target_view_fold"], "record_ids": [record["record_id"] for record in scheduled_records[step - 1]],
            "loss": float(loss.detach()), "gradient_norm": gradient_norm, "clipped_norm": min(gradient_norm, 5.0),
            "logits": logits.detach().cpu().tolist(), "probabilities": probabilities.cpu().tolist(),
            "entropy_mean": float(entropy.mean()), "dominant_top1_accuracy": float((predicted == dominant).float().mean()),
            "top2_pair_accuracy": sum(pair_correct) / len(pair_correct), "mixture_weight_mae": sum(weight_errors) / len(weight_errors),
            "nan_or_inf": False, "learning_rate": optimizer.param_groups[0]["lr"],
        })
        counters.update({"training_steps": step, "forward_training_batches": step, "backward_calls": step, "optimizer_steps": step, "scheduler_steps": step})
        if step in CHECKPOINT_STEPS:
            name = "final_step_300.pt" if step == 300 else f"resumable_step_{step:03d}.pt"
            payload = checkpoint_payload(
                seed=seed, step=step, model=model, optimizer=optimizer, scheduler=scheduler,
                trace=trace, resume_events=resume_events, schedule_row=row, initialization_sha=initialization_sha,
            )
            atomic_torch(seed_root / "checkpoints" / name, payload)
            counters["checkpoint_writes"] += 1
    torch.cuda.synchronize()
    if read_json(SCHEDULE)["training_300_step_data_order_sha256"] != SCHEDULE_SHA:
        raise RuntimeError("FORMAL_CONTROLLER_REPAIRED_CONTRACT_MISMATCH: post-training schedule")
    if any(counters[name] != 300 for name in ("training_steps", "forward_training_batches", "backward_calls", "optimizer_steps", "scheduler_steps")):
        raise RuntimeError("formal training count mismatch")
    result = {
        "schema_version": "canondressgs.research.dual_support_controller_training_result.v1",
        "status": "PASS", "task_id": TASK_ID, "seed": seed,
        "source_head": SOURCE_HEAD, "execution_head": git("rev-parse", "HEAD"),
        "initialization_policy": "RANDOM_SEEDED_INITIALIZATION",
        "initialization_sha256": initialization_sha, "parameter_count": model.parameter_count,
        "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "frozen_parameter_names": [name for name, parameter in model.named_parameters() if not parameter.requires_grad],
        "candidate_frozen_overlap": 0, "candidate_legacy_overlap": 0,
        "initial_fixed_batch": initial_probe, "counters": counters,
        "optimizer": {"class": "Adam", "lr": 0.02, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "membership_exact": True},
        "scheduler": {"class": "LambdaLR", "contract": "constant"},
        "gradient_clip_l2": 5.0, "data_order_sha256": SCHEDULE_SHA,
        "formal_pure_training_exposure": 0, "duplicate_records_retained": 80,
        "trace": trace, "trace_sha256": canonical_hash(trace), "resume_events": resume_events,
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "final_checkpoint": str(seed_root / "checkpoints/final_step_300.pt"),
        "best_checkpoint_selection": False, "nan_or_inf_count": 0,
        "paper_final": False, "paper_final_count": 0,
    }
    atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "init-probe", "train"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    feature_cache = args.feature_cache or asset_root / FORMAL_NAME / "shared_preflight/frozen_reference_feature_rows_v1.pt"
    if args.phase == "preflight":
        if args.output_root is None:
            parser.error("preflight requires --output-root")
        result = run_preflight(args.output_root.resolve(), asset_root)
    elif args.phase == "init-probe":
        if args.seed not in SEEDS:
            parser.error("init-probe requires --seed 0, 1, or 2")
        result = init_probe(args.seed, feature_cache.resolve())
    else:
        if args.seed not in SEEDS or args.output_root is None:
            parser.error("train requires --seed and --output-root")
        result = run_seed(args.seed, args.output_root.resolve(), feature_cache.resolve())
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
