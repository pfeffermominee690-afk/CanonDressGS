"""Render all repaired Controller V2/matched V1 logical evaluation records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scene.gaussian_clothing_residuals import (  # noqa: E402
    CHANNELS,
    GaussianClothingResiduals,
)
from scene.reference_conditioned_dual_support_controller import (  # noqa: E402
    OUTFIT_ORDER,
    stable_top2_selection,
)
from tools.paper import run_controller_v2_repaired_training as training  # noqa: E402
from tools.paper import run_geometry_dual_support_micro_pilot as geometry  # noqa: E402
from tools.paper import run_p0_color_spatial_soft_control_evaluations as sealed  # noqa: E402


TASK_ID = training.TASK_ID
SOURCE_HEAD = training.SOURCE_HEAD
ATTEMPT = training.ATTEMPT
OUTPUT_NAME = training.OUTPUT_NAME
FAMILIES = training.FAMILIES
RISK = PROJECT_ROOT / "paper_protocol/reviewer_risk"
VISUAL_REPRESENTATIVES = (
    RISK / "controller_v2_micro_pilot_visual_representatives.json"
)
FORMAL_OUTPUT = "REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002"
GEOMETRY_CHANNELS = {
    "delta_xyz", "delta_log_scaling", "delta_rotvec"
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value, handle, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def root(output_root: Path) -> Path:
    return output_root / ATTEMPT


def configure() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)


def record_left(row: Mapping[str, Any]) -> str:
    if row.get("pair_id"):
        return row["pair_id"].split("_")[0]
    if row.get("target_outfit"):
        return row["target_outfit"]
    return row["garment_labels"][0]


def v1_branches(
    row: Mapping[str, Any],
    endpoints: Mapping[str, GaussianClothingResiduals],
) -> tuple[tuple[str, GaussianClothingResiduals, float], ...]:
    probabilities = torch.tensor(row["garment_probabilities"])
    selection = stable_top2_selection(probabilities)
    if row["mode"] == "SINGLE_ENDPOINT":
        return ((row["predicted_top1"], endpoints[row["predicted_top1"]], 1.0),)
    if row["mode"] != "DUAL_SUPPORT":
        raise RuntimeError("matched V1 has an unsupported render mode")
    return (
        (
            selection.top1_outfit,
            endpoints[selection.top1_outfit],
            selection.normalized_top2_weight_1,
        ),
        (
            selection.top2_outfit,
            endpoints[selection.top2_outfit],
            selection.normalized_top2_weight_2,
        ),
    )


def hard_geometry_soft_va(
    dominant: GaussianClothingResiduals,
    first: GaussianClothingResiduals,
    second: GaussianClothingResiduals,
    first_weight: float,
) -> GaussianClothingResiduals:
    values = {}
    for name in CHANNELS:
        if name in GEOMETRY_CHANNELS:
            values[name] = getattr(dominant, name)
        else:
            values[name] = (
                first_weight * getattr(first, name)
                + (1.0 - first_weight) * getattr(second, name)
            )
    return GaussianClothingResiduals.from_dict(values)


def v2_branches(
    row: Mapping[str, Any],
    endpoints: Mapping[str, GaussianClothingResiduals],
) -> tuple[tuple[str, GaussianClothingResiduals, float], ...]:
    if row["mode"] == "SINGLE_ENDPOINT":
        outfit = row["predicted_top1"]
        return ((outfit, endpoints[outfit], 1.0),)
    first, second = row["predicted_pair"].split("_")
    first_weight = float(row["predicted_earlier_pair_weight"])
    if row["mode"] == "DUAL_SUPPORT":
        return (
            (first, endpoints[first], first_weight),
            (second, endpoints[second], 1.0 - first_weight),
        )
    if row["mode"] == "HARD_GEOMETRY_SOFT_VA":
        dominant = row["predicted_top1"]
        residual = hard_geometry_soft_va(
            endpoints[dominant], endpoints[first], endpoints[second],
            first_weight,
        )
        return ((dominant, residual, 1.0),)
    raise RuntimeError("Controller V2 has an unsupported render mode")


def render_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "left": record_left(row),
        "target_view_fold": row["target_view_fold"],
        "family": row["family"],
        "mode": row["mode"],
    }
    if row["mode"] == "SINGLE_ENDPOINT":
        value["endpoint"] = row["predicted_top1"]
    elif row["family"] == "V2":
        value.update({
            "predicted_pair": row["predicted_pair"],
            "predicted_earlier_pair_weight":
                row["predicted_earlier_pair_weight"],
            "dominant_outfit": row["predicted_top1"],
        })
    else:
        selection = stable_top2_selection(
            torch.tensor(row["garment_probabilities"])
        )
        value.update({
            "top1": selection.top1_outfit,
            "top2": selection.top2_outfit,
            "weight_1": selection.normalized_top2_weight_1,
            "weight_2": selection.normalized_top2_weight_2,
        })
    return value


def render_paths(attempt: Path, signature_sha256: str) -> tuple[Path, Path, Path]:
    directory = attempt / "renders/cache" / signature_sha256[:2]
    return (
        directory / f"{signature_sha256}_rgb.png",
        directory / f"{signature_sha256}_alpha.png",
        directory / f"{signature_sha256}_diagnostics.json",
    )


def render_or_load(
    runtime: sealed.EvaluationRuntime,
    endpoints: Mapping[str, GaussianClothingResiduals],
    row: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any], bool, str]:
    signature = render_signature(row)
    signature_sha256 = canonical_hash(signature)
    rgb_path, alpha_path, diagnostics_path = render_paths(
        root(Path(runtime.asset_root) / OUTPUT_NAME), signature_sha256
    )
    exists = (
        rgb_path.is_file(), alpha_path.is_file(), diagnostics_path.is_file()
    )
    if any(exists) and not all(exists):
        raise RuntimeError(f"partial render cache entry: {signature_sha256}")
    if all(exists):
        return (
            rgb_path, alpha_path, read_json(diagnostics_path), False,
            signature_sha256,
        )
    branches = (
        v2_branches(row, endpoints)
        if row["family"] == "V2"
        else v1_branches(row, endpoints)
    )
    with torch.inference_mode():
        rgb, alpha, diagnostics = geometry.render_branches(
            runtime, record_left(row), row["target_view_fold"], branches
        )
    sealed.save_new_render(rgb_path, rgb, 3)
    sealed.save_new_render(alpha_path, alpha, 1)
    diagnostics.update({
        "signature": signature,
        "signature_sha256": signature_sha256,
        "family": row["family"],
        "mode": row["mode"],
        "geometry_interpolation": False,
        "target_forward_leakage": 0,
    })
    atomic_json(diagnostics_path, diagnostics)
    return rgb_path, alpha_path, diagnostics, True, signature_sha256


def historical_paths(
    asset_root: Path, row: Mapping[str, Any],
) -> dict[str, Path]:
    role = (
        "pure" if row.get("split") == "FORMAL_PURE_SECONDARY"
        else "mixed"
    )
    safe = row["record_id"].replace("/", "__")
    base = (
        asset_root / FORMAL_OUTPUT / "attempt_004/seed_0" / role
        / "paired_baselines"
    )
    left = record_left(row)
    right = row["pair_id"].split("_")[1] if row.get("pair_id") else left
    result = {
        "oracle_rgb": base / f"{safe}__oracle_rgb.png",
        "oracle_alpha": base / f"{safe}__oracle_alpha.png",
        "source_rgb": base / f"{safe}__source_{left}_rgb.png",
        "target_endpoint_rgb": base / f"{safe}__target_{right}_rgb.png",
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"historical exact baseline missing: {missing}")
    return result


def metric_record(
    runtime: sealed.EvaluationRuntime,
    row: Mapping[str, Any],
    rgb_path: Path,
    alpha_path: Path,
    historical: Mapping[str, Path],
) -> dict[str, float]:
    rgb = sealed.image_tensor(rgb_path, 3).to(runtime.device)
    alpha = sealed.image_tensor(alpha_path, 1).to(runtime.device)
    oracle = sealed.image_tensor(historical["oracle_rgb"], 3).to(runtime.device)
    source = sealed.image_tensor(historical["source_rgb"], 3).to(runtime.device)
    target = sealed.image_tensor(
        historical["target_endpoint_rgb"], 3
    ).to(runtime.device)
    return geometry.image_metrics(
        runtime, record_left(row), row["target_view_fold"],
        rgb, alpha, oracle, source, target,
    )


def all_logical_rows(attempt: Path) -> list[dict[str, Any]]:
    predictions = read_json(attempt / "predictions/test_predictions.json")
    perturbations = read_json(attempt / "perturbations/results.json")
    rows = []
    for row in predictions["primary_test"]:
        rows.append({**row, "render_role": "PRIMARY_TEST"})
    for row in predictions["formal_pure_secondary"]:
        rows.append({**row, "render_role": "FORMAL_PURE_SECONDARY"})
    for row in perturbations["records"]:
        rows.append({
            **row,
            "render_role": "PERTURBATION",
            "split": "PERTURBATION",
        })
    if len(rows) != 5280:
        raise RuntimeError(f"LOGICAL-RENDER-COUNT-MISMATCH: {len(rows)}")
    return rows


def execute(output_root: Path, asset_root: Path) -> dict[str, Any]:
    configure()
    attempt = root(output_root)
    result_path = attempt / "metrics/render_results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    os.environ["CANONDRESSGS_ASSET_ROOT"] = str(asset_root)
    sealed.RUN_BRANCH = training.RUN_BRANCH
    sealed.SOURCE_HEAD = SOURCE_HEAD
    runtime = sealed.EvaluationRuntime(
        attempt / "audits/render_runtime_no_write", asset_root, {}
    )
    endpoints = geometry.endpoint_residuals(runtime)
    rows = all_logical_rows(attempt)
    logical = []
    metric_cache: dict[tuple[str, str], dict[str, float]] = {}
    created = 0
    reused = 0
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        rgb_path, alpha_path, diagnostics, was_created, signature_sha = (
            render_or_load(runtime, endpoints, row)
        )
        created += int(was_created)
        reused += int(not was_created)
        historical = historical_paths(asset_root, row)
        metric_key = (row["record_id"], signature_sha)
        if metric_key not in metric_cache:
            metric_cache[metric_key] = metric_record(
                runtime, row, rgb_path, alpha_path, historical
            )
        logical.append({
            "logical_render_index": index,
            "render_role": row["render_role"],
            "family": row["family"],
            "rotation": row["rotation"],
            "seed": row["seed"],
            "record_id": row["record_id"],
            "variant": row.get("variant"),
            "pair_id": row.get("pair_id"),
            "predicted_pair": row["predicted_pair"],
            "predicted_earlier_pair_weight":
                row.get("predicted_earlier_pair_weight"),
            "mixedness_probability": row.get("mixedness_probability"),
            "thresholds": row.get("thresholds"),
            "compatibility_label": row.get("compatibility_label"),
            "mode": row["mode"],
            "endpoint_provenance":
                "FROZEN_FIVE_OUTFIT_IMMUTABLE_ENDPOINT_BANK",
            "render_source": (
                "NEW_CONTROLLER_RENDER" if was_created
                else "CURRENT_ATTEMPT_EXACT_SIGNATURE_REUSE"
            ),
            "reused": not was_created,
            "signature_sha256": signature_sha,
            "rgb_path": str(rgb_path),
            "rgb_sha256": file_hash(rgb_path),
            "alpha_path": str(alpha_path),
            "alpha_sha256": file_hash(alpha_path),
            "active_gaussian_count":
                diagnostics["active_gaussian_count"],
            "render_time_seconds":
                diagnostics["render_time_seconds"] if was_created else 0.0,
            "source_render_time_seconds":
                diagnostics["render_time_seconds"],
            "peak_vram_bytes": diagnostics["peak_vram_bytes"],
            "metrics": metric_cache[metric_key],
            "oracle_rgb_path": str(historical["oracle_rgb"]),
            "target_endpoint_rgb_path":
                str(historical["target_endpoint_rgb"]),
            "exact_historical_baseline_reuse": True,
            "geometry_interpolation": False,
            "target_forward_leakage": 0,
        })
        if index % 100 == 0:
            print(
                json.dumps({
                    "progress": index, "logical_total": len(rows),
                    "new": created, "reused": reused,
                }),
                flush=True,
            )
    if created + reused != 5280:
        raise RuntimeError("render accounting mismatch")
    fields = (
        "garment_rgb_mae", "garment_lpips", "silhouette_iou",
        "boundary_fscore", "protected_lpips", "identity_metric",
        "outside_garment_opacity",
    )
    aggregates = []
    for family in FAMILIES:
        for role in (
            "PRIMARY_TEST", "FORMAL_PURE_SECONDARY", "PERTURBATION"
        ):
            selected = [
                row for row in logical
                if row["family"] == family and row["render_role"] == role
            ]
            aggregates.append({
                "family": family,
                "render_role": role,
                "logical_count": len(selected),
                "means": {
                    field: statistics.fmean(
                        row["metrics"][field] for row in selected
                    )
                    for field in fields
                },
                "maxima": {
                    field: max(row["metrics"][field] for row in selected)
                    for field in fields
                },
            })
    result = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_render_results.v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "attempt": ATTEMPT,
        "source_head": SOURCE_HEAD,
        "counts": {
            "logical": len(logical),
            "reused": reused,
            "new": created,
            "failed": 0,
            "unique_render_signatures": len({
                row["signature_sha256"] for row in logical
            }),
            "unique_metric_records": len(metric_cache),
            "historical_oracle_target_endpoint_reuse": len(logical),
        },
        "wall_clock_seconds": time.perf_counter() - started,
        "aggregates": aggregates,
        "records": logical,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(result_path, result)
    return result


def fit_panel(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        image, ((width - image.width) // 2, (height - image.height) // 2)
    )
    return canvas


def text_panel(
    title: str, lines: Sequence[str], width: int, height: int
) -> Image.Image:
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill="black", font=font)
    y = 32
    for line in lines:
        for start in range(0, len(line), 35):
            draw.text(
                (10, y), line[start:start + 35], fill="black", font=font
            )
            y += 16
    return panel


def label_panel(image: Image.Image, title: str) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 28), "white")
    result.paste(image, (0, 28))
    ImageDraw.Draw(result).text(
        (8, 8), title, fill="black", font=ImageFont.load_default()
    )
    return result


def target_image(asset_root: Path, row: Mapping[str, Any]) -> Path:
    dominant = OUTFIT_ORDER[
        max(
            range(len(row["target_distribution"])),
            key=lambda index: row["target_distribution"][index],
        )
    ]
    path = (
        asset_root
        / "pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003"
        / "dataset/rgb/edit" / dominant
        / f"{row['target_view_fold']}.png"
    )
    if not path.is_file():
        raise RuntimeError(f"target image missing: {path}")
    return path


def reference_strip(
    row: Mapping[str, Any], width: int, height: int
) -> Image.Image:
    images = [
        fit_panel(Path(source["image_path"]), width // 3, height)
        for source in row["source_references"]
    ]
    result = Image.new("RGB", (width, height), "white")
    for index, image in enumerate(images):
        result.paste(image, (index * (width // 3), 0))
    return result


def make_sheets(output_root: Path, asset_root: Path) -> dict[str, Any]:
    attempt = root(output_root)
    render_result = read_json(attempt / "metrics/render_results.json")
    visual = read_json(VISUAL_REPRESENTATIVES)
    manifest = read_json(
        RISK / "dual_support_controller_training_manifest.json"
    )
    record_index = {
        row["record_id"]: row for row in manifest["query_sets"]
    }
    render_index = {
        (
            row["family"], row["rotation"], row["seed"], row["record_id"]
        ): row
        for row in render_result["records"]
        if row["render_role"] == "PRIMARY_TEST"
    }
    output = []
    panel_width, panel_height = 300, 420
    for spec in visual["sheets"]:
        record = record_index[spec["record_id"]]
        v2 = render_index[
            ("V2", spec["rotation"], spec["seed"], spec["record_id"])
        ]
        v1 = render_index[
            (
                "MATCHED_V1", spec["rotation"], spec["seed"],
                spec["record_id"],
            )
        ]
        historical = historical_paths(asset_root, record)
        dominant = (
            historical["source_rgb"]
            if record["assignment_type"] == "AAB"
            else historical["target_endpoint_rgb"]
        )
        visual_panels = [
            label_panel(
                reference_strip(record, panel_width, panel_height),
                "references",
            ),
            label_panel(
                fit_panel(Path(v2["rgb_path"]), panel_width, panel_height),
                "V2 actual",
            ),
            label_panel(
                fit_panel(Path(v1["rgb_path"]), panel_width, panel_height),
                "matched V1 actual",
            ),
            label_panel(
                fit_panel(
                    target_image(asset_root, record),
                    panel_width, panel_height,
                ),
                "target",
            ),
            label_panel(
                fit_panel(
                    historical["oracle_rgb"], panel_width, panel_height
                ),
                "oracle",
            ),
            label_panel(
                fit_panel(dominant, panel_width, panel_height),
                "dominant endpoint",
            ),
            label_panel(
                text_panel(
                    "predicted pairs / weights",
                    [
                        f"V2 pair={v2['predicted_pair']}",
                        f"V2 w={v2['predicted_earlier_pair_weight']}",
                        f"V1 pair={v1['predicted_pair']}",
                        f"V1 w={v1['predicted_earlier_pair_weight']}",
                    ],
                    panel_width, panel_height,
                ),
                "predicted pairs / weights",
            ),
            label_panel(
                text_panel(
                    "modes / compatibility",
                    [
                        f"V2 mode={v2['mode']}",
                        f"V2 compat={v2['compatibility_label']}",
                        f"V1 mode={v1['mode']}",
                        f"V1 compat={v1['compatibility_label']}",
                    ],
                    panel_width, panel_height,
                ),
                "modes / compatibility",
            ),
        ]
        sheet = Image.new(
            "RGB",
            (
                panel_width * 4,
                (panel_height + 28) * 2 + 60,
            ),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (10, 10),
            (
                f"{spec['sheet_id']} | rotation={spec['rotation']} "
                f"seed={spec['seed']} pair={spec['pair_id']} "
                f"composition={spec['composition']}"
            ),
            fill="black",
            font=ImageFont.load_default(),
        )
        for index, panel in enumerate(visual_panels):
            x = (index % 4) * panel_width
            y = 50 + (index // 4) * (panel_height + 28)
            sheet.paste(panel, (x, y))
        path = (
            attempt / "visuals/main_sheets"
            / f"rotation_{spec['rotation']}" / f"seed_{spec['seed']}"
            / f"{spec['pair_id']}__{spec['composition']}.png"
        )
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        output.append({
            **spec,
            "path": str(path),
            "sha256": file_hash(path),
            "width": sheet.width,
            "height": sheet.height,
            "opened_original_detail": False,
        })
    if len(output) != 240 or len({row["path"] for row in output}) != 240:
        raise RuntimeError("visual sheet accounting mismatch")
    result = {
        "schema_version":
            "canondressgs.research.controller_v2_repaired_visual_manifest.v1",
        "status": "PENDING_MANUAL_REVIEW",
        "task_id": TASK_ID,
        "attempt": ATTEMPT,
        "source_archive_sha256": visual["archive_content_sha256"],
        "sheet_count": len(output),
        "unique_path_count": len({row["path"] for row in output}),
        "items": output,
        "paper_final": False,
        "paper_final_count": 0,
    }
    atomic_json(attempt / "visuals/visual_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("execute", "sheets"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    arguments = parser.parse_args()
    if ATTEMPT != "attempt_004":
        raise RuntimeError("successful repaired rendering is bound to attempt_004")
    output_root = arguments.output_root.resolve()
    asset_root = arguments.asset_root.resolve()
    result = (
        execute(output_root, asset_root)
        if arguments.phase == "execute"
        else make_sheets(output_root, asset_root)
    )
    print(json.dumps({
        "status": result["status"],
        "counts": result.get("counts"),
        "sheet_count": result.get("sheet_count"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
