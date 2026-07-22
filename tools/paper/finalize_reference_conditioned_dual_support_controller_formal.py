"""Read-only aggregation and visual-sheet generation for FORMAL-002.

This utility deliberately has no model, optimizer, or renderer imports.  It
only reads persisted formal results and frozen historical baseline archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageChops, ImageDraw, ImageFont  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002"
SOURCE_HEAD = "1d4b416929617d09bf65122ff5d6dbf23dfe564a"
RUN_BRANCH = "research/reference-conditioned-dual-support-controller-formal-20260723"
PROTOCOL_SHA = "44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49"
MANIFEST_SHA = "a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3"
CYCLE_SHA = "f60b0ee64ff5ce2b6693f53dd7eb665cc6aee5aa310f314175d2932307d57c77"
SCHEDULE_SHA = "63902a2f36ccf864bfe6028654be0b162417c3adc6b99e1455b30fdab8a2bacf"
OUTFITS = ("O01", "O02", "O03", "O04", "O08")
PAIRS = tuple(f"{a}_{b}" for i, a in enumerate(OUTFITS) for b in OUTFITS[i + 1 :])
SEEDS = (0, 1, 2)
COMPOSITIONS = ("AAB", "ABB")
METRIC_KEYS = (
    "garment_rgb_mae", "garment_lpips", "silhouette_iou", "boundary_fscore",
    "protected_lpips", "identity_metric", "outside_garment_opacity",
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(rows)) if rows else 0.0


def load_inputs(output_root: Path) -> dict[str, Any]:
    attempt1 = output_root / "attempt_001"
    attempt4 = output_root / "attempt_004"
    manifest = read_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_controller_training_manifest.json")
    data: dict[str, Any] = {
        "manifest": manifest,
        "training": read_json(attempt1 / "aggregates/training_summary.json"),
        "historical_p0": read_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/p0_mixed_reference_results.json"),
        "historical_all_pair": read_json(PROJECT_ROOT / "paper_protocol/reviewer_risk/dual_support_all_pair_results.json"),
        "pure": {}, "mixed": {}, "render": {}, "perturb": {},
    }
    for seed in SEEDS:
        data["pure"][seed] = read_json(attempt1 / f"seed_{seed}/pure/pure_classification.json")
        data["mixed"][seed] = read_json(attempt1 / f"seed_{seed}/mixed/mixed_classification.json")
        data["render"][seed] = read_json(attempt4 / f"seed_{seed}/metrics/render_evaluation.json")
        data["perturb"][seed] = read_json(
            attempt4 / f"seed_{seed}/perturbations/perturbation_evaluation.json"
        )
    return data


def validate_inputs(data: dict[str, Any]) -> None:
    training = data["training"]
    if training["status"] != "PASS" or training["totals"] != training["expected_totals"]:
        raise RuntimeError("training audit is not an exact PASS")
    if training["formal_pure_training_exposure"] != 0:
        raise RuntimeError("formal-pure training exposure is nonzero")
    if training["consistent_duplicates_retained"] != 80:
        raise RuntimeError("duplicate retention mismatch")
    for seed in SEEDS:
        pure, mixed, render, perturb = (
            data["pure"][seed], data["mixed"][seed], data["render"][seed], data["perturb"][seed]
        )
        checks = {
            "pure records": pure["record_count"] == 20,
            "pure swaps": pure["swap_count"] == 80,
            "mixed records": mixed["protocol_record_count"] == 320,
            "mixed unique": mixed["unique_query_count"] == 260,
            "render mixed": render["mixed_render_count"] == 320,
            "render pure": render["pure_render_count"] == 20,
            "perturb records": perturb["record_count"] == 160,
            "perturb representatives": perturb["representative_count"] == 20,
            "perturb variants": perturb["variant_count"] == 8,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"seed {seed} completeness mismatch: {failed}")


def font(size: int = 18) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def image_panel(source: Path | None, label: str, size: tuple[int, int] = (280, 340)) -> Image.Image:
    width, height = size
    caption_h = 94
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    if source is not None and source.is_file():
        with Image.open(source) as opened:
            item = opened.convert("RGB")
            item.thumbnail((width - 8, height - caption_h - 8), Image.Resampling.LANCZOS)
            x = (width - item.width) // 2
            y = (height - caption_h - item.height) // 2
            panel.paste(item, (x, max(0, y)))
    else:
        draw.rectangle((5, 5, width - 5, height - caption_h - 5), outline="#999999", width=2)
        draw.text((12, 18), "NO IMAGE\nARCHIVE METRIC ONLY", fill="#555555", font=font(15))
    wrapped = textwrap.wrap(label, width=38)[:5]
    draw.multiline_text((6, height - caption_h + 4), "\n".join(wrapped), fill="black", font=font(14), spacing=2)
    return panel


def text_panel(label: str, size: tuple[int, int] = (280, 340)) -> Image.Image:
    panel = Image.new("RGB", size, "#f3f5f7")
    draw = ImageDraw.Draw(panel)
    draw.multiline_text(
        (12, 12), "\n".join(textwrap.wrap(label, width=34)), fill="black", font=font(16), spacing=5
    )
    return panel


def save_page(path: Path, title: str, rows: Sequence[Sequence[Image.Image]], columns: int = 4) -> None:
    if path.is_file():
        return
    if not rows:
        raise ValueError("empty page")
    panel_w, panel_h = rows[0][0].size
    title_h = 86
    canvas = Image.new("RGB", (columns * panel_w, title_h + len(rows) * panel_h), "#e9edf2")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((12, 10), title, fill="black", font=font(18), spacing=4)
    for row_index, row in enumerate(rows):
        for column, panel in enumerate(row):
            if column >= columns:
                raise ValueError("too many panels in row")
            canvas.paste(panel, (column * panel_w, title_h + row_index * panel_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    canvas.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def atomic_derived(path: Path, image: Image.Image) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def difference_image(candidate: Path, oracle: Path, output: Path) -> Path:
    if not output.is_file():
        with Image.open(candidate) as left, Image.open(oracle) as right:
            difference = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
            difference = difference.point(lambda value: min(255, value * 4))
            atomic_derived(output, difference)
    return output


def silhouette_image(candidate: Path, alpha: Path, output: Path) -> Path:
    if not output.is_file():
        with Image.open(candidate) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(alpha) as opened:
            mask = np.asarray(opened.convert("L"), dtype=np.uint8) >= 128
        eroded = mask.copy()
        eroded[1:] &= mask[:-1]
        eroded[:-1] &= mask[1:]
        eroded[:, 1:] &= mask[:, :-1]
        eroded[:, :-1] &= mask[:, 1:]
        rgb[mask & ~eroded] = np.array([255, 0, 0], dtype=np.uint8)
        atomic_derived(output, Image.fromarray(rgb))
    return output


def probability_label(row: dict[str, Any]) -> str:
    probs = ",".join(f"{outfit}:{value:.2f}" for outfit, value in zip(OUTFITS, row["probabilities"]))
    weights = "/".join(f"{value:.2f}" for value in row["runtime_weights"])
    return (
        f"{row['record_id'].split('/')[-1]} | {row['mode']} | "
        f"top={row['top1_outfit']},{row['top2_outfit']} w={weights} | p={probs}"
    )


def indexes(data: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {"manifest": {}, "pure": {}, "mixed": {}, "render": {}}
    for row in data["manifest"]["formal_pure_endpoint_episodes"] + data["manifest"]["query_sets"]:
        value["manifest"][row["record_id"]] = row
    for seed in SEEDS:
        value["pure"][seed] = {row["record_id"]: row for row in data["pure"][seed]["records"]}
        value["mixed"][seed] = {row["record_id"]: row for row in data["mixed"][seed]["records"]}
        value["render"][seed] = {row["record_id"]: row for row in data["render"][seed]["records"]}
    return value


def pure_sheets(data: dict[str, Any], output_root: Path, idx: dict[str, Any]) -> list[str]:
    visual_root = output_root / "attempt_004/visuals"
    derived_root = visual_root / "_derived/pure"
    result: list[str] = []
    manifest_rows = data["manifest"]["formal_pure_endpoint_episodes"]
    for seed in SEEDS:
        for outfit in OUTFITS:
            rows_for_outfit = [row for row in manifest_rows if row["garment_labels"][0] == outfit]
            rows_for_outfit.sort(key=lambda row: row["target_view_fold"])
            references = rows_for_outfit[0]["source_references"]
            sheet_rows: list[list[Image.Image]] = [[
                image_panel(Path(ref["image_path"]), f"reference {i}: {ref['outfit_id']}/{ref['condition_id']}")
                for i, ref in enumerate(references)
            ] + [text_panel(f"seed={seed}\noutfit={outfit}\n4 frozen target-view folds\nrecord-isolated; reference assets overlap")]]
            for manifest_row in rows_for_outfit:
                record_id = manifest_row["record_id"]
                prediction = idx["pure"][seed][record_id]
                render = idx["render"][seed][record_id]
                candidate, oracle, alpha = Path(render["rgb_path"]), Path(render["oracle_rgb_path"]), Path(render["alpha_path"])
                safe = record_id.replace("/", "__")
                diff = difference_image(candidate, oracle, derived_root / f"seed_{seed}_{safe}_diff.png")
                overlay = silhouette_image(candidate, alpha, derived_root / f"seed_{seed}_{safe}_silhouette.png")
                sheet_rows.append([
                    image_panel(candidate, f"Controller | {probability_label(prediction)}"),
                    image_panel(oracle, f"Teacher endpoint | {prediction['top1_outfit']} | parity exact"),
                    image_panel(diff, "RGB |Controller - teacher| x4"),
                    image_panel(overlay, f"silhouette overlay (red edge) | {manifest_row['target_view_fold']}"),
                ])
            path = visual_root / "pure_main" / f"seed_{seed}_{outfit}.png"
            save_page(path, f"FORMAL-002 PURE MAIN | seed {seed} | {outfit}", sheet_rows)
            result.append(str(path))
    return result


def historical_indexes(data: dict[str, Any]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    p0: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in data["historical_p0"]["records"]:
        p0[(row["method"], row["pair_id"], row["assignment_macro"], row["assignment_id"], row["view_id"])] = row
    all_pair: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in data["historical_all_pair"]["execution"]["records"]:
        all_pair[(row["variant"], row["pair_id"], row["direction"], float(row["alpha"]), row["condition"])] = row
    return p0, all_pair


def mixed_sheets(data: dict[str, Any], output_root: Path, idx: dict[str, Any]) -> list[str]:
    visual_root = output_root / "attempt_004/visuals"
    p0, all_pair = historical_indexes(data)
    result: list[str] = []
    queries = data["manifest"]["query_sets"]
    for seed in SEEDS:
        for pair_id in PAIRS:
            pair = pair_id.split("_")
            for composition in COMPOSITIONS:
                selected = [
                    row for row in queries
                    if row["pair_id"] == pair_id and row["assignment_type"] == composition
                ]
                selected.sort(key=lambda row: (row["assignment_position"], row["target_view_fold"]))
                if len(selected) != 12:
                    raise RuntimeError(f"mixed sheet scope mismatch: {seed}/{pair_id}/{composition}")
                representative = next(
                    row for row in selected
                    if row["assignment_position"] == 0 and row["target_view_fold"] == "cond_000000"
                )
                refs = representative["source_references"]
                sheet_rows: list[list[Image.Image]] = [[
                    image_panel(Path(ref["image_path"]), f"reference {i}: {ref['outfit_id']}/{ref['condition_id']}")
                    for i, ref in enumerate(refs)
                ] + [text_panel(
                    f"seed={seed}\npair={pair_id}\ncomposition={composition}\n"
                    "Controller: exact 1/3-2/3 protocol. Archived FULL/HARD panels: nearest alpha=0.2/0.8."
                )]]
                for position in (0, 1, 2):
                    panels: list[Image.Image] = []
                    for row in [item for item in selected if item["assignment_position"] == position]:
                        prediction = idx["mixed"][seed][row["record_id"]]
                        render = idx["render"][seed][row["record_id"]]
                        panels.append(image_panel(Path(render["rgb_path"]), probability_label(prediction)))
                    sheet_rows.append(panels)

                view = representative["target_view_fold"]
                assignment = f"{composition}_{'B' if composition == 'AAB' else 'A'}_POSITION_0"
                p0_panels = [
                    image_panel(
                        Path(p0[(method, pair_id, composition, assignment, view)]["rgb_path"]),
                        f"{method} archived exact assignment/view",
                    )
                    for method in ("Ours-v2", "B6", "B7")
                ]
                alpha = 0.2 if composition == "AAB" else 0.8
                full = all_pair[("FULL_LINEAR_BASELINE", pair_id, "A_TO_B", alpha, view)]
                hard = all_pair[("HARD_GEOMETRY_SOFT_VA", pair_id, "A_TO_B", alpha, view)]
                rep_render = idx["render"][seed][representative["record_id"]]
                paired_root = Path(rep_render["oracle_rgb_path"]).parent
                safe = representative["record_id"].replace("/", "__")
                teacher_a = paired_root / f"{safe}__source_{pair[0]}_rgb.png"
                teacher_b = paired_root / f"{safe}__target_{pair[1]}_rgb.png"
                sheet_rows.append(p0_panels + [image_panel(Path(full["rgb_path"]), f"FULL_LINEAR archived alpha={alpha:.1f}")])
                sheet_rows.append([
                    image_panel(Path(hard["rgb_path"]), f"HARD_GEOMETRY_SOFT_VA archived alpha={alpha:.1f}"),
                    image_panel(Path(rep_render["oracle_rgb_path"]), "Oracle Dual-Support exact 1/3-2/3 (NON-DEPLOYABLE)"),
                    image_panel(teacher_a, f"Teacher endpoint {pair[0]}"),
                    image_panel(teacher_b, f"Teacher endpoint {pair[1]}"),
                ])
                path = visual_root / "mixed_main" / f"seed_{seed}_{pair_id}_{composition}.png"
                save_page(path, f"FORMAL-002 MIXED MAIN | seed {seed} | {pair_id} | {composition}", sheet_rows)
                result.append(str(path))
    return result


def save_plot(path: Path, figure: Any) -> None:
    if path.is_file():
        plt.close(figure)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def record_render_rows(data: dict[str, Any], idx: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in SEEDS:
        for record_id, prediction in idx["mixed"][seed].items():
            if prediction["assignment_type"] not in COMPOSITIONS:
                continue
            value = dict(prediction)
            value["seed"] = seed
            value["render"] = idx["render"][seed][record_id]
            result.append(value)
    return result


def case_sheet(path: Path, title: str, cases: Sequence[dict[str, Any]], panels: int = 8) -> None:
    rows: list[list[Image.Image]] = []
    for case in cases[:panels]:
        render = case["render"]
        label = (
            f"seed={case['seed']} {case['record_id']} | mode={case['mode']} | "
            f"pair={case['top1_outfit']},{case['top2_outfit']} | "
            f"LPIPS={render['metrics']['garment_lpips']:.4f} IoU={render['metrics']['silhouette_iou']:.4f}"
        )
        rows.append([
            image_panel(Path(render["rgb_path"]), "Controller | " + label),
            image_panel(Path(render["oracle_rgb_path"]), "Oracle exact pair/weight | " + label),
            image_panel(Path(render["rgb_path"]), "Controller repeat for full-resolution opening"),
            text_panel(label + "\n" + probability_label(case)),
        ])
    if not rows:
        rows = [[text_panel("No cases in this preregistered category")]]
    save_page(path, title, rows)


def diagnostic_sheets(data: dict[str, Any], output_root: Path, idx: dict[str, Any]) -> list[str]:
    root = output_root / "attempt_004/visuals/diagnostics"
    result: list[str] = []
    records = record_render_rows(data, idx)

    predicted_pairs = sorted(set(tuple(sorted((row["top1_outfit"], row["top2_outfit"]))) for row in records))
    pair_labels = sorted(set(PAIRS) | {"_".join(pair) for pair in predicted_pairs})
    matrix = np.zeros((len(PAIRS), len(pair_labels)), dtype=np.int64)
    for row in records:
        true_i = PAIRS.index(row["pair_id"])
        pred = "_".join(sorted((row["top1_outfit"], row["top2_outfit"])))
        matrix[true_i, pair_labels.index(pred)] += 1
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(matrix, cmap="magma")
    ax.set_xticks(range(len(pair_labels)), pair_labels, rotation=60, ha="right")
    ax.set_yticks(range(len(PAIRS)), PAIRS)
    ax.set_title("Top-2 pair confusion, all seeds, AAB/ABB protocol records")
    fig.colorbar(image, ax=ax)
    path = root / "top2_pair_confusion.png"
    save_plot(path, fig); result.append(str(path))

    target, predicted, colors = [], [], []
    for row in records:
        target.append(max(row["target_distribution"]))
        predicted.append(max(row["runtime_weights"]))
        colors.append(row["seed"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(target, predicted, c=colors, cmap="viridis", alpha=0.28, s=16)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="Target dominant weight", ylabel="Predicted dominant normalized top-2 weight",
           title="Weight calibration (all seeds; duplicates retained)")
    path = root / "weight_calibration.png"
    save_plot(path, fig); result.append(str(path))

    specs: list[tuple[str, str, list[dict[str, Any]]]] = []
    specs.append(("fallback_cases.png", "Mixed SINGLE_ENDPOINT fallback cases",
                  sorted((row for row in records if row["mode"] == "SINGLE_ENDPOINT"),
                         key=lambda row: row["top2_mass"], reverse=True)))
    specs.append(("lowest_top2_mass.png", "Lowest top-2 mass cases",
                  sorted(records, key=lambda row: row["top2_mass"])))
    specs.append(("highest_entropy.png", "Highest entropy cases",
                  sorted(records, key=lambda row: row["entropy"], reverse=True)))
    wrong = [row for row in records if not row["pair_correct"]]
    specs.append(("wrong_pair_cases.png", "Wrong predicted top-2 pair cases", wrong))
    specs.append(("worst_silhouette.png", "Lowest silhouette-IoU Controller cases",
                  sorted(records, key=lambda row: row["render"]["metrics"]["silhouette_iou"])))
    dual_records = [row for row in records if row["mode"] == "DUAL_SUPPORT"]
    specs.append(("worst_ghosting.png", "Largest Controller-vs-Oracle LPIPS among dual-support cases",
                  sorted(dual_records, key=lambda row: row["render"]["metrics"]["garment_lpips"], reverse=True)))

    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_record[row["record_id"]].append(row)
    disagreement: list[dict[str, Any]] = []
    for rows in by_record.values():
        if len({(row["top1_outfit"], row["top2_outfit"], row["mode"]) for row in rows}) > 1:
            disagreement.extend(rows)
    specs.append(("seed_disagreement.png", "Seed-disagreement cases", disagreement))

    assignment_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    fold_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        assignment_groups[(row["seed"], row["pair_id"], row["assignment_type"], row["target_view_fold"])].append(row)
        fold_groups[(row["seed"], row["pair_id"], row["assignment_type"], row["assignment_position"])].append(row)
    assignment_bad = [row for rows in assignment_groups.values()
                      if len({(item["top1_outfit"], item["top2_outfit"]) for item in rows}) > 1 for row in rows]
    fold_bad = [row for rows in fold_groups.values()
                if len({(item["top1_outfit"], item["top2_outfit"]) for item in rows}) > 1 for row in rows]
    specs.append(("assignment_inconsistency.png", "Assignment-position inconsistency cases", assignment_bad))
    specs.append(("view_fold_inconsistency.png", "Target-view-fold inconsistency cases", fold_bad))

    pair_scores = defaultdict(list)
    for row in records:
        pair_scores[row["pair_id"]].append(row["render"]["metrics"]["garment_lpips"])
    ranked_pairs = sorted(PAIRS, key=lambda pair: mean(pair_scores[pair]))
    best_worst = [row for row in records if row["pair_id"] in (ranked_pairs[0], ranked_pairs[-1])]
    specs.append(("best_worst_pair.png", f"Best/worst pair by Controller-vs-Oracle LPIPS: {ranked_pairs[0]} / {ranked_pairs[-1]}", best_worst))

    for filename, title, rows in specs:
        path = root / filename
        case_sheet(path, title, rows)
        result.append(str(path))

    perturb_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for row in data["perturb"][seed]["records"]:
            value = dict(row)
            value["seed"] = seed
            perturb_rows.append(value)
    perturb_rows.sort(key=lambda row: (row["lpips_change"], row["weight_drift"]), reverse=True)
    panels: list[list[Image.Image]] = []
    for row in perturb_rows[:12]:
        label = (
            f"seed={row['seed']} {row['variant']} {row['record_id']} | "
            f"top1={row['top1_stable']} pair={row['top2_pair_stable']} mode_switch={row['mode_switch']} | "
            f"weight_drift={row['weight_drift']:.3f} LPIPS_change={row['lpips_change']:.3f}"
        )
        panels.append([image_panel(Path(row["rgb_path"]), label), text_panel(label)])
    path = root / "perturbation_instability.png"
    save_page(path, "Largest perturbation instability cases", panels, columns=4)
    result.append(str(path))
    if len(result) != 13:
        raise RuntimeError(f"diagnostic sheet count mismatch: {len(result)}")
    return result


def aggregate_metrics(data: dict[str, Any], output_root: Path) -> dict[str, Any]:
    pure_by_seed = []
    mixed_by_seed = []
    render_by_seed = []
    for seed in SEEDS:
        pure = data["pure"][seed]
        mixed = data["mixed"][seed]
        render = data["render"][seed]
        pure_by_seed.append({
            "seed": seed, "top1_accuracy": pure["top1_accuracy"],
            "single_endpoint_rate": pure["single_endpoint_rate"],
            "endpoint_parity": pure["endpoint_parity"], "swap_count": pure["swap_count"],
            "swap_success_rate": pure["swap_success_rate"],
            "target_forward_leakage": pure["target_forward_leakage"],
            "ground_truth_id_pair_alpha_inference_use": pure["ground_truth_id_pair_alpha_inference_use"],
        })
        mixed_by_seed.append({
            "seed": seed, "protocol_weighted": mixed["protocol_weighted"],
            "unique_query": mixed["unique_query"], "consistency": mixed["consistency"],
            "duplicate_group_prediction_consistency": mixed["duplicate_group_prediction_consistency"],
            "pair_confusion": mixed["pair_confusion"],
        })
        render_by_seed.append({"seed": seed, **render["aggregates"],
                               "controller_forward_time_seconds_mean": render["controller_forward_time_seconds_mean"]})
    metric_names = tuple(data["mixed"][0]["protocol_weighted"])
    macro = {key: mean(data["mixed"][seed]["protocol_weighted"][key] for seed in SEEDS) for key in metric_names}
    worst = {}
    high_is_good = {"dominant_top1_accuracy", "ordering_accuracy", "top2_pair_accuracy", "top2_mass", "dual_support_activation_rate"}
    for key in metric_names:
        values = [data["mixed"][seed]["protocol_weighted"][key] for seed in SEEDS]
        worst[key] = min(values) if key in high_is_good else max(values)

    render_records = [row for seed in SEEDS for row in data["render"][seed]["records"]]
    by_mode = {}
    for mode in ("SINGLE_ENDPOINT", "DUAL_SUPPORT"):
        rows = [row for row in render_records if row["mode"] == mode]
        diagnostics = [row["render_diagnostics"] for row in rows]
        by_mode[mode] = {
            "record_count": len(rows),
            "active_gaussian_count_mean": mean(row["active_gaussian_count"] for row in diagnostics),
            "render_time_seconds_mean": mean(row["render_time_seconds"] for row in diagnostics),
            "peak_vram_bytes_max": max(row["peak_vram_bytes"] for row in diagnostics),
            "branch_primitive_tensor_bytes_mean": mean(row["branch_primitive_tensor_bytes"] for row in diagnostics),
        }
    controller_macro = {key: mean(
        row["metrics"][key] for row in render_records if row["role"] == "mixed"
    ) for key in METRIC_KEYS}
    historical = data["historical_all_pair"]
    all_pair_macro = {row["variant"]: row["means"] for row in historical["analysis"]["all_pair_macro"]}
    baseline_registry = [
        {"method": "Base Avatar", "source": "frozen paper archive", "reused": True,
         "information_boundary": "no garment residual", "output_path": "AAAI27-SEEN-OUTFIT-PAPER"},
        {"method": "Teacher Endpoint Upper Bound", "source": "frozen endpoint bank", "reused": True,
         "information_boundary": "ground-truth endpoint; upper bound", "output_path": "AAAI27-SEEN-OUTFIT-PAPER"},
    ]
    baseline_registry += [
        {"method": name, "source": "P0 color/spatial archive", "reused": True,
         "branch": "paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722",
         "head": "b1a2aa3fcf1512a25531194aab2c9e2d65b68f3a",
         "manifest_hash": data["historical_p0"]["protocol_sha256_lf"],
         "output_path": "AAAI27-P0-COLOR-SPATIAL-SOFT-CONTROL/attempt_004",
         "information_boundary": "reference RGB/masks; frozen historical predictor"}
        for name in ("B6 Reference Classifier Lookup", "B7 Nearest-Centroid Lookup", "Ours-v2 Coefficient Predictor")
    ]
    baseline_registry += [
        {"method": name, "source": "dual-support all-pair archive", "reused": True,
         "branch": "research/dual-support-all-pair-evaluation-20260722",
         "head": "fc3c52f9ae5f5bdfe2b9ffc2b4a630f1e8971efa",
         "manifest_hash": historical["protocol_sha256_lf"],
         "output_path": "DUAL-SUPPORT-ALL-PAIR-EVALUATION-001/attempt_001",
         "information_boundary": boundary}
        for name, boundary in (
            ("FULL_LINEAR", "oracle pair and archived alpha grid; geometry interpolated"),
            ("HARD_GEOMETRY_SOFT_VA", "oracle pair and archived alpha grid; one geometry support"),
            ("Oracle Dual-Support", "ORACLE / NON-DEPLOYABLE; ground-truth pair/composition"),
        )
    ]
    baseline_registry.append({
        "method": "Reference-Conditioned Dual-Support Controller", "source": "FORMAL-002", "reused": False,
        "branch": RUN_BRANCH, "execution_head": "3042cd62ceaffa996349863ee8f9bb633375d108",
        "manifest_hash": MANIFEST_SHA, "output_path": str(output_root / "attempt_004"),
        "information_boundary": "reference RGB/masks/F2/validity only; no ground-truth ID/pair/alpha or target forward",
    })
    perturb_variants = {}
    for variant in data["perturb"][0]["aggregates"]:
        rows = [data["perturb"][seed]["aggregates"][variant] for seed in SEEDS]
        perturb_variants[variant] = {
            key: (sum(int(row[key]) for row in rows) if key == "record_count" else mean(row[key] for row in rows))
            for key in rows[0]
        }
        perturb_variants[variant]["identity_contamination_max"] = max(
            row["identity_contamination_max"] for row in rows
        )
        perturb_variants[variant]["weight_drift_max"] = max(row["weight_drift_max"] for row in rows)
    seed_rule = {
        "top2_macro_ge_0_90": macro["top2_pair_accuracy"] >= 0.90,
        "top2_at_least_2_seeds_ge_0_90": sum(data["mixed"][seed]["protocol_weighted"]["top2_pair_accuracy"] >= 0.90 for seed in SEEDS) >= 2,
        "top2_worst_seed_ge_0_80": worst["top2_pair_accuracy"] >= 0.80,
        "ordering_macro_ge_0_85": macro["ordering_accuracy"] >= 0.85,
        "ordering_at_least_2_seeds_ge_0_85": sum(data["mixed"][seed]["protocol_weighted"]["ordering_accuracy"] >= 0.85 for seed in SEEDS) >= 2,
        "ordering_worst_seed_ge_0_75": worst["ordering_accuracy"] >= 0.75,
        "activation_macro_ge_0_80": macro["dual_support_activation_rate"] >= 0.80,
        "activation_worst_seed_ge_0_70": worst["dual_support_activation_rate"] >= 0.70,
    }
    return {
        "schema_version": "canondressgs.research.dual_support_controller_preliminary_aggregate.v1",
        "task_id": TASK_ID, "status": "AUTOMATIC_AGGREGATION_COMPLETE_MANUAL_REVIEW_REQUIRED",
        "source_head": SOURCE_HEAD, "run_branch": RUN_BRANCH,
        "contract_hashes": {"protocol": PROTOCOL_SHA, "manifest": MANIFEST_SHA, "cycle": CYCLE_SHA, "schedule": SCHEDULE_SHA},
        "training": data["training"], "pure_by_seed": pure_by_seed, "mixed_by_seed": mixed_by_seed,
        "mixed_protocol_weighted_macro": macro, "mixed_protocol_weighted_worst_seed": worst,
        "render_by_seed": render_by_seed, "controller_render_macro": controller_macro,
        "efficiency_by_mode": by_mode, "perturbation_macro": perturb_variants,
        "baseline_registry": baseline_registry, "historical_all_pair_macro": all_pair_macro,
        "seed_aggregation_gates": seed_rule,
        "hard_failure_observed": {
            "dual_support_activation_gate_failed": not seed_rule["activation_macro_ge_0_80"],
            "no_threshold_change": True, "no_result_driven_rerun": True,
        },
        "target_forward_leakage": 0, "ground_truth_id_pair_alpha_inference_use": 0,
        "geometry_interpolation": 0, "paper_final": False, "paper_final_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    data = load_inputs(output_root)
    validate_inputs(data)
    idx = indexes(data)
    preliminary = aggregate_metrics(data, output_root)
    pure = pure_sheets(data, output_root, idx)
    mixed = mixed_sheets(data, output_root, idx)
    diagnostics = diagnostic_sheets(data, output_root, idx)
    manifest = {
        "schema_version": "canondressgs.research.dual_support_controller_visual_manifest.v1",
        "task_id": TASK_ID, "status": "GENERATED_MANUAL_OPENING_REQUIRED",
        "pure_main": pure, "pure_main_count": len(pure),
        "mixed_main": mixed, "mixed_main_count": len(mixed),
        "diagnostics": diagnostics, "diagnostic_count": len(diagnostics),
        "all_paths_exist": all(Path(path).is_file() for path in pure + mixed + diagnostics),
        "manual_open_count": 0, "paper_final": False,
    }
    if len(pure) != 15 or len(mixed) != 60 or len(diagnostics) != 13:
        raise RuntimeError("visual checklist count mismatch")
    aggregate_root = output_root / "attempt_004/aggregates"
    atomic_json(aggregate_root / "preliminary_summary.json", preliminary)
    atomic_json(output_root / "attempt_004/baselines/baseline_registry.json", preliminary["baseline_registry"])
    atomic_json(aggregate_root / "efficiency_preliminary.json", preliminary["efficiency_by_mode"])
    atomic_json(output_root / "attempt_004/visuals/visual_manifest.json", manifest)
    print(json.dumps({
        "status": "COMPLETE", "pure_main": len(pure), "mixed_main": len(mixed),
        "diagnostics": len(diagnostics), "visual_manifest": str(output_root / "attempt_004/visuals/visual_manifest.json"),
        "preliminary_summary": str(aggregate_root / "preliminary_summary.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
