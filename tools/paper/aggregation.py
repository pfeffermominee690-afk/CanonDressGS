from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping

from .evaluate_seen_outfit import ALL_METRICS, EPISODE_METRICS, SEEN_OUTFITS


EXPECTED_SEEDS = (0, 1, 2)


def aggregate_evaluations(evaluations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(evaluations)
    seeds = sorted(item["seed"] for item in items)
    if seeds != list(EXPECTED_SEEDS) or len(set(seeds)) != 3:
        raise ValueError("aggregation requires exactly seeds 0, 1, 2")
    per_outfit: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for evaluation in sorted(items, key=lambda item: item["seed"]):
        episodes = evaluation["per_episode_metrics"]
        if len(episodes) != 20:
            raise ValueError("missing episode: each seed requires 20")
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in episodes:
            groups[record["outfit_id"]].append(record)
        if set(groups) != set(SEEN_OUTFITS) or any(len(group) != 4 for group in groups.values()):
            raise ValueError("each seed requires four episodes for every seen outfit")
        outfit_rows = []
        for outfit in SEEN_OUTFITS:
            row = {"seed": evaluation["seed"], "outfit_id": outfit}
            row.update({metric: fmean(item[metric] for item in groups[outfit]) for metric in EPISODE_METRICS})
            per_outfit.append(row)
            outfit_rows.append(row)
        seed_row = {"seed": evaluation["seed"]}
        seed_row.update({metric: fmean(row[metric] for row in outfit_rows) for metric in EPISODE_METRICS})
        seed_row.update({metric: evaluation["metrics"][metric] for metric in ALL_METRICS if metric not in EPISODE_METRICS})
        per_seed.append(seed_row)
    aggregate = {}
    for metric in ALL_METRICS:
        values = [row[metric] for row in per_seed]
        aggregate[metric] = {"mean": fmean(values), "std": stdev(values), "seed_values": values}
    return {
        "status": "PASS", "aggregation_order": ["episode", "outfit_macro", "seed", "seed_mean_std"],
        "pixel_weighted_micro_average": False, "best_seed_selected": False,
        "seeds": list(EXPECTED_SEEDS), "per_episode_metrics": [record for item in items for record in item["per_episode_metrics"]],
        "per_outfit_metrics": per_outfit, "per_seed_metrics": per_seed,
        "aggregate_metrics": aggregate,
        "held_out_included": False,
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_aggregation(result: Mapping[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "raw_metrics.jsonl", output_dir / "per_episode_metrics.csv",
        output_dir / "per_outfit_metrics.csv", output_dir / "per_seed_metrics.csv",
        output_dir / "aggregate_metrics.json", output_dir / "aggregate_metrics.csv",
    ]
    paths[0].write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in result["per_episode_metrics"]), encoding="utf-8")
    _write_csv(paths[1], list(result["per_episode_metrics"]))
    _write_csv(paths[2], list(result["per_outfit_metrics"]))
    _write_csv(paths[3], list(result["per_seed_metrics"]))
    paths[4].write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = [{"metric": key, **value} for key, value in result["aggregate_metrics"].items()]
    for row in rows:
        row["seed_values"] = json.dumps(row["seed_values"])
    _write_csv(paths[5], rows)
    return paths
