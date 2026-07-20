from __future__ import annotations

from typing import Any

from .evaluate_seen_outfit import FIXED_VIEWS, SEEN_OUTFITS


MARKER = "SYNTHETIC_TEST_ONLY"
TEACHERS = {
    "O01": [-2.0, 0.0, 0.0, 0.0],
    "O02": [-1.0, 1.0, 0.0, 0.0],
    "O03": [0.0, -1.0, 1.0, 0.0],
    "O04": [1.0, 0.0, -1.0, 1.0],
    "O08": [2.0, 0.0, 0.0, -1.0],
}


def build_synthetic_records(
    seed: int,
    asset_fingerprint: str,
    *,
    asset_mismatch: bool = False,
    missing_episode: bool = False,
) -> list[dict[str, Any]]:
    if seed not in (0, 1, 2):
        raise ValueError("synthetic fixture seed must be 0, 1, or 2")
    fingerprint = "0" * 64 if asset_mismatch else asset_fingerprint
    records: list[dict[str, Any]] = [{
        "record_type": "metadata", "marker": MARKER, "seed": seed,
        "asset_fingerprint": fingerprint, "target_forward_input_used": False,
        "efficiency": {
            "trainable_parameter_count": 3076, "basis_storage_bytes": 4096,
            "peak_vram_bytes": 0, "training_time_seconds": 10.0 + seed,
            "inference_time_seconds": 0.02 + seed * 0.001,
            "render_time_seconds": 0.03 + seed * 0.001,
        },
    }]
    for outfit_index, outfit in enumerate(SEEN_OUTFITS):
        for view_index, view in enumerate(FIXED_VIEWS):
            if missing_episode and outfit == "O08" and view == "cond_000347":
                continue
            target = TEACHERS[outfit]
            noise = (seed + 1) * (view_index + 1) * 0.0005
            predicted = [value + noise * (1 if index % 2 == 0 else -1) for index, value in enumerate(target)]
            residual = [((outfit_index + 1) * (index + 1) % 17 - 8) / 10.0 for index in range(40)]
            predicted_residual = [value + noise for value in residual]
            records.append({
                "record_type": "episode", "marker": MARKER, "outfit_id": outfit,
                "view_id": view, "target_reference_overlap": 0,
                "predicted_standardized_coefficients": predicted,
                "target_standardized_coefficients": target,
                "coefficient_normalization": {"mean": [0.1, -0.2, 0.0, 0.3], "std": [1.0, 0.8, 1.2, 0.5]},
                "teacher_standardized_coefficients": TEACHERS,
                "predicted_normalized_residual": predicted_residual,
                "target_normalized_residual": residual,
                "render_metrics": {
                    "garment_rgb_mae": 0.08 + noise,
                    "garment_alpha_mae": 0.04 + noise,
                    "edit_reduction": 0.20 - noise,
                    "target_closer_fraction": 0.75 - noise,
                    "protected_rgb_mae": 0.01 + noise,
                    "background_rgb_mae": 0.001 + noise,
                },
            })
            for swapped in SEEN_OUTFITS:
                if swapped != outfit:
                    records.append({
                        "record_type": "swap", "marker": MARKER, "target_outfit": outfit,
                        "source_outfit": swapped, "view_id": view, "correct_wins": True,
                    })
            records.append({"record_type": "permutation", "marker": MARKER, "outfit_id": outfit, "view_id": view, "max_difference": 1e-7})
            records.extend({"record_type": "single_reference", "marker": MARKER, "outfit_id": outfit, "view_id": view, "variant": index, "correct": True} for index in range(2))
            records.extend({"record_type": "two_reference_dropout", "marker": MARKER, "outfit_id": outfit, "view_id": view, "variant": index, "correct": True} for index in range(2))
            records.append({
                "record_type": "replacement", "marker": MARKER, "outfit_id": outfit,
                "view_id": view, "zero_difference": 0.4 + noise, "base_difference": 0.3 + noise,
            })
    records.append({
        "record_type": "held_out", "marker": MARKER, "outfit_id": "O07",
        "status": "Held-out Diagnostic — FAIL", "included_in_seen_aggregation": False,
    })
    return records


def synthetic_table_source(aggregate: dict[str, Any]) -> dict[str, Any]:
    base_metrics = aggregate["aggregate_metrics"]
    methods = {}
    for index, name in enumerate(("B0", "B1", "B2", "B3", "B4", "B5", "Ours")):
        methods[name] = {
            key: {"mean": value["mean"] + index * 0.001, "std": value["std"]}
            for key, value in base_metrics.items()
        }
    methods["B0"].pop("nearest_teacher_accuracy", None)
    ablations = {}
    for index, name in enumerate(("A1-K1", "A1-K2", "A1-K3", "A1-K4", "A2", "A3", "A4", "A5", "A6", "A7-Kref1", "A7-Kref2", "A7-Kref3")):
        ablations[name] = {
            key: {"mean": value["mean"] + index * 0.001, "std": value["std"]}
            for key, value in base_metrics.items()
        }
    return {
        "marker": MARKER, "methods": methods, "ablations": ablations,
        "held_out": {
            "O07 teacher": {"status": "REFERENCE_ONLY"},
            "O07 projection": {"status": "FAIL"},
            "O07 reference prediction": {"status": "FAIL"},
        },
    }
