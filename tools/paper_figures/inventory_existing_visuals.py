#!/usr/bin/env python3
"""Inventory visual assets from explicit, sealed output roots.

The scanner is intentionally allowlist-only. It never follows symlinks, prunes
checkpoint and raw-data directories, and refuses any root that overlaps a
declared active output. Source files are hashed in place; only deterministic
thumbnails are written.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError


VISUAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
DEFAULT_PRUNED_PARTS = {
    "checkpoints",
    "checkpoint",
    "datasets",
    "dataset",
    "downloads",
    "raw",
    "weights",
}
ATTEMPT_RE = re.compile(r"(?i)(attempt_\d+)")
SEED_RE = re.compile(r"(?i)(?:seed[_-]?)(\d+)")
STEP_RE = re.compile(r"(?i)(?:step|iteration|iter)[_-]?(\d+)")
GARMENT_RE = re.compile(r"(?<![A-Za-z0-9])O\d{2}(?![A-Za-z0-9])", re.IGNORECASE)
PAIR_RE = re.compile(r"(?<![A-Za-z0-9])(O\d{2})[_-](O\d{2})(?![A-Za-z0-9])", re.IGNORECASE)
ROTATION_RE = re.compile(r"(?i)(?:rotation|rot)[_-]?(\d+)")
CAMERA_RE = re.compile(r"(?i)(?:camera|cam|view|cond)[_-]?(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail-root", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--active-root", action="append", default=[])
    parser.add_argument("--max-thumbnail-edge", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def stable_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def overlaps(first: Path, second: Path) -> bool:
    a = normalized(first)
    b = normalized(second)
    try:
        return os.path.commonpath([a, b]) in {a, b}
    except ValueError:
        return False


def match_any(value: str, patterns: Iterable[str]) -> bool:
    normalized_value = value.replace(os.sep, "/")
    return any(fnmatch.fnmatch(normalized_value, pattern) for pattern in patterns)


def infer_attempt(relative_path: str, source: Dict[str, Any]) -> Optional[str]:
    match = ATTEMPT_RE.search(relative_path)
    return match.group(1).lower() if match else source.get("attempt_id")


def infer_experiment(relative_path: str, source: Dict[str, Any]) -> str:
    if source.get("experiment_id"):
        return str(source["experiment_id"])
    parts = Path(relative_path).parts
    if parts and parts[0].lower().startswith(("paper-", "subject")):
        return parts[0]
    return str(source["source_id"])


def infer_visual_type(path_text: str, source: Dict[str, Any]) -> str:
    lower = path_text.lower()
    if source.get("license_status") == "LICENSE_RESTRICTED_DO_NOT_EXPORT":
        return "LICENSE_RESTRICTED"
    if "screenshot" in lower or "screen_capture" in lower or "terminal" in lower:
        return "UI_SCREENSHOT"
    if any(token in lower for token in ("document", "report_page", "table_capture")):
        return "DOCUMENT_SCREENSHOT"
    if any(token in lower for token in ("plot", "chart", "curve", "trajectory", "histogram")):
        return "SCIENTIFIC_METRIC_PLOT"
    if any(token in lower for token in ("contact", "sheet", "comparison", "overview", "montage")):
        return "VISUAL_COMPARISON_SHEET"
    if source.get("asset_type"):
        return str(source["asset_type"])
    if any(token in lower for token in ("failure", "diagnostic", "artifact", "error_map")):
        return "FAILURE_DIAGNOSTIC"
    return "SCIENTIFIC_RENDER"


def infer_fields(relative_path: str, source: Dict[str, Any]) -> Dict[str, Any]:
    pair_match = PAIR_RE.search(relative_path)
    garments = sorted({item.upper() for item in GARMENT_RE.findall(relative_path)})
    seed_match = SEED_RE.search(relative_path)
    step_match = STEP_RE.search(relative_path)
    rotation_match = ROTATION_RE.search(relative_path)
    camera_match = CAMERA_RE.search(relative_path)
    experiment_id = infer_experiment(relative_path, source)
    method = source.get("method", "NOT_RECORDED")
    baseline = source.get("baseline_or_ablation", "NOT_RECORDED")
    if experiment_id.startswith("PAPER-"):
        method = experiment_id
        baseline = experiment_id.removeprefix("PAPER-")
    return {
        "attempt_id": infer_attempt(relative_path, source),
        "experiment_id": experiment_id,
        "method": method,
        "baseline_or_ablation": baseline,
        "identity": source.get("identity", "NOT_RECORDED"),
        "garment": garments[0] if len(garments) == 1 else "NOT_RECORDED",
        "garment_pair": (
            f"{pair_match.group(1).upper()}_{pair_match.group(2).upper()}"
            if pair_match
            else "NOT_RECORDED"
        ),
        "reference_set": source.get("reference_set", "NOT_RECORDED"),
        "pose": source.get("pose", "NOT_RECORDED"),
        "camera": camera_match.group(1) if camera_match else "NOT_RECORDED",
        "split": source.get("split", "NOT_RECORDED"),
        "rotation": int(rotation_match.group(1)) if rotation_match else None,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "checkpoint_step": int(step_match.group(1)) if step_match else None,
        "checkpoint_sha256": source.get("checkpoint_sha256", "NOT_RECORDED"),
        "renderer_sha256": source.get("renderer_sha256", "NOT_RECORDED"),
        "renderer_config_sha256": source.get("renderer_config_sha256", "NOT_RECORDED"),
    }


def raster_info(path: Path) -> Tuple[int, int, int]:
    with Image.open(path) as image:
        width, height = image.size
        channels = len(image.getbands())
        image.verify()
    return width, height, channels


def write_thumbnail(source_path: Path, destination: Path, max_edge: int) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.load()
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        original_size = image.size
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        image.save(destination, format="PNG", compress_level=9, optimize=False)
        return {
            "input_size": list(original_size),
            "output_size": list(image.size),
            "resize_filter": "LANCZOS",
            "background": "#FFFFFF",
            "crop": None,
        }


def iter_visual_paths(source: Dict[str, Any], active_roots: Sequence[Path]) -> Iterable[Path]:
    root = Path(source["root"])
    for active_root in active_roots:
        if overlaps(root, active_root):
            raise RuntimeError(f"source root overlaps active root: {root} vs {active_root}")
    if not root.is_dir():
        return
    include = source.get("include", [])
    exclude = source.get("exclude", [])
    pruned = {str(item).lower() for item in source.get("pruned_parts", DEFAULT_PRUNED_PARTS)}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if name.lower() not in pruned
            and not any(overlaps(current_path / name, active) for active in active_roots)
        )
        for filename in sorted(files):
            path = current_path / filename
            if path.suffix.lower() not in VISUAL_SUFFIXES or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if include and not match_any(relative, include):
                continue
            if exclude and match_any(relative, exclude):
                continue
            yield path


def main() -> int:
    args = parse_args()
    spec = load_json(args.spec)
    active_roots = [Path(item) for item in [*spec.get("active_roots", []), *args.active_root]]
    records: List[Dict[str, Any]] = []
    source_audits: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for source in sorted(spec["sources"], key=lambda item: item["source_id"]):
        root = Path(source["root"])
        discovered = 0
        if source.get("metadata_only"):
            source_audits.append({
                "source_id": source["source_id"],
                "root": str(root),
                "status": "METADATA_ONLY_NO_MEDIA_READ",
                "visual_count": 0,
            })
            continue
        if not root.is_dir():
            source_audits.append({
                "source_id": source["source_id"],
                "root": str(root),
                "status": "MISSING_ROOT",
                "visual_count": 0,
            })
            continue
        for original_path in iter_visual_paths(source, active_roots):
            discovered += 1
            relative_path = original_path.relative_to(root).as_posix()
            try:
                original_sha = sha256_file(original_path)
                width, height, channels = raster_info(original_path)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                errors.append({"path": str(original_path), "error": str(error)})
                continue
            inferred = infer_fields(relative_path, source)
            provenance_complete = bool(
                source.get("source_head") and inferred["attempt_id"] and original_sha
            )
            eligibility = source.get("paper_eligibility", "REQUIRES_MANUAL_ADJUDICATION")
            claim_status = source.get("claim_status", eligibility)
            if not provenance_complete:
                eligibility = "UNKNOWN_PROVENANCE_DO_NOT_USE"
                claim_status = "UNKNOWN_PROVENANCE_DO_NOT_USE"
            asset_id_hash = hashlib.sha256(
                f"{source['source_id']}\n{relative_path}\n{original_sha}".encode("utf-8")
            ).hexdigest()[:16]
            asset_id = f"ASSET-{asset_id_hash.upper()}"
            record: Dict[str, Any] = {
                "asset_id": asset_id,
                "source_category": source["source_category"],
                "original_absolute_path": str(original_path),
                "original_relative_path": relative_path,
                "source_branch": source["source_branch"],
                "source_head": source["source_head"],
                "output_root": str(root),
                "task_id": source["task_id"],
                **inferred,
                "original_sha256": original_sha,
                "width": width,
                "height": height,
                "channels": channels,
                "crop": None,
                "resize": None,
                "composition_transform": "IDENTITY",
                "output_sha256": original_sha,
                "asset_type": infer_visual_type(relative_path, source),
                "claim_status": claim_status,
                "license_status": source.get("license_status", "PROJECT_AUTHORIZED_INTERNAL_EVIDENCE"),
                "paper_eligibility": eligibility,
                "selection_rule": source.get("selection_rule", "REQUIRES_MANUAL_ADJUDICATION"),
                "provenance_complete": provenance_complete,
                "notes": source.get("notes", ""),
            }
            if args.thumbnail_root and not args.dry_run:
                thumbnail_path = args.thumbnail_root / source["source_id"] / f"{asset_id}.png"
                transform = write_thumbnail(original_path, thumbnail_path, args.max_thumbnail_edge)
                record["thumbnail_absolute_path"] = str(thumbnail_path)
                record["thumbnail_sha256"] = sha256_file(thumbnail_path)
                record["thumbnail_transform"] = transform
            else:
                record["thumbnail_absolute_path"] = None
                record["thumbnail_sha256"] = None
                record["thumbnail_transform"] = None
            records.append(record)
        source_audits.append({
            "source_id": source["source_id"],
            "root": str(root),
            "status": "AUDITED" if discovered else "AUDITED_NO_VISUALS",
            "visual_count": discovered,
        })

    records.sort(key=lambda item: (item["source_category"], item["original_absolute_path"]))
    sha_groups: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        sha_groups[record["original_sha256"]].append(record["asset_id"])
    duplicates = [
        {"sha256": sha, "asset_ids": sorted(asset_ids), "count": len(asset_ids)}
        for sha, asset_ids in sorted(sha_groups.items())
        if len(asset_ids) > 1
    ]
    type_counts = Counter(record["asset_type"] for record in records)
    eligibility_counts = Counter(record["paper_eligibility"] for record in records)
    result = {
        "schema_version": "paper_figure_asset_registry.v1",
        "task_id": spec["task_id"],
        "audit_date": spec["audit_date"],
        "active_run_disposition": "ACTIVE_RUN_EXCLUDED_PENDING_FINAL_SEAL",
        "active_run_files_consumed": 0,
        "avatarrex_media_exports": 0,
        "sources": source_audits,
        "assets": records,
        "duplicate_sha_groups": duplicates,
        "errors": errors,
        "statistics": {
            "raw_visual_files_discovered": len(records),
            "unique_sha_visuals": len(sha_groups),
            "duplicate_visuals": len(records) - len(sha_groups),
            "asset_type_counts": dict(sorted(type_counts.items())),
            "eligibility_counts": dict(sorted(eligibility_counts.items())),
            "unknown_provenance_assets": sum(not item["provenance_complete"] for item in records),
            "license_restricted_assets": sum(
                item["license_status"] == "LICENSE_RESTRICTED_DO_NOT_EXPORT" for item in records
            ),
        },
    }
    if args.dry_run:
        print(json.dumps(result["statistics"], indent=2, sort_keys=True))
        return 0
    stable_json_write(args.output, result)
    if args.log:
        stable_json_write(args.log, {
            "task_id": spec["task_id"],
            "status": "PASS" if not errors else "PASS_WITH_DECODE_ERRORS_RECORDED",
            "statistics": result["statistics"],
            "errors": errors,
        })
    print(json.dumps(result["statistics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
