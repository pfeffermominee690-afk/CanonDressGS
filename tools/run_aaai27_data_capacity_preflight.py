#!/usr/bin/env python3
"""Freeze AAAI-27 data/capacity gate inputs before any image or Oracle call.

The local mode audits immutable Windows assets, extracts deterministic Jay cells,
and records the generation/model-route decision. The cloud mode adds base,
graph, interpolation, environment, and Git fingerprints. Neither mode performs
network requests, image generation, segmentation, rendering, or optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


TASK_ID = "SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001"
SCHEMA = "canondressgs.aaai27.data_capacity_gate.preflight.v1"
CANDIDATES = ("O01", "O02", "O03", "O04", "O06", "O07", "O08")
CONDITIONS = (
    ("cond_000000", "front"),
    ("cond_000318", "back"),
    ("cond_000017", "left"),
    ("cond_000347", "right"),
)
REQUIRED_MODEL = "gpt-image-2"
EXPECTED_LONG_HEAD = "9fc88033b407136073f7bddc6ffca6dd4dd1e0f5"
EXPECTED_INPUT_SPRINT_HEAD = "fece64eed3b80043cd7703364f668a3ef5f9d65f"
EXPECTED_BRANCH = "sprint/aaai27-20260718"
MAPPING_PATH = Path(
    r"E:\data_pre\audit_subject02_outfit_conversion_pilot_v1\outfit_cell_mapping_v2_normalized.json"
)
UNION_PATH = Path(
    r"E:\data_pre\audit_jay_coverage_supplement_v1\jay_production_union_manifest_v2.json"
)
V4_ROOT = Path(r"E:\data_pre\audit_subject02_direct_edit_v4")
CONDITION_ROOT = Path(r"E:\data_pre\conditions_300_hand_strict_candidates")
BASE_RENDER_ROOT = Path(
    r"E:\data_pre\staging_subject02_layered_composite_v3\mask_backend_v3a\base_assets\condition_contract_projection_v3\base_renders"
)

O01_VISUAL_OBSERVATIONS = {
    "cond_000000": "Opened with Codex view_image: one complete subject02 person, correct front view and pose, gray hoodie/black trousers, full head/hands/feet, white background.",
    "cond_000318": "Opened with Codex view_image: one complete subject02 person, correct back view and arm pose, gray hoodie/black trousers, full body, white background.",
    "cond_000017": "Opened with Codex view_image: one complete subject02 person, correct left view and pose, gray hoodie/black trousers, full body, white background.",
    "cond_000347": "Opened with Codex view_image: one complete subject02 person, correct right view and pose, gray hoodie/black trousers, full body, white background.",
}

PRIMARY_TRAIN = ("O01", "O02", "O04", "O06")
PRIMARY_UNSEEN = ("O03", "O08")
RESERVE = "O07"
GENERATION_HARD_FAILURES = frozenset(
    {
        "pose_changed",
        "view_changed",
        "wrong_body_count",
        "severe_crop",
        "wrong_outfit",
        "large_edit_region_distortion",
        "limb_structure_failure",
        "second_person",
        "major_body_proportion_change",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "cloud"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/aaai27/subject02_data_capacity_gate_v1.yaml"))
    parser.add_argument("--data-pre-root", type=Path, default=Path(r"E:\data_pre"))
    parser.add_argument("--declared-credential-provider", default="UNDECLARED")
    parser.add_argument("--credential-route-verified", action="store_true")
    parser.add_argument("--images-actually-opened", action="store_true")
    parser.add_argument(
        "--module4b-fingerprint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/"
            "SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001/contract/module4b_input_fingerprint.json"
        ),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/chkpnt100000.pth"),
    )
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


def make_output_tree(root: Path) -> None:
    directories = (
        "contract",
        "generation",
        "generation_audit",
        "masks",
        "dataset",
        "checker",
        "support_compatibility",
        "visual_acceptance",
        "selection",
        "paper_evidence",
        "final_adjudication",
        "reference_sets/cells",
    )
    for name in directories:
        (root / name).mkdir(parents=True, exist_ok=True)
    for outfit in CANDIDATES:
        (root / "oracle" / outfit).mkdir(parents=True, exist_ok=True)


def assert_git_contract(repo: Path) -> dict[str, Any]:
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    long_head = git(repo, "rev-parse", "refs/heads/pipeline/full-dressable-20260715")
    status = git(repo, "status", "--short")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Wrong branch: {branch}")
    if long_head != EXPECTED_LONG_HEAD:
        raise RuntimeError(f"Long-term branch changed: {long_head}")
    if status:
        raise RuntimeError(f"Worktree must be clean before formal preflight:\n{status}")
    if not git(repo, "merge-base", "--is-ancestor", EXPECTED_INPUT_SPRINT_HEAD, head) == "":
        # merge-base --is-ancestor emits no stdout on success.
        raise RuntimeError("Unexpected merge-base output")
    return {"branch": branch, "head": head, "long_term_head": long_head, "status_short": status}


def ensure_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        return {"width": image.width, "height": image.height, "mode": image.mode}


def existing_target_is_reusable(
    record: dict[str, Any],
    dimensions: dict[str, Any],
    input_checks: dict[str, bool],
    images_actually_opened: bool,
) -> bool:
    """Apply the frozen reuse gate without inferring an unrecorded model ID."""

    explicit_api_backends = {"openai_images_edit", "gpt_image_2_images_edit"}
    return bool(
        all(input_checks.values())
        and dimensions["width"] == 1024
        and dimensions["height"] == 1536
        and record.get("model_id") == REQUIRED_MODEL
        and record.get("generation_backend") in explicit_api_backends
        and record.get("status") == "SUCCESS"
        and images_actually_opened
    )


def generation_gate_status(failure_flags: set[str]) -> str:
    """Keep protected-only diagnostics non-blocking while preserving hard failures."""

    if failure_flags & GENERATION_HARD_FAILURES:
        return "FAIL"
    if failure_flags:
        return "PROTECTED_ONLY_DIAGNOSTIC_WARN"
    return "PASS"


def select_benchmark(outfit_status: dict[str, str]) -> dict[str, Any]:
    """Apply only the preregistered replacement and five-outfit fallback rules."""

    if set(outfit_status) != set(CANDIDATES):
        raise ValueError("Selection requires exactly the seven preregistered candidates")
    passed = {outfit for outfit, status in outfit_status.items() if status == "OUTFIT_GATE_PASS"}
    train = [outfit for outfit in PRIMARY_TRAIN if outfit in passed]
    promoted: list[str] = []
    for outfit in PRIMARY_UNSEEN:
        if len(train) == len(PRIMARY_TRAIN):
            break
        if outfit in passed:
            train.append(outfit)
            promoted.append(outfit)
    unseen = [outfit for outfit in PRIMARY_UNSEEN if outfit in passed and outfit not in promoted]
    if len(unseen) < 2 and RESERVE in passed:
        unseen.append(RESERVE)
    train = train[:4]
    unseen = unseen[:2]
    selected = train + unseen
    six_outfit_go = len(train) == 4 and len(unseen) == 2
    five_outfit_go = len(train) == 4 and len(unseen) == 1 and len(selected) == 5
    go = six_outfit_go or five_outfit_go
    return {
        "status": "GO" if go else "NO_GO",
        "train_outfits": train if go else [],
        "unseen_outfits": unseen if go else [],
        "reserve_enabled": RESERVE in selected,
        "target_count": 32 * len(selected) if go else 0,
        "selected_outfits": selected if go else [],
        "passed_outfits": sorted(passed),
        "failed_or_nonpass_outfits": sorted(set(CANDIDATES) - passed),
        "five_outfit_fallback": five_outfit_go,
    }


def condition_number(condition_id: str) -> str:
    return condition_id.split("_", 1)[1]


def resolve_local_paths(data_root: Path) -> dict[str, Path]:
    return {
        "candidate_audit": Path("artifacts/aaai27_sprint/candidate_outfit_audit.json"),
        "condition_selection": Path("artifacts/aaai27_sprint/condition_selection_32.json"),
        "gate_manifest": Path("artifacts/aaai27_sprint/target_generation_gate_28_manifest.json"),
        "full_manifest": Path("artifacts/aaai27_sprint/target_generation_full_192_manifest.json"),
        "experiment_registry": Path("artifacts/aaai27_sprint/experiment_registry.json"),
        "mapping": data_root / "audit_subject02_outfit_conversion_pilot_v1" / "outfit_cell_mapping_v2_normalized.json",
        "union": data_root / "audit_jay_coverage_supplement_v1" / "jay_production_union_manifest_v2.json",
        "v4_prompt": data_root / "audit_subject02_direct_edit_v4" / "direct_edit_prompt_contract_v4.json",
        "v4_raw": data_root / "audit_subject02_direct_edit_v4" / "direct_edit_raw_manifest_v4.json",
        "v4_final": data_root / "audit_subject02_direct_edit_v4" / "final_manifest_v4.json",
        "v4_visual": data_root / "audit_subject02_direct_edit_v4" / "visual_acceptance_v4.json",
    }


def build_reference_sets(
    output_root: Path,
    gate_manifest: dict[str, Any],
    mapping: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], Path]]:
    mapping_by_id = {entry["canonical_outfit_id"]: entry for entry in mapping["entries"]}
    gate_records = {(record["outfit_id"], record["condition_id"]): record for record in gate_manifest["records"]}
    all_sets: list[dict[str, Any]] = []
    cell_paths: dict[tuple[str, str], Path] = {}
    for outfit in CANDIDATES:
        entry = mapping_by_id[outfit]
        references = []
        for condition_id, view in CONDITIONS:
            record = gate_records[(outfit, condition_id)]
            reference = record["donor_references"][0]
            sheet = Path(reference["sheet_path"])
            ensure_files([sheet])
            actual_sheet_hash = sha256(sheet)
            if actual_sheet_hash != reference["sheet_sha256"]:
                raise RuntimeError(f"Jay sheet hash mismatch: {sheet}")
            crop = tuple(int(value) for value in entry["crop_box_xyxy"])
            cell_path = output_root / "reference_sets" / "cells" / outfit / f"{condition_id}.png"
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            if cell_path.exists():
                raise FileExistsError(f"Reference cell already exists: {cell_path}")
            with Image.open(sheet) as image:
                if image.size != (1536, 1024):
                    raise ValueError(f"Unexpected sheet size for {sheet}: {image.size}")
                cell = image.convert("RGB").crop(crop)
                if cell.size != (512, 512):
                    raise ValueError(f"Unexpected crop size for {outfit}/{condition_id}: {cell.size}")
                cell.save(cell_path, format="PNG")
            cell_paths[(outfit, condition_id)] = cell_path
            references.append(
                {
                    "condition_id": condition_id,
                    "view": view,
                    "identity": "Jay",
                    "outfit_id": outfit,
                    "cell_id": entry["physical_position_id"],
                    "generator_outfit_id": entry["generator_outfit_id"],
                    "garment_semantics": entry["description_en"],
                    "source_sheet": str(sheet.resolve()),
                    "source_sheet_sha256": actual_sheet_hash,
                    "source_sheet_manifest_sha256": reference["sheet_sha256"],
                    "crop_box_xyxy": list(crop),
                    "cell_path": str(cell_path.resolve()),
                    "cell_sha256": sha256(cell_path),
                    "cell_image": image_metadata(cell_path),
                    "mapping_evidence": {
                        "contract": "outfit_mapping_v2",
                        "visual_consensus_ratio": entry["visual_consensus_ratio"],
                        "Jay_consensus": entry["Jay_consensus"],
                    },
                }
            )
        payload = {
            "schema_version": SCHEMA,
            "task_id": TASK_ID,
            "outfit_id": outfit,
            "identity": "Jay",
            "garment_semantics": entry["description_en"],
            "frozen": True,
            "reference_count": 4,
            "references": references,
        }
        path = output_root / "reference_sets" / f"reference_set_{outfit}.json"
        write_json(path, payload)
        payload["manifest_path"] = str(path.resolve())
        payload["manifest_sha256"] = sha256(path)
        all_sets.append(payload)
    return all_sets, cell_paths


def audit_o01_existing_targets(
    output_root: Path,
    raw_manifest: dict[str, Any],
    final_manifest: dict[str, Any],
    images_actually_opened: bool,
) -> dict[str, Any]:
    raw_by_id = {record["sample_id"]: record for record in raw_manifest["records"]}
    final_by_id = {record["sample_id"]: record for record in final_manifest["records"]}
    records = []
    for condition_id, view in CONDITIONS:
        sample_id = f"{condition_id}_O01"
        raw_record = raw_by_id[sample_id]
        final_record = final_by_id[sample_id]
        raw_path = Path(final_record["raw_path"])
        required_inputs = [
            Path(raw_record["base_image_path"]),
            Path(raw_record["clay_path"]),
            Path(raw_record["garment_reference_path"]),
            raw_path,
        ]
        ensure_files(required_inputs)
        input_checks = {
            "base": sha256(required_inputs[0]) == raw_record["base_sha256"],
            "clay": sha256(required_inputs[1]) == raw_record["clay_sha256"],
            "garment_reference": sha256(required_inputs[2]) == raw_record["garment_reference_sha256"],
        }
        dimensions = image_metadata(raw_path)
        model_verified = raw_record.get("model_id") == REQUIRED_MODEL
        backend_verified = raw_record.get("generation_backend") in {
            "openai_images_edit",
            "gpt_image_2_images_edit",
        }
        raw_success = raw_record.get("status") == "SUCCESS"
        reusable = existing_target_is_reusable(
            raw_record,
            dimensions,
            input_checks,
            images_actually_opened,
        )
        records.append(
            {
                "sample_id": sample_id,
                "condition_id": condition_id,
                "view": view,
                "raw_path": str(raw_path),
                "raw_sha256": sha256(raw_path),
                "raw_manifest_status": raw_record.get("status"),
                "raw_manifest_backend": raw_record.get("generation_backend"),
                "raw_manifest_model_id": raw_record.get("model_id"),
                "required_model_id": REQUIRED_MODEL,
                "input_hash_checks": input_checks,
                "image": dimensions,
                "images_actually_opened": images_actually_opened,
                "inspection_method": "Codex view_image high detail" if images_actually_opened else None,
                "visual_observation": O01_VISUAL_OBSERVATIONS[condition_id] if images_actually_opened else None,
                "model_verified": model_verified,
                "backend_verified": backend_verified,
                "reusable": reusable,
                "reuse_decision": "REUSED_VERIFIED_EXISTING_TARGET" if reusable else "REJECTED_UNVERIFIED_GENERATION_MODEL",
            }
        )
    reused_count = sum(int(record["reusable"]) for record in records)
    payload = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "status": "PASS_REUSE_AUDIT_NO_REUSE" if reused_count == 0 else "PASS_REUSE_AUDIT",
        "required_model": REQUIRED_MODEL,
        "existing_count": 4,
        "reused_count": reused_count,
        "regeneration_required_count": 4 - reused_count,
        "reason": (
            "The frozen V4 records identify codex_builtin_imagegen / UNEXPOSED_PLATFORM_IMAGE_MODEL, "
            "so none can be claimed as verified gpt-image-2 even though all four raw images are visually usable."
        ),
        "records": records,
    }
    write_json(output_root / "O01_existing_target_reuse_audit.json", payload)
    lines = [
        "# O01 Existing Target Reuse Audit",
        "",
        f"- Required model: `{REQUIRED_MODEL}`",
        f"- Existing targets audited: `{len(records)}`",
        f"- Reused: `{reused_count}`",
        f"- Regeneration required: `{4 - reused_count}`",
        "- Status: **PASS_REUSE_AUDIT_NO_REUSE**",
        "",
        "All four raw images were actually opened and are visually usable as O01 full-body views. They are not reusable in this gate because the frozen metadata does not expose or verify `gpt-image-2`. This is a provenance/model-contract rejection, not a visual rejection. The raw files remain unchanged.",
    ]
    write_text(output_root / "O01_EXISTING_TARGET_REUSE_AUDIT.md", "\n".join(lines))
    return payload


def direct_edit_prompt(outfit: dict[str, Any], condition_id: str, view: str, existing_o01: dict[str, Any] | None) -> tuple[str, str]:
    if existing_o01 is not None:
        return existing_o01["prompt"], "exact_frozen_V4_O01_prompt"
    semantics = outfit["description_en"]
    prompt = (
        "Use case: identity-preserve\n"
        "Asset type: CanonDressGS subject02 direct outfit-edit data gate; one vertical full-body image\n"
        f"Primary request: Edit only the clothing worn by subject02 in Image 1. Replace it with the {semantics} shown in Image 3, matching Image 3 exactly in garment category, color, texture, material, fit, sleeves, upper-garment structure, and trouser structure.\n"
        "Input images: Image 1 is the primary identity/geometry image and the only authority for subject identity and composition. Image 2 is a geometry-only clay reference. Image 3 is a garment-appearance-only reference.\n"
        f"Subject and composition invariants: The output must show exactly the same subject02 person as Image 1. Preserve exactly the face, hairstyle, skin tone, body proportions, pose, limb locations, hand positions, shoes, camera angle, {view} view direction, subject scale, image framing, white background, and lighting from Image 1. Image 2 only confirms the original pose, limb geometry, silhouette, orientation, and {view} view. Never copy its gray clay material. Image 3 only defines garment category, color, texture, material, fit, and clothing structures; never copy Jay's face, hair, skin, body proportions, pose, camera, hands, shoes, or background.\n"
        "Editing requirements: Change only the clothing. Remove every visible part of Image 1's original clothing before adding the new outfit. Do not retain a double neckline, collar, sleeves, cuffs, trouser legs, hood, or hem. Keep protected identity regions unchanged.\n"
        f"Output constraints: One complete person only, full body visible, single portrait image, not a sheet or collage. Keep the exact {view} view; do not rotate the person toward another view. Do not change the arms, legs, torso, hands, or pose. Keep the white background unchanged. No text, labels, numbers, watermark, border, extra person, duplicate body, missing limb, extra limb, crop, padding, or reframing. Target aspect ratio 2:3 and target size 1024x1536."
    )
    return prompt, "V4_contract_template_plus_mapping_v2_semantics"


def build_generation_plan(
    repo: Path,
    output_root: Path,
    mapping: dict[str, Any],
    v4_prompt: dict[str, Any],
    reference_cells: dict[tuple[str, str], Path],
    o01_audit: dict[str, Any],
    declared_provider: str,
    route_verified: bool,
) -> dict[str, Any]:
    mapping_by_id = {entry["canonical_outfit_id"]: entry for entry in mapping["entries"]}
    o01_prompts = {record["sample_id"]: record for record in v4_prompt["records"] if record["outfit_id"] == "O01"}
    reused = {record["sample_id"] for record in o01_audit["records"] if record["reusable"]}
    records = []
    for outfit_id in CANDIDATES:
        outfit = mapping_by_id[outfit_id]
        for condition_id, view in CONDITIONS:
            number = condition_number(condition_id)
            sample_id = f"{condition_id}_{outfit_id}"
            base_path = BASE_RENDER_ROOT / f"{condition_id}_rgb.png"
            clay_path = CONDITION_ROOT / "images" / f"{condition_id}.png"
            pose_path = CONDITION_ROOT / "poses" / f"pose_{number}.npz"
            camera_path = CONDITION_ROOT / "cameras" / f"camera_{number}.json"
            reference_path = reference_cells[(outfit_id, condition_id)]
            ensure_files([base_path, clay_path, pose_path, camera_path, reference_path])
            prompt, prompt_source = direct_edit_prompt(
                outfit,
                condition_id,
                view,
                o01_prompts.get(sample_id) if outfit_id == "O01" else None,
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "outfit_id": outfit_id,
                    "condition_id": condition_id,
                    "view": view,
                    "status": "REUSED_VERIFIED_EXISTING_TARGET" if sample_id in reused else "PENDING_GENERATION",
                    "required_model": REQUIRED_MODEL,
                    "endpoint": "/v1/images/edits",
                    "input_roles": {
                        "image_1": "subject02 base render: identity/composition authority and edit target",
                        "image_2": "subject02 clay: target pose/camera/view geometry reference",
                        "image_3": "Jay frozen garment cell: garment appearance only",
                    },
                    "inputs": {
                        "base": {"path": str(base_path), "sha256": sha256(base_path)},
                        "clay": {"path": str(clay_path), "sha256": sha256(clay_path)},
                        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
                        "pose": {"path": str(pose_path), "sha256": sha256(pose_path)},
                        "camera": {"path": str(camera_path), "sha256": sha256(camera_path)},
                    },
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_source": prompt_source,
                    "output_path": str(output_root / "generation" / "targets" / outfit_id / condition_id / "raw_direct_edit.png"),
                    "raw_immutable": True,
                    "target_used_as_forward_condition": False,
                }
            )
    pending_count = sum(record["status"] == "PENDING_GENERATION" for record in records)
    credential_present = bool(os.environ.get("OPENAI_API_KEY"))
    provider_is_official = declared_provider.strip().lower() in {"openai", "openai_official", "official_openai"}
    execution_allowed = bool(credential_present and provider_is_official and route_verified)
    cli_path = Path.home() / ".codex" / "skills" / ".system" / "imagegen" / "scripts" / "image_gen.py"
    if not cli_path.is_file():
        cli_path = Path(r"C:\Users\发如雪\.codex\skills\.system\imagegen\scripts\image_gen.py")
    ensure_files([cli_path])
    route = {
        "required_model": REQUIRED_MODEL,
        "required_endpoint": "/v1/images/edits",
        "bundled_cli": str(cli_path),
        "bundled_cli_sha256": sha256(cli_path),
        "credential_present": credential_present,
        "credential_length": len(os.environ.get("OPENAI_API_KEY", "")),
        "credential_value_recorded": False,
        "declared_credential_provider": declared_provider,
        "provider_is_official_openai": provider_is_official,
        "credential_route_verified": route_verified,
        "execution_allowed": execution_allowed,
        "built_in_imagegen_allowed_as_substitute": False,
        "reason": (
            "The built-in image tool does not expose a verifiable model ID. The bundled explicit-model CLI uses the official OpenAI route, but the currently declared credential is not an official OpenAI credential and no compatible verified route is configured."
            if not execution_allowed
            else "Explicit gpt-image-2 images.edit route and credential have been independently verified."
        ),
    }
    payload = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "created_at": now(),
        "record_count": len(records),
        "reused_count": len(reused),
        "pending_generation_count": pending_count,
        "generation_call_budget": {"primary_max": 28, "retry_max": 4, "total_max": 32, "actual_primary": 0, "actual_retry": 0, "actual_total": 0},
        "model_route": route,
        "records": records,
    }
    write_json(output_root / "generation" / "generation_plan.json", payload)
    write_json(output_root / "generation" / "generation_call_budget.json", payload["generation_call_budget"])
    write_json(output_root / "generation" / "model_route_preflight.json", route)
    return payload


def local_mode(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    output_root = args.output_dir.resolve()
    if output_root.exists():
        raise FileExistsError(f"Append-only output already exists: {output_root}")
    git_state = assert_git_contract(repo)
    make_output_tree(output_root)
    sources = resolve_local_paths(args.data_pre_root.resolve())
    absolute_sources = {name: (repo / path).resolve() if not path.is_absolute() else path for name, path in sources.items()}
    ensure_files(list(absolute_sources.values()) + [(repo / args.config).resolve()])
    candidate_audit = read_json(absolute_sources["candidate_audit"])
    gate_manifest = read_json(absolute_sources["gate_manifest"])
    mapping = read_json(absolute_sources["mapping"])
    v4_prompt = read_json(absolute_sources["v4_prompt"])
    v4_raw = read_json(absolute_sources["v4_raw"])
    v4_final = read_json(absolute_sources["v4_final"])
    if tuple(record["outfit_id"] for record in candidate_audit["records"]) != CANDIDATES:
        raise RuntimeError("Candidate outfit order changed")
    gate_pairs = {(record["outfit_id"], record["condition_id"]) for record in gate_manifest["records"]}
    expected_pairs = {(outfit, condition) for outfit in CANDIDATES for condition, _ in CONDITIONS}
    if gate_pairs != expected_pairs:
        raise RuntimeError("28-gate outfit/condition mapping changed")

    reference_sets, reference_cells = build_reference_sets(output_root, gate_manifest, mapping)
    o01_audit = audit_o01_existing_targets(output_root, v4_raw, v4_final, args.images_actually_opened)
    generation_plan = build_generation_plan(
        repo,
        output_root,
        mapping,
        v4_prompt,
        reference_cells,
        o01_audit,
        args.declared_credential_provider,
        args.credential_route_verified,
    )
    fingerprint = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "created_at": now(),
        "scope": "local_generation_inputs",
        "git": git_state,
        "source_files": {
            name: {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}
            for name, path in absolute_sources.items()
        },
        "config": {"path": str((repo / args.config).resolve()), "sha256": sha256((repo / args.config).resolve())},
        "conditions": [
            {
                "condition_id": condition,
                "view": view,
                "files": {
                    role: {"path": str(path), "sha256": sha256(path)}
                    for role, path in {
                        "clay": CONDITION_ROOT / "images" / f"{condition}.png",
                        "pose": CONDITION_ROOT / "poses" / f"pose_{condition_number(condition)}.npz",
                        "camera": CONDITION_ROOT / "cameras" / f"camera_{condition_number(condition)}.json",
                        "base_render": BASE_RENDER_ROOT / f"{condition}_rgb.png",
                    }.items()
                },
            }
            for condition, view in CONDITIONS
        ],
        "reference_sets": [
            {"outfit_id": item["outfit_id"], "manifest_path": item["manifest_path"], "manifest_sha256": item["manifest_sha256"]}
            for item in reference_sets
        ],
        "o01_reuse_audit_sha256": sha256(output_root / "O01_existing_target_reuse_audit.json"),
        "generation_plan_sha256": sha256(output_root / "generation" / "generation_plan.json"),
        "raw_inputs_modified": False,
        "network_requests": 0,
        "images_generated": 0,
        "optimizer_steps": 0,
    }
    write_json(output_root / "contract" / "aaai_gate_input_fingerprint.local.json", fingerprint)
    blocked = not generation_plan["model_route"]["execution_allowed"]
    decision = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "status": "BLOCKED_GENERATION_MODEL_CONTRACT" if blocked else "READY_FOR_GENERATION",
        "failure_stage": "generation_model_route_preflight" if blocked else None,
        "valid_images_generated": 0,
        "optimizer_steps": 0,
        "actual_api_calls": 0,
        "o01_reused_count": o01_audit["reused_count"],
        "pending_generation_count": generation_plan["pending_generation_count"],
        "stop_rule": "Do not substitute an unexposed built-in model or send a provider token to a different issuer endpoint.",
        "next_authority_required": "Configure and verify an official OpenAI gpt-image-2 images.edit credential/route, or explicitly revise the frozen model contract in a new task.",
    }
    write_json(output_root / "final_adjudication" / "PREFLIGHT_STATUS.json", decision)
    write_text(
        output_root / "final_adjudication" / "PREFLIGHT_STATUS.md",
        "\n".join(
            [
                "# AAAI-27 28-Image Gate Preflight Status",
                "",
                f"- Status: **{decision['status']}**",
                f"- O01 reused: `{decision['o01_reused_count']}`",
                f"- Targets requiring verified generation: `{decision['pending_generation_count']}`",
                "- API calls: `0`",
                "- Generated images: `0`",
                "- Optimizer steps: `0`",
                "",
                generation_plan["model_route"]["reason"],
                "",
                "The preflight stops before generation, masks, checker, support metrics, or Oracle. No model substitution is permitted.",
            ]
        ),
    )
    write_text(
        output_root / "aaai_gate_start_state.txt",
        "\n".join(
            [
                f"task_id={TASK_ID}",
                f"branch={git_state['branch']}",
                f"head={git_state['head']}",
                f"long_term_head={git_state['long_term_head']}",
                f"worktree_clean={not bool(git_state['status_short'])}",
                f"required_model={REQUIRED_MODEL}",
                f"generation_execution_allowed={generation_plan['model_route']['execution_allowed']}",
                "network_requests=0",
                "images_generated=0",
                "optimizer_steps=0",
            ]
        ),
    )
    print(json.dumps(decision, indent=2))


def cloud_mode(args: argparse.Namespace) -> None:
    repo = args.repo_root.resolve()
    output_root = args.output_dir.resolve()
    make_output_tree(output_root)
    local_fingerprint_path = output_root / "contract" / "aaai_gate_input_fingerprint.local.json"
    ensure_files([local_fingerprint_path, args.module4b_fingerprint, args.base_checkpoint, (repo / args.config).resolve()])
    git_state = assert_git_contract(repo)
    module4b = read_json(args.module4b_fingerprint)
    base_hash = sha256(args.base_checkpoint)
    if base_hash != module4b["base_checkpoint"]["sha256"]:
        raise RuntimeError("Base checkpoint hash differs from sealed Module 4B fingerprint")
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as error:  # pragma: no cover - cloud environment evidence
        torch_info = {"import_error": f"{type(error).__name__}: {error}"}
    cloud = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "created_at": now(),
        "scope": "cloud_execution_inputs",
        "git": git_state,
        "base_checkpoint": {"path": str(args.base_checkpoint), "sha256": base_hash, "size": args.base_checkpoint.stat().st_size},
        "base_gaussian_fingerprint_source": {"path": str(args.module4b_fingerprint), "sha256": sha256(args.module4b_fingerprint)},
        "base_gaussian_fingerprint": module4b["aggregate_sha256"],
        "anchor_graph": module4b["anchor_graph"],
        "anchor_to_gaussian_mapping": module4b["anchor_to_gaussian_mapping"],
        "fixture_manifest": module4b["fixture_manifest"],
        "v5_3_loss_config": {"path": str((repo / args.config).resolve()), "sha256": sha256((repo / args.config).resolve())},
        "environment": {"python": platform.python_version(), "executable": os.sys.executable, "platform": platform.platform(), "pytorch": torch_info},
        "network_requests": 0,
        "images_generated": 0,
        "optimizer_steps": 0,
    }
    write_json(output_root / "contract" / "aaai_gate_input_fingerprint.cloud.json", cloud)
    final = {
        "schema_version": SCHEMA,
        "task_id": TASK_ID,
        "local": read_json(local_fingerprint_path),
        "cloud": cloud,
        "input_integrity": "PASS",
        "generation_model_route": read_json(output_root / "generation" / "model_route_preflight.json"),
        "status": read_json(output_root / "final_adjudication" / "PREFLIGHT_STATUS.json")["status"],
    }
    write_json(output_root / "aaai_gate_input_fingerprint.json", final)
    print(json.dumps({"status": final["status"], "input_integrity": "PASS", "output_dir": str(output_root)}, indent=2))


def main() -> int:
    args = parse_args()
    if args.mode == "local":
        local_mode(args)
    else:
        cloud_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
