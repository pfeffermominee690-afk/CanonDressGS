from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"PASS", "WARN", "FAIL"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply actual-image identity adjudication to a V5 fixture without changing pixels"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    decision_path = args.decisions.resolve()
    audit_dir = args.audit_dir.resolve()
    manifest = _read_json(manifest_path)
    decision_document = _read_json(decision_path)
    decisions = decision_document.get("decisions", [])
    by_sample: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        sample_id = decision.get("sample_id")
        status = decision.get("status")
        if not isinstance(sample_id, str) or status not in VALID_STATUSES:
            raise ValueError(f"invalid visual decision: {decision}")
        if sample_id in by_sample:
            raise ValueError(f"duplicate visual decision: {sample_id}")
        if not str(decision.get("observation", "")).strip():
            raise ValueError(f"visual observation is required: {sample_id}")
        by_sample[sample_id] = decision

    prior_excluded = manifest.get("identity_visual_adjudication", {}).get("excluded_samples", [])
    expected = {
        f"{observation['condition_id']}_{outfit['outfit_id']}"
        for outfit in manifest["outfits"]
        for observation in outfit["observations"]
    } | {item["sample_id"] for item in prior_excluded}
    if set(by_sample) != expected:
        raise ValueError(
            f"visual decisions must exactly cover fixture samples; "
            f"missing={sorted(expected - set(by_sample))}, extra={sorted(set(by_sample) - expected)}"
        )

    excluded: list[dict[str, Any]] = list(prior_excluded)
    retained = 0
    for outfit in manifest["outfits"]:
        accepted = []
        for observation in outfit["observations"]:
            sample_id = f"{observation['condition_id']}_{outfit['outfit_id']}"
            decision = by_sample[sample_id]
            observation["identity_audit_status"] = decision["status"]
            if decision["status"] == "FAIL":
                excluded.append({
                    "sample_id": sample_id,
                    "condition_id": observation["condition_id"],
                    "outfit_id": outfit["outfit_id"],
                    "reason": decision["observation"],
                    "policy": "identity FAIL samples cannot enter the V5 fixture",
                })
            else:
                accepted.append(observation)
                retained += 1
        outfit["observations"] = accepted

    manifest["identity_visual_adjudication"] = {
        "status": "FAIL" if excluded else "PASS",
        "source": str(decision_path),
        "images_actually_opened": decision_document.get("images_actually_opened", []),
        "inspection_method": decision_document.get("inspection_method"),
        "retained_samples": retained,
        "excluded_samples": excluded,
    }
    _atomic_json(manifest_path, manifest)

    counts = {status: sum(item["status"] == status for item in decisions) for status in sorted(VALID_STATUSES)}
    report = {
        "schema_version": "subject02.raw_edit_identity_adjudication.v5",
        "adjudicated_at": _now(),
        "status": "FAIL" if excluded else "PASS",
        "counts": counts,
        "retained_samples": retained,
        "excluded_samples": excluded,
        "images_actually_opened": decision_document.get("images_actually_opened", []),
        "inspection_method": decision_document.get("inspection_method"),
        "decisions": decisions,
        "pixel_files_modified": False,
    }
    _atomic_json(audit_dir / "raw_edit_identity_visual_adjudication_v5.json", report)
    builder_manifest_path = audit_dir / "v5_builder_manifest.json"
    if builder_manifest_path.is_file():
        builder_manifest = _read_json(builder_manifest_path)
        builder_manifest.update({
            "status": "IDENTITY_AUDIT_FAIL" if excluded else "IDENTITY_AUDIT_PASS",
            "visual_identity_audit": str((audit_dir / "raw_edit_identity_visual_adjudication_v5.json").resolve()),
            "accepted_record_count": retained,
            "excluded_record_count": len(excluded),
            "excluded_samples": [item["sample_id"] for item in excluded],
        })
        _atomic_json(builder_manifest_path, builder_manifest)
    rows = [
        "# Raw Direct Edit Identity Audit V5",
        "",
        "- raw_direct_edit_generation: **PASS** (12/12 generation/alignment contract)",
        "- identity_safe_pixel_composite: **FAIL** (V4 historical route; stopped and not used)",
        "- region_aware_training_supervision: **BLOCKED_PENDING_VALID_REPLACEMENT**",
        f"- visual adjudication: **{report['status']}**",
        f"- identity counts: PASS={counts['PASS']}, WARN={counts['WARN']}, FAIL={counts['FAIL']}",
        f"- retained in fixture: {retained}/12",
        "",
        "## Actual image inspection",
        "",
        f"Method: {report['inspection_method']}",
        "",
    ]
    rows.extend(f"- `{item}`" for item in report["images_actually_opened"])
    rows.extend(["", "## Per-sample adjudication", ""])
    rows.extend(
        f"- `{item['sample_id']}`: **{item['status']}** — {item['observation']}"
        for item in decisions
    )
    rows.extend(["", "## Fixture exclusion", ""])
    if excluded:
        rows.extend(
            f"- `{item['sample_id']}` excluded: {item['reason']}"
            for item in excluded
        )
    else:
        rows.append("- None.")
    rows.extend([
        "",
        "No Jay pixels, skin plate, composite target, TPS, flow, or image generation was used.",
        "",
    ])
    _atomic_text(audit_dir / "RAW_EDIT_IDENTITY_AUDIT_V5.md", "\n".join(rows))
    print(json.dumps({
        "status": report["status"],
        "counts": counts,
        "retained_samples": retained,
        "excluded_samples": [item["sample_id"] for item in excluded],
    }, indent=2))


if __name__ == "__main__":
    main()
