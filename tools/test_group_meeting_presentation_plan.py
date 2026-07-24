#!/usr/bin/env python3
"""Validate the CanonDressGS group-meeting presentation planning bundle."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_BRANCH = "research/group-meeting-presentation-plan-20260724"
SOURCE_HEAD = "7e2d8efb3925880da4a9b8b85c0ebdce2437bd37"
PROTO = ROOT / "paper_protocol" / "presentation"
DOCS = ROOT / "docs" / "PRESENTATION"


class DuplicateKeyError(ValueError):
    pass


def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(key)
        result[key] = value
    return result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


def run(*args: str) -> tuple[int, str]:
    process = subprocess.run(args, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    return process.returncode, (process.stdout + process.stderr).strip()


def main() -> int:
    json_paths = sorted(PROTO.glob("*.json")) + [ROOT / "project_control_handoff" / "canondressgs_group_meeting_presentation_plan_handoff.json"]
    md_paths = sorted(DOCS.glob("*.md"))
    parsed = {path.name: load(path) for path in json_paths}
    full_doc = parsed["canondressgs_group_meeting_slide_plan.json"]
    compact_doc = parsed["canondressgs_group_meeting_compact_slide_plan.json"]
    assets_doc = parsed["canondressgs_group_meeting_slide_asset_manifest.json"]
    formulas_doc = parsed["canondressgs_group_meeting_formula_manifest.json"]
    claims_doc = parsed["canondressgs_group_meeting_claim_manifest.json"]
    qa_doc = parsed["canondressgs_group_meeting_question_answer_registry.json"]
    missing_doc = parsed["canondressgs_group_meeting_missing_material_registry.json"]
    summary = parsed["canondressgs_group_meeting_plan_final_summary.json"]
    slides = full_doc["slides"] + compact_doc["slides"] + full_doc["appendix"]
    checks: list[tuple[int, str, bool, str]] = []

    def check(number: int, name: str, passed: bool, detail: str = "") -> None:
        checks.append((number, name, bool(passed), detail))

    check(1, "exact source HEAD", full_doc["source_head"] == SOURCE_HEAD)
    pure_heads = {p["source_head"] for s in slides for p in s["source_provenance"] if p["asset_id"].startswith("GM-A0") or "PURE" in p["asset_id"] or "ENDPOINT" in p["asset_id"]}
    check(2, "Pure Endpoint source valid", "ce110887a942cf8db082ba688c8d36d2433bfdbe" in pure_heads)
    check(3, "Headroom source valid", any(a["source_head"] == "674e6092e21eeddeb22e963536247a3385c4e200" for a in assets_doc["assets"]))
    registries = [ROOT / "paper_protocol/reviewer_risk/paper_figure_candidate_registry_pure_endpoint_refresh.json", ROOT / "paper_protocol/reviewer_risk/ppt_material_index_headroom_refresh.json", ROOT / "paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json"]
    check(4, "Figure Bank registries readable", all(path.is_file() and load(path) for path in registries))
    check(5, "all slides have one-sentence conclusions", all(s["one_sentence_conclusion"].strip() for s in slides))
    check(6, "all slides have speaker notes", all(len(s["speaker_notes"]) >= 3 for s in slides))
    result_statuses = {"PAPER_CORE_EVIDENCE", "NEGATIVE_DIAGNOSTIC", "SUPPLEMENTARY_EXTENSION", "HISTORICAL_DESIGN_EVIDENCE"}
    check(7, "all result slides have source assets", all(s["asset_ids"] for s in slides if s["figure_status"] in result_statuses))
    asset_source_ok = True
    for item in assets_doc["assets"]:
        path = ROOT / item["source_path"]
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        asset_source_ok = asset_source_ok and bool(item["source_head"]) and actual_sha == item["source_sha"]
    check(8, "all source assets have SHA and path", asset_source_ok)
    allowed_slide_statuses = {"PAPER_CORE_EVIDENCE", "PAPER_CORE_METHOD_CANDIDATE", "HISTORICAL_DESIGN_EVIDENCE", "NEGATIVE_DIAGNOSTIC", "SUPPLEMENTARY_EXTENSION", "PENDING_LOO", "MULTI_IDENTITY_FOUNDATION", "LIMITATION", "PROJECT_GOVERNANCE_APPENDIX"}
    check(9, "all claims and slides have valid status", all(c["claim_status"] and c["slide_status"] in allowed_slide_statuses for c in claims_doc["claims"]) and all(s["figure_status"] in allowed_slide_statuses for s in slides))
    s19 = next(s for s in full_doc["slides"] if s["slide_id"] == "S19")
    check(10, "Headroom remains negative", s19["figure_status"] == "NEGATIVE_DIAGNOSTIC" and "未改善" in s19["one_sentence_conclusion"])
    allowed_text = " ".join(" ".join(s["allowed_claims"]) for s in slides)
    check(11, "Pure Endpoint not claimed superior to hard lookup", "优于 hard lookup" not in allowed_text and "outperforms hard lookup" not in allowed_text.lower())
    check(12, "LOO result not fabricated", summary["loo"]["attempt"] == "ABSENT" and summary["loo"]["files_read"] == 0 and summary["loo"]["result_fabricated"] == 0)
    controller_primary_slides = [s for s in slides if "GM-A29-CONTROLLER-VISUALS" in s["primary_assets"] or "GM-A30-CONTROLLER-BUDGET" in s["primary_assets"]]
    check(13, "Controller not a successful core method", bool(controller_primary_slides) and all(s["figure_status"] == "NEGATIVE_DIAGNOSTIC" for s in controller_primary_slides))
    subject_allowed = " ".join(" ".join(s["allowed_claims"]) for s in slides if "GM-A31-SUBJECT00-SHEET" in s["asset_ids"])
    check(14, "Subject00 not reported as multi-garment complete", "multi-garment completion" not in subject_allowed.lower())
    avatar_uses = [a for a in assets_doc["assets"] if a["asset_id"] == "GM-A34-AVATARREX-METADATA"]
    check(15, "AvatarReX media references equal zero", bool(avatar_uses) and all(a["asset_type"] == "METADATA_ONLY" and a["license"] == "LICENSE_RESTRICTED_DO_NOT_EXPORT" for a in avatar_uses))
    teacher_allowed = " ".join(" ".join(s["allowed_claims"]) for s in slides if s["slide_id"] in {"S07", "C05"})
    check(16, "Teacher not called Upper Bound in allowed claims", "Upper Bound" not in teacher_allowed)
    check(17, "raw and snapped coefficients distinguished", all(token in next(s for s in full_doc["slides"] if s["slide_id"] == "S15")["one_sentence_conclusion"] for token in ["raw", "snapping"]))
    s19_text = " ".join(s19["main_content"] + [s19["one_sentence_conclusion"]])
    check(18, "full residual failure correctly explained", all(token in s19_text for token in ["full residual", "patch/cloud/mottle"]))
    historical_slides = [s for s in slides if s["figure_status"] == "HISTORICAL_DESIGN_EVIDENCE"]
    historical_ok = True
    for item in historical_slides:
        text = " ".join([item["title_cn"], item["title_en_optional"], item["one_sentence_conclusion"]] + item["allowed_claims"] + item["main_content"])
        historical_ok = historical_ok and ("histor" in text.lower() or "历史" in text)
    check(19, "historical and current protocols separated", bool(historical_slides) and historical_ok)
    full_ids = {s["slide_id"] for s in full_doc["slides"]}
    mapped_full_ids = {full_id for s in compact_doc["slides"] for full_id in s.get("full_slide_mapping", [])}
    mapping_ok = all(s.get("full_slide_mapping") and set(s["full_slide_mapping"]) <= full_ids for s in compact_doc["slides"]) and mapped_full_ids == full_ids
    check(20, "compact/full mapping complete", mapping_ok)
    check(21, "formula variables complete", len(formulas_doc["formulas"]) >= 10 and all(f["variables"] and f["avoid"] for f in formulas_doc["formulas"]))
    check(22, "Q&A has at least 15 items", qa_doc["question_count"] >= 15 and all(q["recommended_answer"] for q in qa_doc["questions"]))
    check(23, "missing materials explicit", missing_doc["missing_material_count"] >= 1 and any(m["dependency"] == "SEALED_LOO" for m in missing_doc["materials"]))
    gap_text = (DOCS / "CANONDRESSGS_CURRENT_PAPER_NARRATIVE_GAP_MAP_20260724.md").read_text(encoding="utf-8")
    required_gap_items = ["title", "abstract", "contributions", "teaser", "Figure 2 pipeline", "controller section", "Dual-Support positioning", "training objective", "experiment questions", "baseline table", "continuous control section", "multi-identity section", "limitations", "conclusion"]
    check(24, "paper narrative gap map complete", all(item in gap_text for item in required_gap_items))
    check(25, "JSON parse and duplicate-key check", len(parsed) == 9)
    prohibited_outputs = list(DOCS.glob("*.pptx")) + list(DOCS.glob("*.pdf")) + list(PROTO.glob("*.pptx")) + list(PROTO.glob("*.pdf"))
    check(26, "Markdown nonempty and prohibited outputs absent", len(md_paths) == 7 and all(path.stat().st_size > 100 for path in md_paths) and not prohibited_outputs)
    ids = [s["slide_id"] for s in slides]
    check(27, "duplicate slide IDs equal zero", len(ids) == len(set(ids)))
    usage_ids = [a["usage_id"] for a in assets_doc["assets"]]
    known_assets = {a["asset_id"] for a in assets_doc["assets"]}
    check(28, "duplicate asset references are legal", len(usage_ids) == len(set(usage_ids)) and all(set(a["backup_asset_ids"]) <= known_assets for a in assets_doc["assets"]))
    scanned_text = "\n".join(path.read_text(encoding="utf-8") for path in json_paths + md_paths)
    credential_patterns = [r"AKIA[0-9A-Z]{16}", r"ghp_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9]{20,}", r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"]
    mutation_zero = all(value == 0 for value in summary["mutation_audit"].values())
    check(29, "credential and zero-mutation audit", not any(re.search(pattern, scanned_text) for pattern in credential_patterns) and mutation_zero)
    diff_code, diff_output = run("git", "diff", "--check")
    check(30, "git diff --check", diff_code == 0, diff_output)
    head_code, head = run("git", "rev-parse", "HEAD")
    origin_code, origin = run("git", "rev-parse", f"origin/{TARGET_BRANCH}")
    check(31, "Windows/origin consistency", head_code == 0 and origin_code == 0 and head == origin, f"HEAD={head}; origin={origin}")
    status_code, status = run("git", "status", "--porcelain")
    check(32, "worktree clean", status_code == 0 and status == "", status)

    failed = [item for item in checks if not item[2]]
    for number, name, passed, detail in checks:
        suffix = f" — {detail}" if detail else ""
        print(f"{number:02d}. {'PASS' if passed else 'FAIL'} {name}{suffix}")
    print(json.dumps({"required": 32, "passed": 32 - len(failed), "failed": len(failed)}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
