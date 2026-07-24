#!/usr/bin/env python3
"""Generate the sealed-evidence CanonDressGS group-meeting presentation plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "docs" / "PRESENTATION"
PROTO_DIR = ROOT / "paper_protocol" / "presentation"
HANDOFF_DIR = ROOT / "project_control_handoff"

TASK_ID = "CANONDRESSGS-GROUP-MEETING-PRESENTATION-PLAN-20260724"
SOURCE_BRANCH = "research/paper-figure-bank-headroom-refresh-20260724"
SOURCE_HEAD = "7e2d8efb3925880da4a9b8b85c0ebdce2437bd37"
TARGET_BRANCH = "research/group-meeting-presentation-plan-20260724"

ALLOWED_STATUSES = {
    "PAPER_CORE_EVIDENCE",
    "PAPER_CORE_METHOD_CANDIDATE",
    "HISTORICAL_DESIGN_EVIDENCE",
    "NEGATIVE_DIAGNOSTIC",
    "SUPPLEMENTARY_EXTENSION",
    "PENDING_LOO",
    "MULTI_IDENTITY_FOUNDATION",
    "LIMITATION",
    "PROJECT_GOVERNANCE_APPENDIX",
}

SOURCES = {
    "figure_bank": {
        "branch": SOURCE_BRANCH,
        "head": SOURCE_HEAD,
        "classification": "HEADROOM_FIGURE_REFRESH_READY",
    },
    "pure": {
        "branch": "research/pure-endpoint-core-method-crossfit-amended-20260724",
        "head": "ce110887a942cf8db082ba688c8d36d2433bfdbe",
        "execution_head": "195fb887f2cac8a72920b499c44bb66700a97e25",
        "classification": "PURE_ENDPOINT_CORE_METHOD_SUPPORTED",
    },
    "headroom": {
        "branch": "research/render-refined-coefficient-headroom-attempt2-20260724",
        "head": "674e6092e21eeddeb22e963536247a3385c4e200",
        "classification": "TEACHER_SPAN_AT_LOCAL_OPTIMUM",
    },
    "geometry": {
        "branch": "research/continuous-control-causal-attribution-20260722",
        "head": "8c43524b7ca0ee8b3c795dbe349466f30b29354f",
        "classification": "GEOMETRY_MAIN_EFFECT",
    },
    "dual": {
        "branch": "research/dual-support-all-pair-evaluation-20260722",
        "head": "d802f427f1e9c23595e1bdb4f10135f7de2c3f08",
        "classification": "DUAL_SUPPORT_ALL_PAIR_PASS",
    },
    "controller": {
        "branch": "research/controller-garment-head-optimization-budget-diagnosis-20260724",
        "head": "af3a4895c2c6c3a12a869907883bf68e2a2ba820",
        "classification": "OPTIMIZER_OR_NORMALIZATION_LIMIT",
    },
    "historical": {
        "branch": "paper/aaai27-frozen-experiment-batches-20260720",
        "head": "c60d61a67a87eb35d5debca7eb111778b637c252",
        "classification": "HISTORICAL_VIEW_TRANSDUCTIVE_EVIDENCE",
    },
    "loo": {
        "branch": "research/loo-few-view-fold-manifest-repair-20260724",
        "head": "2c7c748026307e82c88b7f96bf2dc41a79ba7b6f",
        "classification": "LOO_FEW_VIEW_FOLD_REPAIRED_AND_READY",
    },
    "subject00": {
        "branch": "research/mmlphuman-subject00-one-pass-medium-pilot-20260723",
        "head": "2d0913eb1b163a79e81c1804c216f91b0a737b44",
        "classification": "BASE_AVATAR_MEDIUM_PILOT_PASS_ONLY",
    },
    "avatarrex": {
        "branch": "research/avatarrex-base-avatar-preflight-20260724",
        "head": "a786e581bd36c57b3f32e9a797b32bebcdedca13",
        "classification": "METADATA_ONLY_LICENSE_RESTRICTED",
    },
}


def sha256(path: str) -> str:
    target = ROOT / path
    if not target.is_file():
        return "METADATA_ONLY_NO_LOCAL_FILE"
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset(
    asset_id: str,
    title: str,
    path: str,
    source: str,
    asset_type: str,
    status: str,
    experiment_id: str,
    *,
    attempt: str = "attempt_001",
    license_status: str = "PROJECT_AUTHORIZED_INTERNAL_EVIDENCE",
    eligibility: str = "GROUP_MEETING_AND_CLAIM_BOUNDARY_COMPLIANT",
    notes: str = "Use without changing scientific pixels or reported values.",
) -> dict:
    source_record = SOURCES[source]
    return {
        "asset_id": asset_id,
        "title": title,
        "source_path": path,
        "source_sha": sha256(path),
        "source_branch": source_record["branch"],
        "source_head": source_record["head"],
        "experiment_id": experiment_id,
        "attempt": attempt,
        "asset_type": asset_type,
        "garment": "MULTI_OR_NOT_APPLICABLE",
        "method": source.upper(),
        "seed": "AGGREGATED_OR_NOT_APPLICABLE",
        "rotation": "AGGREGATED_OR_NOT_APPLICABLE",
        "query": "AGGREGATED_OR_NOT_APPLICABLE",
        "figure_status": status,
        "license": license_status,
        "claim_eligibility": eligibility,
        "crop_transform": "NO_SCIENTIFIC_PIXEL_MUTATION; layout crop only after manual review",
        "recommended_size": "16:9 content area; preserve aspect ratio",
        "missing_dependency": None,
        "notes": notes,
    }


ASSETS = {
    a["asset_id"]: a
    for a in [
        asset("GM-A01-ENDPOINT-TEASER", "Pure Endpoint endpoint-only teaser candidate", "paper_draft/figures/pure_endpoint_refresh/candidates/fig01_teaser/pure_endpoint_endpoint_only_v1/contact_sheet_for_manual_adjudication.png", "pure", "CONTACT_SHEET", "PAPER_CORE_METHOD_CANDIDATE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001", notes="Manual five-query adjudication is still required."),
        asset("GM-A02-EARLY-FAILURES", "Early reference-to-residual failure sheet", "paper_draft/figures/contact_sheets/limitations/early_failures.png", "figure_bank", "CONTACT_SHEET", "HISTORICAL_DESIGN_EVIDENCE", "HISTORICAL-METHOD-TRIAGE"),
        asset("GM-A03-METHOD-WIREFRAME", "Current Figure 2 method wireframe", "paper_draft/figures/candidates/fig02_method/layout_wireframe.svg", "figure_bank", "WIREFRAME", "PAPER_CORE_METHOD_CANDIDATE", "FIGURE-BANK-METHOD-PLAN", notes="METHOD_FREEZE_PENDING_LOO; use as a planning wireframe, not a final figure."),
        asset("GM-A04-TEACHER-ENDPOINTS", "Sealed five-garment endpoint renders", "paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_clean_all_garments.png", "pure", "CONTACT_SHEET", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A05-GEOMETRY-CAUSAL-DATA", "Geometry causal attribution sealed summary", "paper_protocol/reviewer_risk/continuous_control_causal_attribution_final_summary.json", "geometry", "STRUCTURED_RESULT_JSON", "PAPER_CORE_EVIDENCE", "AAAI27-CONTINUOUS-CONTROL-CAUSAL-ATTRIBUTION-001", attempt="attempt_002", notes="Present as editable numeric callouts: main effect about 0.9959; sufficiency and necessity 10/10."),
        asset("GM-A06-DUAL-LPIPS", "Dual-Support all-pair LPIPS", "paper_draft/figures/plots/dual_support/dual_support_all_pair_garment_lpips.png", "dual", "PLOT", "SUPPLEMENTARY_EXTENSION", "AAAI27-DUAL-SUPPORT-ALL-PAIR-EVALUATION-001"),
        asset("GM-A07-DUAL-SHEET", "Dual-Support all-pair visual sheet", "paper_draft/figures/contact_sheets/dual_support/dual_support.png", "dual", "CONTACT_SHEET", "SUPPLEMENTARY_EXTENSION", "AAAI27-DUAL-SUPPORT-ALL-PAIR-EVALUATION-001"),
        asset("GM-A08-PURE-TOP1", "Pure Endpoint clean top-1", "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/clean_method_top1.png", "pure", "PLOT", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A09-ENDPOINT-MATCH", "Pure Endpoint exact endpoint match", "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/endpoint_exact_match.png", "pure", "PLOT", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A10-HARD-LOOKUP", "Hard lookup relation candidate", "paper_draft/figures/pure_endpoint_refresh/candidates/fig05_hard_lookup/pure_endpoint_hard_lookup_v1/pure_endpoint_hard_lookup_v1.png", "pure", "CANDIDATE_PANEL", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001", notes="HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY; requires manual adjudication."),
        asset("GM-A11-RAW-SNAPPED-PLOT", "Raw versus realized coefficient error", "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/raw_vs_realized_coefficient_error.png", "pure", "PLOT", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A12-RAW-SNAPPED-SHEET", "Raw versus snapped endpoint sheet", "paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_raw_vs_snapped.png", "pure", "CONTACT_SHEET", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A13-PERTURBATION", "Pure Endpoint perturbation limitation", "paper_draft/figures/pure_endpoint_refresh/candidates/fig06_limitations/pure_endpoint_perturbation_v1/pure_endpoint_perturbation_v1.png", "pure", "CANDIDATE_PANEL", "LIMITATION", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A14-BLUR-FAILURES", "Mild-blur endpoint flips", "paper_draft/figures/pure_endpoint_refresh/contact_sheets/pure_endpoint/pure_endpoint_mild_blur_failures.png", "pure", "CONTACT_SHEET", "LIMITATION", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A15-PARITY-SAFETY", "Pure Endpoint parity and safety", "paper_draft/figures/pure_endpoint_refresh/candidates/supplementary/pure_endpoint_parity_safety_v1/pure_endpoint_parity_safety_v1.png", "pure", "CANDIDATE_PANEL", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A16-HEADROOM-OVERVIEW", "Headroom negative diagnostic overview", "paper_draft/figures/headroom_refresh/candidates/supplementary/headroom_negative_diagnostic_v1/supplementary_negative_diagnostic_v1.png", "headroom", "CANDIDATE_PANEL", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A17-HEADROOM-PARITY", "Teacher versus SVD parity", "paper_draft/figures/headroom_refresh/plots/headroom/teacher_svd_parity.png", "headroom", "PLOT", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A18-HEADROOM-FOUR-METHOD", "Teacher, SVD, refined coefficient, and full residual", "paper_draft/figures/headroom_refresh/plots/headroom/four_method_test_metrics.png", "headroom", "PLOT", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A19-HEADROOM-GATES", "Headroom LPIPS gate summary", "paper_draft/figures/headroom_refresh/plots/headroom/lpips_gate_summary.png", "headroom", "PLOT", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A20-HEADROOM-PER-GARMENT", "Teacher-minus-refined LPIPS by garment", "paper_draft/figures/headroom_refresh/plots/headroom/per_garment_teacher_minus_refined_lpips.png", "headroom", "PLOT", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A21-HEADROOM-DISPLACEMENT", "Coefficient displacement trajectory", "paper_draft/figures/headroom_refresh/plots/headroom/coefficient_displacement_trajectory.png", "headroom", "PLOT", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A22-FULL-RESIDUAL-ARTIFACT", "Full-residual patch/cloud/mottle examples", "paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png", "headroom", "CONTACT_SHEET", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A23-SPAN-NULL", "Span recovery denominator-null explanation", "paper_draft/figures/headroom_refresh/plots/headroom/span_recovery_denominator_null.png", "headroom", "PLOT", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A24-REFINED-DECOMPOSITION", "Refined lookup decomposition", "paper_draft/figures/headroom_refresh/plots/headroom/refined_lookup_decomposition.png", "headroom", "PLOT", "NEGATIVE_DIAGNOSTIC", "COEFFICIENT-HEADROOM-001", attempt="attempt_002"),
        asset("GM-A25-HIST-RANK", "Historical rank 1-4 ablation", "paper_draft/figures/plots/historical_ablations/rank_1_4_edit_reduction.png", "historical", "PLOT", "HISTORICAL_DESIGN_EVIDENCE", "AAAI27-SEEN-OUTFIT-PAPER"),
        asset("GM-A26-HIST-REFERENCE", "Historical reference-count ablation", "paper_draft/figures/plots/historical_ablations/reference_count_1_3_edit_reduction.png", "historical", "PLOT", "HISTORICAL_DESIGN_EVIDENCE", "AAAI27-SEEN-OUTFIT-PAPER"),
        asset("GM-A27-HIST-REPRESENTATION", "Historical representation ablations", "paper_draft/figures/contact_sheets/ablations/representation_ablations.png", "historical", "CONTACT_SHEET", "HISTORICAL_DESIGN_EVIDENCE", "AAAI27-SEEN-OUTFIT-PAPER"),
        asset("GM-A28-O07-LIMITATION", "Historical O07 held-out collapse", "paper_draft/figures/contact_sheets/limitations/o07.png", "historical", "CONTACT_SHEET", "LIMITATION", "AAAI27-SEEN-OUTFIT-PAPER"),
        asset("GM-A29-CONTROLLER-VISUALS", "Controller diagnostic visuals", "paper_draft/figures/contact_sheets/controller/controller_visuals.png", "controller", "CONTACT_SHEET", "NEGATIVE_DIAGNOSTIC", "CONTROLLER-GARMENT-HEAD-OPTIMIZATION-BUDGET-DIAGNOSTIC-001"),
        asset("GM-A30-CONTROLLER-BUDGET", "Controller continuation-budget trajectory", "paper_draft/figures/plots/controller/controller_budget_continuation_trajectory.png", "controller", "PLOT", "NEGATIVE_DIAGNOSTIC", "CONTROLLER-GARMENT-HEAD-OPTIMIZATION-BUDGET-DIAGNOSTIC-001"),
        asset("GM-A31-SUBJECT00-SHEET", "Subject00 base-avatar evidence", "paper_draft/figures/contact_sheets/subject00/subject00.png", "subject00", "CONTACT_SHEET", "MULTI_IDENTITY_FOUNDATION", "MMLPHUMAN-SUBJECT00-ONE-PASS-MEDIUM-PILOT-001"),
        asset("GM-A32-SUBJECT00-PLOT", "Subject00 base-avatar LPIPS progression", "paper_draft/figures/plots/subject00/subject00_step0_to_medium_lpips.png", "subject00", "PLOT", "MULTI_IDENTITY_FOUNDATION", "MMLPHUMAN-SUBJECT00-ONE-PASS-MEDIUM-PILOT-001"),
        asset("GM-A33-LOO-PENDING", "LOO repaired protocol pending slot", "paper_draft/figures/pure_endpoint_refresh/pending/loo/repaired_protocol_pending.json", "loo", "METADATA_ONLY", "PENDING_LOO", "AAAI27-LOO-FEW-VIEW-FOLD-MANIFEST-REPAIR", notes="LOO_ATTEMPT=ABSENT and files_read=0; no partial or active output may be consumed."),
        asset("GM-A34-AVATARREX-METADATA", "AvatarReX license-restricted metadata slot", "paper_draft/figures/pending/avatarrex_license_restricted/README.md", "avatarrex", "METADATA_ONLY", "MULTI_IDENTITY_FOUNDATION", "AAAI27-AVATARREX-BASE-AVATAR-PREFLIGHT-001", license_status="LICENSE_RESTRICTED_DO_NOT_EXPORT", eligibility="TEXT_ONLY_METADATA; MEDIA_USE_FORBIDDEN", notes="AvatarReX media references and exports must remain zero."),
        asset("GM-A35-FIGURE-MAP", "Current paper figure status map", "paper_protocol/reviewer_risk/paper_figure_map_headroom_refreshed.json", "figure_bank", "STRUCTURED_STATUS_JSON", "PROJECT_GOVERNANCE_APPENDIX", "PAPER-FIGURE-HEADROOM-REFRESH"),
        asset("GM-A36-RENDER-EQUIVALENCE", "Render equivalence and cache summary", "paper_draft/figures/pure_endpoint_refresh/plots/pure_endpoint/render_equivalence_and_cache_summary.png", "pure", "PLOT", "PAPER_CORE_EVIDENCE", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
        asset("GM-A37-HIST-ALL", "Historical 51-run ablation summary", "paper_draft/figures/plots/historical_ablations/historical_ablation_edit_reduction.png", "historical", "PLOT", "HISTORICAL_DESIGN_EVIDENCE", "AAAI27-SEEN-OUTFIT-PAPER"),
        asset("GM-A38-PURE-ROTATIONS", "Pure Endpoint cross-fit rotation manifests", "paper_protocol/reviewer_risk/pure_endpoint_rotation_manifests.json", "pure", "STRUCTURED_PROTOCOL_JSON", "PROJECT_GOVERNANCE_APPENDIX", "AAAI27-PURE-ENDPOINT-CORE-METHOD-CROSSFIT-001"),
    ]
}


def slide(
    slide_id: str,
    section: str,
    title_cn: str,
    title_en: str,
    question: str,
    conclusion: str,
    content: list[str],
    formulas: list[str],
    primary: list[str],
    secondary: list[str],
    backup: list[str],
    status: str,
    allowed: list[str],
    forbidden: list[str],
    transition: str,
    seconds: int,
    paper_eligibility: str,
    loo_dependency: str,
    risks: list[str],
    notes: list[str],
    *,
    full_mapping: list[str] | None = None,
) -> dict:
    all_asset_ids = primary + secondary + backup
    provenance = []
    for asset_id in all_asset_ids:
        record = ASSETS[asset_id]
        item = {"asset_id": asset_id, "source_branch": record["source_branch"], "source_head": record["source_head"]}
        if item not in provenance:
            provenance.append(item)
    result = {
        "slide_id": slide_id,
        "section": section,
        "title_cn": title_cn,
        "title_en_optional": title_en,
        "core_question": question,
        "one_sentence_conclusion": conclusion,
        "main_content": content,
        "formulas": formulas,
        "primary_assets": primary,
        "secondary_assets": secondary,
        "backup_assets": backup,
        "formal_paths": [ASSETS[item]["source_path"] for item in all_asset_ids],
        "asset_ids": all_asset_ids,
        "source_provenance": provenance,
        "figure_status": status,
        "allowed_claims": allowed,
        "forbidden_claims": forbidden,
        "speaker_notes": notes,
        "transition": transition,
        "timing_seconds": seconds,
        "paper_eligibility": paper_eligibility,
        "loo_dependency": loo_dependency,
        "risks_todo": risks,
    }
    if full_mapping is not None:
        result["full_slide_mapping"] = full_mapping
    return result


FULL = [
    slide("S01", "Opening", "CanonDressGS 当前进展", "Reference-Controlled Garment Editing in Canonical Gaussian Space", "当前项目真正解决到哪一步？", "当前已支持固定身份、封闭五服装库中的 reference-controlled endpoint selection，但方法价值仍需 LOO 判定。", ["Group Meeting Progress Report", "subject02 / closed five-garment bank", "PAPER_FINAL=0"], [], ["GM-A01-ENDPOINT-TEASER"], ["GM-A35-FIGURE-MAP"], [], "PAPER_CORE_METHOD_CANDIDATE", ["展示当前封闭服装库任务和方法边界"], ["宣称 unseen garment 或 multi-identity garment editing 已完成"], "先从真实任务动机开始。", 40, "CANDIDATE_AFTER_LOO_SCOPE_FREEZE", "WAIT_FOR_LOO_FOR_FINAL_TITLE", ["Figure 1 候选仍需人工选图"], ["开场直接给出任务和边界。", "说明这不是最终论文答辩，而是研究收敛汇报。", "强调 LOO 尚无结果。", "过渡到服装绑定问题。"]),
    slide("S02", "Problem", "为什么可动画 Avatar 仍被采集服装绑定？", "Motivation", "如何在保持身份与动作的同时指定服装？", "目标是用 reference garment 与 target pose/camera 生成同一身份的指定服装状态。", ["输入：reference RGB/mask", "控制：target pose/camera", "输出：same identity, specified garment"], [], ["GM-A01-ENDPOINT-TEASER"], ["GM-A04-TEACHER-ENDPOINTS"], [], "PAPER_CORE_METHOD_CANDIDATE", ["定义 reference-controlled garment state selection"], ["arbitrary garment synthesis", "cross-identity completion"], "任务看似是外观控制，实际同时涉及几何、可见性和外观。", 55, "PAPER_MOTIVATION_CANDIDATE", "NO", ["避免把 closed-bank 说成 open-vocabulary"], ["用输入输出三元组讲任务。", "指出身份和姿态必须被冻结。", "不要使用 arbitrary editing 表述。", "引出三因素耦合。"]),
    slide("S03", "Problem", "核心困难：服装状态不是单一颜色变量", "Core Challenge", "为什么连续 residual 插值容易失败？", "服装同时改变 geometry、visibility 与 appearance，而 Gaussian 几何插值会制造无效中间支持。", ["lookup：稳定但离散", "continuous residual：灵活但产生 artifact", "研究焦点：在可控性与几何有效性之间取舍"], ["F06"], ["GM-A02-EARLY-FAILURES"], ["GM-A05-GEOMETRY-CAUSAL-DATA"], [], "PAPER_CORE_EVIDENCE", ["提出 geometry/visibility/appearance 三因素问题"], ["将失败归因于单一网络容量而无因果证据"], "因此需要把问题拆成可验证的研究问题。", 55, "PAPER_PROBLEM_FORMULATION_CANDIDATE", "NO", ["早期图仅作为历史设计证据"], ["先对比 lookup 与 interpolation。", "再指出三因素并非同等重要。", "预告后续因果实验。", "转入研究问题地图。"]),
    slide("S04", "Problem", "研究问题地图", "Research Questions", "哪些问题已回答，哪些仍开放？", "当前证据回答了 support、geometry 与 seen-endpoint control，LOO 和第二身份仍是开放问题。", ["Q1 support capacity", "Q2 geometry interpolation failure", "Q3 seen endpoint control", "Q4 basis vs hard lookup", "Q5 out-of-bank adaptation", "Q6 second identity"], [], ["GM-A35-FIGURE-MAP"], ["GM-A33-LOO-PENDING"], [], "PROJECT_GOVERNANCE_APPENDIX", ["明确已证与未证问题"], ["提前判定 Q5/Q6 成功"], "下面按实验如何逐步收缩方法空间来讲。", 55, "PRESENTATION_ONLY", "YES_FOR_Q4_Q5", ["最终论文问题表述必须随 LOO 更新"], ["逐项标记 answered/pending。", "强调 Q4 目前只有描述性等价。", "Q5 由 LOO 决定。", "进入真实研究演化。"]),
    slide("S05", "Research Evolution", "方法路线如何被证据逐步收缩", "Historical Research Contraction", "为什么最终路线与最初设想不同？", "历史 reference-to-residual 路线连续失败，capacity 与 causal 诊断把方案收缩到 valid endpoints、显式 basis 和最小控制器。", ["full reference-to-residual → oracle failure", "support capacity PASS → representation/objective diagnosis", "Teacher endpoints → explicit basis → Pure Endpoint", "Headroom negative → LOO remains"], [], ["GM-A02-EARLY-FAILURES"], ["GM-A27-HIST-REPRESENTATION"], ["GM-A29-CONTROLLER-VISUALS"], "HISTORICAL_DESIGN_EVIDENCE", ["历史负结果推动方法收缩"], ["声称从一开始就设计了最终 pipeline"], "收缩后的路线建立在冻结的 avatar 技术底座上。", 75, "PRESENTATION_NARRATIVE_ONLY", "NO", ["必须区分历史协议与当前协议"], ["按时间顺序讲失败链。", "强调 support capacity PASS 排除了表达容量不足。", "把 controller 与 headroom 放在后段。", "引出冻结底座。"]),
    slide("S06", "Method Foundation", "冻结的技术底座", "Frozen Avatar Foundation", "哪些模块固定，哪些模块允许学习？", "MMLP-Human canonical Gaussians、deformation、LBS 与 renderer 均冻结，仅 garment residual 表示和 reference control 按实验契约变化。", ["Frozen: base avatar, deformation, LBS, renderer", "Trainable by stage: garment endpoint residual or small controller", "No target-forward leakage in current protocol"], [], ["GM-A03-METHOD-WIREFRAME"], ["GM-A35-FIGURE-MAP"], [], "PAPER_CORE_METHOD_CANDIDATE", ["说明冻结边界"], ["暗示基础 avatar 在本任务中重新训练", "将 target views 用于 inference"], "在这个底座上先建立每件服装的有效 Teacher Endpoint。", 55, "PAPER_METHOD_CANDIDATE", "NO", ["Figure 2 仍为 METHOD_FREEZE_PENDING_LOO"], ["从 frozen/trainable 两列讲。", "强调 renderer 和 deformation 不改。", "区分 offline endpoint construction 与 inference。", "过渡到 Teacher Endpoint。"]),
    slide("S07", "Method Foundation", "Garment Teacher Endpoint", "Garment Teacher Endpoint", "如何得到稳定的服装状态？", "每件 seen garment 通过多视图 rendering objective 优化 canonical residual，得到可稳定复用的 valid endpoint。", ["每件 garment 独立优化", "canonical residual 与 base state 组合", "Teacher 是训练目标与稳定 endpoint，不是 inference network"], ["F01", "F02"], ["GM-A04-TEACHER-ENDPOINTS"], ["GM-A15-PARITY-SAFETY"], [], "PAPER_CORE_EVIDENCE", ["Teacher Endpoint 是多视图优化得到的稳定状态"], ["称 Teacher 为 Upper Bound", "声称其覆盖 unseen garment"], "但两个有效 endpoint 之间的线性几何并不一定有效。", 70, "PAPER_CORE_EVIDENCE", "NO", ["需避免 upper bound 旧称"], ["先讲优化目标，再讲 residual 组合。", "强调 Teacher 不等于上界。", "指出五个 seen endpoint 的范围。", "引出端点间插值。"]),
    slide("S08", "Causal Diagnosis", "为什么不能直接插值端点？", "Geometry Is the Main Effect", "中间 artifact 的主因是什么？", "因果注入显示 geometry main effect 约 0.9959，geometry sufficiency 与 necessity 均为 10/10。", ["geometry / visibility / appearance factorial injection", "main effect ≈ 0.9959", "sufficiency 10/10; necessity 10/10", "visibility/appearance 不是主因"], ["F06"], ["GM-A05-GEOMETRY-CAUSAL-DATA"], ["GM-A02-EARLY-FAILURES"], [], "PAPER_CORE_EVIDENCE", ["geometry 是当前中间 artifact 主因"], ["推广到所有 avatar/garment 场景", "说 visibility/appearance 永远无影响"], "既然几何不能线性混合，下一步测试保留两套有效支持。", 70, "PAPER_CORE_EVIDENCE", "NO", ["当前没有 Figure Bank 内的专用 geometry 图，建议用数值 callout"], ["先解释 intervention 逻辑。", "重点报 0.9959 与两个 10/10。", "限定为当前协议。", "转到 Dual-Support。"]),
    slide("S09", "Causal Diagnosis", "Dual-Support：保留有效几何支持", "Oracle Dual-Support Mechanism", "不插值 geometry 能否改善 mixed state？", "Oracle Dual-Support 在全部 10 对服装上改善指标与严重 artifact，但它是 oracle mechanism extension，不是成功的自动 controller。", ["保留两套 endpoint geometry", "只对 effective opacity 加权", "LPIPS 0.085303→0.056985", "IoU 0.710052→0.725290; Boundary F 0.300263→0.379103"], ["F07"], ["GM-A06-DUAL-LPIPS"], ["GM-A07-DUAL-SHEET"], ["GM-A29-CONTROLLER-VISUALS"], "SUPPLEMENTARY_EXTENSION", ["oracle Dual-Support 证明 geometry-safe mechanism 有效"], ["自动 reference-conditioned mixed controller 已成功", "Dual-Support 是当前主 pipeline"], "主线仍需要一个可解释、最小化的 endpoint coordinate system。", 75, "SUPPLEMENTARY_METHOD_EXTENSION", "NO", ["O01_O03 与 O02_O03 仍困难", "需持续强调 oracle"], ["先讲结构，再报三组指标。", "指出 active Gaussian 与时间开销增加。", "明确它不进入主线。", "引出显式 basis。"]),
    slide("S10", "Core Method", "Explicit Global Basis", "Explicit Endpoint Coordinate System", "五个 Teacher residual 如何形成显式坐标？", "对五个 centered Teacher residual 做 SVD 得到 rank-4 global basis，它首先是 endpoint 坐标系，adaptation 价值尚待 LOO。", ["five residuals centered by mean", "maximum centered rank is 4", "endpoint coefficients are exact coordinates", "not yet a universal semantic manifold"], ["F03"], ["GM-A17-HEADROOM-PARITY"], ["GM-A04-TEACHER-ENDPOINTS"], [], "PAPER_CORE_METHOD_CANDIDATE", ["rank-4 精确覆盖五个 centered endpoints"], ["称 rank-4 为强压缩", "称其已证明通用 semantic manifold"], "在这个坐标系中，用最小 reference predictor 选择 endpoint。", 65, "PAPER_CORE_METHOD_CANDIDATE", "WAIT_FOR_LOO_FOR_BASIS_CLAIM", ["basis 的论文定位由 LOO 决定"], ["解释 n=5 时 centered rank≤4。", "强调坐标系而非压缩。", "区分 exact endpoint reconstruction 与 generalization。", "转入 Pure Endpoint。"]),
    slide("S11", "Core Method", "Pure Endpoint Pipeline", "Reference-to-Endpoint Selection", "reference 如何控制最终 Gaussian 状态？", "Frozen F2 特征经 mean/max aggregation 和 LayerNorm+Linear(4) 预测系数，再以 endpoint snapping 实现稳定状态。", ["reference RGB/mask → frozen F2", "mean/max aggregation → LN + Linear(4)", "raw coefficient → nearest endpoint coefficient", "snapped endpoint → frozen renderer"], ["F04", "F05"], ["GM-A03-METHOD-WIREFRAME"], ["GM-A12-RAW-SNAPPED-SHEET"], ["GM-A01-ENDPOINT-TEASER"], "PAPER_CORE_METHOD_CANDIDATE", ["最小控制器可靠选择 seen endpoints", "raw 与 realized coefficient 明确分开"], ["把 snapping 描述为精确 continuous regression", "把方法写成 unseen garment synthesis"], "可靠性需要严格的 condition-fold cross-fit 验证。", 75, "PAPER_CORE_METHOD_CANDIDATE", "WAIT_FOR_LOO_FOR_FINAL_FIGURE2", ["Figure 2 状态 METHOD_FREEZE_PENDING_LOO"], ["按箭头顺序讲 pipeline。", "在 raw coefficient 处停顿。", "强调实际 render 使用 snapped endpoint。", "转入协议。"]),
    slide("S12", "Core Method", "Pure Endpoint 实验设计", "Condition-Fold Cross-Fit", "如何避免把服装识别成功误当成泛化？", "4 rotations×3 seeds 的 condition-fold cross-fit 连同 7 methods、perturbation、parity 与 safety 检查，限定了 clean closed-bank 结论。", ["24 completed runs", "4 rotations × 3 seeds", "train/calibration/test condition folds", "7 methods + hard lookup + perturbations", "endpoint parity and identity safety"], [], ["GM-A38-PURE-ROTATIONS"], ["GM-A15-PARITY-SAFETY"], [], "PROJECT_GOVERNANCE_APPENDIX", ["协议支持固定 subject02、五个 seen garments 的闭集评估"], ["把 condition-fold 说成 garment-held-out", "把 hard lookup 当 unseen baseline"], "在这个边界内，Pure Endpoint 的 clean 结果非常明确。", 70, "PAPER_EXPERIMENT_PROTOCOL_CANDIDATE", "NO", ["需口头区分 condition-held-out 与 garment-held-out"], ["先解释 fold 划分单位。", "报 24 runs。", "说明所有比较共享 endpoint realization。", "转入主结果。"]),
    slide("S13", "Core Results", "Pure Endpoint 主结果", "Closed-Bank Endpoint Control", "reference 能否可靠选择五个 seen endpoints？", "24/24 runs 完成，五件 garment 的 clean top-1 与 precision/recall 均为 1，parity 与 identity safety PASS。", ["clean top-1 = 1.0", "per-garment precision/recall = 1.0", "endpoint parity PASS", "identity contamination = 0; severe wrong outfit = 0"], [], ["GM-A08-PURE-TOP1"], ["GM-A09-ENDPOINT-MATCH", "GM-A15-PARITY-SAFETY"], ["GM-A04-TEACHER-ENDPOINTS"], "PAPER_CORE_EVIDENCE", ["closed five-garment bank endpoint control supported"], ["unseen garment generalization supported", "优于 hard lookup"], "但 clean benchmark 上的成功与 hard lookup 功能等价。", 70, "PAPER_CORE_EVIDENCE", "NO", ["clean benchmark 已饱和"], ["先报完成数，再报准确率。", "补充安全性为零污染。", "明确 closed-bank。", "主动引出 hard lookup 等价。"]),
    slide("S14", "Core Results", "Hard Lookup 等价性", "Descriptive Relation to Hard Lookup", "Pure Endpoint 是否优于 hard lookup？", "clean 条件下正确方法选择同一 endpoint 并产生等价 render，因此当前只能描述功能等价，不能宣称优于 lookup。", ["clean agreement", "error overlap under perturbation", "endpoint-equivalent render", "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY"], [], ["GM-A10-HARD-LOOKUP"], ["GM-A36-RENDER-EQUIVALENCE"], ["GM-A08-PURE-TOP1"], "PAPER_CORE_EVIDENCE", ["clean closed-bank 下功能等价"], ["CanonDressGS 优于所有 hard lookup", "将 descriptive relation 写成 superiority"], "区别在于 predictor 输出了连续坐标，但实际成功仍来自 endpoint realization。", 70, "PAPER_CORE_EVIDENCE_DESCRIPTIVE_ONLY", "WAIT_FOR_LOO_FOR_VALUE_BEYOND_LOOKUP", ["Figure 5 仍需人工裁决"], ["先展示 agreement。", "再解释 render equivalence。", "主动说不能宣称 superiority。", "转到 raw coefficient。"]),
    slide("S15", "Core Results", "Raw Coefficient 与 Endpoint Snapping", "Prediction versus Realization", "连续系数预测本身是否精确？", "raw continuous coefficient 误差仍高，而 snapping 将预测可靠地映射到正确 endpoint；当前成功来自正确 realization。", ["raw predicted coefficient", "nearest endpoint coefficient", "realized snapped coefficient", "render uses realized state"], ["F04", "F05"], ["GM-A11-RAW-SNAPPED-PLOT"], ["GM-A12-RAW-SNAPPED-SHEET"], [], "PAPER_CORE_EVIDENCE", ["snapping 有效修正 endpoint realization"], ["raw regression 已精确", "连续 manifold 已学得"], "接下来检查这种 endpoint control 对 reference 扰动是否稳定。", 65, "PAPER_CORE_EVIDENCE", "NO", ["图例必须固定 raw/snapped 术语"], ["先指 raw error。", "再指 realized endpoint。", "解释二者不是同一输出。", "转入 perturbation。"]),
    slide("S16", "Core Results", "Perturbation 与安全性", "Robustness and Abstention", "reference 质量下降时系统如何失败？", "single reference 基本稳定、complete dropout 可安全 abstain，但 mild blur 产生明显 endpoint flip，限制了当前鲁棒性。", ["single reference: largely stable", "mild blur: elevated endpoint flips", "complete dropout: safe abstention", "identity contamination remains zero"], [], ["GM-A13-PERTURBATION"], ["GM-A14-BLUR-FAILURES", "GM-A15-PARITY-SAFETY"], [], "LIMITATION", ["报告具体 perturbation 边界与安全行为"], ["宣称鲁棒 reference understanding 已解决"], "既然 raw coefficient 不精确，下一步自然问题是 rendering loss 能否优化它。", 65, "PAPER_LIMITATION_CANDIDATE", "NO", ["blur failure 必须保留在主汇报"], ["按 single/blur/dropout 顺序讲。", "不要只展示安全项。", "强调 blur flip 是真实限制。", "引出 headroom。"]),
    slide("S17", "Headroom", "为什么做 Coefficient Headroom？", "Can Rendering Loss Refine the Coefficient?", "basis 是否不仅能分类 endpoint，还能提供优化空间？", "Headroom 实验检验固定 rank-4 span 内 test-time coefficient refinement 是否能超过 Teacher Endpoint。", ["Teacher vs exact SVD reconstruction", "refined global coefficient", "unconstrained full residual comparator", "pre-registered LPIPS gates"], ["F08"], ["GM-A24-REFINED-DECOMPOSITION"], ["GM-A17-HEADROOM-PARITY"], [], "NEGATIVE_DIAGNOSTIC", ["提出 seen-endpoint refinement 假设"], ["预设 refinement 会改善结果"], "实验只优化 4 个系数，并用 full residual 检查是否存在更大 headroom。", 55, "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC", "NO", ["不得将 headroom 动机写成最终 pipeline"], ["从 raw coefficient 误差连接动机。", "解释 Teacher/SVD parity 的必要性。", "说明 full residual 是 comparator。", "进入实验设计。"]),
    slide("S18", "Headroom", "Headroom 实验设计", "Sealed Headroom Protocol", "如何区分 basis 限制与优化失败？", "120 个 optimizer runs 在 optimize/calibration/test 上比较 Teacher、SVD、refined coefficient 与 full residual，并进行 λ 选择与门槛审计。", ["120/120 optimizer runs", "36,000 optimizer steps", "960 checkpoints", "Teacher/SVD parity 20/20 PASS", "fixed rank-4 basis; four trainable coefficients"], ["F08"], ["GM-A17-HEADROOM-PARITY"], ["GM-A18-HEADROOM-FOUR-METHOD", "GM-A21-HEADROOM-DISPLACEMENT"], [], "NEGATIVE_DIAGNOSTIC", ["说明 sealed attempt_002 的完整 protocol"], ["将 full residual 结果归因于未完成运行", "暗示 renderer 重新运行于本规划任务"], "完整执行后，结果否定了 seen-endpoint refinement 路线。", 65, "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC", "NO", ["数字应与 sealed summary 原样一致"], ["先报 parity，再报 120/36000/960。", "解释 λ selection。", "强调本次规划未运行任何 optimizer。", "转入结果。"]),
    slide("S19", "Headroom", "Headroom 结果：Teacher Span 已在局部最优", "Teacher Span at Local Optimum", "refinement 是否改善 Teacher Endpoint？", "global rank-4 coefficient refinement 未改善 Teacher，0/5 garments 过门槛；full residual 反而产生 patch/cloud/mottle 与 edge scatter。", ["macro Teacher-minus-refined LPIPS < 0", "0/5 garments pass improvement gate", "full residual severe degradation", "classification: TEACHER_SPAN_AT_LOCAL_OPTIMUM"], [], ["GM-A16-HEADROOM-OVERVIEW"], ["GM-A19-HEADROOM-GATES", "GM-A20-HEADROOM-PER-GARMENT"], ["GM-A22-FULL-RESIDUAL-ARTIFACT", "GM-A23-SPAN-NULL"], "NEGATIVE_DIAGNOSTIC", ["Headroom 没有改善 Teacher Endpoint", "full residual 退化并产生 artifacts"], ["将 Headroom 写成正结果", "声称 basis span recoverable denominator 非零"], "这个负结果直接改变当前 pipeline 的取舍。", 80, "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC", "NO", ["caption 必须写明不支持 render refinement 进入主线"], ["先报 0/5。", "再解释 macro gain 的符号。", "最后展示 full-residual artifact。", "用 classification 收束。"]),
    slide("S20", "Headroom", "Headroom 如何改变方法设计", "Design Decision after a Negative Result", "哪些路线因此被冻结？", "reference→coefficient→test-time refinement 不进入主 pipeline，spatial coefficient field 也因缺乏 full-residual headroom 仅保留 contingency。", ["remove test-time render refinement from main pipeline", "Headroom: CONSUMED_FROM_SEALED_RESULT", "Spatial field: CONTINGENCY_ONLY_NOT_STARTED", "negative result narrows claims and implementation"], [], ["GM-A24-REFINED-DECOMPOSITION"], ["GM-A22-FULL-RESIDUAL-ARTIFACT", "GM-A35-FIGURE-MAP"], [], "LIMITATION", ["负结果支持冻结 refinement 与 local spatial basis"], ["宣称 spatial basis 已测试失败", "宣称所有优化都无效"], "剩下真正能判断 basis 是否有独立价值的问题是 LOO adaptation。", 60, "PAPER_LIMITATION_OR_SUPPLEMENTARY", "NO", ["spatial field 是未启动 contingency，不是负实验"], ["清楚划掉 refinement 分支。", "区分 not supported 与 proven impossible。", "说明 spatial field 没有启动。", "引出 LOO。"]),
    slide("S21", "LOO Decision", "LOO：basis 剩余的核心价值检验", "Leave-One-Garment-Out Basis Adaptation", "explicit basis 能否超越 closed-bank lookup？", "LOO 用四件 garment 构建 rank≤3 basis，并完全排除第五件，直接检验 out-of-bank basis adaptation。", ["build μ_-g and B_-g from four garments", "exclude held-out garment completely", "compare K=1/K=2, hard lookup, oracle projection, full residual", "few-view coefficient fitting"], ["F09", "F10"], ["GM-A33-LOO-PENDING"], ["GM-A28-O07-LIMITATION"], [], "PENDING_LOO", ["LOO 是判断 basis adaptation value 的关键实验"], ["预判 LOO 成功", "把 historical O07 当正式 LOO 结果"], "协议已经修复完成，但当前没有 sealed execution result。", 70, "WAIT_FOR_LOO", "REQUIRED", ["不得读取 partial/active output"], ["先解释 held-out 的信息边界。", "强调 rank≤3。", "说明历史 O07 不等于新 LOO。", "进入当前状态。"]),
    slide("S22", "LOO Decision", "LOO 当前状态", "Ready / Pending Execution", "我们现在能报告什么？", "LOO protocol 已 READY，但 metadata-only 检查显示 LOO_ATTEMPT=ABSENT，因此结果页只保留占位。", ["protocol: LOO_FEW_VIEW_FOLD_REPAIRED_AND_READY", "execution: READY / PENDING EXECUTION", "LOO_ATTEMPT=ABSENT", "files_read=0; no fabricated metrics"], [], ["GM-A33-LOO-PENDING"], ["GM-A35-FIGURE-MAP"], [], "PENDING_LOO", ["只报告 protocol readiness 与 pending 状态"], ["任何 LOO 指标、成功率或图片", "使用正在运行或 partial output"], "在等待 LOO 的同时，历史消融只用于解释设计演化。", 40, "WAIT_FOR_LOO", "REQUIRED", ["汇报当天需再次做 metadata-only 状态确认"], ["一句话说明没有结果。", "报告 files_read=0。", "不要填任何占位数字。", "转入历史设计证据。"]),
    slide("S23", "Historical Evidence", "历史 51 实验如何塑造最小模型", "Historical Ablations", "哪些旧实验仍有设计解释价值？", "rank、mask、mean-only、standardization 与 reference-count 消融解释了为何收缩到显式 basis 与最小 predictor，但它们属于历史 view-transductive 协议。", ["51/51 experiments; 14,400 steps", "rank 1–4", "clothing mask / mean-only / standardization", "reference count 1/2/3", "historical protocol kept separate"], [], ["GM-A37-HIST-ALL"], ["GM-A25-HIST-RANK", "GM-A26-HIST-REFERENCE"], ["GM-A27-HIST-REPRESENTATION"], "HISTORICAL_DESIGN_EVIDENCE", ["历史消融解释设计选择"], ["与 current Pure Endpoint 无标记合并成主表", "把 historical view-transductive 写成 current generalization"], "这些证据也暴露了当前仍必须正面呈现的失败模式。", 70, "HISTORICAL_ONLY", "NO", ["表格必须用 Historical 标签"], ["开头先打 Historical 标签。", "报 51 与 14,400。", "只挑三个最影响设计的消融。", "转入 failure modes。"]),
    slide("S24", "Limitations", "Failure Modes / Limitations", "What Still Breaks", "当前系统最脆弱的地方是什么？", "O07 collapse、blur flip、neural/full-residual artifacts、困难 Dual-Support pair 与第二身份服装容量共同限定当前 claim。", ["historical O07 → O03 collapse", "mild-blur endpoint flip", "neural/full-residual patch-cloud-mottle", "O01_O03 and O02_O03 remain difficult", "Subject00 loose-clothing capacity unproven"], [], ["GM-A28-O07-LIMITATION"], ["GM-A14-BLUR-FAILURES", "GM-A22-FULL-RESIDUAL-ARTIFACT"], ["GM-A07-DUAL-SHEET"], "LIMITATION", ["集中展示已知失败边界"], ["把失败归为展示偶例而忽略定量证据"], "其中第二身份目前只完成了 base-avatar foundation。", 70, "PAPER_LIMITATION_CANDIDATE", "NO", ["避免堆叠太多小图，最终 PPT 需人工选 3 类"], ["按 generalization/robustness/artifact 三类讲。", "每类只给一个结论。", "说明失败如何影响 claim。", "转入 multi-identity。"]),
    slide("S25", "Multi-Identity", "Multi-Identity 基础进展", "Foundations, Not Completion", "第二身份做到哪一步？", "Subject00 short canary 与 medium base-avatar pilot PASS，formal 与三服装端点仍 pending；AvatarReX 仅 preflight ready 且媒体禁止导出。", ["Subject00: short canary PASS", "Subject00: medium base-avatar PASS", "formal 101245-step protocol READY; garment endpoints pending", "AvatarReX: zero-copy/preflight ready; derived template/LBS missing", "AvatarReX media use = 0"], [], ["GM-A31-SUBJECT00-SHEET"], ["GM-A32-SUBJECT00-PLOT", "GM-A34-AVATARREX-METADATA"], [], "MULTI_IDENTITY_FOUNDATION", ["仅称 base-avatar foundation 与 protocol readiness"], ["multi-identity garment editing completed", "展示或导出 AvatarReX media"], "因此论文冻结必须把已完成核心与待验证扩展分开。", 65, "SUPPLEMENTARY_FOUNDATION_ONLY", "NO", ["Subject00 图不能被解释为 garment editing"], ["先讲 Subject00 的 base-avatar 范围。", "再讲 formal 与 garment pending。", "AvatarReX 只讲 metadata。", "转到方法冻结状态。"]),
    slide("S26", "Method Freeze", "当前论文方法冻结状态", "Current Method Freeze", "哪些模块已冻结，哪些等待 LOO？", "Teacher Endpoint、explicit endpoint basis、Pure Endpoint 与 geometry causal 已形成核心候选；basis 的超 lookup 定位和 Figure 2 仍等待 LOO。", ["Frozen candidate: Teacher Endpoint, explicit basis, Pure Endpoint", "Core evidence: geometry causal", "Supplementary: oracle Dual-Support", "Negative supplementary: Controller and Headroom", "Figure 2: METHOD_FREEZE_PENDING_LOO"], [], ["GM-A35-FIGURE-MAP"], ["GM-A03-METHOD-WIREFRAME", "GM-A33-LOO-PENDING"], [], "PROJECT_GOVERNANCE_APPENDIX", ["展示当前冻结矩阵"], ["称论文已 final", "把 Controller 放回主方法"], "冻结矩阵对应一组明确的 reviewer risks。", 65, "PRESENTATION_GOVERNANCE", "REQUIRED_FOR_FINAL_FREEZE", ["PAPER_FINAL 必须保持 0"], ["按 core/supplementary/pending 三列讲。", "点名 Figure 2 状态。", "强调 Controller 不在主线。", "转入风险。"]),
    slide("S27", "Method Freeze", "当前不足与 Reviewer 风险", "Reviewer Risk Map", "投稿前最可能被质疑什么？", "最大风险是 clean closed-bank 饱和且与 hard lookup 等价，LOO 将决定 basis 是否具有独立方法贡献。", ["closed-bank saturation", "no superiority over hard lookup", "rank-4 is not strong compression", "no headroom for refinement", "multi-identity garment editing incomplete", "automatic mixed controller failed", "Teacher/view-transductive boundary"], [], ["GM-A10-HARD-LOOKUP"], ["GM-A23-SPAN-NULL", "GM-A33-LOO-PENDING"], ["GM-A30-CONTROLLER-BUDGET"], "LIMITATION", ["主动呈现 reviewer 风险和对应证据"], ["用工程完成度替代科学贡献", "隐去 hard lookup 等价"], "下一步必须按风险优先级执行，而不是继续扩大方法。", 75, "PAPER_RISK_DISCUSSION", "REQUIRED", ["LOO 失败时需收缩论文主张"], ["先说最大风险。", "再列支撑风险的三条事实。", "说明负结果已阻止无效扩张。", "转入 next steps。"]),
    slide("S28", "Closing", "下一步：先完成决定性实验，再冻结论文", "Next Steps", "今晚之后的最高优先级是什么？", "先完成并 sealed LOO，再据此冻结方法范围、refresh Figure Bank 和重写论文；第二身份与 spatial field 仅按证据门槛推进。", ["1. Execute and seal LOO", "2. Freeze basis claim and Figure 2", "3. Refresh Figure Bank", "4. Rewrite Method/Experiments", "5. Subject00 formal + three-garment canary", "6. Spatial field only if a new oracle headroom gate supports it"], [], ["GM-A35-FIGURE-MAP"], ["GM-A33-LOO-PENDING"], [], "PENDING_LOO", ["按证据门槛安排 next steps"], ["自动启动 PPTX 生成", "在 LOO 前宣布论文完成"], "汇报结束，进入问题讨论。", 55, "PRESENTATION_CLOSING", "REQUIRED", ["NEXT_TASK 仅为人工审阅 outline 与选图"], ["按优先级读，不展开工程细节。", "第一项必须是 LOO。", "明确 PPTX 不在本任务生成。", "收束并进入 Q&A。"]),
]


COMPACT_SPECS = [
    ("C01", "Opening", "研究目标与当前边界", "Task and Boundary", "当前到底支持什么？", "当前支持 subject02 closed-bank seen-endpoint control，LOO 与 multi-identity 仍未完成。", ["reference garment + target pose/camera", "closed five-garment bank", "PAPER_FINAL=0"], [], ["GM-A01-ENDPOINT-TEASER"], ["GM-A35-FIGURE-MAP"], [], "PAPER_CORE_METHOD_CANDIDATE", ["closed-bank endpoint control"], ["unseen garment completion"], "先解释这个任务为什么难。", 50, "CANDIDATE", "WAIT_FOR_LOO_FOR_TITLE", ["人工选 teaser"], ["先讲任务。", "第二句讲边界。", "不要预告 LOO 结果。", "进入困难。"], ["S01", "S02"]),
    ("C02", "Problem", "服装控制的核心困难", "Geometry, Visibility, Appearance", "为什么不能直接连续插值？", "服装耦合 geometry、visibility 和 appearance，其中无效 geometry 是中间 artifact 主因。", ["lookup stable/discrete", "interpolation flexible/artifact-prone"], ["F06"], ["GM-A02-EARLY-FAILURES"], ["GM-A05-GEOMETRY-CAUSAL-DATA"], [], "PAPER_CORE_EVIDENCE", ["三因素与几何主因"], ["泛化到所有场景"], "这推动方法路线发生了连续收缩。", 55, "CORE_PROBLEM", "NO", ["历史图需标签"], ["先讲 tradeoff。", "点出 geometry。", "预告因果证据。", "进入演化。"], ["S03", "S04"]),
    ("C03", "Research Evolution", "从 reference-to-residual 到 endpoint control", "Historical Research Contraction", "失败如何改变方法？", "历史 capacity 与 causal 诊断排除了错误假设，路线收缩为 Teacher endpoints、explicit basis 与最小控制器。", ["full model failures", "support capacity PASS", "Teacher → basis → Pure Endpoint → Headroom → LOO"], [], ["GM-A02-EARLY-FAILURES"], ["GM-A27-HIST-REPRESENTATION"], [], "HISTORICAL_DESIGN_EVIDENCE", ["历史负结果推动收缩"], ["最终方案从一开始已确定"], "先看收缩后的当前 pipeline。", 60, "PRESENTATION_ONLY", "NO", ["协议分层"], ["按箭头讲。", "不展开每个失败。", "强调 causal diagnosis。", "转到 pipeline。"], ["S05"]),
    ("C04", "Core Method", "当前 Pipeline", "Pure Endpoint Pipeline", "reference 如何决定 Gaussian endpoint？", "Frozen F2 预测 raw coefficient，endpoint snapping 选择 valid Teacher state，再由 frozen renderer 输出。", ["reference → F2 → aggregation → Linear(4)", "raw coefficient → endpoint snapping → renderer"], ["F04", "F05"], ["GM-A03-METHOD-WIREFRAME"], ["GM-A12-RAW-SNAPPED-SHEET"], [], "PAPER_CORE_METHOD_CANDIDATE", ["raw/snapped distinction"], ["continuous coefficient is exact"], "这个 pipeline 依赖稳定的 Teacher Endpoint。", 65, "METHOD_CANDIDATE", "WAIT_FOR_LOO_FOR_FIGURE2", ["Figure 2 pending LOO"], ["按输入到输出讲。", "停在 snapping。", "标出 frozen 模块。", "转到 Teacher。"], ["S06", "S11"]),
    ("C05", "Method Foundation", "Teacher Endpoint 与冻结底座", "Valid Garment States", "稳定服装状态如何建立？", "每件 seen garment 通过多视图 rendering objective 优化 canonical residual；Teacher 是 endpoint，不是 Upper Bound。", ["frozen avatar", "per-garment residual optimization", "five valid endpoints"], ["F01", "F02"], ["GM-A04-TEACHER-ENDPOINTS"], ["GM-A15-PARITY-SAFETY"], [], "PAPER_CORE_EVIDENCE", ["Teacher endpoint"], ["Teacher upper bound"], "下一页给出不能插值它们的因果证据。", 60, "CORE_EVIDENCE", "NO", ["术语审计"], ["讲 frozen/trained。", "报五件 garment。", "禁止 upper bound。", "转到 geometry。"], ["S06", "S07"]),
    ("C06", "Causal Diagnosis", "Geometry 是中间 artifact 主因", "Geometry Main Effect", "哪一类通道导致失败？", "geometry main effect 约 0.9959，sufficiency/necessity 均为 10/10。", ["factor injection", "0.9959", "10/10 + 10/10"], ["F06"], ["GM-A05-GEOMETRY-CAUSAL-DATA"], ["GM-A07-DUAL-SHEET"], [], "PAPER_CORE_EVIDENCE", ["current-protocol geometry main effect"], ["visibility/appearance universally irrelevant"], "因此 basis 表示端点坐标，而不是直接插值几何。", 55, "CORE_EVIDENCE", "NO", ["使用数字 callout"], ["一句讲实验。", "一句报数字。", "一句给设计含义。", "转到 basis。"], ["S08", "S09"]),
    ("C07", "Core Method", "Explicit Basis：endpoint 坐标系", "Rank-4 Endpoint Basis", "rank-4 到底代表什么？", "五个 centered Teacher residual 的最大 rank 本就是 4，因此它是精确 endpoint 坐标系，不是强压缩。", ["ΔG≈μ+Bc", "rank=4", "adaptation value pending LOO"], ["F03"], ["GM-A17-HEADROOM-PARITY"], ["GM-A04-TEACHER-ENDPOINTS"], [], "PAPER_CORE_METHOD_CANDIDATE", ["endpoint coordinate system"], ["universal semantic manifold", "strong compression"], "接着看这个坐标系上的严格 cross-fit。", 55, "METHOD_CANDIDATE", "REQUIRED_FOR_BASIS_VALUE", ["LOO decides claim"], ["解释 centered rank。", "区分坐标与泛化。", "不讲压缩优势。", "转入 protocol。"], ["S10"]),
    ("C08", "Core Results", "Pure Endpoint Protocol", "Condition-Fold Cross-Fit", "成功是否来自泄漏或单一划分？", "24 runs 覆盖 4 rotations×3 seeds，并审计 7 methods、perturbation、parity 与 identity safety。", ["24 runs", "4×3", "condition folds", "7 methods + safety"], [], ["GM-A38-PURE-ROTATIONS"], ["GM-A15-PARITY-SAFETY"], [], "PROJECT_GOVERNANCE_APPENDIX", ["closed-bank protocol"], ["garment-held-out"], "协议内的 clean 结果达到满分。", 55, "EXPERIMENT_PROTOCOL", "NO", ["condition vs garment held-out"], ["报设计。", "解释 fold 单位。", "强调安全审计。", "转结果。"], ["S12"]),
    ("C09", "Core Results", "Pure Endpoint 结果", "Closed-Bank Endpoint Control", "seen endpoint 能否可靠选择？", "clean top-1 与每 garment precision/recall 均为 1，parity PASS 且 identity contamination=0。", ["top-1=1", "precision/recall=1", "parity PASS", "identity contamination=0"], [], ["GM-A08-PURE-TOP1"], ["GM-A09-ENDPOINT-MATCH", "GM-A15-PARITY-SAFETY"], [], "PAPER_CORE_EVIDENCE", ["seen endpoint selection supported"], ["unseen garment supported"], "但满分结果必须和 hard lookup 一起解释。", 55, "CORE_EVIDENCE", "NO", ["benchmark saturation"], ["报数字。", "报安全。", "限定 closed-bank。", "转 lookup。"], ["S13"]),
    ("C10", "Core Results", "与 Hard Lookup 功能等价", "Descriptive Equivalence", "为什么这不是 superiority？", "clean 条件下二者选择相同 endpoint 并产生等价 render，所以当前 relation 只能描述性报告。", ["agreement", "render equivalence", "no superiority claim"], [], ["GM-A10-HARD-LOOKUP"], ["GM-A36-RENDER-EQUIVALENCE"], [], "PAPER_CORE_EVIDENCE", ["functional equivalence"], ["better than hard lookup"], "真正的差异需要 out-of-bank LOO 来检验。", 55, "DESCRIPTIVE_ONLY", "REQUIRED_FOR_VALUE_BEYOND_LOOKUP", ["人工裁决 Figure 5"], ["主动承认等价。", "解释同 endpoint 同 render。", "不要回避 lookup。", "连接 LOO。"], ["S14"]),
    ("C11", "Core Results", "Snapping 有效，但 blur 仍脆弱", "Realization and Robustness", "成功来自哪里，失败发生在哪？", "成功主要来自 endpoint snapping 而非精确 continuous regression；mild blur 会触发 endpoint flip。", ["raw coefficient error high", "snapping exact endpoint", "single reference stable", "blur flips; dropout abstains"], ["F05"], ["GM-A11-RAW-SNAPPED-PLOT"], ["GM-A13-PERTURBATION", "GM-A14-BLUR-FAILURES"], [], "LIMITATION", ["snapping and perturbation boundary"], ["robust continuous manifold learned"], "这正是做 Headroom refinement 的原因。", 60, "CORE_EVIDENCE_AND_LIMITATION", "NO", ["不要弱化 blur"], ["先讲 raw/snapped。", "再讲 perturbation。", "突出 blur。", "转 Headroom。"], ["S15", "S16"]),
    ("C12", "Headroom", "Headroom：连续优化是否还有空间？", "Refinement Hypothesis", "rendering loss 能否超过 Teacher？", "实验固定 rank-4 basis，只优化 4 个系数，并用 full residual 作 headroom comparator。", ["Teacher/SVD", "refined coefficient", "full residual", "pre-registered gates"], ["F08"], ["GM-A24-REFINED-DECOMPOSITION"], ["GM-A17-HEADROOM-PARITY"], [], "NEGATIVE_DIAGNOSTIC", ["headroom hypothesis"], ["refinement is main pipeline"], "120 次优化给出了明确负结果。", 50, "SUPPLEMENTARY_NEGATIVE", "NO", ["只讲 sealed attempt_002"], ["连接 raw error。", "解释四方法。", "说明门槛。", "转结果。"], ["S17", "S18"]),
    ("C13", "Headroom", "Headroom 负结果收缩了 Pipeline", "Teacher Span at Local Optimum", "refinement 有改善吗？", "0/5 garments 过门槛，refined coefficient 不改善 Teacher；full residual 产生 patch/cloud/mottle，因此 refinement 不进主线。", ["120/120 runs", "0/5 pass", "macro gain < 0", "full residual artifacts", "TEACHER_SPAN_AT_LOCAL_OPTIMUM"], [], ["GM-A16-HEADROOM-OVERVIEW"], ["GM-A20-HEADROOM-PER-GARMENT", "GM-A22-FULL-RESIDUAL-ARTIFACT"], [], "NEGATIVE_DIAGNOSTIC", ["negative result and design decision"], ["positive headroom", "spatial field tested"], "只剩 LOO 能检验 basis 的独立价值。", 70, "SUPPLEMENTARY_NEGATIVE", "NO", ["caption boundary"], ["先报 0/5。", "再看 artifact。", "给出 pipeline 决策。", "转 LOO。"], ["S19", "S20"]),
    ("C14", "LOO Decision", "LOO 设计与当前状态", "Decisive Pending Test", "basis 能否适配完全排除的 garment？", "四 garment 构建 rank≤3 basis、第五件完全 held out；protocol READY，但 LOO_ATTEMPT=ABSENT。", ["μ_-g, B_-g", "K=1/K=2", "hard lookup / oracle projection / full residual", "READY / PENDING EXECUTION; files_read=0"], ["F09", "F10"], ["GM-A33-LOO-PENDING"], ["GM-A28-O07-LIMITATION"], [], "PENDING_LOO", ["design and pending status"], ["fabricated LOO result"], "等待 LOO 时，历史消融只能解释设计。", 65, "WAIT_FOR_LOO", "REQUIRED", ["汇报前 metadata recheck"], ["讲 held-out 边界。", "报 READY。", "明确 absent。", "不展示任何结果。"], ["S21", "S22"]),
    ("C15", "Historical Evidence", "历史消融：设计证据，不是当前主表", "Historical Design Evidence", "旧协议还能回答什么？", "51 个 view-transductive 实验解释 rank、mask、standardization 与 reference count 选择，但不能与 current protocol 混表。", ["51 runs", "14,400 steps", "rank/mask/mean/std/reference count"], [], ["GM-A37-HIST-ALL"], ["GM-A25-HIST-RANK", "GM-A26-HIST-REFERENCE"], [], "HISTORICAL_DESIGN_EVIDENCE", ["design rationale"], ["current generalization result"], "这些历史结果与新负诊断共同定义限制。", 55, "HISTORICAL_ONLY", "NO", ["Historical 标签"], ["先报历史协议。", "挑关键消融。", "说明不能混表。", "转限制。"], ["S23"]),
    ("C16", "Limitations", "当前限制与第二身份基础", "Limitations and Portability", "投稿前还有哪些实质缺口？", "O07、blur、full-residual artifacts 与 controller failure 均未解决；Subject00 只完成 base-avatar foundation。", ["failure modes", "controller supplementary diagnostic", "Subject00 base only", "AvatarReX media=0"], [], ["GM-A28-O07-LIMITATION"], ["GM-A22-FULL-RESIDUAL-ARTIFACT", "GM-A31-SUBJECT00-SHEET"], ["GM-A29-CONTROLLER-VISUALS"], "LIMITATION", ["current limitations"], ["multi-identity garment editing complete"], "因此当前方法冻结必须保持保守。", 60, "LIMITATION", "NO", ["避免把 Subject00 图写成 garment"], ["按三类失败讲。", "Subject00 只讲 base。", "AvatarReX 不出图。", "转冻结。"], ["S24", "S25"]),
    ("C17", "Method Freeze", "当前方法冻结与 Reviewer 风险", "Method Freeze", "论文主线现在能冻结到哪里？", "核心候选已收敛到 Teacher+basis+Pure Endpoint，但 hard lookup 等价和 LOO pending 阻止最终贡献冻结。", ["core candidate", "supplementary oracle", "negative diagnostics", "Figure 2 pending LOO", "largest reviewer risk"], [], ["GM-A35-FIGURE-MAP"], ["GM-A10-HARD-LOOKUP", "GM-A33-LOO-PENDING"], [], "LIMITATION", ["freeze matrix and reviewer risk"], ["PAPER_FINAL"], "下一步只按决定性风险排序。", 65, "GOVERNANCE", "REQUIRED", ["PAPER_FINAL=0"], ["列 core/supp/pending。", "点名 hard lookup。", "点名 LOO。", "转 next steps。"], ["S26", "S27"]),
    ("C18", "Closing", "下一步", "Next Steps", "最先做什么？", "先 sealed LOO，再冻结 Figure 2 与论文 claim；随后人工选组会素材，不自动生成 PPTX。", ["LOO", "method/figure freeze", "paper rewrite", "Subject00", "manual PPT asset selection"], [], ["GM-A33-LOO-PENDING"], ["GM-A35-FIGURE-MAP"], [], "PENDING_LOO", ["evidence-gated next steps"], ["automatic PPTX generation"], "结束并进入讨论。", 50, "CLOSING", "REQUIRED", ["NEXT_TASK fixed"], ["第一项 LOO。", "第二项 freeze。", "强调人工选图。", "结束。"], ["S28"]),
]

COMPACT = [slide(*spec[:-1], full_mapping=spec[-1]) for spec in COMPACT_SPECS]


APPENDIX = [
    slide("A01", "Appendix", "数据集与服装定义", "Dataset and Garments", "评估对象和闭集边界是什么？", "主实验固定 subject02 与 O01/O02/O03/O04/O08 五件 seen garments。", ["identity", "garment IDs", "reference/query units"], [], ["GM-A04-TEACHER-ENDPOINTS"], ["GM-A38-PURE-ROTATIONS"], [], "PROJECT_GOVERNANCE_APPENDIX", ["定义 closed bank"], ["unseen garment"], "按提问跳转。", 60, "APPENDIX", "NO", ["需最终核对展示名称"], ["只在被问数据集时使用。", "先身份再服装。", "说明 O07 历史角色。"]),
    slide("A02", "Appendix", "Exact Rotation Manifests", "Cross-Fit Rotations", "每个 rotation 如何划分？", "四个 rotation 的 train/calibration/test condition folds 已封存且无重复泄漏。", ["rotation records", "fold membership", "duplicate preservation"], [], ["GM-A38-PURE-ROTATIONS"], [], [], "PROJECT_GOVERNANCE_APPENDIX", ["协议可复核"], ["garment-held-out"], "按提问跳转。", 75, "APPENDIX", "NO", ["只展示摘要，不暴露过密 manifest"], ["说明 fold 单位。", "说明重复处理。", "指向正式 JSON。"]),
    slide("A03", "Appendix", "Pure Endpoint 七方法注册表", "Seven-Method Registry", "比较方法是否同边界？", "七方法共享 frozen endpoint/evaluator 边界，hard lookup relation 仅描述性报告。", ["method IDs", "trainable parameters", "realization rule"], [], ["GM-A38-PURE-ROTATIONS"], ["GM-A10-HARD-LOOKUP"], [], "PROJECT_GOVERNANCE_APPENDIX", ["方法注册完整"], ["superiority claim"], "按提问跳转。", 75, "APPENDIX", "NO", ["最终 PPT 可转为小表"], ["先列方法。", "再列 realization。", "强调公平性。"]),
    slide("A04", "Appendix", "完整 Perturbation 表", "Perturbation Matrix", "每类扰动的分母与失败率是什么？", "single reference、mild blur 与 complete dropout 必须分别报告，不能用总平均掩盖 blur failure。", ["denominators", "endpoint flips", "abstention", "identity safety"], [], ["GM-A13-PERTURBATION"], ["GM-A14-BLUR-FAILURES", "GM-A15-PARITY-SAFETY"], [], "LIMITATION", ["完整 robustness 边界"], ["总体鲁棒性已解决"], "按提问跳转。", 75, "APPENDIX", "NO", ["保持原始分母"], ["按三种扰动讲。", "先失败再安全。", "不做平均美化。"]),
    slide("A05", "Appendix", "历史 Rank Ablation", "Historical Rank Ablation", "rank=4 为什么被保留？", "历史 rank 1–4 结果提供设计线索，但 rank=4 的当前数学理由仍是五 endpoint centered rank 上限。", ["historical trend", "current algebraic rank"], ["F03"], ["GM-A25-HIST-RANK"], ["GM-A37-HIST-ALL"], [], "HISTORICAL_DESIGN_EVIDENCE", ["历史趋势"], ["当前 protocol 主结果"], "按提问跳转。", 60, "APPENDIX", "NO", ["双重理由要分开"], ["先打 Historical 标签。", "再讲 rank 上限。", "不称压缩。"]),
    slide("A06", "Appendix", "Controller 诊断", "Why the Controller Was Frozen", "自动 mixed controller 为什么停止？", "pair identification 未达核心门槛，最新预算诊断指向 optimizer/normalization limit，因此仅保留 supplementary diagnostic。", ["CONTROLLER_V2_CROSSFIT_MICRO_PILOT_FAIL", "OPTIMIZER_OR_NORMALIZATION_LIMIT", "CONTROLLER_TRAINING_REPAIR_MODERATE_VALUE"], [], ["GM-A30-CONTROLLER-BUDGET"], ["GM-A29-CONTROLLER-VISUALS"], [], "NEGATIVE_DIAGNOSTIC", ["controller 未成功"], ["controller 是当前主模块"], "按提问跳转。", 75, "APPENDIX", "NO", ["避免只报 AUROC 而忽略 pair gate"], ["先讲失败门槛。", "再讲预算轨迹。", "说明冻结决策。"]),
    slide("A07", "Appendix", "Dual-Support 全 10 Pair", "All-Pair Oracle Mechanism", "改善是否覆盖所有 pair？", "全部 10 pair 的 aggregate 指标改善，但困难 pair 和计算开销仍存在。", ["per-pair LPIPS", "IoU/Boundary F", "active Gaussian and render cost"], ["F07"], ["GM-A06-DUAL-LPIPS"], ["GM-A07-DUAL-SHEET"], [], "SUPPLEMENTARY_EXTENSION", ["oracle all-pair result"], ["automatic controller"], "按提问跳转。", 75, "APPENDIX", "NO", ["指出困难 pair"], ["先讲 aggregate。", "再点困难 pair。", "最后报 oracle 定位。"]),
    slide("A08", "Appendix", "Headroom λ Selection", "Headroom Selection Protocol", "超参数是否看了 test？", "λ 在规定 split 上选择，test 仅用于 sealed evaluation；120 runs 与 960 checkpoints 完整。", ["optimize/calibration/test", "λ selection", "120 runs", "960 checkpoints"], ["F08"], ["GM-A21-HEADROOM-DISPLACEMENT"], ["GM-A19-HEADROOM-GATES"], [], "NEGATIVE_DIAGNOSTIC", ["selection protocol closed"], ["test-selected λ"], "按提问跳转。", 60, "APPENDIX", "NO", ["口头解释 split"], ["先讲 split。", "再讲 λ。", "报完整运行数。"]),
    slide("A09", "Appendix", "Full-Residual Failure", "Why More Capacity Hurt", "为什么 full residual 反而更差？", "更大自由度在有限 rendering objective 下重开 patch/cloud/mottle 与 edge scatter，说明容量不等于可泛化 headroom。", ["artifact taxonomy", "test degradation", "identity safety still PASS"], [], ["GM-A22-FULL-RESIDUAL-ARTIFACT"], ["GM-A18-HEADROOM-FOUR-METHOD"], [], "NEGATIVE_DIAGNOSTIC", ["full residual degrades"], ["证明所有 residual optimization 必然失败"], "按提问跳转。", 75, "APPENDIX", "NO", ["机制解释保持假设语气"], ["先展示 artifact。", "再联系指标。", "最后限定结论。"]),
    slide("A10", "Appendix", "Subject00 / AvatarReX 基础", "Multi-Identity Foundations", "第二身份的证据与限制是什么？", "Subject00 仅有 base-avatar evidence；AvatarReX 仅 metadata preflight，媒体与 derived assets 均不可作为结果。", ["Subject00 short/medium", "formal and garment pending", "AvatarReX license restricted"], [], ["GM-A31-SUBJECT00-SHEET"], ["GM-A32-SUBJECT00-PLOT", "GM-A34-AVATARREX-METADATA"], [], "MULTI_IDENTITY_FOUNDATION", ["foundation only"], ["multi-garment completion", "AvatarReX media"], "按提问跳转。", 60, "APPENDIX", "NO", ["license audit"], ["先 Subject00。", "再 pending。", "AvatarReX 不展示媒体。"]),
    slide("A11", "Appendix", "Claim Boundary Matrix", "Allowed and Forbidden Claims", "哪些措辞可以进入论文？", "所有核心结论必须绑定 fixed subject02、closed five-garment seen-reference protocol；八类开放问题不得提前宣称。", ["supported", "descriptive only", "pending", "forbidden"], [], ["GM-A35-FIGURE-MAP"], ["GM-A10-HARD-LOOKUP", "GM-A33-LOO-PENDING"], [], "PROJECT_GOVERNANCE_APPENDIX", ["claim audit"], ["PAPER_FINAL"], "按提问跳转。", 75, "APPENDIX", "REQUIRED", ["随 LOO 更新"], ["先讲 supported。", "再讲 pending。", "最后讲 forbidden。"]),
    slide("A12", "Appendix", "执行治理与证据完整性", "Governance and Evidence Completeness", "这些结论是否可追溯？", "所有组会素材绑定 source path/SHA/branch/HEAD；本任务 GPU、训练、推理、renderer、PPTX、Figure Bank mutation 均为 0。", ["provenance", "mutation audit", "LOO files_read=0", "PAPER_FINAL=0"], [], ["GM-A35-FIGURE-MAP"], ["GM-A33-LOO-PENDING"], [], "PROJECT_GOVERNANCE_APPENDIX", ["audit trail complete"], ["scientific work executed in planning task"], "按提问跳转。", 60, "APPENDIX", "NO", ["Cloud sync DNS pending is infrastructure only"], ["展示 provenance 字段。", "报 mutation=0。", "区分 cloud 状态与科学状态。"]),
]


FORMULAS = [
    {"formula_id": "F01", "name": "Teacher Endpoint optimization", "latex": r"\Delta G_g^*=\arg\min_{\Delta G_g}\sum_{v\in\mathcal V_g}\mathcal L_{render}(R(G_0\oplus\Delta G_g;v),I_{g,v})", "full_slides": ["S07"], "compact_slides": ["C05"], "must_explain": True, "timing_seconds": 35, "variables": {"G_0": "frozen base canonical Gaussians", "Delta G_g": "garment residual", "V_g": "teacher optimization views", "R": "frozen renderer"}, "avoid": "Do not call the Teacher an Upper Bound."},
    {"formula_id": "F02", "name": "Residual endpoint", "latex": r"G_g=G_0\oplus\Delta G_g^*", "full_slides": ["S07"], "compact_slides": ["C05"], "must_explain": True, "timing_seconds": 15, "variables": {"G_g": "valid garment endpoint", "oplus": "field-wise canonical residual composition"}, "avoid": "Do not imply interpolation between endpoints."},
    {"formula_id": "F03", "name": "Global affine basis", "latex": r"\Delta G_g\approx\mu+B c_g,\quad \mathrm{rank}(B)=4", "full_slides": ["S10"], "compact_slides": ["C07"], "must_explain": True, "timing_seconds": 30, "variables": {"mu": "mean of five Teacher residuals", "B": "centered SVD basis", "c_g": "endpoint coefficient"}, "avoid": "Rank 4 is not strong compression for five centered samples."},
    {"formula_id": "F04", "name": "Reference coefficient prediction", "latex": r"\hat c=f_\theta(\operatorname{Agg}(F_2(I_{ref},M_{ref})))", "full_slides": ["S11", "S15"], "compact_slides": ["C04"], "must_explain": True, "timing_seconds": 25, "variables": {"F_2": "frozen reference feature extractor", "Agg": "mean/max aggregation", "f_theta": "LayerNorm plus Linear(4)", "hat c": "raw coefficient"}, "avoid": "Do not present raw coefficients as exact."},
    {"formula_id": "F05", "name": "Endpoint realization", "latex": r"g^*=\arg\min_g\|\hat c-c_g\|_2,\qquad c_{realized}=c_{g^*}", "full_slides": ["S11", "S15"], "compact_slides": ["C04", "C11"], "must_explain": True, "timing_seconds": 25, "variables": {"c_g": "registered endpoint coefficient", "g_star": "selected garment endpoint", "c_realized": "snapped coefficient used for rendering"}, "avoid": "Do not conflate prediction with realization."},
    {"formula_id": "F06", "name": "Linear geometry interpolation", "latex": r"G_{geom}(\alpha)=(1-\alpha)G_{geom}^{a}+\alpha G_{geom}^{b}", "full_slides": ["S03", "S08"], "compact_slides": ["C02", "C06"], "must_explain": True, "timing_seconds": 20, "variables": {"alpha": "mix coefficient", "G_geom": "Gaussian geometry channels"}, "avoid": "Do not describe the interpolated support as necessarily valid."},
    {"formula_id": "F07", "name": "Dual-Support effective opacity", "latex": r"\tilde\alpha_i^{(a)}=(1-w)\alpha_i^{(a)},\qquad \tilde\alpha_j^{(b)}=w\alpha_j^{(b)}", "full_slides": ["S09"], "compact_slides": [], "must_explain": False, "timing_seconds": 25, "variables": {"w": "oracle mixture weight", "alpha": "endpoint opacity", "tilde_alpha": "effective opacity"}, "avoid": "Do not call w an automatically predicted successful controller output."},
    {"formula_id": "F08", "name": "Headroom coefficient optimization", "latex": r"c^*=\arg\min_c\mathcal L_{render}(R(G_0\oplus(\mu+Bc)))", "full_slides": ["S17", "S18"], "compact_slides": ["C12"], "must_explain": True, "timing_seconds": 25, "variables": {"c": "four optimized coefficients", "B": "fixed rank-4 basis", "R": "frozen renderer"}, "avoid": "Do not imply this improved Teacher Endpoint."},
    {"formula_id": "F09", "name": "LOO basis", "latex": r"\mu_{-g}=\frac{1}{4}\sum_{h\ne g}\Delta G_h,\qquad B_{-g}=\operatorname{SVD}_{\le3}(\{\Delta G_h-\mu_{-g}\}_{h\ne g})", "full_slides": ["S21"], "compact_slides": ["C14"], "must_explain": True, "timing_seconds": 35, "variables": {"g": "fully held-out garment", "mu_-g": "four-garment mean", "B_-g": "rank at most 3 basis"}, "avoid": "No held-out residual may enter basis construction."},
    {"formula_id": "F10", "name": "LOO coefficient fitting", "latex": r"c_{-g}^*=\arg\min_c\sum_{v\in\mathcal V_{few}}\mathcal L_{render}(R(G_0\oplus(\mu_{-g}+B_{-g}c);v),I_{g,v})", "full_slides": ["S21"], "compact_slides": ["C14"], "must_explain": True, "timing_seconds": 35, "variables": {"V_few": "few-view adaptation set", "c_-g": "held-out garment fitted coefficient"}, "avoid": "Do not report a result before a sealed LOO attempt exists."},
]


QA = [
    ("Q01", "为什么不直接用 hard lookup？", "clean closed-bank 下 hard lookup 与 CanonDressGS 功能等价。继续保留 basis 的唯一科学理由是检验 out-of-bank adaptation；这由 LOO 决定。", ["GM-A10-HARD-LOOKUP", "GM-A33-LOO-PENDING"]),
    ("Q02", "当前方法和分类器有什么区别？", "当前 realized behavior 确实接近 endpoint classifier 加 lookup；显式 coefficient space 的额外价值尚未由 clean benchmark 证明。", ["GM-A11-RAW-SNAPPED-PLOT"]),
    ("Q03", "rank-4 是不是伪低维？", "五个 residual 居中后最大 rank 就是 4，因此不是强压缩；它是精确 endpoint coordinate system。", ["GM-A17-HEADROOM-PARITY"]),
    ("Q04", "为什么 Teacher 不是 Upper Bound？", "Teacher 是固定多视图 objective 下的局部优化 endpoint，不保证对所有 objective、视图或表示全局最优。", ["GM-A04-TEACHER-ENDPOINTS"]),
    ("Q05", "Headroom 为什么失败？", "sealed 结果表明 rank-4 span 内优化没有 LPIPS gain，可能因为 Teacher/SVD 已在该 objective 的局部最优；不把机制推断写成已证明因果。", ["GM-A20-HEADROOM-PER-GARMENT"]),
    ("Q06", "full residual 为什么反而更差？", "更大自由度重新打开了 patch/cloud/mottle 与 edge-scatter 解；有限 rendering objective 未提供足够正则来保证 test 泛化。", ["GM-A22-FULL-RESIDUAL-ARTIFACT"]),
    ("Q07", "LOO 为什么能回答核心问题？", "它从 basis construction 中完全排除目标 garment，再用 few views 拟合系数，直接测试 basis 是否含有可迁移结构。", ["GM-A33-LOO-PENDING"]),
    ("Q08", "如果 LOO 失败，论文还剩什么贡献？", "仍有 geometry causal attribution、valid endpoint construction、closed-bank reference control、hard-lookup equivalence audit 和负诊断；但 basis 主贡献必须显著收缩。", ["GM-A05-GEOMETRY-CAUSAL-DATA", "GM-A10-HARD-LOOKUP"]),
    ("Q09", "Dual-Support 是最终方法吗？", "不是。它是 oracle geometry-safe mechanism，证明保留有效支持有价值；自动 mixed controller 未成功。", ["GM-A06-DUAL-LPIPS", "GM-A29-CONTROLLER-VISUALS"]),
    ("Q10", "Controller 为什么停止？", "核心 pair identification 与 routing gates 未达标；最新诊断是 optimizer/normalization limit，修复价值仅 moderate。", ["GM-A30-CONTROLLER-BUDGET"]),
    ("Q11", "当前是否支持 unseen garment？", "不支持。Pure Endpoint 只覆盖五个 seen endpoints，正式 unseen adaptation 等待 LOO。", ["GM-A08-PURE-TOP1", "GM-A33-LOO-PENDING"]),
    ("Q12", "是否支持第二身份？", "仅支持说 Subject00 base-avatar pilot PASS；第二身份 garment endpoints 和 editing 尚未完成。", ["GM-A31-SUBJECT00-SHEET"]),
    ("Q13", "Teacher 是否使用 test views？", "Teacher 是 offline endpoint construction；当前报告必须按各 sealed protocol 的 view boundary 解释，不能把 Teacher evidence 当 strict unseen-view inference。", ["GM-A38-PURE-ROTATIONS"]),
    ("Q14", "为什么不直接做 spatial local basis？", "Headroom 没有显示 global span 外存在可泛化 full-residual gain，full residual 还严重退化，因此 spatial field 只保留 contingency。", ["GM-A23-SPAN-NULL", "GM-A22-FULL-RESIDUAL-ARTIFACT"]),
    ("Q15", "论文最终主线可能有哪些版本？", "LOO 成功则突出 transferable basis adaptation；LOO 失败则收缩为 endpoint control、geometry diagnosis 与 limitation-focused evidence。", ["GM-A33-LOO-PENDING", "GM-A35-FIGURE-MAP"]),
    ("Q16", "这么多负实验值得吗？", "值得，因为它们分别排除了 support capacity、geometry interpolation、neural decoder、automatic controller 和 test-time refinement 等错误路线。", ["GM-A02-EARLY-FAILURES", "GM-A16-HEADROOM-OVERVIEW"]),
    ("Q17", "当前最主要投稿风险是什么？", "closed-bank benchmark 饱和且与 hard lookup 等价；没有 LOO 正结果时，basis 的独立贡献不足。", ["GM-A10-HARD-LOOKUP", "GM-A33-LOO-PENDING"]),
    ("Q18", "Headroom 负结果会不会否定 explicit basis？", "它否定的是 seen-endpoint test-time refinement value，不否定 basis 作为精确 endpoint coordinate system；adaptation value 仍由 LOO 决定。", ["GM-A16-HEADROOM-OVERVIEW", "GM-A33-LOO-PENDING"]),
]

QA_REGISTRY = [
    {"qa_id": qid, "question": question, "recommended_answer": answer, "evidence_asset_ids": assets, "claim_boundary": "ANSWER_ONLY_WITH_SEALED_EVIDENCE", "risk": "Do not extend beyond the answer wording without new sealed evidence."}
    for qid, question, answer, assets in QA
]


CLAIMS = [
    ("CL01", "Canonical Gaussian support represents five seen garments.", "SUPPORTED", "PAPER_CORE_EVIDENCE", ["S07", "S13"]),
    ("CL02", "Geometry interpolation is the main source of intermediate artifacts in the sealed protocol.", "SUPPORTED", "PAPER_CORE_EVIDENCE", ["S08"]),
    ("CL03", "Teacher Endpoints stably store valid seen garment states.", "SUPPORTED", "PAPER_CORE_EVIDENCE", ["S07"]),
    ("CL04", "Pure Endpoint reliably selects five seen endpoints under clean closed-bank evaluation.", "SUPPORTED", "PAPER_CORE_EVIDENCE", ["S13"]),
    ("CL05", "Pure Endpoint and hard lookup are functionally equivalent on clean closed-bank inputs.", "DESCRIPTIVE_ONLY", "PAPER_CORE_EVIDENCE", ["S14"]),
    ("CL06", "Endpoint snapping is effective while raw continuous coefficients remain inaccurate.", "SUPPORTED", "PAPER_CORE_EVIDENCE", ["S15"]),
    ("CL07", "Global rank-4 coefficient refinement does not improve Teacher Endpoint.", "NEGATIVE_RESULT", "NEGATIVE_DIAGNOSTIC", ["S19"]),
    ("CL08", "Unconstrained full-residual optimization degrades and creates patch/cloud/mottle artifacts.", "NEGATIVE_RESULT", "NEGATIVE_DIAGNOSTIC", ["S19"]),
    ("CL09", "Oracle Dual-Support improves all-pair aggregates without proving an automatic controller.", "SUPPORTED_WITH_ORACLE_BOUNDARY", "SUPPLEMENTARY_EXTENSION", ["S09"]),
    ("CL10", "Unseen garment adaptation is supported.", "PENDING_FORBIDDEN_UNTIL_LOO", "PENDING_LOO", ["S21", "S22"]),
    ("CL11", "Arbitrary garment editing is supported.", "FORBIDDEN", "LIMITATION", ["S27"]),
    ("CL12", "Strict unseen-reference generalization is supported.", "FORBIDDEN", "LIMITATION", ["S27"]),
    ("CL13", "Cross-identity garment editing is complete.", "FORBIDDEN", "MULTI_IDENTITY_FOUNDATION", ["S25"]),
    ("CL14", "Spatially distributed coefficient fields are effective.", "FORBIDDEN_NOT_STARTED", "LIMITATION", ["S20"]),
    ("CL15", "The explicit basis is a universal semantic garment manifold.", "FORBIDDEN", "LIMITATION", ["S10"]),
    ("CL16", "CanonDressGS outperforms all hard lookup baselines.", "FORBIDDEN", "LIMITATION", ["S14"]),
    ("CL17", "Automated mixed-reference control succeeds.", "FORBIDDEN", "NEGATIVE_DIAGNOSTIC", ["S09", "A06"]),
    ("CL18", "Teacher is an Upper Bound.", "FORBIDDEN_TERMINOLOGY", "LIMITATION", ["S07"]),
]

CLAIM_MANIFEST = [
    {"claim_id": cid, "claim": text, "claim_status": claim_status, "slide_status": slide_status, "slides": slides, "evidence_required": True}
    for cid, text, claim_status, slide_status, slides in CLAIMS
]


MISSING = [
    {"material_id": "MISS-01", "title": "Final Figure 1 five-query selection", "priority": "P0", "affected_slides": ["S01", "S02", "C01"], "reason": "Candidate contact sheet exists but requires manual adjudication.", "action": "Select five representative clean queries without changing scientific pixels.", "blocking": False, "dependency": "MANUAL_REVIEW"},
    {"material_id": "MISS-02", "title": "Final Figure 2 method freeze", "priority": "P0", "affected_slides": ["S06", "S11", "S26", "C04", "C17"], "reason": "Figure 2 is METHOD_FREEZE_PENDING_LOO.", "action": "Freeze pipeline after LOO classification.", "blocking": True, "dependency": "SEALED_LOO"},
    {"material_id": "MISS-03", "title": "Sealed LOO result assets", "priority": "P0", "affected_slides": ["S21", "S22", "S27", "S28", "C14", "C17", "C18"], "reason": "LOO_ATTEMPT=ABSENT; files_read=0.", "action": "Execute, seal, audit, and then refresh the placeholder.", "blocking": True, "dependency": "SEALED_LOO"},
    {"material_id": "MISS-04", "title": "Dedicated geometry causal visual", "priority": "P1", "affected_slides": ["S08", "C06"], "reason": "The sealed JSON exists but the current Figure Bank has no dedicated causal summary panel.", "action": "Use numeric callouts now; later manually build an editable chart from sealed values.", "blocking": False, "dependency": "MANUAL_CHART_FROM_SEALED_JSON"},
    {"material_id": "MISS-05", "title": "Final paper baseline table after LOO", "priority": "P0", "affected_slides": ["S27", "C17"], "reason": "Current and historical protocols cannot be merged, and LOO is pending.", "action": "Design the current-protocol table only after LOO.", "blocking": True, "dependency": "SEALED_LOO"},
    {"material_id": "MISS-06", "title": "Subject00 formal and garment endpoints", "priority": "P1", "affected_slides": ["S25", "C16", "A10"], "reason": "Only short/medium base-avatar evidence exists.", "action": "Complete formal base and three-garment canary under their protocols.", "blocking": False, "dependency": "FUTURE_EXPERIMENT"},
    {"material_id": "MISS-07", "title": "AvatarReX template/LBS derived assets", "priority": "P2", "affected_slides": ["S25", "A10"], "reason": "Preflight is ready but derived assets are missing and media export is restricted.", "action": "Keep metadata-only until license-compliant derived assets exist.", "blocking": False, "dependency": "LICENSE_AND_DERIVED_ASSETS"},
    {"material_id": "MISS-08", "title": "Final captions and paper wording", "priority": "P0", "affected_slides": ["S26", "S27", "S28"], "reason": "PAPER_FINAL=0 and method scope depends on LOO.", "action": "Rewrite only after the presentation outline is manually reviewed and LOO is sealed.", "blocking": True, "dependency": "MANUAL_REVIEW_AND_LOO"},
]


GAP_MAP = [
    ("title", "WAIT_FOR_LOO", "Current title may overstate adaptation; freeze after LOO."),
    ("abstract", "REWRITE", "Separate closed-bank endpoint control, geometry evidence, negative diagnostics, and pending LOO."),
    ("contributions", "WAIT_FOR_LOO", "Basis contribution depends on adaptation evidence; hard-lookup relation must remain explicit."),
    ("teaser", "REWRITE", "Use endpoint-only candidate and remove any mixed-controller implication."),
    ("Figure 2 pipeline", "WAIT_FOR_LOO", "METHOD_FREEZE_PENDING_LOO; exclude test-time refinement and automatic mixed controller."),
    ("controller section", "MOVE_TO_SUPPLEMENTARY", "Controller V1/V2 are failed diagnostics, not a core method module."),
    ("Dual-Support positioning", "MOVE_TO_SUPPLEMENTARY", "Position as oracle geometry-safe mechanism extension."),
    ("training objective", "REWRITE", "Distinguish Teacher endpoint construction, predictor training, and optional adaptation objectives."),
    ("experiment questions", "REWRITE", "Organize by support, causal geometry, closed-bank control, hard lookup, headroom, and LOO."),
    ("baseline table", "WAIT_FOR_LOO", "Do not mix historical view-transductive and current cross-fit results."),
    ("continuous control section", "REWRITE", "Retain geometry causal evidence; remove unsupported continuous-manifold success language."),
    ("multi-identity section", "REWRITE", "Subject00 is base-avatar foundation; AvatarReX remains metadata-only."),
    ("limitations", "KEEP", "Strengthen hard-lookup equivalence, blur flips, Headroom artifacts, Controller failure, and multi-identity gaps."),
    ("conclusion", "WAIT_FOR_LOO", "Final scope must follow LOO classification."),
]


def asset_usages() -> list[dict]:
    usages = []
    for deck_name, slides in (("FULL", FULL), ("COMPACT", COMPACT), ("APPENDIX", APPENDIX)):
        for item in slides:
            for purpose, field in (("PRIMARY_ASSET", "primary_assets"), ("SECONDARY_ASSET", "secondary_assets"), ("BACKUP_ASSET", "backup_assets")):
                for index, asset_id in enumerate(item[field], start=1):
                    base = dict(ASSETS[asset_id])
                    base.update({
                        "usage_id": f"{deck_name}-{item['slide_id']}-{purpose}-{index:02d}",
                        "deck": deck_name,
                        "slide_id": item["slide_id"],
                        "purpose": purpose,
                        "backup_asset_ids": item["backup_assets"],
                    })
                    usages.append(base)
    return usages


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def md_slide(item: dict) -> str:
    lines = [
        f"## {item['slide_id']} · {item['title_cn']}",
        f"- **Section / status**: {item['section']} / `{item['figure_status']}`",
        f"- **Core question**: {item['core_question']}",
        f"- **One-sentence conclusion**: {item['one_sentence_conclusion']}",
        f"- **Main content**: {'；'.join(item['main_content'])}",
        f"- **Formulas**: {', '.join(item['formulas']) if item['formulas'] else 'none'}",
        f"- **Primary assets**: {', '.join(item['primary_assets'])}",
        f"- **Secondary assets**: {', '.join(item['secondary_assets']) if item['secondary_assets'] else 'none'}",
        f"- **Backup assets**: {', '.join(item['backup_assets']) if item['backup_assets'] else 'none'}",
        f"- **Formal paths**: {'; '.join(item['formal_paths'])}",
        f"- **Allowed claims**: {'；'.join(item['allowed_claims'])}",
        f"- **Forbidden claims**: {'；'.join(item['forbidden_claims'])}",
        f"- **Transition**: {item['transition']}",
        f"- **Timing**: {item['timing_seconds']} seconds",
        f"- **Paper eligibility**: `{item['paper_eligibility']}`",
        f"- **LOO dependency**: `{item['loo_dependency']}`",
        f"- **Risks / TODO**: {'；'.join(item['risks_todo'])}",
    ]
    if "full_slide_mapping" in item:
        lines.append(f"- **Full mapping**: {', '.join(item['full_slide_mapping'])}")
    return "\n".join(lines)


def build_markdown() -> None:
    storyline = f"""# CanonDressGS Group Meeting PPT Storyline — 2026-07-24

## Governance

- Task: `{TASK_ID}`
- Source: `{SOURCE_BRANCH}@{SOURCE_HEAD}`
- Target: `{TARGET_BRANCH}`
- LOO: `READY / PENDING EXECUTION`; `LOO_ATTEMPT=ABSENT`; files read = 0
- Boundaries: GPU=0, training=0, inference=0, renderer=0, new scientific images=0, image API=0, PPTX=0, paper mutation=0, Figure Bank mutation=0, `PAPER_FINAL=0`

## Story Arc

The presentation follows the actual contraction of the research rather than a retrospective success story:

1. **Problem** — reference garment + target pose/camera should produce the same identity with a specified garment.
2. **Early failures** — full reference-to-residual routes fail despite adequate support capacity.
3. **Causal diagnosis** — geometry interpolation is the dominant source of invalid intermediate support.
4. **Valid states** — per-garment Teacher Endpoints provide stable canonical residual states.
5. **Explicit coordinates** — centered SVD gives a rank-4 endpoint coordinate system, not yet a universal manifold.
6. **Pure Endpoint** — a minimal reference predictor plus endpoint snapping reliably selects five seen states.
7. **Hard lookup audit** — clean closed-bank behavior is functionally equivalent to hard lookup.
8. **Headroom negative result** — coefficient refinement adds no gain; full residual reintroduces artifacts.
9. **Decisive pending test** — LOO is the remaining test of basis value beyond lookup.
10. **Scope freeze** — core, supplementary, negative, pending, and multi-identity foundation evidence remain visibly separated.

## Section Structure

| Section | Full slides | Purpose |
|---|---:|---|
| Opening and problem | S01–S04 | Task, difficulty, and research questions |
| Research evolution and foundation | S05–S10 | Contraction, Teacher, causal geometry, Dual-Support, basis |
| Core method and evidence | S11–S16 | Pipeline, protocol, result, lookup relation, snapping, robustness |
| Headroom decision | S17–S20 | Hypothesis, protocol, negative result, design consequence |
| LOO decision | S21–S22 | Decisive design and honest pending status |
| Historical, limitations, identity | S23–S25 | Design evidence and unresolved scope |
| Freeze and next steps | S26–S28 | Paper state, reviewer risks, priorities |

## Visual Rules for the Later PPTX

- 16:9, white/light-gray background, one conclusion per slide.
- Use existing sealed scientific assets only; do not beautify or regenerate scientific pixels.
- Keep fixed visual labels for Main, Historical, Negative, Pending, and Foundation.
- Keep method/baseline/Teacher colors stable after manual asset selection.
- Show source IDs in notes, not as dense Git metadata on the visible slide.
- Headroom caption must state that refinement did not improve Teacher and full residual produced patch/cloud/mottle artifacts.
"""
    (DOC_DIR / "CANONDRESSGS_GROUP_MEETING_PPT_STORYLINE_20260724.md").write_text(storyline, encoding="utf-8")

    full_md = "# CanonDressGS Group Meeting Full Outline — 2026-07-24\n\n**Plan:** 28 slides, 25–30 minutes, plus A1–A12 appendix.\n\n" + "\n\n".join(md_slide(item) for item in FULL) + "\n\n# Appendix\n\n" + "\n\n".join(md_slide(item) for item in APPENDIX) + "\n"
    (DOC_DIR / "CANONDRESSGS_GROUP_MEETING_FULL_OUTLINE_20260724.md").write_text(full_md, encoding="utf-8")

    compact_md = "# CanonDressGS Group Meeting Compact Outline — 2026-07-24\n\n**Plan:** 18 independently reorganized slides, 15–18 minutes. Every slide maps to the full version.\n\n" + "\n\n".join(md_slide(item) for item in COMPACT) + "\n"
    (DOC_DIR / "CANONDRESSGS_GROUP_MEETING_COMPACT_OUTLINE_20260724.md").write_text(compact_md, encoding="utf-8")

    notes = ["# CanonDressGS Group Meeting Speaker Notes — 2026-07-24", "", "These are speaking prompts, not a verbatim script.", ""]
    for item in FULL + APPENDIX:
        notes.extend([f"## {item['slide_id']} · {item['title_cn']}"] + [f"- {point}" for point in item["speaker_notes"]] + [f"- **Likely follow-up:** {item['risks_todo'][0]}", f"- **Recommended answer boundary:** {'；'.join(item['allowed_claims'])}", f"- **Transition:** {item['transition']}", ""])
    (DOC_DIR / "CANONDRESSGS_GROUP_MEETING_SPEAKER_NOTES_20260724.md").write_text("\n".join(notes), encoding="utf-8")

    qa_lines = ["# CanonDressGS Group Meeting Q&A Preparation — 2026-07-24", "", "All answers are bounded by sealed evidence; LOO metrics must remain absent until a sealed result exists.", ""]
    for item in QA_REGISTRY:
        qa_lines.extend([f"## {item['qa_id']} · {item['question']}", item["recommended_answer"], "", f"Evidence: `{', '.join(item['evidence_asset_ids'])}`", ""])
    (DOC_DIR / "CANONDRESSGS_GROUP_MEETING_QA_PREPARATION_20260724.md").write_text("\n".join(qa_lines), encoding="utf-8")

    missing_lines = ["# CanonDressGS Group Meeting Missing Materials — 2026-07-24", "", "Pending material is explicit and does not block the planning classification.", "", "| ID | Priority | Material | Affected slides | Blocking | Dependency | Action |", "|---|---|---|---|---|---|---|"]
    for item in MISSING:
        missing_lines.append(f"| {item['material_id']} | {item['priority']} | {item['title']} | {', '.join(item['affected_slides'])} | {item['blocking']} | {item['dependency']} | {item['action']} |")
    missing_lines.extend(["", "## LOO Boundary", "", "`LOO_ATTEMPT=ABSENT`; no LOO result files were read. The placeholder is intentionally metadata-only."])
    (DOC_DIR / "CANONDRESSGS_GROUP_MEETING_MISSING_MATERIALS_20260724.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")

    gap_lines = ["# CanonDressGS Current Paper Narrative Gap Map — 2026-07-24", "", "This is an audit only; no paper text is modified.", "", "| Paper element | Action | Reason |", "|---|---|---|"]
    for element, action, reason in GAP_MAP:
        gap_lines.append(f"| {element} | `{action}` | {reason} |")
    gap_lines.extend(["", "## Action Definitions", "", "- `KEEP`: retain with evidence-bound wording.", "- `REWRITE`: current wording conflicts with the sealed narrative.", "- `MOVE_TO_SUPPLEMENTARY`: evidence is useful but not a core method result.", "- `REMOVE`: unsupported material should not remain.", "- `WAIT_FOR_LOO`: the correct wording depends on the sealed LOO classification."])
    (DOC_DIR / "CANONDRESSGS_CURRENT_PAPER_NARRATIVE_GAP_MAP_20260724.md").write_text("\n".join(gap_lines) + "\n", encoding="utf-8")


def build_outputs() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_DIR.mkdir(parents=True, exist_ok=True)
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    build_markdown()

    common = {"schema_version": "canondressgs.presentation.plan.v1", "task_id": TASK_ID, "source_branch": SOURCE_BRANCH, "source_head": SOURCE_HEAD, "target_branch": TARGET_BRANCH, "paper_final": 0}
    write_json(PROTO_DIR / "canondressgs_group_meeting_slide_plan.json", {**common, "version": "FULL", "slide_count": len(FULL), "estimated_minutes": [25, 30], "slides": FULL, "appendix_count": len(APPENDIX), "appendix": APPENDIX})
    write_json(PROTO_DIR / "canondressgs_group_meeting_compact_slide_plan.json", {**common, "version": "COMPACT", "slide_count": len(COMPACT), "estimated_minutes": [15, 18], "slides": COMPACT})
    usages = asset_usages()
    write_json(PROTO_DIR / "canondressgs_group_meeting_slide_asset_manifest.json", {**common, "asset_definition_count": len(ASSETS), "asset_usage_count": len(usages), "directly_usable_asset_count": sum(1 for item in ASSETS.values() if item["source_sha"] != "METADATA_ONLY_NO_LOCAL_FILE" and item["figure_status"] != "PENDING_LOO"), "manual_selection_asset_count": 4, "pending_loo_asset_count": 1, "assets": usages})
    write_json(PROTO_DIR / "canondressgs_group_meeting_formula_manifest.json", {**common, "formula_count": len(FORMULAS), "formulas": FORMULAS})
    write_json(PROTO_DIR / "canondressgs_group_meeting_claim_manifest.json", {**common, "claims": CLAIM_MANIFEST})
    write_json(PROTO_DIR / "canondressgs_group_meeting_question_answer_registry.json", {**common, "question_count": len(QA_REGISTRY), "questions": QA_REGISTRY})
    write_json(PROTO_DIR / "canondressgs_group_meeting_missing_material_registry.json", {**common, "missing_material_count": len(MISSING), "loo_pending_material_count": sum(1 for item in MISSING if item["dependency"] == "SEALED_LOO"), "materials": MISSING})

    output_paths = [
        "docs/PRESENTATION/CANONDRESSGS_GROUP_MEETING_PPT_STORYLINE_20260724.md",
        "docs/PRESENTATION/CANONDRESSGS_GROUP_MEETING_FULL_OUTLINE_20260724.md",
        "docs/PRESENTATION/CANONDRESSGS_GROUP_MEETING_COMPACT_OUTLINE_20260724.md",
        "docs/PRESENTATION/CANONDRESSGS_GROUP_MEETING_SPEAKER_NOTES_20260724.md",
        "docs/PRESENTATION/CANONDRESSGS_GROUP_MEETING_QA_PREPARATION_20260724.md",
        "docs/PRESENTATION/CANONDRESSGS_GROUP_MEETING_MISSING_MATERIALS_20260724.md",
        "docs/PRESENTATION/CANONDRESSGS_CURRENT_PAPER_NARRATIVE_GAP_MAP_20260724.md",
        "paper_protocol/presentation/canondressgs_group_meeting_slide_plan.json",
        "paper_protocol/presentation/canondressgs_group_meeting_compact_slide_plan.json",
        "paper_protocol/presentation/canondressgs_group_meeting_slide_asset_manifest.json",
        "paper_protocol/presentation/canondressgs_group_meeting_formula_manifest.json",
        "paper_protocol/presentation/canondressgs_group_meeting_claim_manifest.json",
        "paper_protocol/presentation/canondressgs_group_meeting_question_answer_registry.json",
        "paper_protocol/presentation/canondressgs_group_meeting_missing_material_registry.json",
        "paper_protocol/presentation/canondressgs_group_meeting_plan_final_summary.json",
        "project_control_handoff/canondressgs_group_meeting_presentation_plan_handoff.json",
    ]
    validation = {
        "status": "32_CHECK_TEST_SUITE_INCLUDED; FINAL_RESULT_REPORTED_AFTER_PUSH",
        "required_check_count": 32,
        "test_command": "python tools/test_group_meeting_presentation_plan.py",
    }
    summary = {
        **common,
        "status": "FINAL",
        "final_classification": "GROUP_MEETING_PRESENTATION_PLAN_READY",
        "full_slide_count": len(FULL),
        "full_estimated_minutes": [25, 30],
        "compact_slide_count": len(COMPACT),
        "compact_estimated_minutes": [15, 18],
        "appendix_count": len(APPENDIX),
        "formula_count": len(FORMULAS),
        "qa_count": len(QA_REGISTRY),
        "claim_count": len(CLAIM_MANIFEST),
        "asset_definition_count": len(ASSETS),
        "asset_usage_count": len(usages),
        "directly_usable_asset_count": sum(1 for item in ASSETS.values() if item["source_sha"] != "METADATA_ONLY_NO_LOCAL_FILE" and item["figure_status"] != "PENDING_LOO"),
        "manual_selection_asset_count": 4,
        "pending_loo_asset_count": 1,
        "loo": {"protocol_status": "LOO_FEW_VIEW_FOLD_REPAIRED_AND_READY", "execution_status": "READY / PENDING EXECUTION", "attempt": "ABSENT", "files_read": 0, "result_fabricated": 0},
        "figure_status": {"Figure 2": "METHOD_FREEZE_PENDING_LOO", "Headroom": "CONSUMED_FROM_SEALED_RESULT", "LOO": "PENDING_LOO_ADAPTATION", "Spatial coefficient field": "CONTINGENCY_ONLY_NOT_STARTED"},
        "mutation_audit": {"gpu": 0, "training": 0, "model_inference": 0, "renderer": 0, "new_scientific_images": 0, "image_api_calls": 0, "pptx_generated": 0, "presentation_pdf_generated": 0, "paper_modified": 0, "experiment_output_mutation": 0, "figure_bank_mutation": 0, "paper_final": 0},
        "cloud_status": "CLOUD_SYNC_PENDING_DNS",
        "outputs": output_paths,
        "validation": validation,
        "next_task": "MANUALLY_REVIEW_GROUP_MEETING_PPT_OUTLINE_AND_SELECT_ASSETS",
    }
    write_json(PROTO_DIR / "canondressgs_group_meeting_plan_final_summary.json", summary)
    handoff = {
        **common,
        "handoff_status": "READY_FOR_MANUAL_REVIEW",
        "final_classification": "GROUP_MEETING_PRESENTATION_PLAN_READY",
        "source_heads": SOURCES,
        "deliverables": output_paths,
        "loo_status": summary["loo"],
        "mutation_audit": summary["mutation_audit"],
        "cloud_status": "CLOUD_SYNC_PENDING_DNS",
        "next_task": summary["next_task"],
        "do_not_auto_execute": "BUILD_EDITABLE_CANONDRESSGS_GROUP_MEETING_PPTX",
        "manual_decisions_required": ["Choose compact or full version", "Confirm talk duration", "Select final assets", "Choose retained technical depth"],
    }
    write_json(HANDOFF_DIR / "canondressgs_group_meeting_presentation_plan_handoff.json", handoff)


if __name__ == "__main__":
    build_outputs()
    print(f"Generated {len(FULL)} full slides, {len(COMPACT)} compact slides, and {len(APPENDIX)} appendix slides.")
