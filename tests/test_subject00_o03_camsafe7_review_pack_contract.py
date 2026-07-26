from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = (
    ROOT
    / "tools"
    / "second_identity"
    / "prepare_subject00_o03_camsafe7_review_pack.py"
)
PDF_BUILDER = (
    ROOT
    / "tools"
    / "second_identity"
    / "build_subject00_o03_camsafe7_review_pdf.py"
)


def source():
    return GENERATOR.read_text(encoding="utf-8")


def test_scripts_are_valid_python():
    ast.parse(source())
    ast.parse(PDF_BUILDER.read_text(encoding="utf-8"))


def test_frozen_source_and_new_branch_are_explicit():
    tree = ast.parse(source())
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"SOURCE_HEAD", "BRANCH"}
    }
    assert values["SOURCE_HEAD"] == "a0461084021e62f0a02e4e1a1f601556a8b4017e"
    assert values["BRANCH"] == (
        "research/subject00-o03-camerasafe7-provisional-teacher-"
        "review-pack-20260727"
    )


def test_exact_safe7_and_excluded_request_contract():
    text = source()
    for slot in ("slot_00", "slot_01", "slot_02", "slot_03", "slot_05", "slot_06", "slot_07"):
        assert slot in text
    assert "subject00_O03_slot04_canary_attempt004_cand00" in text
    assert '"slot04_sample_count"] == 0' in text


def test_exact_eleven_primary_pages_and_fixed_order():
    tree = ast.parse(source())
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    page_specs = ast.literal_eval(assignments["PAGE_SPECS"])
    assert len(page_specs) == 11
    assert page_specs[0][1] == "subject00_O03_camsafe7_teacher_master_overview.png"
    assert page_specs[-1][1] == (
        "subject00_O03_camsafe7_animation_and_slot04_quarantine.png"
    )


def test_page_resolution_contract_is_publication_scale():
    text = source()
    assert "size=(6400, 4000)" in text
    assert "size=(6400, 4800)" in text
    assert "width=6400" not in text  # dimensions are page contracts, not image stretching


def test_generator_has_no_training_renderer_or_inference_call_path():
    text = source()
    tree = ast.parse(text)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "import torch" not in text
    assert "torch.optim" not in text
    assert not any(node.func.attr in {"backward", "step", "render"} for node in calls)
    assert "segmentation" not in text.lower()
    assert "mask inference" not in text.lower()


def test_human_and_scientific_fields_are_frozen_null():
    text = source()
    for field in (
        "garment_visual_pass",
        "identity_visual_pass",
        "silhouette_visual_pass",
        "boundary_visual_pass",
        "protected_region_visual_pass",
        "hands_feet_visual_pass",
        "camera_pose_visual_pass",
        "animation_visual_pass",
        "final_human_visual_decision",
        "scientific_pass",
    ):
        assert f'"{field}": None' in text
    assert '"paper_eligible": False' in text


def test_pdf_builder_has_fixed_eleven_page_order_and_validation():
    text = PDF_BUILDER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    filenames = ast.literal_eval(assignments["PAGE_FILENAMES"])
    assert len(filenames) == 11
    assert "from reportlab.pdfgen import canvas" in text
    assert "from pypdf import PdfReader" in text
    assert "len(reader.pages) != 11" in text


def test_poppler_render_and_visual_qa_are_mandatory():
    text = source()
    assert '"pdftoppm"' in text
    assert '"pdfinfo"' in text
    assert "pdf_render_qa_contact.png" in text
    assert "PASS_POPPLER_RENDER_11_OF_11" in text


def test_required_git_artifacts_are_declared_without_binary_assets():
    text = source()
    for name in (
        "subject00_O03_camsafe7_human_review_upload_manifest_20260727.json",
        "subject00_O03_camsafe7_review_pack_execution_tests_20260727.json",
        "subject00_O03_camsafe7_review_pack_final_summary_20260727.json",
        "subject00_O03_camsafe7_review_pack_handoff_20260727.json",
        "SUBJECT00_O03_CAMSAFE7_REVIEW_PACK_REPORT_20260727.md",
    ):
        assert name in text
    assert "RISK_ROOT / PDF_NAME" not in text


def test_final_classification_and_unique_next_task():
    text = source()
    assert (
        "SUBJECT00_O03_CAMSAFE7_PROVISIONAL_TEACHER_REVIEW_PACK_"
        in text
    )
    assert "READY_FOR_USER_REVIEW" in text
    assert "USER_UPLOAD_AND_REVIEW_SUBJECT00_O03_CAMSAFE7_REVIEW_PAGES" in text
