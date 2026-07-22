from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from tools.paper.figure_builders.build_canondressgs_teaser import (
    BASE_SPEC,
    GARMENTS,
    RESULT_HASHES,
    build,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    REPO_ROOT
    / "paper_protocol/figure_manifests/figure1_canondressgs_teaser_sources.json"
)
PNG_PATH = REPO_ROOT / "paper_draft/aaai27/figures/figure1_canondressgs_teaser.png"
PDF_PATH = REPO_ROOT / "paper_draft/aaai27/figures/figure1_canondressgs_teaser.pdf"


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_teaser_uses_real_reference_images() -> None:
    rows = manifest()["reference_sources"]
    assert [row["garment_id"] for row in rows] == list(GARMENTS)
    assert all(row["role"] == "archived_formal_reference_input" for row in rows)
    assert all("/dataset/rgb/edit/" in row["source_path"] for row in rows)
    assert all("/dataset/masks/target_clothing_mask/" in row["mask_source_path"] for row in rows)


def test_teaser_uses_ours_v2_only() -> None:
    rows = manifest()["edited_results"]
    assert len(rows) == 6
    assert {row["method"] for row in rows} == {"Ours-v2"}
    assert {row["formal_run_id"] for row in rows} == {"P0-FORMAL-OURS-V2-R0"}


def test_teaser_does_not_use_teacher_b6_b7() -> None:
    source_paths = [row["source_path"].lower() for row in manifest()["edited_results"]]
    assert all("teacher" not in path for path in source_paths)
    assert all("/b6/" not in path and "/b7/" not in path for path in source_paths)


def test_teaser_does_not_use_interpolation() -> None:
    data = manifest()
    paths = [row["source_path"].lower() for row in data["edited_results"]]
    assert all("interpolation" not in path for path in paths)
    assert data["checks"]["interpolation_source_count"] == 0
    assert data["checks"]["mixed_reference_source_count"] == 0


def test_teaser_reference_target_do_not_overlap() -> None:
    data = manifest()
    references = {row["condition_id"] for row in data["reference_sources"]}
    targets = {row["condition_id"] for row in data["edited_results"]}
    assert references.isdisjoint(targets)
    assert data["checks"]["reference_target_overlap"] is False


def test_teaser_pose_ids_are_shared_across_garments() -> None:
    rows = manifest()["edited_results"]
    by_garment = {
        garment: {row["condition_id"] for row in rows if row["garment_id"] == garment}
        for garment in GARMENTS
    }
    assert len({tuple(sorted(ids)) for ids in by_garment.values()}) == 1


def test_teaser_pose_hashes_are_different() -> None:
    poses = manifest()["pose_conditions"]
    assert poses["Pose A"]["pose_sha256"] != poses["Pose B"]["pose_sha256"]


def test_teaser_source_hashes_are_recorded() -> None:
    data = manifest()
    rows = data["reference_sources"] + data["edited_results"] + [data["base_avatar"]]
    assert all(len(row["source_sha256"]) == 64 for row in rows)
    for row in rows:
        assert sha256_file(REPO_ROOT / row["cached_path"]) == row["source_sha256"]


def test_teaser_builder_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    build(REPO_ROOT, output / "figure.png", output / "figure.pdf", output / "manifest.json")
    first_hashes = {
        name: sha256_file(output / name)
        for name in ("figure.png", "figure.pdf", "manifest.json")
    }
    build(REPO_ROOT, output / "figure.png", output / "figure.pdf", output / "manifest.json")
    for name in ("figure.png", "figure.pdf", "manifest.json"):
        assert sha256_file(output / name) == first_hashes[name]


def test_teaser_png_and_pdf_exist() -> None:
    assert PNG_PATH.is_file() and PDF_PATH.is_file()
    with Image.open(PNG_PATH) as image:
        assert image.size[0] >= 3000
        assert image.size == (3600, 1714)
    assert len(PdfReader(str(PDF_PATH)).pages) == 1


def test_formal_assets_are_unchanged(tmp_path: Path) -> None:
    data = manifest()
    cached = [REPO_ROOT / row["cached_path"] for row in data["reference_sources"]]
    cached += [REPO_ROOT / row["mask_cached_path"] for row in data["reference_sources"]]
    cached += [REPO_ROOT / row["cached_path"] for row in data["edited_results"]]
    cached += [REPO_ROOT / BASE_SPEC["cached_path"]]
    before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in cached}
    build(REPO_ROOT, tmp_path / "figure.png", tmp_path / "figure.pdf", tmp_path / "manifest.json")
    after = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in cached}
    assert before == after
    assert data["checks"]["formal_asset_mutation_count"] == 0


def test_no_paper_final_is_created() -> None:
    data = manifest()
    assert data["paper_final"] is False
    assert data["status"] == "MANUAL_REVIEW_REQUIRED"
    assert not list((REPO_ROOT / "paper_draft/aaai27").rglob("*PAPER_FINAL*"))


def main() -> None:
    no_arg_tests = [
        test_teaser_uses_real_reference_images,
        test_teaser_uses_ours_v2_only,
        test_teaser_does_not_use_teacher_b6_b7,
        test_teaser_does_not_use_interpolation,
        test_teaser_reference_target_do_not_overlap,
        test_teaser_pose_ids_are_shared_across_garments,
        test_teaser_pose_hashes_are_different,
        test_teaser_source_hashes_are_recorded,
        test_teaser_png_and_pdf_exist,
        test_no_paper_final_is_created,
    ]
    passed = 0
    for test in no_arg_tests:
        test()
        passed += 1
        print(f"PASS {test.__name__}")
    with tempfile.TemporaryDirectory(prefix="canondressgs_teaser_tests_") as tmp:
        tmp_path = Path(tmp)
        deterministic_tmp = tmp_path / "deterministic"
        deterministic_tmp.mkdir()
        test_teaser_builder_is_deterministic(deterministic_tmp)
        passed += 1
        print("PASS test_teaser_builder_is_deterministic")
        formal_tmp = tmp_path / "formal_unchanged"
        formal_tmp.mkdir()
        test_formal_assets_are_unchanged(formal_tmp)
        passed += 1
        print("PASS test_formal_assets_are_unchanged")
    print(f"{passed} passed")


if __name__ == "__main__":
    main()
