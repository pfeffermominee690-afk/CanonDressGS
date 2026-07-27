from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
ATTEMPT_ROOT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001"
    r"\attempt_001_subject00_24_cell_person_garment_masks"
)
UPLOAD_ROOT = ATTEMPT_ROOT / "07_review_assets" / "upload_pack_20260726"
INDEX_PATH = UPLOAD_ROOT / "05_indexes" / "subject00_mask_upload_pack_index.json"
README_PATH = UPLOAD_ROOT / "05_indexes" / "SUBJECT00_MASK_UPLOAD_PACK_README.md"
MANIFEST_PATH = (
    RISK / "subject00_24_cell_mask_human_review_upload_manifest_20260726.json"
)
SUMMARY_PATH = (
    RISK / "subject00_24_cell_mask_human_review_pack_final_summary_20260726.json"
)
TESTS_PATH = RISK / "subject00_24_cell_mask_human_review_pack_tests_20260726.json"
IMMUTABILITY_PATH = (
    RISK / "subject00_24_cell_mask_human_review_pack_immutability_20260726.json"
)
CLASSIFICATION = (
    "SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_PACK_READY_FOR_USER_REVIEW"
)
NEXT_TASK = "USER_UPLOAD_AND_REVIEW_SUBJECT00_MASK_REVIEW_PAGES"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Subject00MaskHumanReviewPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = load(INDEX_PATH)
        cls.manifest = load(MANIFEST_PATH)
        cls.summary = load(SUMMARY_PATH)

    def test_eight_pages_parse_resolution_bytes_and_sha(self) -> None:
        self.assertEqual(self.index["review_page_count"], 8)
        self.assertEqual(len(self.index["pages"]), 8)
        for page in self.index["pages"]:
            path = Path(page["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, page["bytes"])
            self.assertEqual(sha256(path), page["sha256"])
            with Image.open(path) as opened:
                opened.verify()
            with Image.open(path) as opened:
                self.assertEqual(opened.mode, "RGB")
                self.assertEqual(
                    [opened.width, opened.height], page["resolution_wh"]
                )
                self.assertGreaterEqual(opened.width, 4800)
                self.assertGreaterEqual(opened.height, 3200)

    def test_exact_24_cell_six_column_detail_coverage(self) -> None:
        pages = self.index["pages"]
        self.assertEqual(len(pages[0]["request_ids"]), 24)
        detail_ids = [
            request_id
            for page in pages[1:7]
            for request_id in page["request_ids"]
        ]
        self.assertEqual(len(detail_ids), 24)
        self.assertEqual(len(set(detail_ids)), 24)
        manifest_ids = [row["request_id"] for row in self.manifest["records"]]
        self.assertEqual(set(detail_ids), set(manifest_ids))
        self.assertEqual(len(self.index["layout"]["detail_columns"]), 6)
        self.assertEqual(self.index["layout"]["cells_per_detail_page"], 4)
        self.assertEqual(
            [row["garment"] for row in self.manifest["records"]].count("O01"), 8
        )
        self.assertEqual(
            [row["garment"] for row in self.manifest["records"]].count("O03"), 8
        )
        self.assertEqual(
            [row["garment"] for row in self.manifest["records"]].count("O04"), 8
        )

    def test_input_assets_qa_limitations_and_overrides(self) -> None:
        self.assertEqual(self.manifest["person_mask_count"], 24)
        self.assertEqual(self.manifest["garment_mask_count"], 24)
        self.assertEqual(self.manifest["mask_qa_pass_count"], 48)
        for row in self.manifest["records"]:
            self.assertTrue(Path(row["raw"]["path"]).is_file())
            self.assertTrue(Path(row["person_mask"]["path"]).is_file())
            self.assertTrue(Path(row["garment_mask"]["path"]).is_file())
            self.assertTrue(row["person_qa"]["technical_status"].startswith("PASS"))
            self.assertTrue(row["garment_qa"]["technical_status"].startswith("PASS"))
            self.assertTrue(row["pair_qa"]["technical_status"].startswith("PASS"))
        risk = self.index["risk_page_mapping"]
        self.assertEqual(len(risk["limitation_request_ids"]), 9)
        self.assertEqual(len(risk["registration_override_request_ids"]), 2)
        self.assertEqual(len(risk["priority_request_ids"]), 2)

    def test_human_teacher_and_inference_boundaries(self) -> None:
        self.assertEqual(self.manifest["display_role"], "DISPLAY_ONLY_MASK_REVIEW_LAYOUT")
        self.assertTrue(
            all(
                row["person_mask_human_decision"] is None
                and row["garment_mask_human_decision"] is None
                and row["mask_pair_human_decision"] is None
                and not row["mask_accepted"]
                and not row["teacher_target"]
                for row in self.manifest["records"]
            )
        )
        self.assertEqual(self.manifest["mask_accepted_count"], 0)
        self.assertEqual(self.manifest["teacher_target_count"], 0)
        self.assertEqual(self.manifest["segmentation_inference_calls"], 0)
        self.assertEqual(self.manifest["model_forward_calls"], 0)
        script = (
            ROOT
            / "tools"
            / "datasets"
            / "prepare_subject00_24_cell_mask_human_review_upload_pack.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("import torch", script)
        self.assertNotIn("from transformers", script)
        self.assertNotIn("model(", script)

    def test_raw_mask_attempt_immutability(self) -> None:
        immutable = load(IMMUTABILITY_PATH)
        self.assertEqual(immutable["status"], "PASS_NO_MUTATIONS")
        self.assertEqual(immutable["mutation_count"], 0)
        self.assertEqual(immutable["person_mask_mutations"], 0)
        self.assertEqual(immutable["garment_mask_mutations"], 0)
        self.assertEqual(immutable["accepted_raw_mutations"], 0)
        self.assertTrue(
            all(value == 0 for value in immutable["attempt_mutations"].values())
        )

    def test_indexes_reports_tests_classification_and_next_task(self) -> None:
        required = [
            INDEX_PATH,
            README_PATH,
            MANIFEST_PATH,
            SUMMARY_PATH,
            TESTS_PATH,
            IMMUTABILITY_PATH,
            RISK / "subject00_mask_upload_pack_index_20260726.json",
            RISK / "SUBJECT00_MASK_UPLOAD_PACK_README_20260726.md",
            RISK / "SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_PACK_REPORT_20260726.md",
            ROOT
            / "project_control_handoff"
            / "subject00_24_cell_mask_human_review_pack_handoff_20260726.json",
            ROOT
            / "docs"
            / "PAPER"
            / "AAAI27_SUBJECT00_24_CELL_MASK_HUMAN_REVIEW_PACK_REPORT_20260726.md",
        ]
        self.assertTrue(all(path.is_file() for path in required))
        tests = load(TESTS_PATH)
        self.assertEqual(tests["status"], "PASS")
        self.assertEqual(tests["pass_count"], tests["test_count"])
        self.assertEqual(tests["fail_count"], 0)
        self.assertEqual(self.summary["final_classification"], CLASSIFICATION)
        self.assertEqual(self.summary["next_task"], NEXT_TASK)
        self.assertEqual(self.summary["paper_modifications"], 0)
        self.assertFalse(self.summary["paper_final"])


if __name__ == "__main__":
    unittest.main()
