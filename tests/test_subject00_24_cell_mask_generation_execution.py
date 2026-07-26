from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
MANIFEST = RISK / "subject00_24_cell_mask_generation_execution_manifest_20260726.json"
ATTEMPT = Path(
    r"E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-MASKS-001"
    r"\attempt_001_subject00_24_cell_person_garment_masks"
)
EXECUTOR = (
    ROOT / "tools" / "datasets" / "execute_subject00_24_cell_masks_from_frozen_contract.py"
)
EXECUTOR_SHA = "b602df821f30046941bd98956d5afc552f36c7c72f1fe4af5c58621713a36693"
CLASSIFICATION = (
    "SUBJECT00_24_CELL_MASK_GENERATION_TECHNICAL_PASS_PENDING_HUMAN_REVIEW"
)
NEXT_TASK = "USER_REVIEW_SUBJECT00_24_CELL_GENERATED_MASKS"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Subject00MaskGenerationExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load(MANIFEST)
        cls.summary = load(
            RISK
            / "subject00_24_cell_mask_generation_execution_final_summary_20260726.json"
        )
        cls.person = load(
            RISK / "subject00_24_cell_person_mask_qa_results_20260726.json"
        )
        cls.garment = load(
            RISK / "subject00_24_cell_garment_mask_qa_results_20260726.json"
        )
        cls.pair = load(
            RISK / "subject00_24_cell_mask_pair_qa_results_20260726.json"
        )

    def test_frozen_executor_state_and_exact_formal_paths(self) -> None:
        self.assertEqual(sha256(EXECUTOR), EXECUTOR_SHA)
        state = load(ATTEMPT / "08_logs" / "execution_state.json")
        self.assertEqual(list(state["records"]), self.manifest["request_order"])
        self.assertEqual(len(state["records"]), 24)
        self.assertTrue(
            all(
                row["status"] == "MASK_PAIR_GENERATED_QA_PENDING_HUMAN"
                and row["retry_count"] == 0
                for row in state["records"].values()
            )
        )
        expected_person = {
            str(Path(row["person_mask_output_path"]).resolve())
            for row in self.manifest["records"]
        }
        expected_garment = {
            str(Path(row["garment_mask_output_path"]).resolve())
            for row in self.manifest["records"]
        }
        actual_person = {
            str(path.resolve())
            for path in (ATTEMPT / "03_person_masks").rglob("*.png")
        }
        actual_garment = {
            str(path.resolve())
            for path in (ATTEMPT / "04_garment_masks").rglob("*.png")
        }
        self.assertEqual(actual_person, expected_person)
        self.assertEqual(actual_garment, expected_garment)

    def test_all_48_masks_are_native_l_binary_and_pairs_are_subsets(self) -> None:
        for record in self.manifest["records"]:
            raw_path = Path(record["accepted_raw_path"])
            person_path = Path(record["person_mask_output_path"])
            garment_path = Path(record["garment_mask_output_path"])
            with Image.open(raw_path) as opened:
                raw_size = opened.size
            with Image.open(person_path) as opened:
                self.assertEqual(opened.mode, "L")
                self.assertEqual(opened.size, raw_size)
                person = np.asarray(opened)
            with Image.open(garment_path) as opened:
                self.assertEqual(opened.mode, "L")
                self.assertEqual(opened.size, raw_size)
                garment = np.asarray(opened)
            self.assertEqual(sorted(np.unique(person).tolist()), [0, 255])
            self.assertEqual(sorted(np.unique(garment).tolist()), [0, 255])
            self.assertGreater(np.count_nonzero(person), 0)
            self.assertGreater(np.count_nonzero(garment), 0)
            self.assertEqual(np.count_nonzero((garment > 0) & (person == 0)), 0)
            self.assertGreater(np.count_nonzero((person > 0) & (garment == 0)), 0)

    def test_technical_qa_and_cross_view_pass(self) -> None:
        self.assertEqual((self.person["pass_count"], self.person["fail_count"]), (24, 0))
        self.assertEqual((self.garment["pass_count"], self.garment["fail_count"]), (24, 0))
        self.assertEqual((self.pair["pass_count"], self.pair["fail_count"]), (24, 0))
        self.assertEqual(self.pair["garment_outside_person_pixel_total"], 0)
        self.assertEqual(self.pair["protected_region_nonempty_count"], 24)
        cross = load(
            RISK / "subject00_24_cell_mask_cross_view_qa_20260726.json"
        )
        self.assertTrue(cross["automatic_status"].startswith("PASS"))
        self.assertTrue(
            all(
                group["automatic_status"].startswith("PASS")
                and group["record_count"] == 8
                for group in cross["garments"].values()
            )
        )

    def test_review_package_and_human_boundary(self) -> None:
        review = load(
            RISK / "subject00_24_cell_mask_human_review_manifest_20260726.json"
        )
        self.assertEqual(review["display_role"], "DISPLAY_ONLY_MASK_REVIEW_LAYOUT")
        self.assertEqual(review["contract_required_artifact_count"], 90)
        self.assertEqual(review["actual_materialized_artifact_count"], 92)
        self.assertEqual(review["limitation_review_page_count"], 9)
        self.assertEqual(review["registration_override_review_page_count"], 2)
        self.assertTrue(all(Path(row["path"]).is_file() for row in review["artifacts"]))
        self.assertIsNone(review["person_mask_human_decision"])
        self.assertIsNone(review["garment_mask_human_decision"])
        self.assertIsNone(review["mask_pair_human_decision"])
        self.assertEqual(review["mask_accepted_count"], 0)

    def test_legacy_immutability_and_teacher_boundary(self) -> None:
        immutable = load(
            ATTEMPT
            / "02_input_bindings"
            / "post_execution_immutability_registry.json"
        )
        self.assertEqual(immutable["status"], "PASS_NO_MUTATIONS")
        self.assertEqual(immutable["checked_file_count"], 320)
        self.assertEqual(immutable["mutation_count"], 0)
        self.assertEqual(immutable["accepted_raw_mutations"], 0)
        self.assertTrue(all(value == 0 for value in immutable["attempt_mutations"].values()))
        teacher = load(
            RISK
            / "subject00_24_cell_mask_teacher_compatibility_registry_20260726.json"
        )
        self.assertEqual(teacher["teacher_target_count"], 0)
        self.assertFalse(teacher["teacher_target_root_created"])
        self.assertFalse(teacher["teacher_target_creation_authorized"])
        self.assertFalse(teacher["teacher_endpoint_optimization_authorized"])

    def test_reports_structured_checks_classification_and_next_task(self) -> None:
        required = [
            RISK / "subject00_24_cell_mask_generation_execution_registry_20260726.json",
            RISK / "subject00_24_cell_person_mask_qa_results_20260726.json",
            RISK / "subject00_24_cell_garment_mask_qa_results_20260726.json",
            RISK / "subject00_24_cell_mask_pair_qa_results_20260726.json",
            RISK / "subject00_24_cell_mask_cross_view_qa_20260726.json",
            RISK / "subject00_24_cell_mask_human_review_manifest_20260726.json",
            RISK / "subject00_24_cell_mask_teacher_compatibility_registry_20260726.json",
            RISK / "subject00_24_cell_mask_generation_execution_tests_20260726.json",
            RISK / "subject00_24_cell_mask_generation_execution_final_summary_20260726.json",
            RISK / "SUBJECT00_24_CELL_MASK_GENERATION_EXECUTION_REPORT_20260726.md",
            ROOT
            / "project_control_handoff"
            / "subject00_24_cell_mask_generation_execution_handoff_20260726.json",
            ROOT
            / "docs"
            / "PAPER"
            / "AAAI27_SUBJECT00_24_CELL_MASK_GENERATION_EXECUTION_REPORT_20260726.md",
        ]
        self.assertTrue(all(path.is_file() for path in required))
        tests = load(
            RISK / "subject00_24_cell_mask_generation_execution_tests_20260726.json"
        )
        self.assertEqual(tests["status"], "PASS")
        self.assertEqual(tests["pass_count"], tests["test_count"])
        self.assertEqual(tests["fail_count"], 0)
        self.assertEqual(self.summary["final_classification"], CLASSIFICATION)
        self.assertEqual(self.summary["next_task"], NEXT_TASK)
        self.assertEqual(self.summary["model_download_bytes"], 0)
        self.assertEqual(self.summary["environment_install_calls"], 0)
        self.assertEqual(self.summary["automatic_retry_calls"], 0)
        self.assertEqual(self.summary["paper_modifications"], 0)
        self.assertFalse(self.summary["paper_final"])


if __name__ == "__main__":
    unittest.main()
