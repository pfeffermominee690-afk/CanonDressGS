from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "paper_protocol" / "datasets"
GARMENT_IDS = ["O01", "O02", "O03", "O04", "O08"]
SKELETON_DIRS = {
    "raw_references",
    "identity_manifest",
    "condition_manifests",
    "condition_previews",
    "generation_contract",
    "generated_candidates",
    "screening",
    "accepted_endpoints",
    "teacher_inputs",
    "endpoint_bank",
    "reports",
    "provenance",
}
JSON_OUTPUTS = [
    "shared_five_garment_wardrobe.json",
    "shared_condition_pose_view_protocol.json",
    "subject00_garment_preparation_manifest.json",
    "avatarrex_zero_copy_adapter_contract.json",
    "avatarrex_camera_pose_split.json",
    "multi_identity_generation_provenance_schema.json",
    "multi_identity_screening_contract.json",
    "multi_identity_benchmark_final_summary.json",
]
REPORTS = [
    "MULTI_IDENTITY_GARMENT_BENCHMARK_PLAN_20260724.md",
    "SUBJECT00_GARMENT_BANK_PREPARATION_20260724.md",
    "AVATARREX_ZERO_COPY_STANDARDIZATION_20260724.md",
    "CROSS_IDENTITY_WARDROBE_FAIRNESS_20260724.md",
    "SYNTHETIC_GARMENT_DATA_CLAIM_BOUNDARY_20260724.md",
]


def load_json(name: str) -> dict:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultiIdentityFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wardrobe = load_json("shared_five_garment_wardrobe.json")
        cls.condition = load_json("shared_condition_pose_view_protocol.json")
        cls.subject00 = load_json("subject00_garment_preparation_manifest.json")
        cls.adapter = load_json("avatarrex_zero_copy_adapter_contract.json")
        cls.split = load_json("avatarrex_camera_pose_split.json")
        cls.provenance = load_json("multi_identity_generation_provenance_schema.json")
        cls.screening = load_json("multi_identity_screening_contract.json")
        cls.summary = load_json("multi_identity_benchmark_final_summary.json")
        cls.handoff = json.loads(
            (ROOT / "project_control_handoff/multi_identity_garment_foundations_handoff.json").read_text(
                encoding="utf-8"
            )
        )

    def test_required_json_and_reports_exist(self) -> None:
        for name in JSON_OUTPUTS:
            self.assertTrue((DATASETS / name).is_file(), name)
        for name in REPORTS:
            path = ROOT / "docs" / "DATASET" / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(len(path.read_text(encoding="utf-8").strip()), 200, name)

    def test_wardrobe_is_exactly_five_frozen_garments(self) -> None:
        self.assertEqual(self.wardrobe["garment_ids"], GARMENT_IDS)
        self.assertEqual(self.wardrobe["garment_count"], 5)
        self.assertEqual([row["garment_id"] for row in self.wardrobe["garments"]], GARMENT_IDS)
        required = {
            "semantic_description",
            "upper_garment_type",
            "lower_garment_type",
            "colors",
            "material_appearance",
            "fit",
            "sleeve_length",
            "hem",
            "forbidden_attributes",
            "reference_prompt",
            "negative_prompt",
            "historical_generation",
            "screening_rule_set",
        }
        for row in self.wardrobe["garments"]:
            self.assertTrue(required.issubset(row), row["garment_id"])
            self.assertEqual(
                row["historical_generation"]["provenance_completeness"],
                "LIMITED_GARMENT_GENERATION_PROVENANCE",
            )
            self.assertIsNone(row["historical_generation"]["model_id"])
            self.assertIsNone(row["historical_generation"]["seed"])

    def test_shared_slot_reference_endpoint_and_teacher_budgets(self) -> None:
        self.assertEqual(self.condition["condition_slot_count"], 8)
        self.assertEqual(self.condition["reference_slot_count_per_garment"], 4)
        self.assertEqual(self.condition["endpoint_count_per_garment"], 8)
        shared = self.summary["fairness_contract"]["shared"]
        self.assertEqual(shared["condition_slot_count"], 8)
        self.assertEqual(shared["reference_count_per_garment"], 4)
        self.assertEqual(shared["endpoint_count_per_garment"], 8)
        self.assertEqual(shared["endpoint_teacher_optimizer_steps_per_garment"], 1200)
        self.assertEqual(self.subject00["teacher_input_contract"]["accepted_endpoints_required"], 40)

    def test_semantic_slots_replace_cross_identity_frame_ids(self) -> None:
        self.assertEqual(self.condition["unit"], "SEMANTIC_POSE_SLOT")
        self.assertFalse(
            self.condition["deterministic_identity_matching"]["same_raw_frame_id_across_identities_required"]
        )
        self.assertEqual(len({row["semantic_pose_slot"] for row in self.condition["slots"]}), 8)

    def test_generation_schema_is_complete_and_append_only(self) -> None:
        required = set(self.provenance["required"])
        expected = {
            "identity_id",
            "garment_id",
            "condition_id",
            "semantic_pose_slot",
            "source_image_sha256",
            "source_mask_sha256",
            "prompt_sha256",
            "negative_prompt_sha256",
            "provider",
            "model_id",
            "api_base",
            "seed",
            "image_parameters",
            "request_timestamp",
            "response_metadata",
            "output_sha256",
            "retry_reason",
            "moderation_or_error_state",
        }
        self.assertTrue(expected.issubset(required))
        policy = self.provenance["x_append_only_policy"]
        self.assertTrue(policy["failure_records_must_be_retained"])
        self.assertFalse(policy["records_may_be_updated_in_place"])
        self.assertTrue(policy["retry_creates_new_attempt_index"])

    def test_screening_forbids_vlm_only_or_single_score_acceptance(self) -> None:
        policy = self.screening["decision_policy"]
        self.assertFalse(policy["automatic_acceptance_allowed"])
        self.assertFalse(policy["vlm_only_automatic_acceptance"])
        self.assertFalse(policy["single_score_automatic_acceptance"])
        self.assertGreaterEqual(policy["minimum_independent_human_reviewers"], 2)
        self.assertTrue(self.screening["record_policy"]["failed_screening_records_retained"])

    def test_avatarrex_adapter_split_and_smoke_are_strict(self) -> None:
        self.assertEqual(self.adapter["status"], "PASS")
        self.assertEqual(self.adapter["standardization_mode"], "READ_ONLY_ZERO_COPY_LOADER_ADAPTER")
        self.assertEqual(self.adapter["loader_smoke"]["status"], "PASS")
        self.assertEqual(self.adapter["loader_smoke"]["decoded_record_count"], 9)
        self.assertEqual(len(self.adapter["camera_mapping"]), 16)
        self.assertIn("/mask/pha/", self.adapter["mask_resolver"])
        self.assertEqual(self.split["camera_split"]["train_count"], 12)
        self.assertEqual(self.split["camera_split"]["heldout_count"], 4)
        self.assertEqual(self.split["camera_split"]["overlap"], [])
        self.assertEqual(self.split["pose_split"]["heldout_count"], 95)
        self.assertGreaterEqual(self.split["pose_split"]["heldout_pair_min_temporal_distance"], 11)
        self.assertEqual(self.split["pose_split"]["train_heldout_overlap"], [])
        self.assertEqual(self.split["pose_split"]["train_buffer_overlap"], [])
        self.assertEqual(self.split["pose_split"]["temporal_leakage"], 0)

    def test_all_execution_and_copy_counts_are_zero(self) -> None:
        self.assertTrue(all(value == 0 for value in self.summary["execution_counts"].values()))
        self.assertTrue(all(value == 0 for value in self.adapter["counts"].values()))
        self.assertEqual(self.adapter["loader_smoke"]["rgb_copies"], 0)
        self.assertEqual(self.adapter["loader_smoke"]["mask_copies"], 0)
        self.assertEqual(self.adapter["loader_smoke"]["raw_writes"], 0)
        self.assertTrue(all(value == 0 for value in self.subject00["counts"].values()))

    def test_raw_archive_tree_and_metadata_were_reverified_unchanged(self) -> None:
        integrity = self.summary["avatarrex"]["raw_integrity_verification"]
        self.assertEqual(integrity["verification_timing"], "REHASHED_AFTER_ZERO_COPY_LOADER_SMOKE")
        self.assertEqual(integrity["archive_bytes"], 12569755256)
        self.assertEqual(
            integrity["archive_sha256"],
            "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1",
        )
        self.assertEqual(integrity["raw_file_count"], 60834)
        self.assertEqual(integrity["raw_apparent_bytes"], 19135049684)
        self.assertEqual(
            integrity["raw_full_content_fingerprint"],
            "00482b7c98f6f46773fd13a3f33ebe278b9353e09fdb51fa9ed72583f7b27b15",
        )
        self.assertTrue(integrity["all_frozen_values_match"])
        self.assertEqual(integrity["raw_data_mutations"], 0)
        self.assertEqual(self.handoff["raw_integrity_verification"], integrity)

    def test_readiness_and_claim_boundaries_are_exact(self) -> None:
        self.assertEqual(
            self.subject00["readiness"],
            "SUBJECT00_GARMENT_BANK_PREPARATION_READY_PENDING_FORMAL_BASE",
        )
        self.assertFalse(self.subject00["evidence"]["formal_final_checkpoint_exists"])
        self.assertEqual(self.subject00["condition_source_contract"]["medium"], "PREVIEW_ONLY")
        self.assertIn("AVATARREX_ZERO_COPY_STANDARDIZATION_READY", self.summary["per_identity_readiness"]["avatarrex_lbn1"])
        self.assertIn("AVATARREX_BASE_AVATAR_PREPARATION_REQUIRED", self.summary["per_identity_readiness"]["avatarrex_lbn1"])
        self.assertEqual(
            self.summary["overall_classification"],
            "MULTI_IDENTITY_GARMENT_BENCHMARK_FOUNDATIONS_READY",
        )
        forbidden = self.summary["claim_boundary"]["forbidden"]
        self.assertIn("captured multi-outfit dataset", forbidden)
        self.assertIn("true cross-identity garment ground truth", forbidden)
        self.assertIn("AVATARREX MULTI-GARMENT DATASET READY", forbidden)

    def test_directory_skeletons_are_complete_and_data_free(self) -> None:
        forbidden_suffixes = {".png", ".jpg", ".jpeg", ".pth", ".pt", ".ckpt", ".npz"}
        for identity in ("subject00", "avatarrex_lbn1"):
            root = ROOT / "multi_garment_benchmark" / identity
            self.assertEqual({path.name for path in root.iterdir() if path.is_dir()}, SKELETON_DIRS)
            self.assertTrue((root / "README.md").is_file())
            for directory in SKELETON_DIRS:
                self.assertTrue((root / directory / ".gitkeep").is_file())
            self.assertFalse(any(path.suffix.lower() in forbidden_suffixes for path in root.rglob("*")))

    def test_generation_record_adapter_only_prepares_deterministic_records(self) -> None:
        module = load_module(
            "identity_garment_generation_record_adapter",
            "tools/datasets/identity_garment_generation_record_adapter.py",
        )
        seed = module.derive_seed("subject00", "O01", "front", 0)
        self.assertEqual(seed, module.derive_seed("subject00", "O01", "front", 0))
        self.assertNotEqual(seed, module.derive_seed("subject00", "O01", "front", 1))
        record = {
            "record_id": "subject00-O01-front-0",
            "attempt_index": 0,
            "identity_id": "subject00",
            "garment_id": "O01",
            "condition_id": "future-formal-condition",
            "semantic_pose_slot": "front",
            "source_image_sha256": "a" * 64,
            "source_mask_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "negative_prompt_sha256": "d" * 64,
            "provider": "pinned-provider",
            "model_id": "pinned-model-revision",
            "api_base": "pinned-api-base",
            "image_parameters": {"width": 1024, "height": 1536},
        }
        normalized = module.validate_and_normalize(record)
        self.assertEqual(normalized["event_type"], "REQUEST_PREPARED")
        self.assertIsNone(normalized["request_timestamp"])
        self.assertIsNone(normalized["output_sha256"])

    def test_builder_and_record_adapter_have_no_execution_clients(self) -> None:
        sources = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "tools/datasets/build_multi_identity_garment_foundations.py",
                "tools/datasets/identity_garment_generation_record_adapter.py",
            )
        )
        forbidden = ("requests.post", "urllib.request", "openai", "optimizer.step", "torch.", "shutil.copy")
        self.assertFalse(any(token in sources for token in forbidden))

    def test_sealed_artifact_hashes_are_valid(self) -> None:
        for payload in (
            self.wardrobe,
            self.condition,
            self.subject00,
            self.screening,
            self.summary,
            self.handoff,
        ):
            expected = payload.pop("content_sha256")
            try:
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), expected)
            finally:
                payload["content_sha256"] = expected

    def test_no_paper_final_artifact_exists(self) -> None:
        self.assertEqual(self.summary["paper_final"], 0)
        self.assertEqual(self.handoff["paper_final"], 0)
        self.assertFalse(any("PAPER_FINAL" in path.name for path in ROOT.rglob("*")))
        self.assertFalse(self.handoff["control_center_modified"])


if __name__ == "__main__":
    unittest.main()
