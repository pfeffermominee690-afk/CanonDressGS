from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "paper_protocol/datasets"
GARMENTS = ["O01", "O03", "O04"]
SLOTS = [
    "front",
    "front_left_three_quarter",
    "front_right_three_quarter",
    "left",
    "right",
    "back_left_three_quarter",
    "back_right_three_quarter",
    "back",
]
OUTPUTS = [
    "subject00_three_garment_canary_protocol.yaml",
    "subject00_three_garment_condition_manifest.json",
    "subject00_three_garment_generation_contract.json",
    "subject00_three_garment_screening_contract.json",
    "subject00_three_garment_teacher_contract.json",
    "subject00_three_garment_success_gates.json",
    "subject00_three_garment_protocol_final_summary.json",
]
REPORTS = [
    "SUBJECT00_THREE_GARMENT_CANARY_PROTOCOL_20260724.md",
    "SUBJECT00_THREE_GARMENT_GEOMETRY_CHALLENGES_20260724.md",
    "SUBJECT00_THREE_GARMENT_SCREENING_PROTOCOL_20260724.md",
    "SUBJECT00_THREE_GARMENT_TEACHER_CONTRACT_20260724.md",
]


def load_json(name: str) -> dict:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def load_module(relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Subject00ThreeGarmentCanaryProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wardrobe = load_json("shared_five_garment_wardrobe.json")
        cls.protocol = load_json(OUTPUTS[0])
        cls.manifest = load_json(OUTPUTS[1])
        cls.generation = load_json(OUTPUTS[2])
        cls.screening = load_json(OUTPUTS[3])
        cls.teacher = load_json(OUTPUTS[4])
        cls.gates = load_json(OUTPUTS[5])
        cls.summary = load_json(OUTPUTS[6])

    def test_required_outputs_and_reports_exist(self) -> None:
        for name in OUTPUTS:
            self.assertTrue((DATASETS / name).is_file(), name)
        for name in REPORTS:
            path = ROOT / "docs/DATASET" / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(len(path.read_text(encoding="utf-8").strip()), 300)

    def test_only_o01_o03_o04_are_selected_without_semantic_changes(self) -> None:
        self.assertEqual(self.summary["selected_garments"], GARMENTS)
        source = {row["garment_id"]: row for row in self.wardrobe["garments"]}
        self.assertEqual(
            self.summary["selected_garment_semantics"],
            [source[garment_id] for garment_id in GARMENTS],
        )
        self.assertEqual(self.wardrobe["garment_ids"], ["O01", "O02", "O03", "O04", "O08"])

    def test_reference_endpoint_and_pending_condition_counts_are_exact(self) -> None:
        self.assertEqual(self.protocol["semantic_slots"], SLOTS)
        self.assertEqual(self.protocol["reference_slots"], ["front", "left", "back", "right"])
        self.assertEqual(len(self.manifest["records"]), 24)
        self.assertEqual(sum(row["reference_role"] for row in self.manifest["records"]), 12)
        self.assertTrue(self.manifest["reference_is_endpoint_subset"])
        self.assertEqual(self.manifest["materialized_condition_record_count"], 0)
        self.assertTrue(all(row["materialization_status"] == "PENDING_FORMAL_BASE" for row in self.manifest["records"]))

    def test_mapping_is_deterministic_strict_train_only_and_quality_blind(self) -> None:
        mapping = self.manifest["pose_camera_mapping"]
        boundary = mapping["strict_split_boundary"]
        self.assertEqual(boundary["condition_and_teacher_target_pose_semantics"], "STRICT_TRAIN_ONLY")
        self.assertEqual(boundary["condition_and_teacher_target_camera_semantics"], "STRICT_TRAIN_ONLY")
        self.assertFalse(boundary["strict_heldout_pose_or_camera_as_teacher_target"])
        self.assertFalse(mapping["global_pose_selection_rule"]["future_generation_quality_used"])
        self.assertFalse(mapping["per_slot_camera_selection_rule"]["manual_best_looking_frame_selection"])
        for row in mapping["mappings"]:
            self.assertEqual(row["camera_candidate_set"]["heldout_camera_ids_forbidden"], [0, 4, 8, 12, 16, 20])

    def test_formal_and_generation_dependencies_remain_pending(self) -> None:
        self.assertFalse(self.protocol["formal_base"]["checkpoint_available"])
        self.assertEqual(self.protocol["formal_base"]["medium_use"], "PREVIEW_ONLY")
        self.assertFalse(self.protocol["formal_base"]["medium_may_be_promoted_to_formal"])
        self.assertEqual(self.generation["status"], "GENERATION_BACKEND_DEPENDENCY_PENDING")
        self.assertFalse(self.generation["medium_preview_as_formal_allowed"])
        self.assertTrue(self.generation["record_policy"]["append_only"])

    def test_screening_requires_automated_and_two_independent_humans(self) -> None:
        policy = self.screening["decision_policy"]
        self.assertTrue(policy["automated_checks_must_pass"])
        self.assertGreaterEqual(policy["minimum_independent_human_reviewers"], 2)
        self.assertTrue(policy["reviewers_work_independently"])
        self.assertFalse(policy["vlm_only_acceptance_allowed"])
        self.assertFalse(policy["automatic_acceptance_allowed"])
        self.assertEqual(set(self.screening["garment_specific_geometry_metrics"]), set(GARMENTS))

    def test_teacher_contract_is_1200_step_cached_surface_attachment(self) -> None:
        self.assertEqual(self.teacher["optimizer"]["optimizer_steps"], 1200)
        self.assertEqual(self.teacher["checkpoint_cadence"][-1], 1200)
        self.assertEqual(self.teacher["base_contract"]["template_lbs_mode"], "surface_attachment_cached")
        self.assertTrue(all(value == 0 for value in self.teacher["topology_and_lbs_invariants"].values()))
        self.assertEqual(self.teacher["teacher_training_runs_in_this_task"], 0)
        self.assertEqual(self.teacher["optimizer_created_in_this_task"], 0)

    def test_all_execution_and_mutation_counts_are_zero(self) -> None:
        self.assertTrue(all(value == 0 for value in self.summary["counts"].values()))
        self.assertTrue(all(value == 0 for value in self.summary["immutability"].values()))
        root = ROOT / "multi_garment_benchmark/subject00"
        media = {".png", ".jpg", ".jpeg", ".pth", ".pt", ".ckpt", ".npz"}
        self.assertFalse(any(path.suffix.lower() in media for path in root.rglob("*") if path.is_file()))
        self.assertEqual(len([path for path in (root / "generated_candidates").rglob("*") if path.is_file() and path.name != ".gitkeep"]), 0)
        self.assertEqual(len([path for path in (root / "accepted_endpoints").rglob("*") if path.is_file() and path.name != ".gitkeep"]), 0)

    def test_claim_classification_and_next_task_are_exact(self) -> None:
        self.assertEqual(
            self.summary["classification"],
            "SUBJECT00_THREE_GARMENT_PROTOCOL_READY_PENDING_FORMAL_BASE",
        )
        self.assertEqual(
            self.summary["next_task"],
            "RUN_SUBJECT00_FORMAL_BASE_THEN_FREEZE_THREE_GARMENT_CONDITIONS",
        )
        self.assertFalse(self.summary["next_task_started"])
        self.assertEqual(self.summary["paper_final"], 0)

    def test_adapter_is_deterministic_and_has_no_execution_client(self) -> None:
        module = load_module("tools/datasets/subject00_three_garment_generation_record_adapter.py")
        record = {
            "record_id": "subject00-O01-front-0",
            "attempt_index": 0,
            "identity_id": "subject00",
            "garment_id": "O01",
            "condition_id": "subject00/O01/front",
            "semantic_pose_slot": "front",
            "source_rgb_sha256": "a" * 64,
            "source_alpha_sha256": "b" * 64,
            "prompt_sha256": "c" * 64,
            "negative_prompt_sha256": "d" * 64,
            "provider": "frozen-provider",
            "model_id": "frozen-model-revision",
            "base_url_identifier": "frozen-non-secret-id",
            "quality": "frozen-quality",
            "image_size": {"width": 1024, "height": 1536},
            "retry_parent_record_id": "NONE_FOR_FIRST_ATTEMPT",
        }
        normalized = module.validate_and_normalize(record)
        self.assertEqual(normalized["seed"], module.derive_seed("O01", "front", 0))
        self.assertFalse(normalized["request_submitted"])
        source = (ROOT / "tools/datasets/subject00_three_garment_generation_record_adapter.py").read_text(encoding="utf-8")
        for token in ("requests", "urllib", "openai", "torch", "optimizer"):
            self.assertNotIn(token, source)

    def test_content_hashes_are_valid(self) -> None:
        payloads = [
            self.protocol, self.manifest, self.generation, self.screening,
            self.teacher, self.gates, self.summary,
        ]
        for payload in payloads:
            expected = payload.pop("content_sha256")
            try:
                encoded = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), expected)
            finally:
                payload["content_sha256"] = expected


if __name__ == "__main__":
    unittest.main()
