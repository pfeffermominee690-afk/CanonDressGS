from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.multi_identity_review.review_server import (
    adjudication_contexts,
    candidate_queue_for_reviewer,
    confined_path,
    validate_review_transition,
)
from tools.multi_identity_review.workflow import (
    GROUP_DIMENSIONS,
    REVIEW_DIMENSIONS,
    SLOTS,
    ContractError,
    append_jsonl_event,
    build_dry_run_request,
    candidate_filename,
    evaluate_garment_acceptance,
    resolve_adjudication,
    resolve_initial_decision,
    scan_credential_leaks,
    seal_accepted_manifest,
    seal_candidate_manifest,
    seal_manifest,
    seal_rejected_manifest,
    validate_request,
    validate_request_event_sequence,
    validate_review_event,
    verify_sealed_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "paper_protocol" / "datasets"
CONTRACTS = (
    "multi_identity_generation_backend_contract.json",
    "multi_identity_generation_request_schema.json",
    "multi_identity_generation_seed_map.json",
    "multi_identity_generation_budget.json",
    "multi_identity_retry_contract.json",
    "multi_identity_automatic_screening_contract.json",
    "multi_identity_human_review_contract.json",
    "multi_identity_group_review_contract.json",
    "multi_identity_candidate_acceptance_contract.json",
    "multi_identity_generation_dry_run.json",
    "multi_identity_generation_backend_final_summary.json",
)
REPORTS = (
    "MULTI_IDENTITY_GENERATION_BACKEND_CONTRACT_20260724.md",
    "MULTI_IDENTITY_GENERATION_FAILURE_AND_RETRY_20260724.md",
    "MULTI_IDENTITY_HUMAN_REVIEW_WORKFLOW_20260724.md",
    "MULTI_IDENTITY_CANDIDATE_ACCEPTANCE_20260724.md",
)


def load_json(name: str) -> dict:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def review(decision: str, stage: str = "INITIAL") -> dict:
    return {
        "decision": decision,
        "review_stage": stage,
    }


class MultiIdentityGenerationBackendReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = load_json("multi_identity_generation_backend_contract.json")
        cls.request_schema = load_json("multi_identity_generation_request_schema.json")
        cls.seed = load_json("multi_identity_generation_seed_map.json")
        cls.budget = load_json("multi_identity_generation_budget.json")
        cls.retry = load_json("multi_identity_retry_contract.json")
        cls.screening = load_json("multi_identity_automatic_screening_contract.json")
        cls.human = load_json("multi_identity_human_review_contract.json")
        cls.group = load_json("multi_identity_group_review_contract.json")
        cls.acceptance = load_json("multi_identity_candidate_acceptance_contract.json")
        cls.dry_run = load_json("multi_identity_generation_dry_run.json")
        cls.summary = load_json("multi_identity_generation_backend_final_summary.json")
        cls.handoff = json.loads(
            (ROOT / "project_control_handoff/multi_identity_generation_backend_handoff.json").read_text(
                encoding="utf-8"
            )
        )
        wardrobe = load_json("shared_five_garment_wardrobe.json")
        cls.garments = {row["garment_id"]: row for row in wardrobe["garments"]}

    def test_required_artifacts_parse_and_reports_exist(self) -> None:
        for name in CONTRACTS:
            self.assertIsInstance(load_json(name), dict, name)
        for name in REPORTS:
            path = ROOT / "docs" / "DATASET" / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(len(path.read_text(encoding="utf-8")), 800, name)
        self.assertTrue((ROOT / "tools/multi_identity_review/index.html").is_file())
        self.assertTrue((ROOT / "tools/multi_identity_review/app.js").is_file())
        self.assertTrue((ROOT / "tools/multi_identity_review/styles.css").is_file())

    def test_backend_is_selection_required_without_guessed_wire_values(self) -> None:
        self.assertEqual(self.backend["status"], "GENERATION_BACKEND_SELECTION_REQUIRED")
        prospective = self.backend["prospective_contract"]
        for field in ("provider", "model_id_and_revision", "base_url_identifier", "wire_endpoint", "wire_api_schema"):
            self.assertIsNone(prospective[field])
        self.assertEqual(self.backend["next_task"], "USER_SELECT_GENERATION_PROVIDER_AND_MODEL")
        self.assertEqual(
            self.backend["historical_execution_provenance"][0]["provenance_completeness"],
            "LIMITED_GARMENT_GENERATION_PROVENANCE",
        )

    def test_two_dry_runs_are_deterministic_blocked_and_seedless(self) -> None:
        expected_coordinates = (
            ("subject00", "O01", "front"),
            ("avatarrex_lbn1", "O03", "back"),
        )
        self.assertEqual(len(self.dry_run["requests"]), 2)
        for stored, (identity_id, garment_id, slot) in zip(
            self.dry_run["requests"], expected_coordinates
        ):
            rebuilt = build_dry_run_request(
                identity_id=identity_id,
                garment_id=garment_id,
                semantic_pose_slot=slot,
                candidate_index=0,
                garment_prompt=self.garments[garment_id]["reference_prompt"],
                negative_prompt=self.garments[garment_id]["negative_prompt"],
            )
            self.assertEqual(stored, rebuilt)
            validate_request(stored)
            self.assertNotIn("seed", stored)
            self.assertFalse(stored["network_call_performed"])
            self.assertTrue(all(source["status"] == "FORMAL_BASE_PENDING" for source in stored["source_inputs"].values()))
            self.assertTrue(all(source["sha256"] is None for source in stored["source_inputs"].values()))
            self.assertTrue(stored["paths"]["output_path"].endswith(".png"))

    def test_seed_contract_forbids_fabricated_seed(self) -> None:
        self.assertEqual(self.seed["seed_field_policy"], "OMIT_FROM_REQUEST_UNLESS_PROVIDER_SUPPORT_IS VERIFIED")
        self.assertIsNone(self.seed["seed_map"])
        self.assertFalse(self.seed["bitwise_reproducibility_claim_allowed"])
        invalid = dict(self.dry_run["requests"][0])
        invalid["seed"] = 7
        with self.assertRaises(ContractError):
            validate_request(invalid)

    def test_budget_is_fixed_and_identity_symmetric(self) -> None:
        options = {row["name"]: row for row in self.budget["options"]}
        self.assertEqual(options["MINIMAL"]["valid_candidate_budget"], 80)
        self.assertEqual(options["STANDARD"]["valid_candidate_budget"], 160)
        self.assertEqual(options["ROBUST"]["valid_candidate_budget"], 320)
        self.assertEqual(self.budget["recommended_option"], "STANDARD")
        self.assertTrue(self.budget["fairness_rules"]["same_n_for_both_identities"])
        self.assertFalse(self.budget["fairness_rules"]["failed_request_counts_as_valid_candidate"])

    def test_retry_contract_is_narrow_and_failure_preserving(self) -> None:
        self.assertEqual(self.retry["maximum_retries_after_initial_attempt"], 2)
        self.assertEqual(
            set(self.retry["retryable_error_classes"]),
            {"NETWORK", "TIMEOUT", "HTTP_5XX", "PROVIDER_TRANSIENT_ERROR"},
        )
        self.assertIn("SCIENTIFIC_QUALITY_REJECTION", self.retry["non_retryable_error_classes"])
        self.assertTrue(self.retry["append_only_policy"]["failed_request_record_retained"])
        self.assertTrue(self.retry["append_only_policy"]["hash_chain_required"])

    def test_append_only_event_hash_chain_and_transition_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            first = append_jsonl_event(path, {"request_id": "r", "event_type": "REQUEST_PREPARED"})
            size_after_first = path.stat().st_size
            second = append_jsonl_event(path, {"request_id": "r", "event_type": "REQUEST_FAILED"})
            self.assertGreater(path.stat().st_size, size_after_first)
            self.assertEqual(second["previous_event_hash"], first["event_hash"])
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            validate_request_event_sequence(events)
            with self.assertRaises(ContractError):
                validate_request_event_sequence(events + [{"request_id": "r", "event_type": "REQUEST_SUCCEEDED"}])

    def test_manifest_sealing_and_traceable_filename(self) -> None:
        name = candidate_filename("subject00", "O01", "front", 0, "mir_331bd6c164cec9062c22dd86")
        self.assertIn("subject00__O01__front__candidate00__mir_", name)
        sealed = seal_manifest({"candidate_sha256": "a" * 64, "state": "CANDIDATE_SEALED"})
        self.assertTrue(verify_sealed_manifest(sealed))
        changed = dict(sealed)
        changed["state"] = "ACCEPT"
        self.assertFalse(verify_sealed_manifest(changed))
        request = copy.deepcopy(self.dry_run["requests"][0])
        request["mode"] = "AUTHORIZED_EXECUTION"
        request["status"] = "EXECUTION_AUTHORIZED"
        request["backend"] = {
            "provider": "selected-provider",
            "model_id": "selected-model-revision",
            "base_url_identifier": "selected-base",
            "endpoint": "/selected-image-edit-endpoint",
            "selection_status": "GENERATION_BACKEND_CONTRACT_READY",
        }
        request["reproducibility"]["seed_support"] = "UNSUPPORTED"
        for index, source in enumerate(request["source_inputs"].values(), start=1):
            source["status"] = "READY_AND_HASHED"
            source["sha256"] = f"{index:064x}"
        request["request_parameters"]["wire_size"] = "provider-frozen-size"
        request["request_parameters"]["quality"] = "provider-frozen-quality"
        candidate = seal_candidate_manifest(
            request,
            candidate_path=request["paths"]["output_path"],
            candidate_sha256="b" * 64,
            request_event_hash="c" * 64,
        )
        accepted = seal_accepted_manifest(
            candidate,
            review_resolution_event_hash="d" * 64,
            group_resolution_event_hash="e" * 64,
            group_review_status="PASS",
        )
        rejected = seal_rejected_manifest(
            candidate,
            rejection_event_hash="f" * 64,
            rejection_class="SCIENTIFIC_QUALITY_REJECTION",
        )
        self.assertEqual(accepted["candidate_sha256"], candidate["candidate_sha256"])
        self.assertFalse(accepted["candidate_pixels_copied"])
        self.assertTrue(rejected["candidate_retained"])
        self.assertEqual(accepted["candidate_manifest_sha256"], rejected["candidate_manifest_sha256"])

    def test_automatic_screening_keeps_per_check_evidence(self) -> None:
        names = {row["name"] for row in self.screening["checks"]}
        expected = {
            "file_decodable", "dimensions", "channels", "alpha_validity", "nonempty_image",
            "multi_person", "text_or_watermark", "background_compliance", "body_completeness",
            "hands", "feet", "anatomy", "identity_face_similarity", "garment_semantic_match",
            "view_consistency", "blur", "cross_view_garment_consistency",
        }
        self.assertEqual(names, expected)
        self.assertFalse(self.screening["decision_boundary"]["automatic_acceptance_allowed"])
        self.assertFalse(self.screening["record_schema"]["aggregate_score_may_replace_per_check_records"])

    def test_independent_review_adjudication_and_group_forms(self) -> None:
        self.assertEqual(resolve_initial_decision(review("ACCEPT"), review("ACCEPT")), "ACCEPT")
        self.assertEqual(resolve_initial_decision(review("MAYBE"), review("ACCEPT")), "ADJUDICATION_REQUIRED")
        self.assertEqual(resolve_initial_decision(review("REJECT"), review("ACCEPT")), "REJECT")
        self.assertEqual(
            resolve_adjudication(review("MAYBE"), review("ACCEPT"), review("ACCEPT", "ADJUDICATION")),
            "ACCEPT",
        )
        base = {
            "candidate_id": "candidate",
            "reviewer_id": "REVIEWER_A",
            "review_stage": "INITIAL",
            "decision": "MAYBE",
            "scores": {name: "UNCERTAIN" for name in REVIEW_DIMENSIONS},
            "comment": "",
            "timestamp": "2026-07-24T00:00:00Z",
        }
        validate_review_event(base)
        group = {
            **base,
            "group_id": "subject00__O01",
            "scores": {name: "PASS" for name in GROUP_DIMENSIONS},
        }
        group.pop("candidate_id")
        validate_review_event(group, group=True)

    def test_reviewer_queue_hides_peer_decisions(self) -> None:
        candidates = [{
            "candidate_id": "c1",
            "image_path": "c1.png",
            "reviews": {"REVIEWER_B": "REJECT"},
            "decision_history": ["REJECT"],
        }]
        queue = candidate_queue_for_reviewer(candidates, "REVIEWER_A")
        self.assertEqual(queue["decision_visibility"], "PEER_INITIAL_DECISIONS_HIDDEN")
        self.assertNotIn("reviews", queue["candidates"][0])
        self.assertNotIn("decision_history", queue["candidates"][0])
        events = [
            {"candidate_id": "c1", "reviewer_id": "REVIEWER_A", "review_stage": "INITIAL", "decision": "MAYBE"},
            {"candidate_id": "c1", "reviewer_id": "REVIEWER_B", "review_stage": "INITIAL", "decision": "ACCEPT"},
        ]
        contexts = adjudication_contexts(events, group=False)
        adjudicator_queue = candidate_queue_for_reviewer(
            candidates,
            "ADJUDICATOR",
            adjudication_contexts=contexts,
        )
        self.assertEqual(adjudicator_queue["candidates"][0]["adjudication_context"], {"REVIEWER_A": "MAYBE", "REVIEWER_B": "ACCEPT"})

    def test_duplicate_review_and_premature_adjudication_are_rejected(self) -> None:
        reviewer_a = {
            "candidate_id": "c1",
            "reviewer_id": "REVIEWER_A",
            "review_stage": "INITIAL",
            "decision": "MAYBE",
        }
        reviewer_b = {
            "candidate_id": "c1",
            "reviewer_id": "REVIEWER_B",
            "review_stage": "INITIAL",
            "decision": "ACCEPT",
        }
        adjudicator = {
            "candidate_id": "c1",
            "reviewer_id": "ADJUDICATOR",
            "review_stage": "ADJUDICATION",
            "decision": "ACCEPT",
        }
        with self.assertRaises(ValueError):
            validate_review_transition([reviewer_a], reviewer_a, group=False)
        with self.assertRaises(ValueError):
            validate_review_transition([reviewer_a], adjudicator, group=False)
        validate_review_transition([reviewer_a, reviewer_b], adjudicator, group=False)

    def test_media_path_is_read_only_and_confined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "media"
            root.mkdir()
            inside = root / "candidate.png"
            inside.write_bytes(b"not-a-real-image")
            outside = Path(directory) / "outside.png"
            outside.write_bytes(b"outside")
            self.assertEqual(confined_path(root, "candidate.png"), inside.resolve())
            with self.assertRaises(ValueError):
                confined_path(root, "../outside.png")

    def test_eight_of_eight_acceptance_and_duplicate_rejection(self) -> None:
        records = [
            {
                "semantic_pose_slot": slot,
                "decision": "ACCEPT",
                "candidate_sha256": f"{index:064x}",
                "identity_contamination": 0,
                "severe_anatomy_failure": 0,
                "text_or_watermark": 0,
                "full_body_completeness": "PASS",
            }
            for index, slot in enumerate(SLOTS, start=1)
        ]
        self.assertEqual(evaluate_garment_acceptance(records, "PASS")["status"], "ACCEPT")
        records[-1]["candidate_sha256"] = records[0]["candidate_sha256"]
        self.assertEqual(
            evaluate_garment_acceptance(records, "PASS")["status"],
            "GARMENT_SLOT_GENERATION_INCOMPLETE",
        )

    def test_tool_is_local_only_and_has_candidate_and_group_modes(self) -> None:
        server = (ROOT / "tools/multi_identity_review/review_server.py").read_text(encoding="utf-8")
        workflow = (ROOT / "tools/multi_identity_review/workflow.py").read_text(encoding="utf-8")
        app = (ROOT / "tools/multi_identity_review/app.js").read_text(encoding="utf-8")
        self.assertIn('(\"127.0.0.1\", args.port)', server)
        self.assertIn("/api/group-reviews", server)
        self.assertIn("/api/group-reviews", app)
        self.assertIn("groupDimensions", app)
        forbidden_client_tokens = ("requests.post", "urllib.request", "OpenAI(", ".images.generate", ".images.edit")
        self.assertFalse(any(token in workflow for token in forbidden_client_tokens))

    def test_credential_scanner_reports_rule_without_secret_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.txt"
            path.write_text("authorization: Bearer " + "x" * 24, encoding="utf-8")
            findings = scan_credential_leaks([path])
            self.assertTrue(findings)
            encoded = json.dumps(findings)
            self.assertNotIn("x" * 24, encoded)
        new_files = [ROOT / relative for relative in (
            [f"paper_protocol/datasets/{name}" for name in CONTRACTS]
            + [f"docs/DATASET/{name}" for name in REPORTS]
            + [
                "project_control_handoff/multi_identity_generation_backend_handoff.json",
                "tools/multi_identity_review/workflow.py",
                "tools/multi_identity_review/review_server.py",
                "tools/multi_identity_review/index.html",
                "tools/multi_identity_review/app.js",
                "tools/multi_identity_review/styles.css",
            ]
        )]
        self.assertEqual(scan_credential_leaks(new_files), [])

    def test_all_execution_counts_and_paper_final_are_zero(self) -> None:
        for payload in (self.backend, self.summary, self.handoff):
            self.assertTrue(all(value == 0 for value in payload["execution_counts"].values()))
        self.assertEqual(self.summary["frozen_source_contracts_modified"], 0)
        self.assertEqual(self.summary["final_classification"], "GENERATION_BACKEND_SELECTION_REQUIRED")
        self.assertEqual(self.handoff["next_task"], "USER_SELECT_GENERATION_PROVIDER_AND_MODEL")
        self.assertFalse(any("PAPER_FINAL" in path.name for path in ROOT.rglob("*")))


if __name__ == "__main__":
    unittest.main()
