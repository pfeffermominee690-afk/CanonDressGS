import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol" / "reviewer_risk"
SOURCE_HEAD = "ff56ebfaf7b733adbb41799e248d01d0e7801ea8"
PRIMARY_NAMES = [
    "Base Avatar",
    "Teacher Endpoint",
    "Outfit-ID Oracle",
    "Reference Classifier Lookup",
    "Nearest-Centroid Lookup",
    "Linear Coefficient Predictor",
    "CanonDressGS-Endpoint",
]
SOURCE_HASHES = {
    "paper_protocol/reviewer_risk/pure_endpoint_protocol_artifact_hash_manifest.json": "9325900a8bff9060108066042df3d2e7bec8461b0d7fb0a23180f83fda691f57",
    "paper_protocol/reviewer_risk/pure_endpoint_direct_residual_decoder_audit.json": "fc7ed9c720defdd0b0883bb5692b28c43a550cd904fc36b05d7dbc309049619f",
    "paper_protocol/reviewer_risk/pure_endpoint_direct_residual_decoder_contract.json": "73318509f85d4a530fc79f35091c7a9d3e0384afd34ae5ae61ab6d638f41a5f9",
    "paper_protocol/reviewer_risk/pure_endpoint_baseline_registry_repaired.json": "83d79b79b63545819929b5107e22012fc7af5a395d418cafd81da2f2298ccff4",
    "paper_protocol/reviewer_risk/pure_endpoint_execution_contract_repaired.json": "317a2efe9214f250577890aeca44fa8e620db687a79befb8a689e25d5175519f",
    "paper_protocol/reviewer_risk/pure_endpoint_contract_repair_tests.json": "8e8561b1f5e3265c9c73dcf3941e5bdff6dc0d499f438d4e01304af292c6e2bb",
    "paper_protocol/reviewer_risk/pure_endpoint_contract_repair_final_summary.json": "e713d65077cd7f29bf8dcfc64e3f590235df1fba305f569d0efd30e08b1f2086",
    "project_control_handoff/pure_endpoint_contract_repair_handoff.json": "bd231f1709224a701672a4a3d3c3381f40281ff5aa68469377c934761e6e8c0e",
}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(name):
    return json.loads(
        (RISK / name).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


def sha256(relative):
    data = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def null_paths(value, prefix="$"):
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        result = []
        for key, child in value.items():
            result.extend(null_paths(child, f"{prefix}.{key}"))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(null_paths(child, f"{prefix}[{index}]"))
        return result
    return []


class PureEndpointProtocolAmendmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adjudication = read_json("pure_endpoint_direct_decoder_adjudication.json")
        cls.primary = read_json("pure_endpoint_primary_baseline_registry_amended.json")
        cls.supplementary = read_json("pure_endpoint_supplementary_baseline_registry.json")
        cls.counts = read_json("pure_endpoint_expected_counts_amended.json")
        cls.contract = read_json("pure_endpoint_execution_contract_amended.json")
        cls.tests_payload = read_json("pure_endpoint_protocol_amendment_tests.json")
        cls.summary = read_json("pure_endpoint_protocol_amendment_final_summary.json")
        cls.handoff = json.loads(
            (ROOT / "project_control_handoff" / "pure_endpoint_protocol_amendment_handoff.json").read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )

    def test_exact_source_and_frozen_source_hashes(self):
        self.assertEqual(self.adjudication["source_head"], SOURCE_HEAD)
        for relative, expected in SOURCE_HASHES.items():
            self.assertEqual(sha256(relative), expected, relative)

    def test_hash_closure_still_passes(self):
        manifest = read_json("pure_endpoint_protocol_artifact_hash_manifest.json")
        self.assertEqual(manifest["closure_status"], "PASS_INDEPENDENT_MANIFEST")
        self.assertEqual(manifest["artifact_count"], 11)
        self.assertEqual(manifest["original_artifact_mutation_count"], 0)

    def test_primary_registry_is_exactly_seven_methods(self):
        names = [row["paper_name"] for row in self.primary["methods"]]
        self.assertEqual(names, PRIMARY_NAMES)
        self.assertNotIn("Direct Residual Decoder", names)
        self.assertEqual(self.primary["method_count"], 7)

    def test_direct_decoder_is_preserved_in_supplement_only(self):
        row = self.supplementary["methods"][0]
        self.assertEqual(row["formal_name"], "Direct Residual Decoder (Historical V7)")
        self.assertFalse(row["primary_table_eligible"])
        self.assertFalse(row["execution_required_in_pure_endpoint_crossfit"])
        self.assertEqual(row["execution_count_in_pure_endpoint_crossfit"], 0)
        self.assertEqual(row["stage_b_result"], "NOT_RUN_STAGE_A_FAILED")

    def test_no_invented_architecture_or_predictor_rename(self):
        self.assertFalse(self.adjudication["decision_guards"]["architecture_invented"])
        self.assertFalse(self.adjudication["decision_guards"]["baseline_renamed"])
        self.assertEqual(PRIMARY_NAMES[5], "Linear Coefficient Predictor")
        shared = self.primary["methods"][5:7]
        self.assertEqual({row["model_family"] for row in shared}, {"SHARED_LINEAR_ENDPOINT_PREDICTOR"})

    def test_expected_counts_are_arithmetically_closed(self):
        training = self.counts["training"]
        self.assertEqual(training["total_training_runs"], 2 * 4 * 3)
        self.assertEqual(training["optimizer_steps"], 24 * 300)
        self.assertEqual(training["checkpoint_writes"], 24 * 6)
        self.assertEqual(training["direct_decoder_runs"], 0)
        evaluation = self.counts["evaluation"]
        self.assertEqual(evaluation["primary_test_inferences"], 7 * 4 * 20 * 3)
        self.assertEqual(evaluation["formal_pure_inferences"], 7 * 20 * 3)
        self.assertEqual(evaluation["perturbation_inferences"], 7 * 20 * 3 * 6)
        rendering = self.counts["rendering"]
        self.assertEqual(rendering["logical_renders"], 4620)
        self.assertEqual(
            rendering["unique_physical_renders"] + rendering["render_cache_reuses"],
            rendering["logical_renders"],
        )

    def test_visual_contract(self):
        visual = self.counts["visual_review"]
        self.assertEqual(visual["method_columns"], PRIMARY_NAMES)
        self.assertEqual(visual["main_sheet_count"], 60)
        self.assertEqual(visual["formal_pure_sheet_count"], 60)
        self.assertEqual(visual["method_column_count"], 7)

    def test_amended_contract_inherits_frozen_science(self):
        self.assertEqual(
            self.contract["classification"],
            "PURE_ENDPOINT_EXECUTION_CONTRACT_AMENDED_AND_READY",
        )
        self.assertTrue(self.contract["execution_authorized"])
        self.assertFalse(self.contract["amendment"]["direct_decoder_execution_required"])
        self.assertEqual(self.contract["model_and_training_contract"]["steps"], 300)
        self.assertEqual(self.contract["model_and_training_contract"]["seeds"], [0, 1, 2])
        self.assertFalse(self.contract["model_and_training_contract"]["model_contract_changed"])
        self.assertFalse(self.contract["model_and_training_contract"]["training_budget_changed"])
        self.assertTrue(self.contract["success_gates"]["unchanged"])
        self.assertEqual(self.contract["evaluator_contract"]["protocol_weighted_denominator"], 80)
        self.assertEqual(self.contract["evaluator_contract"]["unique_query_denominator"], 20)

    def test_actual_execution_is_zero_and_paper_not_final(self):
        for payload in (self.adjudication, self.counts, self.contract):
            actual = payload.get("actual_execution_counts", payload.get("actual_amendment_execution_counts"))
            self.assertTrue(all(value == 0 for value in actual.values()))
            self.assertFalse(payload["paper_final"])

    def test_generated_core_json_has_no_nulls(self):
        for payload in (
            self.adjudication,
            self.primary,
            self.supplementary,
            self.counts,
            self.contract,
            self.tests_payload,
            self.summary,
            self.handoff,
        ):
            self.assertEqual(null_paths(payload), [])
        self.assertEqual(
            self.handoff["final_summary"]["sha256_lf"],
            sha256("paper_protocol/reviewer_risk/pure_endpoint_protocol_amendment_final_summary.json"),
        )
        for reference in self.summary["references"].values():
            self.assertEqual(reference["sha256_lf"], sha256(reference["relative_path"]))

    def test_source_artifacts_have_no_worktree_mutation(self):
        paths = list(SOURCE_HASHES)
        result = subprocess.run(
            ["git", "diff", "--exit-code", SOURCE_HEAD, "--", *paths],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_markdown_outputs_are_nonempty(self):
        names = [
            "AAAI27_DIRECT_DECODER_BASELINE_ADJUDICATION_20260724.md",
            "AAAI27_PURE_ENDPOINT_PRIMARY_BASELINE_SCOPE_AMENDMENT_20260724.md",
            "AAAI27_PURE_ENDPOINT_HARD_LOOKUP_CLAIM_BOUNDARY_20260724.md",
        ]
        for name in names:
            self.assertGreater((ROOT / "docs" / "PAPER" / name).stat().st_size, 200)


if __name__ == "__main__":
    unittest.main()
