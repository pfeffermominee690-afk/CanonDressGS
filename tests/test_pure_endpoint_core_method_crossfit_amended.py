from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper import run_pure_endpoint_core_method_crossfit_amended as runner


class PureEndpointAmendedContractTests(unittest.TestCase):
    def test_frozen_registry_and_schedule_contracts(self) -> None:
        registry = runner.registry_audit()
        schedules = runner.schedule_audit()
        self.assertEqual(registry["status"], "PASS")
        self.assertEqual(registry["methods"], list(runner.METHODS))
        self.assertEqual(schedules["status"], "PASS")
        self.assertEqual(schedules["actual"], runner.SCHEDULE_HASHES)

    def test_frozen_model_shapes_parameter_counts_and_zero_initialization(self) -> None:
        audit = runner.model_audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["classifier"]["parameters"], 3589)
        self.assertEqual(audit["shared_coefficient"]["parameters"], 3076)
        model = runner.OursV2CandidateAdapter(input_dim=512)
        output = model(torch.zeros((3, 256)), torch.ones((3, 1)))
        self.assertEqual(output.standardized_coefficients.shape, (4,))
        self.assertEqual(int(torch.count_nonzero(output.standardized_coefficients)), 0)

    def test_expected_counts_are_amended_contract_counts(self) -> None:
        counts = runner.expected_counts()
        self.assertEqual(counts["registry"]["primary_method_count"], 7)
        self.assertEqual(counts["registry"]["independent_trainable_model_family_count"], 2)
        self.assertEqual(counts["training"]["total_training_runs"], 24)
        self.assertEqual(counts["training"]["optimizer_steps"], 7200)
        self.assertEqual(counts["training"]["checkpoint_writes"], 144)
        self.assertEqual(counts["evaluation"]["total_logical_method_output_inferences"], 4620)
        self.assertEqual(counts["rendering"]["unique_physical_renders"], 1460)
        self.assertEqual(counts["rendering"]["render_cache_reuses"], 3160)
        self.assertEqual(counts["visual_review"]["main_sheet_count"], 60)
        self.assertEqual(counts["visual_review"]["formal_pure_sheet_count"], 60)

    def test_render_signature_contract_has_exact_physical_count(self) -> None:
        signatures = set()
        queries = runner.unique_queries()
        self.assertEqual(len(queries), 20)
        for query in queries:
            for seed in runner.SEEDS:
                for method in runner.METHODS:
                    signatures.add(runner.physical_render_key(
                        method=method, phase="primary", rotation=query["rotation"], seed=seed,
                        garment=query["garment"], condition=query["condition"], variant="clean",
                    ))
                    signatures.add(runner.physical_render_key(
                        method=method, phase="formal_pure", rotation=query["rotation"], seed=seed,
                        garment=query["garment"], condition=query["condition"], variant="clean",
                    ))
                    for variant in runner.PERTURBATIONS:
                        signatures.add(runner.physical_render_key(
                            method=method, phase="perturbation", rotation=query["rotation"], seed=seed,
                            garment=query["garment"], condition=query["condition"], variant=variant,
                        ))
        self.assertEqual(len(signatures), 1460)

    def test_perturbation_contract_is_exact_and_complete_dropout_is_separate(self) -> None:
        self.assertEqual(runner.PERTURBATIONS, (
            "mild_blur", "mask_erosion", "mask_dilation", "assignment_permutation",
            "reference_dropout", "single_reference",
        ))
        self.assertEqual(runner.perturbation_selection("assignment_permutation"), ("normal", (2, 0, 1)))
        self.assertEqual(runner.perturbation_selection("reference_dropout"), ("normal", (0, 1)))
        self.assertEqual(runner.perturbation_selection("single_reference"), ("normal", (0,)))
        with self.assertRaises(ValueError):
            runner.perturbation_selection("complete_dropout")

    def test_strict_json_duplicate_key_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                runner.read_json(path)

    def test_success_gate_enum_and_paper_final_are_frozen(self) -> None:
        gates = runner.read_json(runner.RISK / "pure_endpoint_success_gates.json")
        self.assertEqual(gates["decision_rules"], {
            "PURE_ENDPOINT_CORE_METHOD_PARTIAL": "at least one but not all rotation clauses pass",
            "PURE_ENDPOINT_CORE_METHOD_SUPPORTED": "all primary gate clauses pass on frozen crossfit results",
            "REFERENCE_CONTROL_NOT_SUPPORTED": "reference identification fails while Outfit-ID Oracle endpoint parity and rendering remain valid",
        })
        self.assertIs(gates["paper_final"], False)
        self.assertEqual(gates["paper_final_count"], 0)

    def test_historical_blockers_and_v7_execution_zero_are_preserved(self) -> None:
        contract = runner.amended_contract()
        supplementary = runner.read_json(
            runner.RISK / "pure_endpoint_supplementary_baseline_registry.json"
        )
        self.assertEqual(contract["historical_preservation"]["historical_classifications"], [
            "PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH",
            "PURE_ENDPOINT_DIRECT_DECODER_CONTRACT_BLOCKED",
        ])
        self.assertEqual(contract["historical_preservation"]["blocked_artifact_mutation_count"], 0)
        self.assertEqual(supplementary["methods"][0]["execution_count_in_pure_endpoint_crossfit"], 0)
        self.assertEqual(supplementary["methods"][0]["formal_name"], "Direct Residual Decoder (Historical V7)")


if __name__ == "__main__":
    unittest.main()
