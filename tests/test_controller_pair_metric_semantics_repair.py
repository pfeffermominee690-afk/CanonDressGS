import math
import unittest

from tools.paper import run_controller_pair_metric_semantics_repair as repair


class ControllerPairMetricSemanticsRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = repair.read_json(
            repair.RISK / "dual_support_controller_training_manifest.json"
        )
        cls.predictions = repair.read_json(
            repair.RISK / "controller_v2_repaired_test_predictions.json"
        )
        cls.historical = repair.read_json(
            repair.RISK / "controller_v2_repaired_final_summary.json"
        )
        cls.perturbation = repair.read_json(
            repair.RISK / "controller_v2_repaired_perturbation_results.json"
        )
        cls.visual = repair.read_json(
            repair.RISK / "controller_v2_repaired_visual_review.json"
        )
        cls.compatibility = repair.compatibility_labels()
        cls.audit, cls.visible, cls.identifiability = repair.protocol_semantics(
            cls.manifest, cls.compatibility
        )
        cls.pure, cls.mixed, cls.routing = repair.corrected_metrics(
            cls.predictions, cls.compatibility
        )
        cls.gates = repair.gates_and_interpretation(
            cls.pure,
            cls.mixed,
            cls.routing,
            cls.historical,
            cls.perturbation,
            cls.visual,
        )

    def test_visible_garment_cardinality_comes_only_from_reference_slots(self) -> None:
        rows = self.visible["records"]
        query_pure = [
            row for row in rows
            if row["record_scope"] == "PAIR_QUERY"
            and row["assignment_type"] in repair.PURE_TYPES
        ]
        query_mixed = [
            row for row in rows
            if row["record_scope"] == "PAIR_QUERY"
            and row["assignment_type"] in repair.MIXED_TYPES
        ]
        formal_pure = [row for row in rows if row["record_scope"] == "FORMAL_PURE"]
        self.assertEqual((len(query_pure), len(query_mixed), len(formal_pure)), (80, 240, 20))
        self.assertTrue(all(row["visible_cardinality"] == 1 for row in query_pure + formal_pure))
        self.assertTrue(all(row["visible_cardinality"] == 2 for row in query_mixed))
        self.assertTrue(all(row["compatibility_gt_by_rotation"] == "NOT_APPLICABLE_TO_PURE" for row in query_pure))
        self.assertEqual(self.visible["inconsistency_count"], 0)

    def test_latent_secondary_has_four_uniform_candidates_per_observable(self) -> None:
        classes = self.identifiability["equivalence_classes"]
        self.assertEqual(len(classes), 20)
        for row in classes:
            self.assertEqual(row["latent_secondary_candidate_count"], 4)
            self.assertEqual(sorted(row["latent_secondary_counts"].values()), [1, 1, 1, 1])
            self.assertAlmostEqual(row["entropy_bits"], 2.0)
            self.assertAlmostEqual(row["uniform_random_accuracy"], 0.25)
            self.assertAlmostEqual(row["majority_accuracy"], 0.25)
            self.assertAlmostEqual(row["bayes_optimal_accuracy"], 0.25)
            self.assertFalse(row["latent_secondary_consistent_across_duplicates"])
        self.assertEqual(
            self.identifiability["protocol_semantic_classification"],
            "LATENT_SECONDARY_NOT_INPUT_IDENTIFIABLE",
        )

    def test_historical_08125_is_derived_from_actual_protocol_counts(self) -> None:
        derivations = self.identifiability["historical_0_8125_derivations"]
        self.assertEqual(len(derivations), 3)
        for row in derivations.values():
            self.assertEqual((row["mixed_record_count"], row["pure_record_count"]), (240, 80))
            self.assertAlmostEqual(row["pure_latent_secondary_accuracy"], 0.25)
            self.assertAlmostEqual(row["combined_all_record_expected_score"], 0.8125)
        self.assertEqual(
            self.identifiability["interpretation"],
            "LATENT_LABEL_CHANCE_CEILING_NOT_REPRESENTATION_UPPER_BOUND",
        )

    def test_corrected_pure_and_mixed_denominators_and_values(self) -> None:
        expected_pair = {
            "V2": (0.7180555555555556, [0.7277777777777777, 0.6888888888888889, 0.7166666666666667, 0.7388888888888889]),
            "MATCHED_V1": (0.7583333333333333, [0.7333333333333333, 0.6888888888888889, 0.8444444444444444, 0.7666666666666667]),
        }
        for family in repair.FAMILIES:
            primary = self.pure["families"][family]["primary_pure"]
            formal = self.pure["families"][family]["formal_pure_secondary"]
            mixed = self.mixed["families"][family]
            self.assertEqual((primary["record_count"], mixed["record_count"], formal["record_count"]), (240, 720, 240))
            self.assertAlmostEqual(primary["top1_visible_garment_accuracy"], 1.0)
            self.assertAlmostEqual(mixed["unordered_top2_pair_accuracy"], expected_pair[family][0])
            self.assertEqual(
                [mixed["per_rotation"][str(index)]["unordered_top2_pair_accuracy"] for index in range(4)],
                expected_pair[family][1],
            )
            self.assertEqual(mixed["protocol_weighted"]["record_count"], 720)
            self.assertEqual(mixed["unique_query"]["unique_query_count"], 240)
            self.assertAlmostEqual(mixed["rotation_macro"], mixed["global_macro"])
        self.assertAlmostEqual(self.pure["families"]["V2"]["primary_pure"]["single_endpoint_rate"], 1.0)
        self.assertAlmostEqual(self.pure["families"]["MATCHED_V1"]["primary_pure"]["single_endpoint_rate"], 0.9)
        self.assertAlmostEqual(self.pure["families"]["V2"]["formal_pure_secondary"]["single_endpoint_rate"], 1.0)
        self.assertAlmostEqual(self.pure["families"]["MATCHED_V1"]["formal_pure_secondary"]["single_endpoint_rate"], 0.9333333333333333)

    def test_task_conditional_metric_is_secondary_and_cardinality_selected(self) -> None:
        archive = self.mixed["task_conditional_identification_accuracy"]
        self.assertEqual(archive["role"], "SECONDARY_SUMMARY_ONLY")
        self.assertTrue(archive["offline_gt_cardinality_selects_rule"])
        for family in repair.FAMILIES:
            row = archive["families"][family]
            pair = self.mixed["families"][family]["unordered_top2_pair_accuracy"]
            self.assertAlmostEqual(row["metric"], 0.25 + 0.75 * pair)
            self.assertEqual(row["record_count"], 960)
            self.assertEqual(row["protocol_weighted"]["record_count"], 960)

    def test_routing_and_weight_keep_frozen_cell_macro_contract(self) -> None:
        row = self.routing["families"]["V2"]
        self.assertEqual(row["record_count"], 720)
        self.assertEqual(row["pure_records_in_denominators"], 0)
        self.assertEqual(row["denominator_classification"], "ROUTING_DENOMINATOR_ALREADY_VALID")
        self.assertEqual(row["rotation_seed_cell_count"], 12)
        self.assertAlmostEqual(row["correct_compatible_dual_rate"], 0.18937812421128092)
        self.assertAlmostEqual(row["correct_incompatible_hard_rate"], 0.3138888888888889)
        self.assertAlmostEqual(row["incompatible_dual_rate"], 0.0)
        self.assertAlmostEqual(row["wrong_pair_dual_rate"], 0.07282421740626076)
        self.assertAlmostEqual(row["mixed_false_single_rate"], 0.8305555555555555)
        self.assertAlmostEqual(row["correct_pair_weight"]["mae"], 0.15805836898719514)
        self.assertAlmostEqual(row["correct_pair_weight"]["rmse"], 0.1981430412872887)
        self.assertEqual(row["correct_pair_weight"]["count"], 517)
        self.assertEqual(row["correct_order_weight"]["count"], 405)
        self.assertFalse(math.isclose(row["record_micro"]["correct_pair_weight"]["mae"], row["correct_pair_weight"]["mae"]))

    def test_gate_reinterpretation_preserves_historical_fail(self) -> None:
        self.assertEqual(self.gates["historical_overall_classification"], repair.HISTORICAL_CLASSIFICATION)
        self.assertEqual(self.gates["primary_failure_interpretation"], "MIXED_PAIR_IDENTIFICATION_PARTIAL")
        self.assertEqual(self.gates["next_task"], "DIAGNOSE_CONTROLLER_GARMENT_HEAD_OPTIMIZATION_BUDGET")
        self.assertFalse(self.gates["next_task_started"])
        self.assertEqual(len(self.gates["original_gate_table"]), len(self.gates["semantically_corrected_gate_table"]))
        self.assertFalse(self.gates["information_ablation_definition_changed"])
        self.assertFalse(self.gates["preserved_visual_gates"]["denominator_changed"])

    def test_future_evaluator_contract_forbids_latent_pure_pair_metric(self) -> None:
        contract = repair.evaluator_contract()
        self.assertIn(
            "exact_unordered_top2_pair_accuracy",
            contract["pure_garment_identification"]["forbidden_metrics"],
        )
        self.assertIn(
            "unordered_top2_pair_accuracy",
            contract["mixed_pair_identification"]["primary_metrics"],
        )
        self.assertEqual(contract["formal_pure_20"], "SECONDARY_ENDPOINT_SAFETY_ONLY")
        self.assertEqual(contract["historical_v1_96_25"], "CLOSED_WARDROBE_PROTOCOL_FIT_RESULT")
        self.assertEqual(
            contract["controller_crossfit_boundary"],
            "CONDITION_FOLD_HELD_OUT_WITH_REFERENCE_ASSET_OVERLAP",
        )
        self.assertFalse(contract["denominator_selection_in_inference"])

    def test_all_markdown_reports_render_nonempty(self) -> None:
        claims = repair.claim_manifest()
        reports = repair.markdown_reports(
            self.audit,
            self.identifiability,
            self.pure,
            self.mixed,
            self.routing,
            self.gates,
            claims,
            repair.evaluator_contract(),
        )
        self.assertEqual(len(reports), 4)
        self.assertTrue(all(len(value.strip()) > 500 for value in reports.values()))
        joined = "\n".join(reports.values())
        self.assertIn("LATENT_SECONDARY_NOT_INPUT_IDENTIFIABLE", joined)
        self.assertIn("MIXED_PAIR_IDENTIFICATION_PARTIAL", joined)
        self.assertIn("CLOSED_WARDROBE_PROTOCOL_FIT_RESULT", joined)


if __name__ == "__main__":
    unittest.main()
