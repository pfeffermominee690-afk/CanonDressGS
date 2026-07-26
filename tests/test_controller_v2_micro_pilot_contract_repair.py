"""Static tests for the pre-result Controller V2 contract repair.

These tests intentionally use no torch, model, optimizer, renderer, or metric
runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import unittest
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"
DOCS = ROOT / "docs/PAPER"
HANDOFF = ROOT / "project_control_handoff"

SOURCE_HEAD = "b15f26e51958820d361b8992fab5ddec0e592fb5"
OUTFITS = ["O01", "O02", "O03", "O04", "O08"]
FOLDS = ["cond_000000", "cond_000318", "cond_000017", "cond_000347"]
EXPECTED_LF = {
    "dual_support_controller_protocol.yaml": "44ef0c53f7a5ec4fe19733371ffefb1be14d5b6902e7f4060707edd4660d9f49",
    "dual_support_controller_training_manifest.json": "a0dbfe98a25b7f82f198405cf41c1b8625b6faa63499396a78683d5d326ce8e3",
}


def read(name: str) -> Any:
    return json.loads((RISK / name).read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    payload = (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ControllerV2ContractRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repaired = read("controller_v2_micro_pilot_contract_repaired.json")
        cls.manifests = read("controller_v2_micro_pilot_rotation_manifests.json")
        cls.schedules = read("controller_v2_micro_pilot_batch_schedules.json")
        cls.optimizer = read("controller_v2_micro_pilot_optimizer_contract.json")
        cls.loss = read("controller_v2_micro_pilot_loss_contract.json")
        cls.calibration = read("controller_v2_micro_pilot_calibration_contract.json")
        cls.denominators = read("controller_v2_micro_pilot_evaluator_denominators.json")
        cls.perturb = read("controller_v2_micro_pilot_perturbation_representatives.json")
        cls.visual = read("controller_v2_micro_pilot_visual_representatives.json")
        cls.summary = read("controller_v2_micro_pilot_contract_repair_summary.json")

    def test_exact_source_head_and_branch(self) -> None:
        merge_base = subprocess.check_output(
            ["git", "merge-base", SOURCE_HEAD, "HEAD"], cwd=ROOT, text=True
        ).strip()
        self.assertEqual(merge_base, SOURCE_HEAD)
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip()
        self.assertEqual(
            branch, "research/controller-v2-micro-pilot-contract-repair-20260723"
        )
        self.assertEqual(self.repaired["source_head"], SOURCE_HEAD)

    def test_all_expected_files_parse_and_markdown_nonempty(self) -> None:
        expected_json = [
            "controller_v2_micro_pilot_contract_repaired.json",
            "controller_v2_micro_pilot_rotation_manifests.json",
            "controller_v2_micro_pilot_batch_schedules.json",
            "controller_v2_micro_pilot_optimizer_contract.json",
            "controller_v2_micro_pilot_loss_contract.json",
            "controller_v2_micro_pilot_calibration_contract.json",
            "controller_v2_micro_pilot_evaluator_denominators.json",
            "controller_v2_micro_pilot_perturbation_representatives.json",
            "controller_v2_micro_pilot_visual_representatives.json",
            "controller_v2_micro_pilot_contract_repair_summary.json",
        ]
        self.assertEqual(len(expected_json), 10)
        for name in expected_json:
            self.assertIsInstance(read(name), dict)
        self.assertIsInstance(json.loads((HANDOFF / "controller_v2_micro_pilot_contract_repair_handoff.json").read_text(encoding="utf-8")), dict)
        for name in (
            "AAAI27_CONTROLLER_V2_MICRO_PILOT_CONTRACT_REPAIR_20260723.md",
            "AAAI27_CONTROLLER_V2_MATCHED_V1_TRAINING_CONTRACT_20260723.md",
            "AAAI27_CONTROLLER_V2_CALIBRATION_AND_EVALUATOR_CONTRACT_20260723.md",
        ):
            text = (DOCS / name).read_text(encoding="utf-8")
            self.assertGreater(len(text.strip()), 100)

    def test_protocol_and_manifest_lf_hashes(self) -> None:
        for name, expected in EXPECTED_LF.items():
            payload = (RISK / name).read_bytes().replace(b"\r\n", b"\n")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)
        cycle = read("dual_support_controller_training_cycle.json")
        schedule = read("dual_support_controller_training_schedule_300_steps.json")
        self.assertEqual(
            cycle["training_cycle_sha256"],
            "f60b0ee64ff5ce2b6693f53dd7eb665cc6aee5aa310f314175d2932307d57c77",
        )
        self.assertEqual(
            schedule["training_300_step_data_order_sha256"],
            "63902a2f36ccf864bfe6028654be0b162417c3adc6b99e1455b30fdab8a2bacf",
        )

    def test_thirty_field_audit_is_complete(self) -> None:
        checks = self.repaired["contract_checks"]
        self.assertEqual(len(checks), 30)
        self.assertEqual([row["id"] for row in checks], list(range(1, 31)))
        self.assertTrue(all(row["final_state"] == "PASS" for row in checks))
        self.assertTrue(all(row["repaired_value"] is not None for row in checks))
        self.assertEqual(
            self.repaired["repaired_audit"],
            {"pass": 30, "missing": 0, "ambiguous": 0, "total": 30},
        )

    def test_four_rotation_exact_disjoint_manifests(self) -> None:
        source = read("dual_support_controller_training_manifest.json")
        source_ids = {row["record_id"] for row in source["query_sets"]}
        self.assertEqual(len(self.manifests["rotations"]), 4)
        for rotation in self.manifests["rotations"]:
            parts = rotation["partitions"]
            train = set(parts["train"]["record_ids"])
            calibration = set(parts["calibration"]["record_ids"])
            test = set(parts["test"]["record_ids"])
            self.assertEqual((len(train), len(calibration), len(test)), (160, 80, 80))
            self.assertFalse(train & calibration)
            self.assertFalse(train & test)
            self.assertFalse(calibration & test)
            self.assertEqual(train | calibration | test, source_ids)
            self.assertEqual(rotation["record_intersections"], {
                "train_calibration": 0, "train_test": 0, "calibration_test": 0,
            })
            self.assertEqual(rotation["logical_query_intersections"], {
                "train_calibration": 0, "train_test": 0, "calibration_test": 0,
            })
            for name, expected in (("train", (40, 120, 130)), ("calibration", (20, 60, 65)), ("test", (20, 60, 65))):
                part = parts[name]
                self.assertEqual(
                    (part["pure_count"], part["mixed_count"], part["unique_logical_query_count"]),
                    expected,
                )
                self.assertTrue(all(value == part["record_count"] // 5 for value in part["per_dominant_outfit_counts"].values()))
            manifest_sha = rotation["rotation_manifest_sha256"]
            scientific = {key: value for key, value in rotation.items() if key != "rotation_manifest_sha256"}
            self.assertEqual(canonical_sha(scientific), manifest_sha)

    def test_duplicates_are_preserved(self) -> None:
        for rotation in self.manifests["rotations"]:
            parts = rotation["partitions"]
            self.assertEqual(parts["train"]["consistent_duplicate_record_count"], 40)
            self.assertEqual(parts["calibration"]["consistent_duplicate_record_count"], 20)
            self.assertEqual(parts["test"]["consistent_duplicate_record_count"], 20)
            self.assertTrue(rotation["consistent_duplicates"]["preserved_as_independent_protocol_records"])
            self.assertFalse(rotation["consistent_duplicates"]["deduplicated"])

    def test_balanced_batches_complete_cycles_and_exposure_formula(self) -> None:
        manifest_index = {
            row["record_id"]: row
            for row in read("dual_support_controller_training_manifest.json")["query_sets"]
        }
        for rotation in self.schedules["rotations"]:
            cycle = rotation["cycle"]
            self.assertEqual((cycle["batch_size"], cycle["batch_count"]), (5, 32))
            ids = []
            for batch in cycle["batches"]:
                self.assertEqual(len(batch["records"]), 5)
                self.assertEqual([row["dominant_outfit"] for row in batch["records"]], OUTFITS)
                ids.extend(row["record_id"] for row in batch["records"])
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(ids), 160)
            expected_train = set(
                self.manifests["rotations"][rotation["rotation"]]["partitions"]["train"]["record_ids"]
            )
            self.assertEqual(set(ids), expected_train)
            self.assertTrue(all(record_id in manifest_index for record_id in ids))
            self.assertEqual(rotation["optimizer_steps"], 4 * 32 + (11 * 32) // 16)
            self.assertEqual(rotation["clean_record_exposures_per_run"], 750)
            self.assertEqual(rotation["exposure_histogram"], {"4": 50, "5": 110})

    def test_clean_order_equality_and_seed_independence(self) -> None:
        for rotation in self.schedules["rotations"]:
            family = rotation["family_training_plans"]
            self.assertEqual(
                family["V2"]["clean_schedule_sha256"],
                family["MATCHED_V1"]["clean_schedule_sha256"],
            )
            seed_hashes = rotation["seed_clean_data_order_sha256"]
            self.assertEqual(len(set(seed_hashes.values())), 1)
            identities = [
                value
                for values in rotation["future_initialization_identity_sha256"].values()
                for value in values.values()
            ]
            self.assertEqual(len(identities), len(set(identities)))

    def test_checkpoint_schedule(self) -> None:
        for rotation in self.schedules["rotations"]:
            self.assertEqual(rotation["checkpoint_steps"], [0, 30, 60, 90, 120, 150])
            self.assertEqual(rotation["checkpoint_count_per_run"], 6)
            self.assertEqual(rotation["final_checkpoint_rule"], "EVALUATE_STEP_150_ONLY_NO_BEST_OR_EARLY_SELECTION")

    def test_optimizer_provenance_has_no_unresolved_defaults(self) -> None:
        optimizer = self.optimizer["optimizer"]
        expected = {
            "class", "learning_rate", "weight_decay", "betas", "epsilon",
            "amsgrad", "maximize", "foreach", "capturable", "differentiable",
            "fused", "parameter_group_count", "parameter_group_rule",
            "optimizer_state_keys_per_parameter", "defaults_used_as_provenance",
        }
        self.assertEqual(set(optimizer), expected)
        self.assertEqual(optimizer["class"], "torch.optim.Adam")
        self.assertEqual(optimizer["learning_rate"], 0.02)
        self.assertEqual(optimizer["weight_decay"], 0.0)
        self.assertEqual(optimizer["betas"], [0.9, 0.999])
        self.assertEqual(optimizer["epsilon"], 1e-8)
        self.assertFalse(optimizer["defaults_used_as_provenance"])
        self.assertFalse(self.optimizer["runtime"]["mixed_precision"])
        self.assertEqual(self.optimizer["runtime"]["gradient_accumulation_steps"], 1)
        self.assertEqual(self.optimizer["runtime"]["gradient_clipping"]["max_norm"], 5.0)

    def test_static_random_loss_interface(self) -> None:
        self.assertEqual(
            self.loss["lambdas"],
            {"lambda_cons": 0.1, "lambda_mix": 1.0, "lambda_weight": 1.0},
        )
        self.assertEqual(self.loss["pair_weight"]["beta"], 0.1)
        self.assertEqual(self.loss["consistency"]["mixedness"]["beta"], 0.1)
        self.assertEqual(self.loss["consistency"]["pair_weights"]["beta"], 0.1)
        rng = random.Random(20260723)
        garment_logits = [[rng.uniform(-2, 2) for _ in range(5)] for _ in range(5)]
        mixed_logits = [rng.uniform(-2, 2) for _ in range(5)]
        weight_logits = [[rng.uniform(-2, 2) for _ in range(10)] for _ in range(5)]
        garment_probabilities = []
        for row in garment_logits:
            exp = [math.exp(value - max(row)) for value in row]
            garment_probabilities.append([value / sum(exp) for value in exp])
        mixed_probabilities = [1 / (1 + math.exp(-value)) for value in mixed_logits]
        weight_probabilities = [
            [1 / (1 + math.exp(-value)) for value in row] for row in weight_logits
        ]
        self.assertEqual((len(garment_probabilities), len(garment_probabilities[0])), (5, 5))
        self.assertEqual(len(mixed_probabilities), 5)
        self.assertEqual((len(weight_probabilities), len(weight_probabilities[0])), (5, 10))
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in garment_probabilities))

    def test_nuisance_severity_and_round_robin(self) -> None:
        definitions = self.loss["nuisance_augmentation"]["definitions"]
        self.assertEqual(definitions["blur"]["kernel_size"], 11)
        self.assertEqual(definitions["blur"]["sigma"], 3.0)
        self.assertEqual(definitions["mask_erosion"]["iterations"], 3)
        self.assertEqual(definitions["mask_dilation"]["iterations"], 3)
        self.assertEqual(definitions["assignment_permutation"]["rule"], "REFERENCE_ROW_AND_VALIDITY_INDEX_ORDER_[2,0,1]")
        expected = ["blur", "mask_erosion", "mask_dilation", "assignment_permutation"]
        for rotation in self.schedules["rotations"]:
            observed = [
                step["v2_nuisance_type"] for step in rotation["schedule"]["steps"]
            ]
            self.assertTrue(all(value == expected[index % 4] for index, value in enumerate(observed)))
            self.assertEqual(Counter(observed), Counter({
                "blur": 38, "mask_erosion": 38, "mask_dilation": 37,
                "assignment_permutation": 37,
            }))

    def test_information_ablation_training_boundary(self) -> None:
        consistency = self.loss["consistency"]
        self.assertEqual(consistency["information_ablation_exposure"], 0)
        self.assertFalse(self.calibration["information_ablation_used"])
        self.assertEqual(self.calibration["threshold_selection_count_in_this_task"], 0)

    def test_threshold_grids_and_tie_break(self) -> None:
        self.assertEqual(self.calibration["mixedness_threshold_grid"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.assertEqual(self.calibration["pair_confidence_threshold_grid"], [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        self.assertEqual(self.calibration["candidate_count_per_rotation_seed"], 81)
        candidates = self.calibration["all_candidates"]
        objective = (0.0, 0.0, 0.0, -1.0, -1.0, 0.0)
        selected = min(
            ((objective + (-row["tau_pair"], -row["tau_mix"]), row) for row in candidates),
            key=lambda item: item[0],
        )[1]
        self.assertEqual(selected, {"tau_pair": 0.5, "tau_mix": 0.9})

    def test_evaluator_denominators(self) -> None:
        self.assertEqual(self.denominators["primary_gate_aggregation"], "PROTOCOL_WEIGHTED_MACRO")
        self.assertEqual(self.denominators["mandatory_secondary_aggregation"], "UNIQUE_QUERY_MACRO")
        for rotation in self.denominators["rotations"]:
            self.assertEqual(rotation["protocol_weighted"]["all_test_records"], 80)
            self.assertEqual(rotation["unique_query"]["all_test_queries"], 65)
            self.assertEqual(rotation["protocol_weighted"]["compatible_mixed_test_records"], 48)
            self.assertEqual(rotation["protocol_weighted"]["incompatible_mixed_test_records"], 12)

    def test_perturbation_and_visual_representatives(self) -> None:
        self.assertEqual(len(self.perturb["rotations"]), 4)
        for rotation in self.perturb["rotations"]:
            rows = rotation["representatives"]
            self.assertEqual(len(rows), 20)
            self.assertEqual(
                {(row["pair_id"], row["composition"]) for row in rows},
                {
                    (pair, composition)
                    for pair in (
                        "O01_O02", "O01_O03", "O01_O04", "O01_O08",
                        "O02_O03", "O02_O04", "O02_O08", "O03_O04",
                        "O03_O08", "O04_O08",
                    )
                    for composition in ("AAB", "ABB")
                },
            )
        self.assertEqual(self.perturb["queries_both_families_all_rotations_seeds"], 2880)
        self.assertEqual(self.visual["main_sheet_count"], 240)
        self.assertEqual(len(self.visual["sheets"]), 240)
        self.assertEqual(self.visual["grade_schema"]["scale"], {
            "0": "NONE", "1": "MINOR", "2": "MODERATE", "3": "SEVERE",
        })

    def test_matched_v1_comparability(self) -> None:
        self.assertEqual(self.optimizer["matched_v1_parameter_count"], 3589)
        self.assertEqual(self.loss["matched_v1_loss"]["type"], "SOFT_TARGET_CROSS_ENTROPY_ONLY")
        self.assertEqual(self.loss["matched_v1_loss"]["fallback"], "HISTORICAL_0.90_0.10")
        self.assertFalse(self.loss["matched_v1_loss"]["v2_consistency"])

    def test_expected_counts_are_non_null(self) -> None:
        counts = self.repaired["expected_future_counts"]
        self.assertFalse(any(value is None for value in counts["global"].values()))
        self.assertEqual(counts["global"]["total_training_runs"], 24)
        self.assertEqual(counts["global"]["main_visual_sheets"], 240)
        self.assertEqual(counts["global"]["total_future_evaluation_inference_including_calibration"], 6240)
        self.assertEqual(counts["global"]["total_future_logical_renders"], 5280)

    def test_hashes_are_complete_and_recomputable(self) -> None:
        for rotation in self.schedules["rotations"]:
            self.assertEqual(canonical_sha(rotation["cycle"]), rotation["cycle_sha256"])
            self.assertEqual(canonical_sha(rotation["schedule"]), rotation["schedule_sha256"])
            self.assertTrue(all(rotation["family_training_plans"][family]["training_plan_sha256"] for family in ("V2", "MATCHED_V1")))
        for name, expected in self.repaired["component_hashes"].items():
            self.assertEqual(canonical_sha(read(name)), expected)
        global_payload = {
            "source_head": self.repaired["source_head"],
            "frozen_hashes": self.repaired["frozen_upstream_hashes"],
            "component_hashes": self.repaired["component_hashes"],
            "field_audit": self.repaired["contract_checks"],
            "expected_counts": self.repaired["expected_future_counts"],
        }
        self.assertEqual(canonical_sha(global_payload), self.repaired["global_contract_sha256"])

    def test_no_execution_no_mutation_and_no_paper_final(self) -> None:
        execution = self.repaired["current_execution_counts"]
        self.assertTrue(all(value == 0 for value in execution.values()))
        mutations = self.repaired["frozen_mutations"]
        self.assertTrue(all(value == 0 for value in mutations.values()))
        self.assertEqual(self.repaired["paper_final_count"], 0)
        self.assertFalse(self.repaired["paper_final"])
        self.assertFalse(self.repaired["attempt_001_preservation"]["attempt_002_created"])
        self.assertEqual(len(self.repaired["attempt_001_preservation"]["file_sha256"]), 4)


if __name__ == "__main__":
    unittest.main()
