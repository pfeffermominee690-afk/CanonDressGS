import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"
FIGURES = ROOT / "paper_draft/figures/loo_method_freeze"
ALLOWED_CLASSIFICATIONS = {
    "LOO_METHOD_FREEZE_FIGURE_REFRESH_READY",
    "LOO_FIGURE_REFRESH_SOURCE_NOT_SEALED",
    "LOO_METHOD_FREEZE_CONTRACT_BLOCKED",
    "LOO_METHOD_FREEZE_FIGURE_REFRESH_INCONCLUSIVE",
}
GARMENTS = {"O01", "O02", "O03", "O04", "O08"}


def load_json(path: Path):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise AssertionError(f"duplicate key {key!r} in {path}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class LooMethodFreezeFigureRefreshTest(unittest.TestCase):
    def test_01_sealed_source_gate_and_counts(self):
        source = load_json(RISK / "paper_figure_loo_refresh_source_registry.json")
        gate = source["source_gate"]
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["classification"], "LOO_BASIS_CAPACITY_LIMITED")
        self.assertEqual(gate["attempt"], "attempt_003")
        self.assertTrue(gate["attempt_003_sealed"])
        self.assertEqual(gate["counts"]["total_optimizer_runs"], 120)
        self.assertEqual(gate["counts"]["optimizer_steps"], 36000)
        self.assertEqual(gate["counts"]["checkpoint_writes"], 720)
        self.assertEqual(gate["counts"]["evaluation_inference"], 960)
        self.assertEqual(gate["counts"]["total_logical_renders"], 54960)
        self.assertEqual(gate["counts"]["unique_physical_renders"], 54845)
        self.assertEqual(gate["counts"]["K_shared_static_cache_hits"], 115)
        self.assertEqual(gate["counts"]["visual_sheets"], 26)
        self.assertEqual(gate["reporting_count_key_repair"], {"guaranteed": 100, "shared": 15, "divergent": 5})

    def test_02_source_scope_and_mutation_baselines(self):
        source = load_json(RISK / "paper_figure_loo_refresh_source_registry.json")
        self.assertEqual(source["attempt_001_scientific_results_consumed"], 0)
        self.assertEqual(source["attempt_002_scientific_results_consumed"], 0)
        self.assertFalse(source["garment_cherry_picking"])
        self.assertFalse(source["failed_garments_hidden"])
        self.assertEqual(len(source["visual_sheets"]), 26)
        for baseline in source["baselines"].values():
            self.assertTrue(baseline["clean"])
            self.assertEqual(baseline["mutation_count"], 0)

    def test_03_asset_registry_hash_closure(self):
        registry = load_json(RISK / "paper_figure_loo_refresh_asset_registry.json")
        self.assertEqual(registry["status"], "PASS")
        self.assertEqual(registry["asset_count"], len(registry["assets"]))
        self.assertGreaterEqual(registry["asset_count"], 14)
        for row in registry["assets"]:
            path = ROOT / row["relative_path"]
            self.assertTrue(path.is_file(), row["relative_path"])
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertEqual(sha256(path), row["sha256"])
            self.assertFalse(row["ai_generated"])
            self.assertFalse(row["scientific_pixel_transform"])

    def test_04_representative_visuals_are_byte_identical_and_complete(self):
        registry = load_json(RISK / "paper_figure_loo_refresh_asset_registry.json")
        rows = [row for row in registry["assets"] if row["role"] == "REPRESENTATIVE_VISUAL_COMPARISON_SHEET"]
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["garment"] for row in rows}, GARMENTS)
        self.assertEqual({row["rotation"] for row in rows}, {"R0"})
        self.assertEqual({row["selection_rule"] for row in rows}, {"ALL_GARMENTS_FIXED_ROTATION_R0_LEXICAL_ORDER"})
        for row in rows:
            self.assertEqual(row["sha256"], row["source_sha256"])
        transforms = load_json(RISK / "paper_figure_loo_refresh_transform_registry.json")
        copies = [row for row in transforms["transforms"] if row["transform_type"] == "BYTE_IDENTICAL_COPY_NO_PIXEL_TRANSFORM"]
        self.assertEqual(len(copies), 5)
        for row in copies:
            self.assertEqual(row["source_sha256"], row["output_sha256"])
            self.assertIsNone(row["crop"])
            self.assertIsNone(row["resize"])

    def test_05_all_required_figure_roles_exist(self):
        registry = load_json(RISK / "paper_figure_loo_refresh_asset_registry.json")
        roles = {row["role"] for row in registry["assets"]}
        required = {
            "LOO_OVERALL_CLASSIFICATION_PANEL",
            "PER_GARMENT_ORACLE_CAPACITY_GAP",
            "K2_VS_HARD_LOOKUP",
            "K1_VS_K2_VIEW_BUDGET",
            "THREE_METHOD_CAPACITY_COMPARISON",
            "SPAN_DISTANCE_CAPACITY_PANEL",
            "LOO_SUCCESS_GATE_SUMMARY",
            "REPRESENTATIVE_VISUAL_COMPARISON_SHEET",
            "BASIS_ROLE_ADJUDICATION",
            "FIGURE2_ENDPOINT_METHOD_FREEZE_CANDIDATE",
        }
        self.assertTrue(required.issubset(roles), required - roles)

    def test_06_figure_state_freeze(self):
        figure_map = load_json(RISK / "paper_figure_map_loo_method_frozen.json")
        self.assertEqual(figure_map["figure_1"]["status"], "REQUIRES_MANUAL_ADJUDICATION")
        self.assertFalse(figure_map["figure_1"]["final_claimed"])
        self.assertEqual(figure_map["figure_2"]["status"], "ENDPOINT_METHOD_FREEZE_CANDIDATE_READY")
        self.assertEqual(figure_map["figure_3"]["status"], "READY_FROM_HISTORICAL_EVIDENCE")
        self.assertEqual(figure_map["figure_4"]["status"], "READY_FROM_HISTORICAL_EVIDENCE")
        self.assertEqual(figure_map["figure_5"]["status"], "HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY")
        self.assertEqual(figure_map["figure_6"]["status"], "LIMITATION_CANDIDATE_READY")
        self.assertEqual(figure_map["headroom"], "CONSUMED_FROM_SEALED_NEGATIVE_RESULT")
        self.assertEqual(figure_map["loo"], "CONSUMED_FROM_SEALED_CAPACITY_LIMITED_RESULT")
        self.assertEqual(figure_map["subject00"], "SECOND_IDENTITY_VALIDATION_PENDING")

    def test_07_main_method_freeze(self):
        method = load_json(RISK / "canondressgs_main_method_freeze.json")
        self.assertEqual(method["formal_basis_role"], "EXPLICIT_TEACHER_DERIVED_ENDPOINT_COORDINATE_SYSTEM")
        self.assertEqual(method["coefficient_semantics"]["raw"], "RAW_PREDICTED_COEFFICIENT")
        self.assertEqual(method["coefficient_semantics"]["realized"], "REALIZED_SNAPPED_ENDPOINT_COEFFICIENT")
        self.assertFalse(method["coefficient_semantics"]["equal"])
        self.assertEqual(method["dual_support_role"], "ORACLE_OR_USER_SPECIFIED_GEOMETRY_SAFE_EXTENSION")
        excluded = set(method["excluded_from_main_pipeline"])
        self.assertIn("LOO Few-View Adaptation", excluded)
        self.assertIn("Full-Residual Optimization", excluded)
        self.assertIn("Spatially Distributed Clothing Coefficients", excluded)
        self.assertIn("Automatic Dual-Support Controller", excluded)

    def test_08_contribution_freeze(self):
        contribution = load_json(RISK / "canondressgs_contribution_freeze.json")
        self.assertEqual(len(contribution["main_contributions"]), 3)
        self.assertIn("Gaussian geometry interpolation", contribution["main_contributions"][1])
        self.assertIn("closed-wardrobe", contribution["main_contributions"][2])
        excluded = set(contribution["not_positive_contributions"])
        self.assertIn("LOO adaptation", excluded)
        self.assertIn("semantic garment manifold", excluded)
        self.assertIn("unseen garment editing", excluded)

    def test_09_current_protocol_tables_are_separated(self):
        table = load_json(RISK / "current_protocol_baseline_table_source.json")
        main = table["main_subject02_table"]
        self.assertFalse(main["historical_51_experiments_included"])
        self.assertEqual(
            [row["paper_name"] for row in main["rows"]],
            [
                "Base Avatar",
                "Teacher Endpoint",
                "Reference Classifier Lookup",
                "Nearest-Centroid Lookup",
                "CanonDressGS Endpoint",
            ],
        )
        analysis = {row["analysis"]: row for row in table["analysis_table"]["rows"]}
        self.assertEqual(set(analysis), {"Geometry causal", "Dual-Support", "Headroom", "LOO"})
        self.assertEqual(analysis["LOO"]["classification"], "LOO_BASIS_CAPACITY_LIMITED")
        self.assertEqual(set(analysis["LOO"]["modes"]), {"deployable_adaptation", "oracle_projection", "hard_lookup"})

    def test_10_loo_scope_adjudication(self):
        scope = load_json(RISK / "loo_method_scope_adjudication.json")
        self.assertEqual(scope["classification"], "LOO_BASIS_CAPACITY_LIMITED")
        self.assertEqual(scope["basis_role"], "ENDPOINT_COORDINATE_SYSTEM")
        self.assertEqual(scope["adaptation_space_role"], "REJECTED_AS_CURRENT_MAIN_METHOD")
        self.assertEqual(scope["oracle_projection"]["status"], "OFFLINE_ORACLE_ONLY")
        self.assertEqual(scope["hard_lookup"]["status"], "CLOSED_WARDROBE_DESCRIPTIVE_RELATION")

    def test_11_execution_boundaries_docs_and_change_scope(self):
        summary = load_json(RISK / "paper_figure_loo_method_freeze_final_summary.json")
        self.assertIn(summary["classification"], ALLOWED_CLASSIFICATIONS)
        self.assertFalse(summary["PAPER_FINAL"])
        self.assertEqual(summary["paper_final_count"], 0)
        self.assertTrue(all(value == 0 for value in summary["resource_counts"].values()))
        self.assertTrue(all(value == 0 for value in summary["mutation_audit"].values()))
        required_docs = {
            "AAAI27_LOO_FIGURE_REFRESH_20260725.md",
            "AAAI27_CANONDRESSGS_MAIN_METHOD_FREEZE_20260725.md",
            "AAAI27_CANONDRESSGS_CONTRIBUTION_FREEZE_20260725.md",
            "AAAI27_CURRENT_PROTOCOL_BASELINE_TABLE_PLAN_20260725.md",
            "AAAI27_ENDPOINT_COORDINATE_SYSTEM_SCOPE_20260725.md",
        }
        self.assertTrue(all((ROOT / "docs/PAPER" / name).is_file() for name in required_docs))
        status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
        changed = [line[3:].replace("\\", "/") for line in status.splitlines()]
        self.assertFalse(any(path.lower().endswith(".tex") for path in changed))
        self.assertFalse(any(path.startswith("paper_draft/") and not path.startswith("paper_draft/figures/loo_method_freeze/") for path in changed))

    def test_12_json_integrity_and_no_checkpoint_payload(self):
        new_json = list(FIGURES.rglob("*.json"))
        new_json.extend(
            RISK / name
            for name in (
                "paper_figure_loo_refresh_source_registry.json",
                "paper_figure_loo_refresh_transform_registry.json",
                "paper_figure_loo_refresh_asset_registry.json",
                "paper_figure_map_loo_method_frozen.json",
                "canondressgs_main_method_freeze.json",
                "canondressgs_contribution_freeze.json",
                "current_protocol_baseline_table_source.json",
                "loo_method_scope_adjudication.json",
                "paper_figure_loo_method_freeze_tests.json",
                "paper_figure_loo_method_freeze_final_summary.json",
            )
        )
        new_json.append(ROOT / "project_control_handoff/paper_figure_loo_method_freeze_handoff.json")
        for path in new_json:
            load_json(path)
        self.assertFalse(any(FIGURES.rglob("*.pt")))
        self.assertFalse(any(FIGURES.rglob("*.pth")))
        self.assertFalse(any(FIGURES.rglob("*.ckpt")))


if __name__ == "__main__":
    unittest.main()
