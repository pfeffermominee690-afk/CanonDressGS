import json
import subprocess
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "paper_protocol/reviewer_risk"
FIGURES = ROOT / "paper_draft/figures/headroom_refresh"
PARENT = "1fe425d2cc3cb3efd372334e1d845e63bf9d630a"
HEADROOM = "674e6092e21eeddeb22e963536247a3385c4e200"


def load_json(path: Path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


class HeadroomFigureRefreshTest(unittest.TestCase):
    def test_source_gate(self):
        gate = load_json(RISK / "paper_figure_headroom_source_gate.json")
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["classification"], "TEACHER_SPAN_AT_LOCAL_OPTIMUM")
        self.assertEqual(gate["teacher_svd_parity"], "PASS")
        self.assertEqual(gate["headroom_source_head"], HEADROOM)
        self.assertEqual(gate["actual_counts"], {
            "checkpoint_writes": 960,
            "optimization_runs": 120,
            "optimizer_steps": 36000,
            "renderer_calls": 36662,
        })
        self.assertTrue(gate["branch_pushed_and_clean"])

    def test_no_compute_or_loo_read(self):
        gate = load_json(RISK / "paper_figure_headroom_source_gate.json")
        for key in ("gpu_used",):
            self.assertFalse(gate[key])
        for key in ("training_runs_started", "inference_runs_started", "renderer_calls_started", "checkpoints_written", "checkpoint_payload_files_read", "loo_attempt_files_read"):
            self.assertEqual(gate[key], 0)

    def test_append_only_parent(self):
        output = subprocess.check_output(["git", "diff", "--name-status", PARENT], cwd=ROOT, text=True)
        for line in output.splitlines():
            self.assertTrue(line.startswith("A\t"), line)

    def test_source_manifest_and_sheet_hashes(self):
        manifest = load_json(RISK / "paper_figure_headroom_source_manifest.json")
        self.assertEqual(manifest["source_head"], HEADROOM)
        self.assertEqual(len(manifest["visual_sheet_sources"]), 20)
        self.assertEqual(manifest["checkpoint_payload_files_read"], 0)
        self.assertEqual(manifest["loo_attempt_files_read"], 0)
        for row in manifest["visual_sheet_sources"]:
            self.assertEqual(len(row["sha256"]), 64)
            self.assertTrue(row["sealed_path"].startswith("/root/autodl-tmp/"))

    def test_required_plot_set(self):
        registry = load_json(RISK / "paper_figure_headroom_plot_registry.json")
        required = {
            "teacher_svd_parity",
            "four_method_test_metrics",
            "lpips_gate_summary",
            "per_garment_teacher_minus_refined_lpips",
            "coefficient_displacement_trajectory",
            "full_residual_artifact_examples",
            "span_recovery_denominator_null",
            "refined_lookup_decomposition",
            "supplementary_negative_diagnostic_v1",
        }
        self.assertEqual(registry["plot_count"], 9)
        self.assertEqual({row["plot_id"] for row in registry["plots"]}, required)
        for row in registry["plots"]:
            self.assertEqual(row["classification"], "SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC")
            self.assertEqual(row["paper_final"], 0)
            path = ROOT / row["output"]
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                image.verify()

    def test_figure_states(self):
        figure_map = load_json(RISK / "paper_figure_map_headroom_refreshed.json")
        self.assertEqual(figure_map["figure_2"]["status"], "METHOD_FREEZE_PENDING_LOO")
        self.assertEqual(figure_map["headroom"], "CONSUMED_FROM_SEALED_RESULT")
        self.assertEqual(figure_map["loo"], "PENDING_LOO_ADAPTATION")
        self.assertEqual(figure_map["spatial_coefficient_field"], "CONTINGENCY_ONLY_NOT_STARTED")
        self.assertEqual(figure_map["paper_final"], 0)

    def test_ppt_index_without_ppt(self):
        index = load_json(RISK / "ppt_material_index_headroom_refresh.json")
        self.assertFalse(index["ppt_generated"])
        self.assertGreaterEqual(len(index["materials"]), 8)
        for item in index["materials"]:
            self.assertFalse(item["main_paper_candidate"])
            self.assertTrue(item["negative_diagnostic"])
            self.assertTrue(item["question"])
            self.assertTrue(item["recommended_slide_title"])
            self.assertEqual(len(item["source_sha256"]), 64)
        self.assertFalse(any(ROOT.rglob("*.ppt")))
        self.assertFalse(any(ROOT.rglob("*.pptx")))

    def test_caption_and_reports(self):
        caption = (FIGURES / "candidates/supplementary/headroom_negative_diagnostic_v1/caption_skeleton.md").read_text(encoding="utf-8")
        reports = [
            ROOT / "docs/PAPER/AAAI27_HEADROOM_FIGURE_REFRESH_20260724.md",
            ROOT / "docs/PAPER/AAAI27_HEADROOM_NEGATIVE_DIAGNOSTIC_FIGURE_PLAN_20260724.md",
            ROOT / "docs/PAPER/AAAI27_HEADROOM_PPT_MATERIAL_INDEX_20260724.md",
        ]
        text = caption + "\n" + "\n".join(path.read_text(encoding="utf-8") for path in reports)
        for phrase in ("did not improve the Teacher Endpoint", "patch/cloud/mottle", "does not support adding render refinement", "PAPER_FINAL=0"):
            self.assertIn(phrase, text)

    def test_mutation_audit_and_summary(self):
        audit = load_json(RISK / "paper_figure_headroom_mutation_audit.json")
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["original_figure_bank_mutation_count"], 0)
        self.assertEqual(audit["pure_endpoint_refresh_mutation_count"], 0)
        self.assertEqual(audit["headroom_source_mutation_count"], 0)
        summary = load_json(RISK / "paper_figure_headroom_refresh_final_summary.json")
        self.assertEqual(summary["classification"], "HEADROOM_FIGURE_REFRESH_READY")
        self.assertFalse(summary["paper_final"])
        self.assertEqual(summary["paper_final_count"], 0)
        self.assertFalse(summary["ppt_generated"])

    def test_json_duplicate_keys_and_no_checkpoint_payload(self):
        paths = list(FIGURES.rglob("*.json"))
        paths += list(RISK.glob("*headroom*refresh*.json"))
        paths += [RISK / "paper_figure_map_headroom_refreshed.json", RISK / "paper_figure_pending_refresh_registry_headroom.json"]
        for path in paths:
            load_json(path)
        self.assertFalse(any(FIGURES.rglob("*.pth")))
        self.assertFalse(any(FIGURES.rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
