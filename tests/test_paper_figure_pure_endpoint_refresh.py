from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "paper_draft/figures/pure_endpoint_refresh"
RISK_ROOT = ROOT / "paper_protocol/reviewer_risk"
PARENT_HEAD = "06261aea35d26006db250df65707d1a09a0bf098"


def load_json(path: Path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PureEndpointFigureRefreshTest(unittest.TestCase):
    def test_duplicate_key_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"key": 1, "key": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                load_json(path)

    def test_all_refresh_json_parses_without_duplicate_keys(self):
        paths = list(FIGURE_ROOT.rglob("*.json"))
        paths += list(RISK_ROOT.glob("paper_figure_*pure_endpoint*.json"))
        paths += list(RISK_ROOT.glob("paper_figure_pending_refresh_registry_amended.json"))
        self.assertGreater(len(paths), 20)
        for path in paths:
            load_json(path)

    def test_append_only_against_parent(self):
        output = subprocess.check_output(
            ["git", "diff", "--name-status", PARENT_HEAD], cwd=ROOT, text=True
        )
        for line in output.splitlines():
            self.assertTrue(line.startswith("A\t"), line)

    def test_asset_registry_closure(self):
        registry = load_json(
            RISK_ROOT / "paper_figure_asset_registry_pure_endpoint_refresh.json"
        )
        self.assertEqual(len(registry["assets"]), 3046)
        self.assertEqual(len(registry["derived_assets"]), 57)
        source_ids = {item["asset_id"] for item in registry["assets"]}
        self.assertEqual(len(source_ids), 3046)
        for item in registry["derived_assets"]:
            path = ROOT / item["original_path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, item["original_bytes"])
            self.assertEqual(sha256(path), item["original_sha256"])

    def test_candidate_provenance_closure(self):
        assets = load_json(
            RISK_ROOT / "paper_figure_asset_registry_pure_endpoint_refresh.json"
        )
        candidates = load_json(
            RISK_ROOT / "paper_figure_candidate_registry_pure_endpoint_refresh.json"
        )
        source_ids = {item["asset_id"] for item in assets["assets"]}
        self.assertEqual(len(candidates["candidates"]), 4)
        self.assertEqual(candidates["paper_final"], 0)
        for candidate in candidates["candidates"]:
            self.assertTrue(set(candidate["source_asset_ids"]).issubset(source_ids))
            panel = ROOT / candidate["repo_panel_path"]
            self.assertTrue(panel.is_file())
            self.assertEqual(sha256(panel), candidate["panel_sha256"])
            self.assertEqual(candidate["provenance_status"], "CLOSED")

    def test_plot_format_and_semantics(self):
        plot_root = FIGURE_ROOT / "plots/pure_endpoint"
        manifest = load_json(plot_root / "metric_plot_manifest.json")
        self.assertEqual(len(manifest["plots"]), 8)
        for plot in manifest["plots"]:
            plot_id = plot["plot_id"]
            for suffix in ("png", "svg", "pdf", "source.json"):
                self.assertTrue((plot_root / f"{plot_id}.{suffix}").is_file())
        endpoint = load_json(plot_root / "endpoint_exact_match.source.json")
        self.assertEqual(
            set(endpoint["not_applicable"]), {"Base Avatar", "Teacher Endpoint"}
        )
        raw = load_json(plot_root / "raw_vs_realized_coefficient_error.source.json")
        self.assertEqual(raw["series"][0]["values"][0]["y"], 71.44514598846436)
        self.assertEqual(raw["series"][1]["values"][1]["y"], 0.0)
        self.assertEqual(raw["endpoint_exact_match"]["Linear continuous"], 0.0)
        self.assertEqual(raw["endpoint_exact_match"]["Canon endpoint snap"], 1.0)
        flips = load_json(plot_root / "endpoint_flip_by_perturbation.source.json")
        self.assertIn("complete_dropout", flips["not_applicable"])
        overlap = load_json(plot_root / "hard_lookup_error_overlap_perturbation.source.json")
        values = [[value["y"] for value in series["values"]] for series in overlap["series"]]
        self.assertEqual(values, [[318, 312, 318], [0, 6, 0], [13, 0, 42], [29, 42, 0]])

    def test_png_svg_pdf_decode(self):
        for path in FIGURE_ROOT.rglob("*.png"):
            with Image.open(path) as image:
                image.load()
                self.assertGreater(image.width * image.height, 0)
        for path in FIGURE_ROOT.rglob("*.svg"):
            self.assertEqual(ET.parse(path).getroot().tag.rsplit("}", 1)[-1], "svg")
        for path in FIGURE_ROOT.rglob("*.pdf"):
            self.assertGreater(path.stat().st_size, 1024)
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))

    def test_transform_safety(self):
        registry = load_json(
            RISK_ROOT / "paper_figure_transform_registry_pure_endpoint_refresh.json"
        )
        self.assertEqual(len(registry["transforms"]), 20)
        for transform in registry["transforms"]:
            self.assertFalse(transform["asymmetric_crop"])
            self.assertFalse(transform["method_specific_enhancement"])
            self.assertFalse(transform["ai_generated"])

    def test_contact_and_candidate_counts(self):
        contacts = load_json(
            RISK_ROOT / "paper_figure_pure_endpoint_contact_sheet_registry.json"
        )
        self.assertEqual(contacts["contact_sheet_count"], 8)
        self.assertEqual(len(list((FIGURE_ROOT / "contact_sheets/pure_endpoint").glob("*.png"))), 8)
        candidate_pngs = list((FIGURE_ROOT / "candidates").rglob("*.png"))
        self.assertEqual(len(candidate_pngs), 4)

    def test_map_and_pending_states(self):
        figure_map = load_json(
            RISK_ROOT / "paper_figure_map_pure_endpoint_refreshed.json"
        )
        self.assertEqual(figure_map["figure_2"]["status"], "METHOD_FREEZE_PENDING_HEADROOM_AND_LOO")
        self.assertEqual(figure_map["figure_3"]["status"], "READY_FROM_HISTORICAL_EVIDENCE")
        self.assertEqual(figure_map["figure_4"]["status"], "READY_FROM_HISTORICAL_EVIDENCE")
        self.assertEqual(
            figure_map["headroom"],
            "PENDING_COEFFICIENT_HEADROOM_ACTIVE_OUTPUT_EXCLUDED",
        )
        self.assertEqual(figure_map["loo"], "PENDING_LOO_ADAPTATION")
        self.assertEqual(figure_map["paper_final"], 0)

    def test_claim_boundary(self):
        claims = load_json(
            RISK_ROOT / "paper_figure_claim_boundary_pure_endpoint_refresh.json"
        )
        self.assertEqual(claims["status"], "PASS")
        allowed = " ".join(claims["allowed_claims"]).lower()
        for forbidden in claims["forbidden_claims"]:
            self.assertNotIn(forbidden.lower(), allowed)

    def test_repo_binary_budget(self):
        binaries = [
            path for path in FIGURE_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg"}
        ]
        self.assertLessEqual(sum(path.stat().st_size for path in binaries), 50 * 1024 * 1024)
        self.assertLessEqual(max(path.stat().st_size for path in binaries), 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
