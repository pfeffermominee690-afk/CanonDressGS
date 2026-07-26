#!/usr/bin/env python3
"""Contract tests for the GS-VTON license/data-conversion preflight."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "AAAI27-GSVTON-LICENSE-DATA-CONVERSION-PREFLIGHT-001"
SOURCE_BRANCH = "research/external-baseline-feasibility-from-bundle-rerun-20260726"
SOURCE_HEAD = "572637f08aef21fde35dcd7f8879ff74549c2afc"
NEW_BRANCH = "research/gsvton-license-data-conversion-preflight-20260726"
PINNED_COMMIT = "96964b0a6528089123cc27a3ff3e3eb46505cf6e"
OFFICIAL_REPO = "https://github.com/yukangcao/GS-VTON"
OFFICIAL_SOURCE = Path(r"E:\model_train\_gsvton_official_source_preflight")

OUT = ROOT / "artifacts" / "gsvton_preflight" / TASK_ID
LICENSE_PATH = OUT / "gsvton_license_audit.json"
DEPENDENCY_PATH = OUT / "gsvton_dependency_matrix.json"
WEIGHTS_PATH = OUT / "gsvton_required_weights_manifest.json"
CONVERSION_PATH = OUT / "gsvton_subject02_conversion_manifest_draft.json"
REPORT_PATH = OUT / "GSVTON_LICENSE_AND_USAGE_RISK_REPORT_20260726.md"
TEST_REPORT_PATH = OUT / "gsvton_preflight_test_results.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace"))
    return proc.stdout.decode(errors="replace").strip()


class GSVTONPreflightContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.license = load(LICENSE_PATH)
        cls.deps = load(DEPENDENCY_PATH)
        cls.weights = load(WEIGHTS_PATH)
        cls.conversion = load(CONVERSION_PATH)
        cls.report = REPORT_PATH.read_text(encoding="utf-8")

    def test_01_source_branch_head(self):
        self.assertEqual(self.license["source_branch"], SOURCE_BRANCH)
        self.assertEqual(self.license["source_head"], SOURCE_HEAD)
        self.assertEqual(self.license["new_branch"], NEW_BRANCH)
        self.assertEqual(git(ROOT, "branch", "--show-current"), NEW_BRANCH)
        self.assertEqual(git(ROOT, "cat-file", "-t", SOURCE_HEAD), "commit")
        self.assertEqual(
            subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE_HEAD, "HEAD"], cwd=ROOT).returncode,
            0,
        )

    def test_02_official_repo(self):
        self.assertEqual(self.license["official_repository"], OFFICIAL_REPO)
        self.assertTrue(self.license["official_source_policy"]["official_source_only"])
        self.assertTrue(OFFICIAL_SOURCE.exists())
        self.assertEqual(git(OFFICIAL_SOURCE, "remote", "get-url", "origin"), OFFICIAL_REPO + ".git")

    def test_03_pinned_commit(self):
        self.assertEqual(self.license["pinned_commit"], PINNED_COMMIT)
        self.assertEqual(git(OFFICIAL_SOURCE, "rev-parse", "HEAD"), PINNED_COMMIT)
        self.assertEqual(self.license["pinned_commit_metadata"]["subject"], "update accepted ijcv")

    def test_04_official_source_only(self):
        policy = self.license["official_source_policy"]
        self.assertTrue(policy["shallow_checkout_used"])
        self.assertTrue(policy["checked_out_exact_commit"])
        self.assertFalse(policy["non_official_forks_used"])
        self.assertFalse(policy["external_source_modified"])

    def test_05_license_search_completeness(self):
        self.assertTrue(all(self.license["license_search_checklist"].values()))
        self.assertTrue(all(not row["found"] for row in self.license["root_license_files"]))
        self.assertGreaterEqual(len(self.license["nested_license_files"]), 6)
        decision = self.license["official_code_license_decision"]
        self.assertEqual(decision["license_status"], "LICENSE_NOT_SPECIFIED")
        self.assertFalse(decision["execution_authorized_by_license"])

    def test_06_dependency_parse(self):
        self.assertEqual(self.deps["dependency_status"], "PARSED_DRAFT_ONLY_ENV_NOT_CREATED")
        self.assertEqual(self.deps["python_requirement"], "3.8")
        self.assertIn("2.2.1", self.deps["pytorch_requirement"])
        self.assertEqual(self.deps["cuda_requirement"], "11.8")
        groups = {row["group"] for row in self.deps["dependency_groups"]}
        self.assertTrue({"base_runtime", "diffusion_lora_and_vton", "3dgs_and_cuda_extensions"}.issubset(groups))

    def test_07_required_weight_manifest(self):
        self.assertEqual(self.weights["required_weight_count"], 10)
        self.assertEqual(len(self.weights["entries"]), 10)
        self.assertEqual(self.weights["known_public_lower_bound_bytes"], 51622353397)
        self.assertTrue(all(row["download_bytes_this_task"] == 0 for row in self.weights["entries"]))

    def test_08_zero_download_bytes(self):
        for artifact in [self.license, self.deps, self.weights, self.conversion]:
            self.assertEqual(artifact["operations"]["checkpoint_download_bytes"], 0)
            self.assertEqual(artifact["operations"].get("dataset_download_bytes", 0), 0)

    def test_09_subject02_field_mapping_completeness(self):
        self.assertEqual(self.conversion["status"], "DRAFT_FIELD_MAPPING_COMPLETE_NO_DATA_MUTATION")
        fields = {row["gsvton_required_field"] for row in self.conversion["field_mappings"]}
        required = {"cloth_path", "DATA/images target view", "target foreground mask for target-space metrics", "DATA camera/cameras.json", "DATA pose/images.txt binding", "gs_source"}
        self.assertTrue(required.issubset(fields))
        for key in ["garment_reference", "target_rgb", "target_mask", "pose", "camera"]:
            self.assertIn("sha256", self.conversion["source_assets"][key])

    def test_10_no_target_leakage(self):
        assets = self.conversion["source_assets"]
        self.assertNotEqual(assets["garment_reference"]["sha256"], assets["target_rgb"]["sha256"])
        self.assertNotEqual(assets["garment_reference"]["sha256"], assets["target_mask"]["sha256"])
        self.assertEqual(self.conversion["target_leakage_status"], "PASS_NO_TARGET_RGB_OR_MASK_USED_AS_GARMENT_REFERENCE")

    def test_11_camera_conversion_specification(self):
        self.assertEqual(self.conversion["camera_conversion_status"], "SPECIFIED_NOT_EXECUTED")
        camera = self.conversion["source_assets"]["camera"]
        self.assertEqual(camera["view_name"], "right")
        self.assertEqual(camera["fov_degrees"], 28.0)
        self.assertEqual(camera["renderer"], "pytorch3d_FoVPerspectiveCameras")

    def test_12_mask_semantics(self):
        self.assertEqual(self.conversion["mask_semantics_status"], "SPECIFIED_FOREGROUND_NOT_HUMAN_PARSE_LABELS")
        self.assertIn("not GS-VTON human parsing labels", self.conversion["source_assets"]["target_mask"]["mask_semantics"])

    def test_13_static_3dgs_initialization_decision(self):
        static = self.conversion["static_3dgs_initialization_requirement"]
        self.assertIn("point_cloud/iteration_30000/point_cloud.ply", static["required_format"])
        self.assertEqual(static["mmlphuman_initialization_compatibility"], "RETRAIN_STATIC_3DGS_REQUIRED")
        self.assertEqual(static["required_adapter"], "NO_DIRECT_ADAPTER_EXECUTED; field mapping alone insufficient")

    def test_14_storage_estimate_completeness(self):
        estimate = self.weights["storage_estimate"]
        keys = {
            "checkpoint_known_public_lower_bound_bytes",
            "environment_estimated_bytes",
            "converted_subject02_logical_referenced_bytes",
            "static_3dgs_initialized_subject_root_lower_bound_bytes",
            "gs_vton_stage1_stage2_outputs_lower_bound_bytes",
            "compiled_extensions_source_and_build_cache_lower_bound_bytes",
            "safety_reserve_bytes",
            "estimated_total_storage_bytes",
        }
        self.assertTrue(keys.issubset(estimate))
        self.assertEqual(estimate["converted_subject02_logical_referenced_bytes"], 374573)
        self.assertGreaterEqual(estimate["estimated_total_storage_bytes"], 93791543234)

    def test_15_one_reference_protocol(self):
        protocol = self.conversion["micro_canary_protocol"]
        self.assertEqual(protocol["garment"], "O03")
        self.assertEqual(protocol["reference_count"], 1)
        self.assertEqual(protocol["reference_top1"], 0.95)
        self.assertFalse(protocol["hyperparameter_sweep"])

    def test_16_no_endpoint_lpips(self):
        self.assertTrue(self.conversion["micro_canary_protocol"]["endpoint_lpips_forbidden"])
        self.assertTrue(self.conversion["target_space_metrics"]["endpoint_lpips_forbidden"])
        self.assertNotIn("Endpoint LPIPS allowed", self.report)

    def test_17_training_zero(self):
        for artifact in [self.license, self.deps, self.weights, self.conversion]:
            self.assertEqual(artifact["operations"]["training_steps"], 0)

    def test_18_inference_zero(self):
        for artifact in [self.license, self.deps, self.weights, self.conversion]:
            self.assertEqual(artifact["operations"]["renderer_inferences"], 0)

    def test_19_data_mutation_zero(self):
        for artifact in [self.license, self.deps, self.weights, self.conversion]:
            self.assertEqual(artifact["operations"]["data_mutations"], 0)
        self.assertEqual(self.conversion["output_bytes_created"], 0)

    def test_20_paper_modification_zero(self):
        for artifact in [self.license, self.deps, self.weights, self.conversion]:
            self.assertEqual(artifact["operations"]["paper_modifications"], 0)
            self.assertFalse(artifact["operations"]["paper_final"])
        self.assertEqual(git(ROOT, "diff", "--name-only", SOURCE_HEAD, "--", "paper_draft"), "")

    def test_21_final_classification(self):
        for artifact in [self.license, self.conversion]:
            self.assertEqual(artifact["final_classification"], "GSVTON_MICRO_CANARY_PREFLIGHT_MULTIPLE_BLOCKERS")

    def test_22_next_task_uniqueness(self):
        next_task = "USER_OBTAIN_GSVTON_AUTHOR_PERMISSION_OR_SELECT_FULL_AVATAR_ONLY"
        self.assertEqual(self.license["next_task"], next_task)
        self.assertEqual(self.conversion["next_task"], next_task)
        self.assertNotIn("next_tasks", self.license)
        self.assertNotIn("next_tasks", self.conversion)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GSVTONPreflightContractTests)
    test_ids = [case.id() for case in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failures = {case.id() for case, _ in result.failures + result.errors}
    report = {
        "schema_version": "canondressgs.gsvton_preflight_test_results.v1",
        "task_id": TASK_ID,
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "test_count": result.testsRun,
        "pass_count": result.testsRun - len(failures),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "contract_checks": [
            {"name": test_id.split(".")[-1], "status": "FAIL" if test_id in failures else "PASS"}
            for test_id in test_ids
        ],
        "operations": {
            "checkpoint_download_bytes": 0,
            "dataset_download_bytes": 0,
            "training_steps": 0,
            "renderer_inferences": 0,
            "data_mutations": 0,
            "paper_modifications": 0,
            "paper_final": False,
        },
        "final_classification": "GSVTON_MICRO_CANARY_PREFLIGHT_MULTIPLE_BLOCKERS",
        "next_task": "USER_OBTAIN_GSVTON_AUTHOR_PERMISSION_OR_SELECT_FULL_AVATAR_ONLY",
        "paper_final": False,
    }
    TEST_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
