from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBJECT_ROOT = REPO_ROOT / "paper_protocol" / "subject00_formal_base"
STORAGE_ROOT = REPO_ROOT / "paper_protocol" / "storage"
MANIFEST = SUBJECT_ROOT / "subject00_formal_base_101245_execution_manifest_20260726.json"
HANDOFF = SUBJECT_ROOT / "subject00_formal_base_101245_handoff_launch_20260726.json"
AUTH = STORAGE_ROOT / "avatarrex_cloud_duplicate_deletion_authorization_20260726.json"
DELETION = STORAGE_ROOT / "avatarrex_cloud_duplicate_deletion_execution_20260726.json"
WRAPPER = REPO_ROOT / "tools" / "second_identity" / "run_subject00_formal_base_101245.py"
LAUNCHER = REPO_ROOT / "tools" / "second_identity" / "launch_subject00_formal_base_101245.sh"


EXPECTED_BRANCH = "research/subject00-formal-base-101245-execution-20260726"
EXPECTED_OUTPUT_ROOT = "/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001"
EXPECTED_ATTEMPT_ROOT = f"{EXPECTED_OUTPUT_ROOT}/attempt_001"
EXPECTED_SHA = "531bd1c71ad9b35f6ae0e2595ee531aa7ba1f83c242f18f2d0505b2dcd5fbcc1"
EXPECTED_BYTES = 12569755256
EXPECTED_CONFIG_CRLF_SHA = "be28dacbea5a3e4b35336556f752b1548ebb8c3c0b355be8348e405df49bd356"
EXPECTED_CONFIG_LF_SHA = "7ffd81771c3e25e8f4fdef4a7e7e628539642a8107f71c77ce0f748303a572d5"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def config_hashes() -> tuple[str, str]:
    payload = (REPO_ROOT / "config" / "subject00_surface_lbs_formal_strict_split.yaml").read_bytes()
    lf_payload = payload.replace(b"\r\n", b"\n")
    crlf_payload = lf_payload.replace(b"\n", b"\r\n")
    return hashlib.sha256(lf_payload).hexdigest(), hashlib.sha256(crlf_payload).hexdigest()


def main() -> None:
    manifest = load_json(MANIFEST)
    handoff = load_json(HANDOFF)
    auth = load_json(AUTH)
    deletion = load_json(DELETION)

    require(auth["deletion_authorized"] is True, "authorization missing")
    require(auth["deletion_executed"] is False, "authorization record must remain pre-delete")
    require(deletion["deletion_authorized"] is True, "deletion authorization missing")
    require(deletion["deletion_executed"] is True, "deletion execution missing")
    require(deletion["cloud_duplicate"]["target_path"] == "/root/autodl-tmp/avatarrex_lbn1.7z", "delete target changed")
    require(deletion["cloud_duplicate"]["bytes_before_delete"] == EXPECTED_BYTES, "deleted archive size changed")
    require(deletion["cloud_duplicate"]["sha256_before_delete"] == EXPECTED_SHA, "deleted archive SHA changed")
    require(deletion["cloud_duplicate"]["delete_command_executed"] == "rm -- /root/autodl-tmp/avatarrex_lbn1.7z", "delete command changed")
    require(deletion["cloud_duplicate"]["exists_after_delete"] is False, "deleted path not recorded absent")
    if Path("/root/autodl-tmp").exists():
        require(not Path("/root/autodl-tmp/avatarrex_lbn1.7z").exists(), "deleted cloud duplicate still exists")

    require(manifest["execution_branch"] == EXPECTED_BRANCH, "execution branch changed")
    require(manifest["authorized_source_head"] == "e5c98482160b2625a9ac6e8af388bc5350eced46", "source head changed")
    require(manifest["training_authorized"] is True, "training authorization missing")
    require(manifest["optimizer_steps_at_manifest_commit"] == 0, "optimizer steps before launch changed")
    require(manifest["output_root_created_at_manifest_commit"] is False, "output root should not preexist at manifest commit")
    require(manifest["subject"]["valid_rgb_mask_pairs"] == 59704, "valid pair count changed")
    require(manifest["subject"]["missing_pairs"] == 296, "missing pair count changed")
    require(manifest["subject"]["denominator"] == 59704, "denominator changed")

    inputs = manifest["immutable_inputs"]
    require(inputs["data_manifest"]["sha256"] == "e61ca6061f0a85be9c5c6e1d9163341c10749bc1a4ff1b14d27f64b5705aa99e", "data manifest SHA changed")
    require(inputs["calibration"]["sha256"] == "4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7", "calibration SHA changed")
    require(inputs["smpl"]["sha256"] == "ac2738c308ad1a9cc02e7b63323c75e0eab28bddc88c57bb5887e07bf8ea28d2", "smpl SHA changed")
    require(inputs["template"]["sha256"] == "f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031", "template SHA changed")
    require(inputs["surface_lbs_weights"]["file_sha256"] == "56aa68a9d4baade67621fa2bfac462ac88074eeaf7c9bfdbe86f2360b91261a1", "LBS file SHA changed")
    require(inputs["initialization_checkpoint"]["sha256"] == "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a", "init checkpoint SHA changed")
    require(inputs["initialization_checkpoint"]["step"] == 0, "init checkpoint step changed")

    config_lf, config_crlf = config_hashes()
    require(config_lf == EXPECTED_CONFIG_LF_SHA, "config LF/Git SHA changed")
    require(config_crlf == EXPECTED_CONFIG_CRLF_SHA, "config CRLF contract SHA changed")
    require(inputs["config"]["git_blob_and_cloud_lf_sha256"] == EXPECTED_CONFIG_LF_SHA, "manifest config LF SHA changed")
    require(inputs["config"]["windows_crlf_contract_sha256"] == EXPECTED_CONFIG_CRLF_SHA, "manifest config CRLF SHA changed")

    storage = manifest["storage_gate"]
    require(storage["free_bytes"] >= storage["formal_storage_gate_bytes"], "free bytes below formal storage gate")
    require(
        storage["projected_minimum_free_bytes_after_deletion"]
        >= storage["required_projected_minimum_free_bytes_after_deletion"],
        "projected post-delete minimum below required gate",
    )
    require(manifest["output_root_gate"]["output_root"] == EXPECTED_OUTPUT_ROOT, "output root changed")
    require(manifest["output_root_gate"]["attempt_root"] == EXPECTED_ATTEMPT_ROOT, "attempt root changed")
    require(manifest["output_root_gate"]["output_root_exists_before_launch"] is False, "output root preexists")
    if Path("/root/autodl-tmp").exists():
        require(not Path(EXPECTED_OUTPUT_ROOT).exists(), "output root exists before launch")

    wrapper_source = WRAPPER.read_text(encoding="utf-8")
    require("run_subject00_formal_strict_split as formal" in wrapper_source, "wrapper must import formal runner")
    require("medium.train_step" not in wrapper_source, "wrapper must not copy train step")
    require("optimizer.step" not in wrapper_source, "wrapper must not own optimizer stepping")
    require("FORMAL_BASE_OUTPUT_ROOT" in wrapper_source, "wrapper output binding missing")
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    require("subject00_formal_base_101245_001" in launcher_source, "session name missing")
    require("run_subject00_formal_base_101245.py --phase train" in launcher_source, "launch command missing")
    require("progress_step_records.jsonl" in launcher_source, "progress log binding missing")

    require(handoff["training_authorized"] is True, "handoff training authorization missing")
    require(handoff["optimizer_steps_before_launch"] == 0, "handoff optimizer step count changed")
    require(handoff["output_root_created_before_launch"] is False, "handoff output root state changed")
    require(handoff["run_root"] == EXPECTED_ATTEMPT_ROOT, "handoff run root changed")

    print("SUBJECT00_FORMAL_BASE_101245_LAUNCH_GATE_PASS")


if __name__ == "__main__":
    main()
