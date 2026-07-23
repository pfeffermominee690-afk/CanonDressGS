import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "paper_protocol" / "datasets"


def _load(name: str) -> dict:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def test_required_artifacts_exist_and_are_nonempty() -> None:
    paths = [
        ROOT / "docs/DATASET/AVATARREX_LICENSE_CLOSURE_20260724.md",
        ROOT / "docs/SECOND_IDENTITY/AVATARREX_BASE_AVATAR_PREFLIGHT_20260724.md",
        ROOT / "docs/SECOND_IDENTITY/AVATARREX_TEMPLATE_AND_LBS_ROUTE_20260724.md",
        ROOT / "docs/SECOND_IDENTITY/AVATARREX_STRICT_SPLIT_AUDIT_20260724.md",
        ROOT / "project_control_handoff/avatarrex_base_avatar_preflight_handoff.json",
    ]
    json_names = [
        "avatarrex_license_and_citation_contract.json",
        "avatarrex_zero_copy_runtime_adapter.json",
        "avatarrex_calibration_runtime_contract.json",
        "avatarrex_smplx_runtime_contract.json",
        "avatarrex_template_route_audit.json",
        "avatarrex_lbs_route_audit.json",
        "avatarrex_base_avatar_preflight_contract.json",
        "avatarrex_canary_training_contract_draft.json",
        "avatarrex_base_avatar_preflight_final_summary.json",
        "avatarrex_base_avatar_runtime_audit.json",
    ]
    paths.extend(DATASETS / name for name in json_names)

    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    for name in json_names:
        assert isinstance(_load(name), dict)


def test_dataset_license_is_primary_and_blocks_redistribution() -> None:
    contract = _load("avatarrex_license_and_citation_contract.json")
    evidence = {row["path"]: row for row in contract["official_evidence"]}
    decision = contract["license_text_decision"]

    assert evidence["AVATARREX_DATASET.md"]["role"] == "PRIMARY_DATASET_LICENSE_AND_CITATION_EVIDENCE"
    assert evidence["LICENSE"]["role"] == "CODE_LICENSE_ONLY_NOT_A_DATASET_LICENSE_SUBSTITUTE"
    assert contract["governance"]["code_license_equals_dataset_license"] is False
    assert decision["internal_single_site_derived_asset_preparation"].startswith("PERMITTED")
    assert decision["public_raw_data"] == "PROHIBITED"
    assert decision["public_derived_assets"].startswith("PROHIBITED")
    assert decision["public_data_derived_checkpoints"].startswith("TREATED_AS_DERIVED_DATA_AND_PROHIBITED")
    assert decision["written_permission_required_before_public_or_third_party_release"] is True
    assert contract["citation_contract"]["publication_must_cite"] == ["AvatarReX", "Animatable Gaussians"]
    assert contract["web_page_evidence"]["avatarrex_historical_project_domain"]["status"] == (
        "DOMAIN_CONTENT_DRIFT_EXCLUDED_FROM_LICENSE_EVIDENCE"
    )


def test_zero_copy_runtime_and_camera_contracts() -> None:
    adapter = _load("avatarrex_zero_copy_runtime_adapter.json")
    camera = _load("avatarrex_calibration_runtime_contract.json")
    runtime = _load("avatarrex_base_avatar_runtime_audit.json")

    assert adapter["standardization_mode"] == "READ_ONLY_ZERO_COPY_LOADER_ADAPTER"
    assert adapter["immutability"]["full_rgb_copy_count"] == 0
    assert adapter["immutability"]["full_mask_copy_count"] == 0
    assert adapter["immutability"]["raw_mutation"] == 0
    assert adapter["loader_smoke"]["decoded_record_count"] == 9
    assert camera["camera_count"] == 16
    assert camera["raw_extrinsic_convention"].startswith("x_camera = R @ x_world + T")
    assert camera["imgSize_order"] == "width_height"
    assert camera["distortion"]["all_coefficients_zero"] is True
    assert all(row["status"] == "PASS" for row in camera["camera_rows"])
    assert runtime["status"] == "DATASET_AND_PARAMETER_ADAPTER_SMOKE_PASS"
    assert runtime["blocker"] == "TEMPLATE_AND_LBS_ASSETS_REQUIRED"
    assert runtime["mutation_audit"]["raw_tree_metadata_unchanged"] is True


def test_smpl_schema_neutral_compatibility_and_unknown_gender() -> None:
    contract = _load("avatarrex_smplx_runtime_contract.json")

    assert contract["status"] == "PASS"
    assert contract["gender"] == "UNKNOWN"
    assert contract["neutral_model_compatibility"]["status"] == "PASS"
    assert contract["neutral_model_compatibility"]["does_not_establish_gender"] is True
    assert contract["neutral_model_compatibility"]["vertex_count"] == 10475
    assert contract["neutral_model_compatibility"]["face_count"] == 20908
    assert contract["mmlp_mapping"]["jaw_pose_consumed_by_current_mmlp_body_deformation"] is False
    assert contract["mmlp_mapping"]["expression_consumed_by_current_mmlp_body_deformation"] is False


def test_template_and_lbs_routes_do_not_claim_missing_assets() -> None:
    template = _load("avatarrex_template_route_audit.json")
    lbs = _load("avatarrex_lbs_route_audit.json")

    assert template["status"] == "ROUTE_SELECTED_ASSET_NOT_ACQUIRED"
    assert template["recommended_route"]["candidate"] == "A_OFFICIAL_PREPROCESSED_LOOSE_CLOTHING_TEMPLATE"
    assert template["expected_hash_path_contract"]["official_archive_expected_sha256"] is None
    assert template["expected_hash_path_contract"]["template_sha256"] is None
    assert template["counts"] == {"large_downloads": 0, "raw_mutation": 0, "template_generation": 0}
    body_fallback = next(row for row in template["candidates"] if row["id"] == "C_SMPLX_BODY_SURFACE_FALLBACK")
    assert body_fallback["equivalent_to_loose_clothing_template"] is False
    assert body_fallback["admitted_use"] == "CANARY_ONLY_AFTER_EXPLICIT_FALLBACK_APPROVAL"

    assert lbs["status"] == "ROUTE_SELECTED_ASSET_NOT_GENERATED"
    assert lbs["recommended_route"]["candidate"] == "MMLP_POINTINTERPOLANT_VOLUME"
    assert lbs["gaussian_lifecycle"]["clone_split_densification_in_current_train_path"].startswith("ABSENT")
    assert lbs["counts"]["lbs_generation"] == 0
    assert lbs["counts"]["training"] == 0


def test_strict_split_regeneration_and_availability() -> None:
    split = _load("avatarrex_camera_pose_split.json")
    runtime = _load("avatarrex_base_avatar_runtime_audit.json")["strict_split_audit"]

    assert split["camera_split"]["train_count"] == 12
    assert split["camera_split"]["heldout_camera_ids"] == [
        "camera_01",
        "camera_02",
        "camera_09",
        "camera_11",
    ]
    assert split["camera_split"]["overlap"] == []
    assert split["pose_split"]["train_count"] == 856
    assert split["pose_split"]["heldout_count"] == 95
    assert split["pose_split"]["buffer_excluded_count"] == 950
    assert split["pose_split"]["heldout_pair_min_temporal_distance"] == 11
    assert split["pose_split"]["train_heldout_overlap"] == []
    assert split["pose_split"]["train_buffer_overlap"] == []
    assert split["pose_split"]["temporal_leakage"] == 0
    assert all(runtime["checks"].values())
    assert runtime["frozen_split_content_sha256"] == runtime["regenerated_split_sha256"]
    assert set(runtime["per_pose_valid_camera_counts"]["train_poses"].values()) == {856, 16}
    assert set(runtime["per_pose_valid_camera_counts"]["heldout_poses"].values()) == {95, 16}


def test_canary_draft_has_no_defaults_or_split_leakage() -> None:
    canary = _load("avatarrex_canary_training_contract_draft.json")
    split = _load("avatarrex_camera_pose_split.json")
    subset = canary["deterministic_training_subset"]
    records = subset["records"]
    train_frames = set(split["pose_split"]["train_frame_ids"])
    buffer_frames = set(split["pose_split"]["buffer_excluded_frame_ids"])
    heldout_frames = set(split["pose_split"]["heldout_frame_ids"])
    train_cameras = set(split["camera_split"]["train_camera_ids"])

    assert canary["status"] == "PENDING_DERIVED_ASSET_AND_RUNTIME_CLOSURE"
    assert canary["execution_authorized"] is False
    assert canary["step_budget"]["value"] is None
    assert canary["checkpoint_cadence"]["value"] is None
    assert canary["optimization_signal_thresholds"]["values"] is None
    assert subset["record_count"] == subset["unique_record_count"] == 64
    assert len({(row["canonical_camera_id"], row["frame_id"]) for row in records}) == 64
    assert all(row["canonical_camera_id"] in train_cameras for row in records)
    assert all(row["frame_id"] in train_frames for row in records)
    assert not {row["frame_id"] for row in records} & buffer_frames
    assert not {row["frame_id"] for row in records} & heldout_frames
    assert canary["rerun_policy"]["result_driven_reruns"] == "FORBIDDEN"
    assert canary["claim_boundary"]["formal_claim"] is False


def test_preflight_classification_and_all_forbidden_counts() -> None:
    preflight = _load("avatarrex_base_avatar_preflight_contract.json")
    summary = _load("avatarrex_base_avatar_preflight_final_summary.json")
    runtime = _load("avatarrex_base_avatar_runtime_audit.json")
    counts = runtime["mutation_audit"]

    assert preflight["classification"] == "AVATARREX_BASE_AVATAR_PREFLIGHT_READY_FOR_DERIVED_ASSETS"
    assert preflight["admission"]["derived_asset_preparation"] is True
    assert preflight["admission"]["short_canary"] is False
    assert preflight["admission"]["formal_training"] is False
    assert preflight["gates"]["model_construction"] == "PENDING_TEMPLATE_AND_LBS_ASSETS"
    assert preflight["gates"]["zero_step_render"] == "PENDING_TEMPLATE_AND_LBS_ASSETS"
    for key in (
        "large_downloads",
        "full_rgb_copies",
        "full_mask_copies",
        "template_generation",
        "lbs_generation",
        "training_runs",
        "backward_calls",
        "optimizer_created",
        "checkpoint_writes",
        "formal_renderer_runs",
        "image_generation_api_calls",
        "raw_mutation",
        "paper_final",
    ):
        assert counts[key] == 0
    assert summary["classification"] == preflight["classification"]
    assert summary["paper_final"] == 0
    assert summary["next_task"] == "PREPARE_AVATARREX_LBN1_BASE_AVATAR_DERIVED_ASSETS_WITHOUT_TRAINING"
    assert summary["next_task_automatic_execution"] is False
