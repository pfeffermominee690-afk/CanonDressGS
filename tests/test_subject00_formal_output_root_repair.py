from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RISK_ROOT = REPO_ROOT / "paper_protocol/reviewer_risk"
CANONICAL_ROOT = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001"
)
REJECTED_ALIAS = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001"
)
EXECUTION_BRANCH = (
    "research/mmlphuman-subject00-formal-output-root-repaired-run-20260725"
)


def load_strict(path: Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


def test_repair_artifacts_parse_strictly() -> None:
    for name in (
        "subject00_formal_output_root_adjudication.json",
        "subject00_formal_output_root_schema.json",
        "subject00_formal_output_root_reference_audit.json",
        "subject00_formal_execution_contract_repaired.json",
    ):
        load_strict(RISK_ROOT / name)


def test_adjudication_freezes_one_root_without_scientific_drift() -> None:
    adjudication = load_strict(
        RISK_ROOT / "subject00_formal_output_root_adjudication.json"
    )
    assert adjudication["canonical_formal_output_root"] == CANONICAL_ROOT
    assert adjudication["rejected_output_root_alias"] == REJECTED_ALIAS
    assert adjudication["classification"] == (
        "REQUEST_LEVEL_OUTPUT_ROOT_ALIAS_CONFLICT"
    )
    assert adjudication["scientific_semantic_drift"] == 0
    assert adjudication["candidate_roots_pre_execution"] == {
        "canonical_exists": False,
        "rejected_alias_exists": False,
    }
    assert adjudication["write_policy"] == "CANONICAL_ROOT_ONLY"


def test_reference_audit_applies_required_authority_order() -> None:
    audit = load_strict(
        RISK_ROOT / "subject00_formal_output_root_reference_audit.json"
    )
    assert audit["authority_order"] == [
        "SEALED_FORMAL_PROTOCOL",
        "FORMAL_EXECUTION_CONTRACT",
        "PREPARED_RUNNER_TEST_CONSENSUS",
        "NATURAL_LANGUAGE_REQUEST",
    ]
    assert all(
        item["observed_root"] == CANONICAL_ROOT
        for item in audit["authoritative_root_sources"]
    )
    assert audit["request_alias_source"]["observed_root"] == REJECTED_ALIAS


def test_repaired_execution_contract_preserves_frozen_science() -> None:
    contract = load_strict(
        RISK_ROOT / "subject00_formal_execution_contract_repaired.json"
    )
    assert contract["execution_branch"] == EXECUTION_BRANCH
    assert contract["prepared_experiment_head"] == (
        "a0d8e64bd99517589870313ce2c71803cd737f7e"
    )
    assert contract["canonical_formal_output_root"] == CANONICAL_ROOT
    assert contract["rejected_output_root_alias"] == REJECTED_ALIAS
    assert contract["scientific_semantic_drift"] == 0
    assert contract["formal_counts"] == {
        "strict_train_records": 20249,
        "passes": 5,
        "optimizer_steps": 101245,
        "checkpoint_steps": [0, 20249, 40498, 60747, 80996, 100000, 101245],
        "train_cameras": 18,
        "heldout_cameras": 6,
        "train_poses": 1130,
        "heldout_poses": 125,
        "buffer_poses": 1245,
    }
    assert contract["attempt_policy"]["attempt_002_allowed"] is False
    assert contract["initialization"]["step"] == 0
    assert contract["initialization"]["sha256"] == (
        "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
    )


def test_runner_consumes_repaired_contract_and_rejects_alias() -> None:
    source = (
        REPO_ROOT
        / "tools/second_identity/run_subject00_formal_strict_split.py"
    ).read_text(encoding="utf-8")
    assert EXECUTION_BRANCH in source
    assert CANONICAL_ROOT.split("/")[-1] in source
    assert REJECTED_ALIAS.split("/")[-1] in source
    assert "REPAIRED_EXECUTION_CONTRACT_PATH" in source
    assert "scientific drift" in source
    assert "conflicting non-protocol output root exists" in source
