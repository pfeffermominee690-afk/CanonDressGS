from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


CACHE_KEY_SCHEMA = "canondressgs.paper.loo_cache_key.v2"
PLAN_SCHEMA = "canondressgs.paper.loo_cache_key_plan.v2"
HARD_LOOKUP = "REFERENCE_NEAREST_HARD_LOOKUP"

CACHE_IDENTITY_FIELDS = (
    "split",
    "held_out_garment",
    "condition",
    "checkpoint_or_final_state_role",
    "selected_known_endpoint",
    "basis_hash",
    "coefficient_or_residual_state_identity",
    "camera_metadata_sha256",
    "pose_metadata_sha256",
    "renderer_contract_sha256",
    "background_identity",
    "resolution_identity",
    "output_semantic_role",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def cache_key_v2(request: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": CACHE_KEY_SCHEMA,
        **{field: request[field] for field in CACHE_IDENTITY_FIELDS},
    }
    return canonical_sha256(identity), identity


def _selected_rows(replay: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {str(row["task_id"]): row for row in replay["rows"]}
    if len(rows) != 40:
        raise ValueError("hard-lookup replay must contain exactly 40 task rows")
    return rows


def _static_state_identity(method: str, task: Mapping[str, Any], endpoint: str) -> str:
    held_out = str(task["held_out_garment"])
    if method == HARD_LOOKUP:
        return f"STATIC/{method}/KNOWN_ENDPOINT/{endpoint}"
    return f"STATIC/{method}/HELD_OUT/{held_out}"


def _basis_hash(method: str, task: Mapping[str, Any]) -> str:
    basis_methods = {
        "ORACLE_PROJECTION_LOO_BASIS",
        "CONVEX_COMBINATION_ORACLE",
        "FEW_VIEW_LOW_DIMENSIONAL_ADAPTATION",
        "ZERO_COEFFICIENT_INITIALIZATION_DIAGNOSTIC",
    }
    if method in basis_methods:
        return str(task["basis_input_contract_sha"])
    return "NOT_APPLICABLE"


def build_cache_key_plan(
    *,
    tasks: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    expected_counts: Mapping[str, int],
    static_methods: Sequence[str],
    optimizer_methods: Sequence[str],
    milestones: Sequence[int],
    condition_metadata: Mapping[str, Mapping[str, Any]],
    renderer_contract_sha256: str,
    background_identity: str,
    resolution_identity: str,
    include_runtime_index: bool = False,
) -> dict[str, Any]:
    replay_rows = _selected_rows(replay)
    request_digest = hashlib.sha256()
    key_digest = hashlib.sha256()
    key_counts: Counter[str] = Counter()
    first_request_for_key: dict[str, str] = {}
    first_identity_for_key: dict[str, dict[str, Any]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    per_track_logical: Counter[str] = Counter()
    per_track_keys: dict[str, set[str]] = defaultdict(set)
    phase_counts: Counter[str] = Counter()
    budget_counts: Counter[str] = Counter()
    static_index: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    runtime_index: dict[str, str] = {}
    logical_count = 0

    def register(request: dict[str, Any]) -> None:
        nonlocal logical_count
        cache_key, render_identity = cache_key_v2(request)
        request["cache_key_v2"] = cache_key
        request["render_identity_sha256"] = canonical_sha256(render_identity)
        encoded = _canonical_bytes(request)
        request_digest.update(encoded + b"\n")
        key_digest.update(cache_key.encode("ascii") + b"\n")
        logical_count += 1
        logical_request_id = str(request["logical_request_id"])
        if logical_request_id in runtime_index:
            raise RuntimeError(f"duplicate logical request id: {logical_request_id}")
        runtime_index[logical_request_id] = cache_key
        phase_counts[str(request["phase"])] += 1
        budget_counts[f"K{int(request['K'])}"] += 1
        track = str(request["track_id"])
        per_track_logical[track] += 1
        per_track_keys[track].add(cache_key)
        key_counts[cache_key] += 1
        if cache_key in first_request_for_key:
            if first_identity_for_key[cache_key] != render_identity:
                raise RuntimeError("CACHE_KEY_V2 identity collision")
            duplicate_rows.append({
                "cache_key_v2": cache_key,
                "source_logical_request_id": first_request_for_key[cache_key],
                "duplicate_logical_request_id": request["logical_request_id"],
                "track_id": track,
            })
        else:
            first_request_for_key[cache_key] = str(request["logical_request_id"])
            first_identity_for_key[cache_key] = render_identity
        if request["phase"] == "STATIC_TEST_EVALUATION":
            static_index[(
                str(request["method"]), str(request["held_out_garment"]),
                str(request["rotation"]), int(request["K"]),
            )] = {
                "cache_key_v2": cache_key,
                "logical_request_id": request["logical_request_id"],
                "selected_known_endpoint": request["selected_known_endpoint"],
            }

    for task in tasks:
        task_id = str(task["task_id"])
        held_out = str(task["held_out_garment"])
        rotation = str(task["rotation"])
        budget = int(task["K"])
        split = f"LOO-{held_out}"
        replay_row = replay_rows[task_id]
        endpoint = str(replay_row["selected_known_endpoint"])

        for method in optimizer_methods:
            selected = endpoint if method != "ZERO_COEFFICIENT_INITIALIZATION_DIAGNOSTIC" else "NOT_APPLICABLE"
            for step in range(1, 301):
                for condition in task["selected_adaptation_conditions"]:
                    metadata = condition_metadata[str(condition)]
                    register({
                        "schema_version": "canondressgs.paper.loo_logical_render_request.v2",
                        "logical_request_id": f"ADAPT/{task_id}/{method}/step_{step:06d}/{condition}",
                        "phase": "ADAPTATION_LOSS_RENDER",
                        "track_id": f"ADAPTATION/{method}",
                        "task_id": task_id,
                        "held_out_garment": held_out,
                        "rotation": rotation,
                        "K": budget,
                        "method": method,
                        "split": split,
                        "condition": str(condition),
                        "checkpoint_or_final_state_role": f"PRE_UPDATE_STEP_{step:06d}",
                        "selected_known_endpoint": selected,
                        "basis_hash": _basis_hash(method, task),
                        "coefficient_or_residual_state_identity": f"OPTIMIZER_STATE/{task_id}/{method}/step_{step:06d}",
                        "camera_metadata_sha256": metadata["camera_metadata_sha256"],
                        "pose_metadata_sha256": metadata["pose_metadata_sha256"],
                        "renderer_contract_sha256": renderer_contract_sha256,
                        "background_identity": background_identity,
                        "resolution_identity": resolution_identity,
                        "output_semantic_role": "ADAPTATION_RENDERING_LOSS_INPUT",
                    })

        test_condition = str(task["test_conditions"][0])
        test_metadata = condition_metadata[test_condition]
        for method in optimizer_methods:
            selected = endpoint if method != "ZERO_COEFFICIENT_INITIALIZATION_DIAGNOSTIC" else "NOT_APPLICABLE"
            for step in milestones:
                register({
                    "schema_version": "canondressgs.paper.loo_logical_render_request.v2",
                    "logical_request_id": f"EVAL/{task_id}/{method}/step_{int(step):06d}/{test_condition}",
                    "phase": "OPTIMIZED_CHECKPOINT_TEST_EVALUATION",
                    "track_id": f"CHECKPOINT_EVALUATION/{method}",
                    "task_id": task_id,
                    "held_out_garment": held_out,
                    "rotation": rotation,
                    "K": budget,
                    "method": method,
                    "split": split,
                    "condition": test_condition,
                    "checkpoint_or_final_state_role": f"CHECKPOINT_STEP_{int(step):06d}",
                    "selected_known_endpoint": selected,
                    "basis_hash": _basis_hash(method, task),
                    "coefficient_or_residual_state_identity": f"CHECKPOINT/{task_id}/{method}/step_{int(step):06d}",
                    "camera_metadata_sha256": test_metadata["camera_metadata_sha256"],
                    "pose_metadata_sha256": test_metadata["pose_metadata_sha256"],
                    "renderer_contract_sha256": renderer_contract_sha256,
                    "background_identity": background_identity,
                    "resolution_identity": resolution_identity,
                    "output_semantic_role": "FORMAL_TEST_METRIC_INPUT",
                })

        for method in static_methods:
            if method == HARD_LOOKUP:
                selected = endpoint
            elif method == "RESIDUAL_NEAREST_ORACLE":
                selected = "DEFERRED_OFFLINE_ORACLE_ENDPOINT"
            else:
                selected = "NOT_APPLICABLE"
            register({
                "schema_version": "canondressgs.paper.loo_logical_render_request.v2",
                "logical_request_id": f"STATIC/{task_id}/{method}/{test_condition}",
                "phase": "STATIC_TEST_EVALUATION",
                "track_id": f"STATIC/{method}",
                "task_id": task_id,
                "held_out_garment": held_out,
                "rotation": rotation,
                "K": budget,
                "method": method,
                "split": split,
                "condition": test_condition,
                "checkpoint_or_final_state_role": "STATIC_FINAL_STATE",
                "selected_known_endpoint": selected,
                "basis_hash": _basis_hash(method, task),
                "coefficient_or_residual_state_identity": _static_state_identity(method, task, selected),
                "camera_metadata_sha256": test_metadata["camera_metadata_sha256"],
                "pose_metadata_sha256": test_metadata["pose_metadata_sha256"],
                "renderer_contract_sha256": renderer_contract_sha256,
                "background_identity": background_identity,
                "resolution_identity": resolution_identity,
                "output_semantic_role": "FORMAL_TEST_METRIC_INPUT",
            })

    per_track = []
    for track in sorted(per_track_logical):
        logical = per_track_logical[track]
        unique = len(per_track_keys[track])
        per_track.append({
            "track_id": track,
            "logical_request_count": logical,
            "unique_physical_key_count": unique,
            "cache_hit_count": logical - unique,
        })

    k_pair_rows = []
    for method in static_methods:
        for held_out in ("O01", "O02", "O03", "O04", "O08"):
            for rotation in ("R0", "R1", "R2", "R3"):
                one = static_index[(method, held_out, rotation, 1)]
                two = static_index[(method, held_out, rotation, 2)]
                k_pair_rows.append({
                    "method": method,
                    "held_out_garment": held_out,
                    "rotation": rotation,
                    "K1_cache_key_v2": one["cache_key_v2"],
                    "K2_cache_key_v2": two["cache_key_v2"],
                    "same_key": one["cache_key_v2"] == two["cache_key_v2"],
                    "K1_selected_known_endpoint": one["selected_known_endpoint"],
                    "K2_selected_known_endpoint": two["selected_known_endpoint"],
                })

    hard_pairs = [row for row in k_pair_rows if row["method"] == HARD_LOOKUP]
    hard_shared = sum(bool(row["same_key"]) for row in hard_pairs)
    hard_divergent = len(hard_pairs) - hard_shared
    guaranteed_rows = [row for row in k_pair_rows if row["method"] != HARD_LOOKUP]
    guaranteed_hits = sum(bool(row["same_key"]) for row in guaranteed_rows)
    unique_count = len(key_counts)
    hit_count = logical_count - unique_count
    checks = {
        "logical_request_count": logical_count == int(expected_counts["total_logical_renders"]),
        "unique_physical_key_count": unique_count == int(expected_counts["unique_physical_renders"]),
        "cache_hit_count": hit_count == int(expected_counts["K_shared_static_cache_hits"]),
        "logical_minus_unique_equals_hits": logical_count - unique_count == hit_count,
        "guaranteed_k_invariant_hits_100": guaranteed_hits == 100,
        "hard_lookup_k_shared_hits_15": hard_shared == 15,
        "hard_lookup_k_divergent_pairs_5": hard_divergent == 5,
        "duplicate_rows_complete": len(duplicate_rows) == hit_count,
        "duplicate_key_multiplicity_two": max(key_counts.values(), default=0) == 2,
        "all_duplicate_identities_exact": all(key_counts[row["cache_key_v2"]] == 2 for row in duplicate_rows),
    }
    result = {
        "schema_version": PLAN_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "cache_key_schema": CACHE_KEY_SCHEMA,
        "cache_identity_fields": list(CACHE_IDENTITY_FIELDS),
        "logical_request_count": logical_count,
        "unique_physical_key_count": unique_count,
        "duplicate_key_count": sum(1 for count in key_counts.values() if count > 1),
        "cache_hit_count": hit_count,
        "guaranteed_k_invariant_hits": guaranteed_hits,
        "hard_lookup_k_shared_hits": hard_shared,
        "hard_lookup_k_divergent_pairs": hard_divergent,
        "phase_counts": dict(sorted(phase_counts.items())),
        "budget_request_counts": dict(sorted(budget_counts.items())),
        "per_track_counts": per_track,
        "K1_K2_static_key_pairs": k_pair_rows,
        "hard_lookup_selected_endpoint_pairs": hard_pairs,
        "duplicate_key_rows": duplicate_rows,
        "key_multiplicity_histogram": {
            str(multiplicity): count
            for multiplicity, count in sorted(Counter(key_counts.values()).items())
        },
        "logical_request_key_index_sha256": canonical_sha256(runtime_index),
        "logical_request_aggregate_sha256": request_digest.hexdigest(),
        "logical_cache_key_sequence_sha256": key_digest.hexdigest(),
        "unique_cache_key_set_sha256": canonical_sha256(sorted(key_counts)),
        "checks": checks,
    }
    result["deterministic_plan_sha256"] = canonical_sha256(result)
    if result["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("LOO cache-key plan mismatch: " + ", ".join(failed))
    if include_runtime_index:
        result["_runtime_logical_request_key_index"] = runtime_index
    return result


def request_count_derivation(
    tasks: Sequence[Mapping[str, Any]], optimizer_methods: Sequence[str],
    static_methods: Sequence[str], milestones: Sequence[int],
) -> dict[str, int]:
    adaptation = sum(
        int(task["K"]) * 300 * len(optimizer_methods) for task in tasks
    )
    optimized = len(tasks) * len(optimizer_methods) * len(milestones)
    static = len(tasks) * len(static_methods)
    return {
        "adaptation_view_logical_renders": adaptation,
        "optimized_checkpoint_evaluation_inferences": optimized,
        "static_oracle_evaluation_inferences": static,
        "total_logical_renders": adaptation + optimized + static,
    }
