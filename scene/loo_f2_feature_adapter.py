from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

try:
    import torch
except ModuleNotFoundError:  # Static governance tests run without PyTorch on Windows.
    torch = None  # type: ignore[assignment]


REFERENCE_RECORD_SCHEMA_VERSION = "canondressgs.loo.f2_reference_record.v1"
REFERENCE_SET_SCHEMA_VERSION = "canondressgs.loo.f2_reference_set.v1"
AGGREGATION_RULE = "FROZEN_PER_REFERENCE_F2_MEAN_MAX_V1"
ADAPTATION_ROLE = "ADAPTATION_REFERENCE_SET"
CALIBRATION_INITIALIZATION_ROLE = "CALIBRATION_INITIALIZATION_REFERENCE_SET"
AUTHORIZED_ROLES = {
    ADAPTATION_ROLE: (1, 2),
    CALIBRATION_INITIALIZATION_ROLE: (4,),
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

REFERENCE_METADATA_KEYS = frozenset({
    "schema_version",
    "outfit_id",
    "condition_id",
    "query_order",
    "source_image_path",
    "source_image_sha256",
    "source_mask_path",
    "source_mask_sha256",
    "cache_key",
})


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_string(record: Mapping[str, Any], name: str) -> str:
    value = record[name]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def validate_reference_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("reference metadata must be a mapping")
    keys = set(record)
    missing = REFERENCE_METADATA_KEYS - keys
    unknown = keys - REFERENCE_METADATA_KEYS
    if missing:
        raise ValueError(f"reference metadata missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"reference metadata unknown keys: {sorted(unknown)}")
    result = dict(record)
    if result["schema_version"] != REFERENCE_RECORD_SCHEMA_VERSION:
        raise ValueError("reference metadata schema_version mismatch")
    for name in (
        "outfit_id", "condition_id", "source_image_path", "source_mask_path",
        "source_image_sha256", "source_mask_sha256", "cache_key",
    ):
        _required_string(result, name)
    if not isinstance(result["query_order"], int) or isinstance(result["query_order"], bool):
        raise TypeError("query_order must be an integer")
    for name in ("source_image_sha256", "source_mask_sha256", "cache_key"):
        if SHA256_PATTERN.fullmatch(result[name]) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
    cache_payload = {
        "schema_version": REFERENCE_RECORD_SCHEMA_VERSION,
        "outfit_id": result["outfit_id"],
        "condition_id": result["condition_id"],
        "source_image_sha256": result["source_image_sha256"],
        "source_mask_sha256": result["source_mask_sha256"],
    }
    if result["cache_key"] != canonical_sha256(cache_payload):
        raise ValueError("reference metadata cache_key mismatch")
    return result


def build_reference_metadata(
    context: Mapping[str, Any], outfit_id: str, condition_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    if not condition_ids:
        raise ValueError("reference conditions must be nonempty")
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("reference conditions must be unique and ordered")
    records = []
    for query_order, condition_id in enumerate(condition_ids):
        sample = context["samples"][f"{outfit_id}/{condition_id}"]
        source = sample.get("source_record")
        if not isinstance(source, Mapping):
            raise ValueError("reference sample lacks source_record metadata")
        if source.get("condition_id") != condition_id:
            raise ValueError("source_record condition_id mismatch")
        checksums = source.get("checksums")
        if not isinstance(checksums, Mapping):
            raise ValueError("source_record lacks checksums")
        image_sha = checksums.get("target_edit_rgb")
        mask_sha = checksums.get("target_clothing_mask")
        cache_payload = {
            "schema_version": REFERENCE_RECORD_SCHEMA_VERSION,
            "outfit_id": outfit_id,
            "condition_id": condition_id,
            "source_image_sha256": image_sha,
            "source_mask_sha256": mask_sha,
        }
        records.append(validate_reference_metadata({
            **cache_payload,
            "query_order": query_order,
            "source_image_path": source.get("target_edit_rgb"),
            "source_mask_path": source.get("target_clothing_mask"),
            "cache_key": canonical_sha256(cache_payload),
        }))
    return tuple(records)


@dataclass(frozen=True)
class F2ReferenceFeatureRecord:
    schema_version: str
    outfit_id: str
    condition_id: str
    query_order: int
    source_image_path: str
    source_image_sha256: str
    source_mask_path: str
    source_mask_sha256: str
    cache_key: str
    per_reference_f2: Any


@dataclass(frozen=True)
class F2ReferenceFeatureSet:
    schema_version: str
    role: str
    aggregation_rule: str
    view_count: int
    condition_ids: tuple[str, ...]
    query_order: tuple[int, ...]
    records: tuple[F2ReferenceFeatureRecord, ...]
    features_per_view: Any
    set_mean: Any
    set_max: Any
    aggregated_feature: Any
    dtype: str
    device: str
    contiguous: bool

    def feature_vector(self) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch is required for F2 feature access")
        if self.aggregated_feature.ndim != 2 or self.aggregated_feature.shape[0] != 1:
            raise ValueError("aggregated_feature must have explicit shape [1,D]")
        return self.aggregated_feature[0]


def _producer_field_names(value: Any) -> set[str]:
    try:
        return {field.name for field in fields(value)}
    except TypeError as error:
        raise TypeError("F2 producer must return a dataclass record") from error


def _validate_producer_output(value: Any, view_count: int, feature_dim: int) -> None:
    required = {
        "per_reference_f2", "set_mean", "set_max", "set_feature",
        "clothing_mean", "clothing_max", "resized_clothing_mask",
        "pooling_denominator", "valid_mask",
    }
    actual = _producer_field_names(value)
    if actual != required:
        raise ValueError(
            f"F2 producer fields mismatch: expected={sorted(required)} actual={sorted(actual)}"
        )
    expected_per_reference = (view_count, 2 * feature_dim)
    if tuple(value.per_reference_f2.shape) != expected_per_reference:
        raise ValueError(
            "F2 producer per_reference_f2 shape mismatch: "
            f"expected={expected_per_reference} actual={tuple(value.per_reference_f2.shape)}"
        )
    expected_set = (1, 4 * feature_dim)
    for name in ("set_feature",):
        if tuple(getattr(value, name).shape) != expected_set:
            raise ValueError(f"F2 producer {name} shape mismatch")
    for name in ("set_mean", "set_max"):
        if tuple(getattr(value, name).shape) != (1, 2 * feature_dim):
            raise ValueError(f"F2 producer {name} shape mismatch")


def extract_reference_feature_set(
    extractor: Any,
    reference_images: Any,
    reference_clothing_masks: Any,
    reference_valid_mask: Any,
    metadata_records: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> F2ReferenceFeatureSet:
    if torch is None:
        raise RuntimeError("PyTorch is required for F2 extraction")
    if role not in AUTHORIZED_ROLES:
        raise ValueError(f"unauthorized F2 reference role: {role}")
    records = tuple(validate_reference_metadata(record) for record in metadata_records)
    view_count = len(records)
    if view_count not in AUTHORIZED_ROLES[role]:
        raise ValueError(
            f"{role} requires view_count in {AUTHORIZED_ROLES[role]}, got {view_count}"
        )
    if [record["query_order"] for record in records] != list(range(view_count)):
        raise ValueError("reference query order mismatch")
    condition_ids = tuple(record["condition_id"] for record in records)
    if len(set(condition_ids)) != view_count:
        raise ValueError("reference condition order contains duplicates")
    if reference_images.ndim != 4 or tuple(reference_images.shape[:2]) != (view_count, 3):
        raise ValueError("reference_images must have explicit shape [V,3,H,W]")
    if reference_clothing_masks.ndim != 4 or tuple(reference_clothing_masks.shape[:2]) != (view_count, 1):
        raise ValueError("reference_clothing_masks must have explicit shape [V,1,H,W]")
    if tuple(reference_valid_mask.shape) != (view_count, 1):
        raise ValueError("reference_valid_mask must have explicit shape [V,1]")
    tensors = (reference_images, reference_clothing_masks, reference_valid_mask)
    if not all(torch.is_floating_point(value) for value in tensors):
        raise TypeError("F2 interface tensors must be floating point")
    if len({value.dtype for value in tensors}) != 1:
        raise TypeError("F2 interface dtype mismatch")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("F2 interface device mismatch")
    if not all(value.is_contiguous() for value in tensors):
        raise ValueError("F2 interface requires contiguous tensors")
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("F2 interface tensors must be finite")
    if not torch.equal(reference_valid_mask, torch.ones_like(reference_valid_mask)):
        raise ValueError("LOO F2 reference sets require every declared view to be valid")

    feature_dim = int(extractor.feature_dim)
    if view_count <= 3:
        produced = extractor(
            reference_images, reference_clothing_masks, reference_valid_mask,
        )
        _validate_producer_output(produced, view_count, feature_dim)
        features_per_view = produced.per_reference_f2
    else:
        singleton_rows = []
        for index in range(view_count):
            produced = extractor(
                reference_images[index:index + 1],
                reference_clothing_masks[index:index + 1],
                reference_valid_mask[index:index + 1],
            )
            _validate_producer_output(produced, 1, feature_dim)
            singleton_rows.append(produced.per_reference_f2)
        features_per_view = torch.cat(singleton_rows, dim=0)

    set_mean = features_per_view.mean(dim=0, keepdim=True)
    set_max = features_per_view.amax(dim=0, keepdim=True)
    aggregated_feature = torch.cat((set_mean, set_max), dim=-1)
    if tuple(features_per_view.shape) != (view_count, 2 * feature_dim):
        raise ValueError("features_per_view shape mismatch after explicit extraction")
    if tuple(aggregated_feature.shape) != (1, 4 * feature_dim):
        raise ValueError("aggregated_feature shape mismatch after explicit aggregation")
    if aggregated_feature.dtype != reference_images.dtype:
        raise TypeError("F2 producer changed dtype")
    if aggregated_feature.device != reference_images.device:
        raise ValueError("F2 producer changed device")
    if not aggregated_feature.is_contiguous():
        raise ValueError("aggregated F2 feature must be contiguous")

    typed_records = tuple(
        F2ReferenceFeatureRecord(
            **record, per_reference_f2=features_per_view[index:index + 1],
        )
        for index, record in enumerate(records)
    )
    return F2ReferenceFeatureSet(
        schema_version=REFERENCE_SET_SCHEMA_VERSION,
        role=role,
        aggregation_rule=AGGREGATION_RULE,
        view_count=view_count,
        condition_ids=condition_ids,
        query_order=tuple(range(view_count)),
        records=typed_records,
        features_per_view=features_per_view,
        set_mean=set_mean,
        set_max=set_max,
        aggregated_feature=aggregated_feature,
        dtype=str(aggregated_feature.dtype),
        device=str(aggregated_feature.device),
        contiguous=True,
    )
