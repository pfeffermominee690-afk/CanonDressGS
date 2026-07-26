#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?mode required}"
PARENT="/root/autodl-tmp/canondressgs_work/teacher_targets/SUBJECT00-24CELL-001"
TARGET="${PARENT}/attempt_001"
EXTRACTING="${PARENT}/.attempt_001.extracting"
UPLOAD="${PARENT}/.subject00_24cell_teacher_targets_attempt001.tar.zst.uploading"
ARCHIVE="${PARENT}/subject00_24cell_teacher_targets_attempt001.tar.zst"
MEMBERS="${PARENT}/.subject00_24cell_teacher_targets_attempt001.members"
TEMP_TAR="${PARENT}/.subject00_24cell_teacher_targets_attempt001.tar"
RESULT="${PARENT}/subject00_teacher_target_cloud_execution_result_20260727.json"
WORKTREE="/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_subject00_teacher_target_materialization_quarantine"
PYTHON="/root/autodl-tmp/conda_envs/mmlphuman/bin/python"
CHECKPOINT="/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001/attempt_001/checkpoints/step_060747.pth"
SOURCE_HEAD="a434ae7a78fe898be2658180f20bbcd4391a64c0"
CHECKPOINT_SHA="2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7"

preflight() {
  test ! -e "${TARGET}"
  test ! -e "${EXTRACTING}"
  test ! -e "${UPLOAD}"
  test ! -e "${ARCHIVE}"
  test ! -e "${TEMP_TAR}"
  test ! -e "${RESULT}"
  test "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" = "${CHECKPOINT_SHA}"
  test "$(git -C "${WORKTREE}" rev-parse HEAD)" = "${SOURCE_HEAD}"
  test -z "$(git -C "${WORKTREE}" status --short)"
  test "$(ps -eo args | grep -E '[p]ython([^ ]*)? .*train\.py|[p]ython([^ ]*)? .*optimizer' | wc -l)" -eq 0
  command -v tar >/dev/null
  "${PYTHON}" -c 'import ctypes.util,torch,numpy,PIL; assert ctypes.util.find_library("zstd"); print("RUNTIME_PASS",torch.__version__,numpy.__version__,PIL.__version__,ctypes.util.find_library("zstd"))'
  mkdir -p "${PARENT}"
  printf 'PREFLIGHT_PASS free_bytes=%s\n' "$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
}

execute() {
  local expected_sha="${2:?archive SHA required}"
  local expected_bytes="${3:?archive bytes required}"
  local expected_members="${4:?archive member count required}"
  test ! -e "${TARGET}"
  test ! -e "${EXTRACTING}"
  test -f "${UPLOAD}"
  test "$(stat -c %s "${UPLOAD}")" = "${expected_bytes}"
  test "$(sha256sum "${UPLOAD}" | awk '{print $1}')" = "${expected_sha}"
  mv "${UPLOAD}" "${ARCHIVE}"
  test "$(stat -c %s "${ARCHIVE}")" = "${expected_bytes}"
  test "$(sha256sum "${ARCHIVE}" | awk '{print $1}')" = "${expected_sha}"

  "${PYTHON}" - "${ARCHIVE}" "${TEMP_TAR}" <<'PY'
from pathlib import Path
import ctypes
import ctypes.util
import sys

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
if target_path.exists():
    raise SystemExit(f"temporary tar already exists: {target_path}")
library_path = ctypes.util.find_library("zstd")
if not library_path:
    raise SystemExit("libzstd not found")
zstd = ctypes.CDLL(library_path)

class InBuffer(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]

class OutBuffer(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]

zstd.ZSTD_createDStream.restype = ctypes.c_void_p
zstd.ZSTD_freeDStream.argtypes = [ctypes.c_void_p]
zstd.ZSTD_initDStream.argtypes = [ctypes.c_void_p]
zstd.ZSTD_initDStream.restype = ctypes.c_size_t
zstd.ZSTD_DStreamInSize.restype = ctypes.c_size_t
zstd.ZSTD_DStreamOutSize.restype = ctypes.c_size_t
zstd.ZSTD_decompressStream.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(OutBuffer),
    ctypes.POINTER(InBuffer),
]
zstd.ZSTD_decompressStream.restype = ctypes.c_size_t
zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
zstd.ZSTD_isError.restype = ctypes.c_uint
zstd.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
zstd.ZSTD_getErrorName.restype = ctypes.c_char_p

stream = zstd.ZSTD_createDStream()
if not stream:
    raise SystemExit("ZSTD_createDStream failed")
try:
    code = zstd.ZSTD_initDStream(stream)
    if zstd.ZSTD_isError(code):
        raise SystemExit(zstd.ZSTD_getErrorName(code).decode())
    input_size = int(zstd.ZSTD_DStreamInSize())
    output_size = int(zstd.ZSTD_DStreamOutSize())
    last_code = None
    with source_path.open("rb") as source, target_path.open("xb") as target:
        while True:
            chunk = source.read(input_size)
            if not chunk:
                break
            input_memory = ctypes.create_string_buffer(chunk)
            input_buffer = InBuffer(
                ctypes.cast(input_memory, ctypes.c_void_p), len(chunk), 0
            )
            while input_buffer.pos < input_buffer.size:
                output_memory = ctypes.create_string_buffer(output_size)
                output_buffer = OutBuffer(
                    ctypes.cast(output_memory, ctypes.c_void_p), output_size, 0
                )
                last_code = zstd.ZSTD_decompressStream(
                    stream, ctypes.byref(output_buffer), ctypes.byref(input_buffer)
                )
                if zstd.ZSTD_isError(last_code):
                    raise SystemExit(zstd.ZSTD_getErrorName(last_code).decode())
                target.write(output_memory.raw[: output_buffer.pos])
    if last_code != 0:
        raise SystemExit(f"incomplete zstd frame: remaining={last_code}")
finally:
    zstd.ZSTD_freeDStream(stream)
print("LIBZSTD_DECOMPRESSION_PASS", target_path.stat().st_size)
PY

  "${PYTHON}" - "${TEMP_TAR}" "${EXTRACTING}" "${MEMBERS}" "${expected_members}" <<'PY'
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

archive = Path(sys.argv[1])
extracting = Path(sys.argv[2])
members_path = Path(sys.argv[3])
expected_members = int(sys.argv[4])
if extracting.exists():
    raise SystemExit(f"extracting root already exists: {extracting}")
extracting.mkdir()
names = []
with archive.open("rb") as raw:
    with tarfile.open(fileobj=raw, mode="r|") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or not path.parts
            ):
                raise SystemExit(f"unsafe or non-file archive member: {member.name}")
            name = path.as_posix()
            if name in names:
                raise SystemExit(f"duplicate archive member: {name}")
            names.append(name)
            destination = extracting.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise SystemExit(f"archive member unreadable: {name}")
            with destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
if len(names) != expected_members:
    raise SystemExit(f"archive member count {len(names)} != {expected_members}")
members_path.write_text("\n".join(names) + "\n", encoding="utf-8")
print("PATH_SAFETY_AND_EXTRACTION_PASS", len(names))
PY
  rm -f -- "${TEMP_TAR}"
  "${PYTHON}" - "${EXTRACTING}" "${expected_members}" <<'PY'
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys

root = Path(sys.argv[1])
expected_members = int(sys.argv[2])
manifest_path = root / "12_portable_archive" / "payload_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest["listed_file_count"] + 1 != expected_members:
    raise SystemExit("payload/archive member count mismatch")
for entry in manifest["files"]:
    rel = PurePosixPath(entry["path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit(f"unsafe manifest path: {rel}")
    path = root.joinpath(*rel.parts)
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise SystemExit(f"file missing or bytes mismatch: {rel}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(f"SHA mismatch: {rel}")
print("PAYLOAD_SHA_PASS", len(manifest["files"]))
PY
  local extracted_bytes
  extracted_bytes="$(du -sb "${EXTRACTING}" | awk '{print $1}')"
  mv "${EXTRACTING}" "${TARGET}"

  "${PYTHON}" - "${WORKTREE}" "${TARGET}" <<'PY'
from pathlib import Path
import json
import sys

repo = Path(sys.argv[1])
target = Path(sys.argv[2])
sys.path.insert(0, str(repo))
from scene.full_dressable_dataset import (  # noqa: E402
    DUAL_TARGET_FIELDS,
    FullDressableInferenceDataset,
    FullDressableTrainingDataset,
)

manifest_path = target / "10_final_registry" / "subject00_22_training_full_dataset_v1.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
index_root = target / "10_final_registry" / "indexes"
index_names = {
    "all": "all_provenance_records.json",
    "training": "training_records.json",
    "evaluation": "evaluation_records.json",
    "quarantine": "review_only_quarantine_records.json",
    "O03": "O03_training_records.json",
}
indexes = {
    key: json.loads((index_root / name).read_text(encoding="utf-8"))
    for key, name in index_names.items()
}
expected = {"all": 24, "training": 22, "evaluation": 22, "quarantine": 2, "O03": 7}
if {key: value["count"] for key, value in indexes.items()} != expected:
    raise SystemExit("index counts changed")
if set(indexes["training"]["request_ids"]) & set(indexes["quarantine"]["request_ids"]):
    raise SystemExit("quarantine leaked into training index")
training = FullDressableTrainingDataset(manifest_path, "train", reference_count=1, seed=0)
inference = FullDressableInferenceDataset(manifest_path, "train", reference_count=1, seed=0)
if len(training) != 22 or len(inference) != 22:
    raise SystemExit("formal loader count is not 22")
conditions = {item["condition_id"]: item for item in manifest["conditions"]}
samples = []
for index in range(len(training)):
    sample = training[index]
    infer = inference[index]
    request_id = sample["target_condition_id"]
    condition = conditions[request_id]
    h, w = int(condition["height"]), int(condition["width"])
    required = {
        "target_edit_rgb",
        "target_base_rgb",
        "target_foreground_mask",
        "target_clothing_mask",
        *DUAL_TARGET_FIELDS,
    }
    if not required <= set(sample):
        raise SystemExit(f"missing loader fields: {request_id}")
    if tuple(sample["target_edit_rgb"].shape) != (3, h, w):
        raise SystemExit(f"edit RGB shape mismatch: {request_id}")
    if tuple(sample["target_base_rgb"].shape) != (3, h, w):
        raise SystemExit(f"base RGB shape mismatch: {request_id}")
    mask_fields = (
        "target_foreground_mask",
        "target_clothing_mask",
        "target_edit_mask",
        "target_edit_core_mask",
        "target_preserve_mask",
        "target_transition_mask",
        "target_protected_mask",
        "target_base_foreground_mask",
        "target_old_clothing_mask",
        "target_revealed_skin_mask",
    )
    if any(tuple(sample[field].shape) != (1, h, w) for field in mask_fields):
        raise SystemExit(f"mask shape mismatch: {request_id}")
    if infer["target_condition_id"] != request_id or set(DUAL_TARGET_FIELDS) & set(infer):
        raise SystemExit(f"inference boundary violation: {request_id}")
    samples.append(
        {
            "index": index,
            "request_id": request_id,
            "width": w,
            "height": h,
            "training_loader_status": "PASS",
            "inference_boundary_status": "PASS",
        }
    )
distribution = {}
for condition in manifest["conditions"]:
    key = f"{condition['width']}x{condition['height']}"
    distribution[key] = distribution.get(key, 0) + 1
if distribution != {"1349x1166": 21, "1350x1165": 1}:
    raise SystemExit(f"training mixed resolution changed: {distribution}")
result = {
    "schema_version": "canondressgs.subject00.teacher_target.loader_smoke.v1",
    "task_id": "AAAI27-SUBJECT00-TEACHER-TARGET-MATERIALIZATION-WITH-QUARANTINE-001",
    "created_at": "2026-07-27T00:00:00Z",
    "status": "PASS_ZERO_OPTIMIZER_CPU_LOADER_SMOKE",
    "loader_path": "scene/full_dressable_dataset.py",
    "loader_sha256": "786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508",
    "schema": "canondressgs.full_dataset.v1",
    "loader_provenance_count": 24,
    "loader_training_count": 22,
    "loader_evaluation_count": 22,
    "loader_quarantine_count": 2,
    "loader_O03_count": 7,
    "training_denominator": 22,
    "O03_training_denominator": 7,
    "training_dataset_len": len(training),
    "inference_dataset_len": len(inference),
    "mixed_resolution_distribution_training": distribution,
    "batch_policy": "BATCH_SIZE_ONE_NATIVE_RESOLUTION_SUPPORTED",
    "resize_crop_padding_calls": 0,
    "optimizer_steps": 0,
    "gpu_calls": 0,
    "teacher_endpoint_calls": 0,
    "samples": samples,
}
output = target / "08_quality_audit" / "cloud_loader_smoke.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY

  local free_after
  free_after="$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')"
  local checkpoint_after
  checkpoint_after="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  test "${checkpoint_after}" = "${CHECKPOINT_SHA}"
  test "$(ps -eo args | grep -E '[p]ython([^ ]*)? .*train\.py|[p]ython([^ ]*)? .*optimizer' | wc -l)" -eq 0

  "${PYTHON}" - \
    "${TARGET}" "${ARCHIVE}" "${RESULT}" "${expected_sha}" "${expected_bytes}" \
    "${expected_members}" "${extracted_bytes}" "${free_after}" "${checkpoint_after}" <<'PY'
from pathlib import Path
import json
import sys

target = Path(sys.argv[1])
archive = Path(sys.argv[2])
result_path = Path(sys.argv[3])
expected_sha = sys.argv[4]
expected_bytes = int(sys.argv[5])
expected_members = int(sys.argv[6])
extracted_bytes = int(sys.argv[7])
free_after = int(sys.argv[8])
checkpoint_after = sys.argv[9]

records = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((target / "02_records").glob("*_teacher_target_record.json"))
]
cameras = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((target / "06_camera").glob("*_camera.json"))
]
manifest = json.loads(
    (target / "10_final_registry" / "subject00_22_training_full_dataset_v1.json").read_text(
        encoding="utf-8"
    )
)
loader = json.loads(
    (target / "08_quality_audit" / "cloud_loader_smoke.json").read_text(encoding="utf-8")
)
quarantine_ids = {
    "subject00_O03_slot04_canary_attempt004_cand00",
    "subject00_O01_slot04_remaining_attempt005_cand00",
}
required_fields = {
    "rgb",
    "target_edit_rgb",
    "target_base_rgb",
    "foreground_mask",
    "clothing_mask",
    "target_edit_mask",
    "target_edit_core_mask",
    "target_preserve_mask",
    "target_transition_mask",
    "target_protected_mask",
    "target_foreground_mask",
    "target_base_foreground_mask",
    "target_clothing_mask",
    "target_old_clothing_mask",
    "target_revealed_skin_mask",
}
observations = [
    observation
    for outfit in manifest["outfits"]
    for observation in outfit["observations"]
]
no_fake_k = all(
    camera["camera_model"] is None
    and camera["K"] is None
    and camera["K_target"] is None
    and camera["T_pixel"] is None
    and camera["w2c"] is None
    and camera["c2w"] is None
    for camera in cameras
    if camera["request_id"] in quarantine_ids
)
manifest_ids = {condition["condition_id"] for condition in manifest["conditions"]}
result = {
    "status": "PASS",
    "schema_version": "canondressgs.subject00.teacher_target.cloud_extraction.v1",
    "task_id": "AAAI27-SUBJECT00-TEACHER-TARGET-MATERIALIZATION-WITH-QUARANTINE-001",
    "created_at": "2026-07-27T00:00:00Z",
    "cloud_target_root": str(target),
    "cloud_archive_path": str(archive),
    "upload_calls": 1,
    "upload_bytes": expected_bytes,
    "cloud_archive_sha256": expected_sha,
    "cloud_archive_sha_status": "PASS",
    "archive_member_count": expected_members,
    "duplicate_member_count": 0,
    "path_traversal_member_count": 0,
    "cloud_extraction_status": "PASS_EXACT_ATOMIC",
    "cloud_extracted_file_count": expected_members,
    "cloud_file_sha_status": "PASS_ALL_LISTED_FILES",
    "cloud_extracted_bytes": extracted_bytes,
    "cloud_free_bytes_after": free_after,
    "operational_margin_status": "PASS" if free_after > 20_000_000_000 else "FAIL",
    "loader_smoke_status": loader["status"],
    "loader_result": loader,
    "no_fake_quarantine_K": no_fake_k,
    "required_schema_fields_status": (
        "PASS" if len(observations) == 22 and all(required_fields <= set(x) for x in observations)
        else "FAIL"
    ),
    "mixed_resolution_status": (
        "PASS"
        if sorted((x["width"], x["height"]) for x in manifest["conditions"]).count((1350, 1165)) == 1
        else "FAIL"
    ),
    "limitation_propagation_count": sum(bool(record["limitations"]) for record in records),
    "human_override_propagation_count": sum(
        record["human_override_status"] is not None for record in records
    ),
    "quarantine_excluded_from_loader": not (manifest_ids & quarantine_ids),
    "formal_base_status": "USER_AUTHORIZED_PAUSED",
    "formal_base_durable_resume_step": 60747,
    "formal_base_resume_authorized": False,
    "formal_base_checkpoint_sha256_after": checkpoint_after,
    "formal_base_process_count": 0,
    "optimizer_steps": 0,
    "gpu_calls": 0,
    "teacher_endpoint_calls": 0,
    "paper_modifications": 0,
}
if not all(
    (
        no_fake_k,
        result["required_schema_fields_status"] == "PASS",
        result["mixed_resolution_status"] == "PASS",
        result["limitation_propagation_count"] == 9,
        result["human_override_propagation_count"] == 2,
        result["quarantine_excluded_from_loader"],
        result["operational_margin_status"] == "PASS",
        result["loader_smoke_status"] == "PASS_ZERO_OPTIMIZER_CPU_LOADER_SMOKE",
    )
):
    raise SystemExit(f"cloud final QA failed: {result}")
result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
PY
  printf 'CLOUD_EXECUTION_PASS result=%s\n' "${RESULT}"
}

case "${MODE}" in
  preflight) preflight ;;
  execute) execute "$@" ;;
  *) printf 'unknown mode: %s\n' "${MODE}" >&2; exit 2 ;;
esac
