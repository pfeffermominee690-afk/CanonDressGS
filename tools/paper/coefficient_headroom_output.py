from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from PIL import Image


REPAIR_TASK_ID = "AAAI27-COEFFICIENT-HEADROOM-EXECUTION-REPAIR-001"
OUTPUT_NAME = "COEFFICIENT-HEADROOM-001"
PRESERVED_ATTEMPT = "attempt_001"
WRITABLE_ATTEMPT = "attempt_002"
ERROR_CODES = {
    "HEADROOM_OUTPUT_PARENT_MISSING",
    "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
    "HEADROOM_OUTPUT_PATH_COLLISION",
    "HEADROOM_OUTPUT_TARGET_EXISTS",
    "HEADROOM_OUTPUT_ATOMIC_WRITE_FAILED",
    "HEADROOM_OUTPUT_VALIDATION_FAILED",
    "HEADROOM_OUTPUT_REGISTRY_UPDATE_FAILED",
    "HEADROOM_OUTPUT_PLAN_MISMATCH",
    "HEADROOM_OUTPUT_TEMP_FILE_LEAK",
}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class HeadroomOutputError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown Headroom output error code: {code}")
        self.code = code
        self.context = {
            "task_id": REPAIR_TASK_ID,
            **{key: value for key, value in context.items() if value is not None},
        }
        super().__init__(f"{code}: {message}; context={json.dumps(self.context, sort_keys=True, default=str)}")


def strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_context(
    path: Path,
    *,
    artifact_type: str,
    writer_id: str,
    phase: str,
    attempt_id: str,
    exception_class: str | None = None,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "artifact_type": artifact_type,
        "writer_id": writer_id,
        "phase": phase,
        "attempt_id": attempt_id,
        "exception_class": exception_class,
    }


def _parts(raw: str | os.PathLike[str]) -> tuple[str, ...]:
    text = os.fspath(raw)
    return tuple(part for part in text.replace("\\", "/").split("/") if part not in ("", "."))


def _reject_untrusted_path(raw: str | os.PathLike[str]) -> None:
    text = os.fspath(raw)
    windows = PureWindowsPath(text)
    if ".." in _parts(text) or (windows.drive and not windows.is_absolute()):
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
            "path traversal or drive-relative path rejected",
            path=text,
            exception_class="ValueError",
        )


def resolve_attempt_root(
    output_root: str | os.PathLike[str],
    *,
    output_name: str = OUTPUT_NAME,
    attempt_id: str = WRITABLE_ATTEMPT,
    for_write: bool = True,
) -> Path:
    for identifier in (output_name, attempt_id):
        if not _IDENTIFIER.fullmatch(identifier) or identifier in {".", ".."}:
            raise HeadroomOutputError(
                "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
                "invalid output or attempt identifier",
                path=identifier,
                attempt_id=attempt_id,
                exception_class="ValueError",
            )
    if for_write and attempt_id == PRESERVED_ATTEMPT:
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
            "preserved attempt is read-only",
            path=attempt_id,
            attempt_id=attempt_id,
            exception_class="PermissionError",
        )
    root = Path(output_root).resolve(strict=False)
    candidate = (root / output_name / attempt_id).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
            "attempt root escaped output root",
            path=candidate,
            attempt_id=attempt_id,
            exception_class=type(error).__name__,
        ) from error
    return candidate


def assert_path_inside_attempt(attempt_root: Path, target: Path) -> Path:
    _reject_untrusted_path(target)
    attempt = attempt_root.resolve(strict=False)
    if attempt.name == PRESERVED_ATTEMPT or PRESERVED_ATTEMPT in _parts(target):
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
            "attempt_001 is immutable",
            path=target,
            attempt_id=attempt.name,
            exception_class="PermissionError",
        )
    candidate = target if target.is_absolute() else attempt / target
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(attempt)
    except ValueError as error:
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
            "target escaped current attempt root",
            path=candidate,
            attempt_id=attempt.name,
            exception_class=type(error).__name__,
        ) from error
    if not relative.parts:
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
            "attempt root is not a file target",
            path=candidate,
            attempt_id=attempt.name,
            exception_class="IsADirectoryError",
        )
    return resolved


def resolve_output_path(attempt_root: Path, *relative_parts: str) -> Path:
    if not relative_parts:
        raise ValueError("at least one relative path component is required")
    for part in relative_parts:
        _reject_untrusted_path(part)
        if PurePosixPath(part).is_absolute() or PureWindowsPath(part).is_absolute():
            raise HeadroomOutputError(
                "HEADROOM_OUTPUT_PATH_OUTSIDE_ATTEMPT",
                "absolute output component rejected",
                path=part,
                attempt_id=attempt_root.name,
                exception_class="ValueError",
            )
    return assert_path_inside_attempt(attempt_root, attempt_root.joinpath(*relative_parts))


def ensure_parent_directory(attempt_root: Path, target: Path) -> Path:
    safe_target = assert_path_inside_attempt(attempt_root, target)
    parent = safe_target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_PARENT_MISSING",
            "parent directory closure failed",
            path=parent,
            attempt_id=attempt_root.name,
            exception_class="NotADirectoryError",
        )
    assert_path_inside_attempt(attempt_root, safe_target)
    return parent


def ensure_directory(attempt_root: Path, directory: Path) -> Path:
    return ensure_parent_directory(attempt_root, directory / ".headroom_parent_contract")


def assert_no_existing_target(attempt_root: Path, target: Path) -> None:
    safe_target = assert_path_inside_attempt(attempt_root, target)
    if safe_target.exists() or safe_target.is_symlink():
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_TARGET_EXISTS",
            "existing target overwrite rejected",
            path=safe_target,
            attempt_id=attempt_root.name,
            exception_class="FileExistsError",
        )


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers via os.fsync.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_with(
    attempt_root: Path,
    target: Path,
    producer: Callable[[Path], None],
    validator: Callable[[Path], None],
    *,
    artifact_type: str,
    writer_id: str,
    phase: str,
    allow_replace: bool = False,
) -> dict[str, Any]:
    safe_target = assert_path_inside_attempt(attempt_root, target)
    context = _error_context(
        safe_target,
        artifact_type=artifact_type,
        writer_id=writer_id,
        phase=phase,
        attempt_id=attempt_root.name,
    )
    ensure_parent_directory(attempt_root, safe_target)
    if not allow_replace:
        assert_no_existing_target(attempt_root, safe_target)
    temporary = safe_target.with_name(f".{safe_target.name}.{uuid.uuid4().hex}.tmp")
    assert_path_inside_attempt(attempt_root, temporary)
    try:
        producer(temporary)
        if not temporary.is_file():
            raise FileNotFoundError("producer did not create temporary file")
        _fsync_file(temporary)
        try:
            validator(temporary)
        except Exception as error:
            raise HeadroomOutputError(
                "HEADROOM_OUTPUT_VALIDATION_FAILED",
                "temporary artifact failed validation",
                **_error_context(
                    safe_target,
                    artifact_type=artifact_type,
                    writer_id=writer_id,
                    phase=phase,
                    attempt_id=attempt_root.name,
                    exception_class=type(error).__name__,
                ),
            ) from error
        digest = file_sha256(temporary)
        size = temporary.stat().st_size
        if allow_replace:
            os.replace(temporary, safe_target)
        else:
            try:
                os.link(temporary, safe_target)
            except FileExistsError as error:
                raise HeadroomOutputError(
                    "HEADROOM_OUTPUT_TARGET_EXISTS",
                    "target appeared before atomic publish",
                    **context,
                ) from error
            temporary.unlink()
        _fsync_directory(safe_target.parent)
        if not safe_target.is_file() or file_sha256(safe_target) != digest:
            raise OSError("published artifact checksum mismatch")
        return {
            "writer_id": writer_id,
            "artifact_type": artifact_type,
            "phase": phase,
            "attempt_id": attempt_root.name,
            "relative_path": safe_target.relative_to(attempt_root.resolve(strict=False)).as_posix(),
            "bytes": size,
            "sha256": digest,
            "atomic_publish": "os.replace" if allow_replace else "same_directory_hardlink",
            "validation": "PASS",
        }
    except HeadroomOutputError:
        raise
    except Exception as error:
        raise HeadroomOutputError(
            "HEADROOM_OUTPUT_ATOMIC_WRITE_FAILED",
            "atomic write failed",
            **_error_context(
                safe_target,
                artifact_type=artifact_type,
                writer_id=writer_id,
                phase=phase,
                attempt_id=attempt_root.name,
                exception_class=type(error).__name__,
            ),
        ) from error
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_bytes(
    attempt_root: Path,
    target: Path,
    value: bytes,
    *,
    artifact_type: str = "binary",
    writer_id: str = "atomic_write_bytes",
    phase: str = "unspecified",
    allow_replace: bool = False,
    validator: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    validation = validator or (lambda path: path.stat().st_size >= 0)
    return atomic_write_with(
        attempt_root,
        target,
        lambda path: _write_bytes(path, value),
        validation,
        artifact_type=artifact_type,
        writer_id=writer_id,
        phase=phase,
        allow_replace=allow_replace,
    )


def atomic_write_text(
    attempt_root: Path,
    target: Path,
    value: str,
    *,
    writer_id: str = "atomic_write_text",
    phase: str = "unspecified",
    allow_replace: bool = False,
) -> dict[str, Any]:
    payload = (value.rstrip() + "\n").encode("utf-8")

    def validate(path: Path) -> None:
        path.read_text(encoding="utf-8")

    return atomic_write_bytes(
        attempt_root,
        target,
        payload,
        artifact_type="text",
        writer_id=writer_id,
        phase=phase,
        allow_replace=allow_replace,
        validator=validate,
    )


def atomic_write_json(
    attempt_root: Path,
    target: Path,
    value: Any,
    *,
    writer_id: str = "atomic_write_json",
    phase: str = "unspecified",
    allow_replace: bool = False,
    required_fields: Sequence[str] = (),
) -> dict[str, Any]:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")

    def validate(path: Path) -> None:
        parsed = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)
        if required_fields and (
            not isinstance(parsed, Mapping) or not set(required_fields).issubset(parsed)
        ):
            raise ValueError(f"required JSON fields missing: {list(required_fields)}")

    return atomic_write_bytes(
        attempt_root,
        target,
        payload,
        artifact_type="json",
        writer_id=writer_id,
        phase=phase,
        allow_replace=allow_replace,
        validator=validate,
    )


def atomic_append_jsonl(
    attempt_root: Path,
    target: Path,
    value: Mapping[str, Any],
    *,
    writer_id: str = "atomic_append_jsonl",
    phase: str = "optimization",
) -> dict[str, Any]:
    safe_target = assert_path_inside_attempt(attempt_root, target)
    existing = safe_target.read_bytes() if safe_target.is_file() else b""
    line = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8") + b"\n"

    def validate(path: Path) -> None:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                json.loads(raw, object_pairs_hook=strict_object)

    return atomic_write_bytes(
        attempt_root,
        safe_target,
        existing + line,
        artifact_type="jsonl",
        writer_id=writer_id,
        phase=phase,
        allow_replace=safe_target.exists(),
        validator=validate,
    )


def validate_png(path: Path) -> None:
    if path.stat().st_size <= 0:
        raise ValueError("PNG is empty")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.format != "PNG" or image.width <= 0 or image.height <= 0:
            raise ValueError("invalid PNG dimensions or format")
        if image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
            raise ValueError(f"invalid PNG mode: {image.mode}")


def atomic_save_png(
    attempt_root: Path,
    target: Path,
    image: Image.Image,
    *,
    writer_id: str = "atomic_save_png",
    phase: str = "render_output",
    allow_replace: bool = False,
) -> dict[str, Any]:
    def produce(path: Path) -> None:
        image.save(path, format="PNG", compress_level=6)

    return atomic_write_with(
        attempt_root,
        target,
        produce,
        validate_png,
        artifact_type="png",
        writer_id=writer_id,
        phase=phase,
        allow_replace=allow_replace,
    )


def atomic_save_figure(
    attempt_root: Path,
    target: Path,
    figure: Any,
    *,
    writer_id: str = "atomic_save_figure",
    phase: str = "visualization",
    allow_replace: bool = False,
    **savefig_kwargs: Any,
) -> dict[str, Any]:
    suffix = target.suffix.lower()
    validator = validate_png if suffix == ".png" else validate_pdf
    artifact_type = "png" if suffix == ".png" else "pdf"
    return atomic_write_with(
        attempt_root,
        target,
        lambda path: figure.savefig(path, **savefig_kwargs),
        validator,
        artifact_type=artifact_type,
        writer_id=writer_id,
        phase=phase,
        allow_replace=allow_replace,
    )


def validate_svg(path: Path) -> None:
    root = ET.parse(path).getroot()
    if path.stat().st_size <= 0 or not root.tag.lower().endswith("svg"):
        raise ValueError("invalid SVG")


def atomic_write_svg(
    attempt_root: Path,
    target: Path,
    value: str,
    *,
    writer_id: str = "atomic_write_svg",
    phase: str = "visualization",
    allow_replace: bool = False,
) -> dict[str, Any]:
    return atomic_write_bytes(
        attempt_root,
        target,
        value.encode("utf-8"),
        artifact_type="svg",
        writer_id=writer_id,
        phase=phase,
        allow_replace=allow_replace,
        validator=validate_svg,
    )


def validate_pdf(path: Path) -> None:
    value = path.read_bytes()
    if not value.startswith(b"%PDF-") or b"%%EOF" not in value[-1024:]:
        raise ValueError("invalid PDF header or trailer")


def atomic_write_pdf(
    attempt_root: Path,
    target: Path,
    value: bytes,
    *,
    writer_id: str = "atomic_write_pdf",
    phase: str = "visualization",
    allow_replace: bool = False,
) -> dict[str, Any]:
    return atomic_write_bytes(
        attempt_root,
        target,
        value,
        artifact_type="pdf",
        writer_id=writer_id,
        phase=phase,
        allow_replace=allow_replace,
        validator=validate_pdf,
    )


def atomic_copy(
    attempt_root: Path,
    source: Path,
    target: Path,
    *,
    writer_id: str = "atomic_copy",
    phase: str = "contract_snapshot",
) -> dict[str, Any]:
    source_sha = file_sha256(source)

    def validate(path: Path) -> None:
        if file_sha256(path) != source_sha:
            raise ValueError("copy checksum mismatch")

    return atomic_write_with(
        attempt_root,
        target,
        lambda path: shutil.copyfile(source, path),
        validate,
        artifact_type="contract_snapshot",
        writer_id=writer_id,
        phase=phase,
        allow_replace=False,
    )


def temporary_file_leaks(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".tmp") or (path.name.startswith(".") and ".tmp" in path.name))
    )


def _windows_component_legal(component: str) -> bool:
    stem = component.split(".", 1)[0].upper()
    return (
        component not in {"", ".", ".."}
        and not any(character in '<>:"/\\|?*\0' for character in component)
        and not component.endswith((" ", "."))
        and stem not in _WINDOWS_RESERVED
        and len(component) <= 255
    )


def audit_path_plan(
    *,
    attempt_root_posix: str,
    path_records: Sequence[Mapping[str, Any]],
    renderer_registry_keys: Sequence[str],
    expected_counts: Mapping[str, int],
) -> dict[str, Any]:
    relative_paths = [str(record["relative_path"]) for record in path_records]
    traversal = [path for path in relative_paths if ".." in _parts(path)]
    absolute = [
        path for path in relative_paths
        if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()
    ]
    attempt_001 = [path for path in relative_paths if PRESERVED_ATTEMPT in _parts(path)]
    normalized = [PurePosixPath(path).as_posix() for path in relative_paths]
    duplicate_count = len(normalized) - len(set(normalized))
    casefold_groups: dict[str, set[str]] = {}
    for path in normalized:
        casefold_groups.setdefault(path.casefold(), set()).add(path)
    collisions = [sorted(values) for values in casefold_groups.values() if len(values) > 1]
    root = PurePosixPath(attempt_root_posix)
    absolute_targets = [(root / path).as_posix() for path in normalized]
    foreign = [path for path in absolute_targets if not path.startswith(root.as_posix() + "/")]
    parent_paths = sorted({PurePosixPath(path).parent.as_posix() for path in normalized})
    windows_legal = all(
        all(_windows_component_legal(component) for component in PurePosixPath(path).parts)
        for path in normalized
    )
    linux_legal = all(
        "\0" not in path
        and len((root / path).as_posix()) <= 4096
        and all(len(component.encode("utf-8")) <= 255 for component in PurePosixPath(path).parts)
        for path in normalized
    )
    max_path_length = max((len(path) for path in absolute_targets), default=0)
    registry_duplicate_count = len(renderer_registry_keys) - len(set(renderer_registry_keys))
    aggregate_rows = [
        f"{record['artifact_type']}\0{PurePosixPath(str(record['relative_path'])).as_posix()}"
        for record in path_records
    ] + [f"renderer_registry_key\0{key}" for key in renderer_registry_keys]
    aggregate_sha = hashlib.sha256("\n".join(sorted(aggregate_rows)).encode("utf-8")).hexdigest()
    counts = {
        "expected_path_count": len(path_records),
        "unique_path_count": len(set(normalized)),
        "duplicate_path_count": duplicate_count,
        "parent_directory_count": len(parent_paths),
        "collision_count": len(collisions),
        "traversal_count": len(traversal) + len(absolute),
        "attempt_001_target_count": len(attempt_001),
        "foreign_output_root_count": len(foreign),
        "renderer_registry_key_count": len(renderer_registry_keys),
        "renderer_registry_duplicate_count": registry_duplicate_count,
        "max_path_length": max_path_length,
    }
    valid = (
        duplicate_count == 0
        and not collisions
        and not traversal
        and not absolute
        and not attempt_001
        and not foreign
        and registry_duplicate_count == 0
        and windows_legal
        and linux_legal
        and len(renderer_registry_keys) == int(expected_counts["renderer_calls"])
    )
    return {
        "schema_version": "canondressgs.paper.coefficient_headroom_attempt_002_output_path_plan.v1",
        "task_id": REPAIR_TASK_ID,
        "status": "PASS" if valid else "FAIL",
        "attempt_id": WRITABLE_ATTEMPT,
        "attempt_root": attempt_root_posix,
        "materialized": False,
        "counts": counts,
        "expected_execution_counts": dict(expected_counts),
        "windows_legality": "PASS" if windows_legal else "FAIL",
        "windows_max_path_260": "PASS" if max_path_length <= 260 else "REQUIRES_LONG_PATH_SUPPORT",
        "linux_legality": "PASS" if linux_legal else "FAIL",
        "deterministic_aggregate_sha256": aggregate_sha,
        "collisions": collisions,
        "parent_directories": parent_paths,
        "paths": list(path_records),
        "renderer_registry_keys": list(renderer_registry_keys),
    }


def synthetic_write_smoke(smoke_root: Path) -> dict[str, Any]:
    attempt = resolve_attempt_root(
        smoke_root, output_name="synthetic", attempt_id=WRITABLE_ATTEMPT, for_write=True
    )
    if attempt.exists():
        shutil.rmtree(attempt)
    receipts = []
    image = Image.new("RGB", (2, 2), color=(17, 34, 51))
    receipts.append(atomic_save_png(attempt, attempt / "nested/png/synthetic.png", image, phase="smoke"))
    receipts.append(atomic_write_json(
        attempt,
        attempt / "nested/json/payload.json",
        {"status": "PASS", "nested": {"value": 1}},
        phase="smoke",
        required_fields=("status",),
    ))
    receipts.append(atomic_write_svg(
        attempt,
        attempt / "nested/svg/synthetic.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2"/></svg>',
        phase="smoke",
    ))
    receipts.append(atomic_write_pdf(
        attempt,
        attempt / "nested/pdf/synthetic.pdf",
        b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\n%%EOF\n",
        phase="smoke",
    ))
    receipts.append(atomic_write_json(
        attempt,
        attempt / "05_checkpoints/metadata/checkpoint_metadata.json",
        {"status": "PASS", "state_keys": ["trainable_state", "optimizer_state"]},
        phase="smoke",
        required_fields=("status", "state_keys"),
    ))
    receipts.append(atomic_write_bytes(
        attempt,
        attempt / "nested/binary/payload.bin",
        b"coefficient-headroom-smoke\n",
        phase="smoke",
    ))
    receipts.append(atomic_write_json(
        attempt,
        attempt / "08_metrics/synthetic_metrics.json",
        {"status": "PASS", "rows": []},
        phase="smoke",
    ))
    receipts.append(atomic_save_png(
        attempt,
        attempt / "11_visual_sheets/synthetic_sheet.png",
        image,
        phase="smoke",
    ))
    leaks_before_cleanup = temporary_file_leaks(attempt)
    aggregate_sha = canonical_sha256([
        {key: receipt[key] for key in ("relative_path", "bytes", "sha256")}
        for receipt in receipts
    ])
    shutil.rmtree(smoke_root)
    return {
        "status": "PASS" if not leaks_before_cleanup and not smoke_root.exists() else "FAIL",
        "receipt_count": len(receipts),
        "aggregate_sha256": aggregate_sha,
        "temporary_file_leaks": leaks_before_cleanup,
        "cleanup_complete": not smoke_root.exists(),
        "renderer_calls": 0,
        "optimizer_creations": 0,
        "optimizer_steps": 0,
        "gpu_calls": 0,
    }
