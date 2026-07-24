from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from PIL import Image


OUTPUT_NAME = "LOO-BASIS-ADAPTATION-001"
ATTEMPT_NAME = "attempt_002"
TEMPORARY_MARKER = ".loo-tmp-"


class LOOOutputError(RuntimeError):
    pass


def strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attempt_path(asset_root: Path) -> Path:
    return asset_root.resolve() / OUTPUT_NAME / ATTEMPT_NAME


def assert_path_inside_attempt(attempt: Path, target: Path) -> Path:
    root = attempt.resolve()
    absolute = target.absolute()
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise LOOOutputError(f"LOO_OUTPUT_PATH_OUTSIDE_ATTEMPT: {absolute}") from error
    if root.name != ATTEMPT_NAME or root.parent.name != OUTPUT_NAME:
        raise LOOOutputError(f"invalid LOO attempt root: {root}")
    for parent in (absolute.parent, *absolute.parent.parents):
        if parent == root.parent:
            break
        if parent.exists() and parent.is_symlink():
            raise LOOOutputError(f"symlink is forbidden inside attempt: {parent}")
    return absolute


def ensure_directory(attempt: Path, directory: Path) -> Path:
    target = assert_path_inside_attempt(attempt, directory)
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        raise LOOOutputError(f"failed to create output directory: {target}")
    return target


def ensure_parent(attempt: Path, target: Path) -> Path:
    absolute = assert_path_inside_attempt(attempt, target)
    ensure_directory(attempt, absolute.parent)
    return absolute


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write_with(
    attempt: Path,
    target: Path,
    writer: Callable[[Path], None],
    validator: Callable[[Path], None] | None = None,
    *,
    allow_replace: bool = False,
) -> dict[str, Any]:
    destination = ensure_parent(attempt, target)
    if destination.exists() and not allow_replace:
        raise FileExistsError(destination)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{destination.name}{TEMPORARY_MARKER}", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        writer(temporary)
        if not temporary.is_file():
            raise LOOOutputError(f"writer did not create temporary file: {temporary}")
        if validator is not None:
            validator(temporary)
        _fsync_file(temporary)
        if destination.exists() and not allow_replace:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }


def atomic_write_bytes(
    attempt: Path, target: Path, value: bytes, *, allow_replace: bool = False
) -> dict[str, Any]:
    return atomic_write_with(
        attempt, target, lambda path: path.write_bytes(value), allow_replace=allow_replace
    )


def atomic_write_text(
    attempt: Path, target: Path, value: str, *, allow_replace: bool = False
) -> dict[str, Any]:
    encoded = value.encode("utf-8")
    return atomic_write_bytes(attempt, target, encoded, allow_replace=allow_replace)


def atomic_write_json(
    attempt: Path, target: Path, value: Any, *, allow_replace: bool = False
) -> dict[str, Any]:
    text = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, default=str
    ) + "\n"

    def validate(path: Path) -> None:
        parsed = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_object)
        if canonical_sha256(parsed) != canonical_sha256(value):
            raise LOOOutputError("JSON roundtrip changed the payload")

    return atomic_write_with(
        attempt,
        target,
        lambda path: path.write_text(text, encoding="utf-8", newline="\n"),
        validate,
        allow_replace=allow_replace,
    )


def atomic_append_jsonl(attempt: Path, target: Path, value: Mapping[str, Any]) -> None:
    destination = ensure_parent(attempt, target)
    line = json.dumps(dict(value), sort_keys=True, ensure_ascii=False, default=str) + "\n"
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_torch_save(
    attempt: Path,
    target: Path,
    value: Any,
    validator: Callable[[Path], None],
    *,
    allow_replace: bool = False,
) -> dict[str, Any]:
    import torch

    return atomic_write_with(
        attempt,
        target,
        lambda path: torch.save(value, path),
        validator,
        allow_replace=allow_replace,
    )


def validate_png(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.width <= 0 or image.height <= 0 or image.mode not in {"L", "RGB", "RGBA"}:
            raise LOOOutputError(f"invalid PNG payload: {path}")


def atomic_save_png(
    attempt: Path, target: Path, image: Image.Image, *, allow_replace: bool = False
) -> dict[str, Any]:
    return atomic_write_with(
        attempt,
        target,
        lambda path: image.save(path, format="PNG", compress_level=6),
        validate_png,
        allow_replace=allow_replace,
    )


def temporary_file_leaks(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(path) for path in root.rglob("*")
        if path.is_file() and TEMPORARY_MARKER in path.name
    )


def audit_relative_paths(paths: Iterable[str]) -> dict[str, Any]:
    rows = []
    seen: set[str] = set()
    for raw in paths:
        value = raw.replace("\\", "/")
        parts = Path(value).parts
        valid = (
            bool(parts)
            and not Path(value).is_absolute()
            and ".." not in parts
            and value not in seen
            and not value.startswith("/")
        )
        rows.append({"path": value, "valid": valid, "duplicate": value in seen})
        seen.add(value)
    return {
        "status": "PASS" if all(row["valid"] for row in rows) else "FAIL",
        "path_count": len(rows),
        "collision_count": len(rows) - len(seen),
        "rows": rows,
        "aggregate_sha256": canonical_sha256([row["path"] for row in rows]),
    }
