from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASS = "PASS"
MISMATCH = "PAPER_ASSET_MISMATCH"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_metadata_fingerprint(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = item.stat()
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def source_bundle_fingerprint(repo_root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        blob = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"], cwd=repo_root,
        )
        file_hash = hashlib.sha256(blob).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_path(template: str, repo_root: Path, output_root: Path | None) -> Path:
    values = {"CANONDRESSGS_REPO_ROOT": str(repo_root)}
    if output_root is not None:
        values["CANONDRESSGS_OUTPUT_ROOT"] = str(output_root)
    resolved = template
    for name, value in values.items():
        resolved = resolved.replace("${" + name + "}", value)
    if "${" in resolved:
        raise ValueError(f"unresolved path template: {template}")
    return Path(resolved)


def _checkpoint_field(path: Path, field: str) -> str:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    value: Any = payload
    for part in field.split("."):
        value = value[part]
    if not isinstance(value, str):
        raise TypeError(f"checkpoint field is not a string: {field}")
    return value


def _commit_exists(repo_root: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def verify_manifest(
    manifest: Mapping[str, Any], repo_root: Path, output_root: Path | None,
    *, verify_external: bool,
) -> dict[str, Any]:
    results = []
    for asset in manifest["assets"]:
        asset_id = asset["asset_id"]
        mode = asset["verification"]
        expected = asset["fingerprint"]
        actual: str | None = None
        skipped = False
        error: str | None = None
        try:
            if mode == "file_sha256":
                if not verify_external:
                    skipped = True
                else:
                    actual = sha256_file(resolve_path(asset["path"], repo_root, output_root))
            elif mode == "checkpoint_field":
                if not verify_external:
                    skipped = True
                else:
                    source = resolve_path(asset["source_path"], repo_root, output_root)
                    actual = _checkpoint_field(source, asset["field"])
            elif mode == "git_commit":
                actual = expected if _commit_exists(repo_root, expected) else "MISSING_COMMIT"
            elif mode == "source_bundle_sha256":
                actual = source_bundle_fingerprint(repo_root, asset["paths"])
            elif mode == "tree_metadata_sha256":
                if not verify_external:
                    skipped = True
                else:
                    actual = tree_metadata_fingerprint(
                        resolve_path(asset["path"], repo_root, output_root)
                    )
            else:
                raise ValueError(f"unknown verification mode: {mode}")
        except Exception as exception:  # evidence must survive a failed preflight
            error = f"{type(exception).__name__}: {exception}"
        passed = skipped or (error is None and actual == expected)
        results.append({
            "asset_id": asset_id,
            "verification": mode,
            "expected": expected,
            "actual": actual,
            "skipped_external": skipped,
            "pass": passed,
            "error": error,
        })
    failed = [item["asset_id"] for item in results if not item["pass"]]
    return {
        "status": PASS if not failed else MISMATCH,
        "asset_count": len(results),
        "verified_external": verify_external,
        "failed_assets": failed,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen seen-outfit paper assets")
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "paper_protocol/frozen_asset_manifest.json",
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(os.environ["CANONDRESSGS_OUTPUT_ROOT"])
        if "CANONDRESSGS_OUTPUT_ROOT" in os.environ else None,
    )
    parser.add_argument("--allow-missing-external", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = verify_manifest(
        manifest, args.repo_root.resolve(), args.output_root,
        verify_external=not args.allow_missing_external,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != PASS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
