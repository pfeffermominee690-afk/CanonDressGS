#!/usr/bin/env python3
"""Compare deterministic artifact directories by filename and SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--rerun-dir", type=Path, required=True)
    parser.add_argument("--suffix", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, suffixes: List[str]) -> Dict[str, str]:
    files = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
    if ".png" in suffixes:
        from PIL import Image

        for path in files:
            if path.suffix.lower() == ".png":
                with Image.open(path) as image:
                    image.verify()
    return {path.name: sha(path) for path in sorted(files)}


def main() -> int:
    args = parse_args()
    suffixes = [suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}" for suffix in args.suffix]
    original = inventory(args.original_dir, suffixes)
    rerun = inventory(args.rerun_dir, suffixes)
    missing = sorted(set(original) - set(rerun))
    extra = sorted(set(rerun) - set(original))
    mismatched = sorted(name for name in set(original) & set(rerun) if original[name] != rerun[name])
    result = {
        "schema_version": "paper_generated_artifact_idempotence.v1",
        "status": "PASS" if not (missing or extra or mismatched) else "FAIL",
        "file_count": len(original),
        "suffixes": suffixes,
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "original_hashes": original,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("status", "file_count", "missing", "extra", "mismatched")}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
