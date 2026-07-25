#!/usr/bin/env python3
"""Read-only cloud checkpoint inventory for Subject00 storage adjudication."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickletools
import re
import subprocess
import sys
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".npz"}
TEXT_SUFFIXES = {
    ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".toml",
    ".csv", ".tsv", ".tex", ".log", ".cfg", ".ini", ".sh", ".py",
}
MAX_TEXT_BYTES = 128 * 1024 * 1024
EXPECTED_SHORT_CANARY_PATH = (
    "/root/autodl-tmp/canondressgs_work/outputs/"
    "SUBJECT00-MMLPHUMAN-SHORT-CANARY-001/attempt_001/"
    "checkpoints/step_000000.pth"
)
EXPECTED_SHORT_CANARY_SHA256 = (
    "29d0edb11474f4384b6d409141d17f5da91c98824bd070196d50804cd7cb480a"
)


def utc_iso(timestamp: float | None = None) -> str:
    value = dt.datetime.now(dt.timezone.utc) if timestamp is None else dt.datetime.fromtimestamp(timestamp, dt.timezone.utc)
    return value.isoformat()


def run_lines(command: list[str], cwd: str | None = None) -> list[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.splitlines()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_cache(path: str | None) -> dict[str, dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def save_hash_cache(path: str | None, cache: dict[str, dict[str, Any]]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, sort_keys=True)
        handle.write("\n")


def filesystem_info(path: str) -> dict[str, Any]:
    try:
        line = run_lines(["findmnt", "-n", "-T", path, "-o", "SOURCE,FSTYPE,TARGET"])[0]
        source, fs_type, target = line.split(None, 2)
    except Exception:
        source, fs_type, target = "UNKNOWN", "UNKNOWN", "UNKNOWN"
    stats = os.statvfs(path)
    return {
        "source": source,
        "type": fs_type,
        "mountpoint": target,
        "total_bytes": stats.f_frsize * stats.f_blocks,
        "free_bytes": stats.f_frsize * stats.f_bavail,
    }


def discover_roots() -> tuple[list[str], list[dict[str, str]]]:
    requested = [
        "/root/autodl-tmp/canondressgs_work/outputs",
        "/root/autodl-tmp/canondressgs_work/checkpoints",
        "/root/autodl-tmp/canondressgs_work/models",
        "/root/autodl-tmp/diffusion_piper_handoff",
        "/root/autodl-tmp/datasets",
        "/root/autodl-tmp/outputs",
        "/root/autodl-tmp/checkpoints",
        "/root/autodl-tmp/models",
        "/root/autodl-tmp/.subject00_cloud_verify_THUMAN4-SUBJECT00-CLOUD-VERIFY-EXTRACT-001",
    ]
    roots: list[str] = []
    audit: list[dict[str, str]] = []
    for item in requested:
        exists = os.path.isdir(item)
        audit.append({"path": item, "state": "SCANNED" if exists else "ABSENT"})
        if exists:
            roots.append(os.path.realpath(item))
    return sorted(set(roots)), audit


def iter_checkpoint_paths(roots: Iterable[str]) -> Iterable[tuple[str, Path]]:
    seen: set[str] = set()
    for root in roots:
        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if not os.path.islink(os.path.join(current, name))]
            for name in files:
                path = Path(current, name)
                if path.suffix.lower() not in CHECKPOINT_SUFFIXES or path.is_symlink():
                    continue
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                yield root, Path(real)


def pickle_strings(blob: bytes) -> list[str]:
    values: list[str] = []
    try:
        for opcode, argument, _ in pickletools.genops(blob):
            if opcode.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"} and isinstance(argument, str):
                if argument not in values:
                    values.append(argument)
                if len(values) >= 300:
                    break
    except Exception:
        pass
    return values


def probe_checkpoint(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    keys: list[str] = []
    method = "UNSUPPORTED_CONTAINER"
    error: str | None = None
    try:
        if suffix == ".safetensors":
            with path.open("rb") as handle:
                header_size = int.from_bytes(handle.read(8), "little")
                if header_size > 64 * 1024 * 1024:
                    raise ValueError("safetensors header exceeds audit limit")
                header = json.loads(handle.read(header_size).decode("utf-8"))
            keys = [str(key) for key in header if key != "__metadata__"][:300]
            method = "SAFETENSORS_HEADER"
        elif suffix in {".pth", ".pt", ".ckpt", ".npz"} and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if suffix == ".npz":
                    keys = [name.rsplit("/", 1)[-1].rsplit(".", 1)[0] for name in names][:300]
                    method = "NPZ_MEMBER_NAMES"
                else:
                    data_pickle = next((name for name in names if name.endswith("data.pkl")), None)
                    if data_pickle:
                        info = archive.getinfo(data_pickle)
                        if info.file_size <= 64 * 1024 * 1024:
                            keys = pickle_strings(archive.read(data_pickle))
                            method = "PYTORCH_ZIP_PICKLE_OPCODE_STRINGS"
                        else:
                            method = "PYTORCH_ZIP_DATA_PICKLE_TOO_LARGE"
                    else:
                        keys = names[:300]
                        method = "ZIP_MEMBER_NAMES"
        elif path.stat().st_size <= 64 * 1024 * 1024 and suffix in {".pth", ".pt", ".ckpt"}:
            keys = pickle_strings(path.read_bytes())
            method = "LEGACY_PICKLE_OPCODE_STRINGS"
        elif suffix == ".bin":
            method = "OPAQUE_BIN_NO_EXECUTION"
        else:
            method = "LARGE_LEGACY_CONTAINER_NOT_LOADED"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    lowered = {key.lower() for key in keys}
    joined = "\n".join(lowered)
    model_markers = (
        "state_dict", "model", "network", "gaussian", "canonical", "decoder",
        "encoder", "weight", "bias", "params", "residual",
    )
    optimizer_markers = ("optimizer", "optim_state", "param_groups", "exp_avg")
    continuation_markers = (
        "scheduler", "lr_scheduler", "rng", "random_state", "sampler", "epoch",
        "global_step", "data_position", "batch_idx", "iteration",
    )
    model_detected = any(marker in joined for marker in model_markers)
    optimizer_detected = any(marker in joined for marker in optimizer_markers)
    continuation_detected = any(marker in joined for marker in continuation_markers)
    return {
        "metadata_probe_method": method,
        "metadata_probe_error": error,
        "detected_key_sample": keys[:100],
        "model_state": "DETECTED" if model_detected else ("UNKNOWN" if method.startswith(("OPAQUE", "LARGE", "UNSUPPORTED")) else "NOT_DETECTED"),
        "optimizer_state": "DETECTED" if optimizer_detected else ("UNKNOWN" if method.startswith(("OPAQUE", "LARGE", "UNSUPPORTED")) else "NOT_DETECTED"),
        "scheduler_rng_or_data_position_state": "DETECTED" if continuation_detected else ("UNKNOWN" if method.startswith(("OPAQUE", "LARGE", "UNSUPPORTED")) else "NOT_DETECTED"),
    }


def parse_step(path: str) -> int | None:
    name = os.path.basename(path)
    patterns = [r"(?:step|iter(?:ation)?|epoch)[_-]?(\d+)", r"(?:^|[_-])(\d{3,})(?:\D|$)"]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def path_roles(root: str, path: str) -> tuple[str, str | None, str]:
    relative = os.path.relpath(path, root)
    parts = relative.split(os.sep)
    attempt_index = next((index for index, part in enumerate(parts) if re.fullmatch(r"attempt[_-]?\d+", part, flags=re.IGNORECASE)), None)
    attempt = parts[attempt_index] if attempt_index is not None else None
    if attempt_index is not None:
        experiment_parts = parts[:attempt_index]
        attempt_root = os.path.join(root, *parts[: attempt_index + 1])
    else:
        checkpoint_markers = {"checkpoints", "checkpoint", "ckpts", "models", "weights"}
        marker_index = next((index for index, part in enumerate(parts[:-1]) if part.lower() in checkpoint_markers), None)
        experiment_parts = parts[:marker_index] if marker_index is not None else parts[:-1]
        attempt_root = os.path.dirname(path)
    logical_experiment = "/".join(experiment_parts) if experiment_parts else os.path.basename(os.path.dirname(path))
    return logical_experiment, attempt, attempt_root


def collect_open_files(checkpoint_paths: set[str]) -> dict[str, list[dict[str, Any]]]:
    opened: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proc_root = Path("/proc")
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        fd_root = process_dir / "fd"
        try:
            command = (process_dir / "comm").read_text(errors="replace").strip()
            cmdline = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            for fd in fd_root.iterdir():
                try:
                    target = os.path.realpath(fd)
                except OSError:
                    continue
                if target in checkpoint_paths:
                    opened[target].append({
                        "pid": int(process_dir.name),
                        "command": command,
                        "cmdline_redacted_to_executable": cmdline.split(" ", 1)[0] if cmdline else "",
                    })
        except (OSError, PermissionError):
            continue
    return opened


def worktree_paths_for_reference_scan(repo_root: str) -> list[str]:
    paths = [os.path.realpath(repo_root)]
    git_file = Path(repo_root, ".git")
    try:
        common_dir = run_lines(["git", "-C", repo_root, "rev-parse", "--git-common-dir"])[0]
        common_path = os.path.realpath(os.path.join(repo_root, common_dir))
        if os.path.basename(common_path) == "canondressgs.git":
            bare = common_path
        else:
            bare = common_path
        lines = run_lines(["git", f"--git-dir={bare}", "worktree", "list", "--porcelain"])
        current_path: str | None = None
        for line in lines + [""]:
            if line.startswith("worktree "):
                current_path = line.split(" ", 1)[1]
            elif line.startswith("branch ") and current_path:
                branch = line.split("refs/heads/", 1)[-1]
                if branch in {
                    "research/paper-figure-p0-closure-prep-20260725",
                    "research/subject00-checkpoint-reclamation-adjudication-20260725",
                    "research/mmlphuman-subject00-storage-migration-adjudication-20260725",
                } and os.path.isdir(current_path):
                    paths.append(os.path.realpath(current_path))
            elif not line:
                current_path = None
    except Exception:
        pass
    return sorted(set(paths))


def iter_reference_files(repo_roots: Iterable[str], scan_roots: Iterable[str]) -> Iterable[Path]:
    seen: set[str] = set()
    for repo in repo_roots:
        try:
            tracked = run_lines(["git", "-C", repo, "ls-files"])
        except Exception:
            tracked = []
        for relative in tracked:
            path = Path(repo, relative)
            if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
                continue
            real = os.path.realpath(path)
            if real not in seen and path.stat().st_size <= MAX_TEXT_BYTES:
                seen.add(real)
                yield Path(real)
    for root in scan_roots:
        for current, directories, files in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if not os.path.islink(os.path.join(current, name))]
            for name in files:
                path = Path(current, name)
                if path.suffix.lower() not in TEXT_SUFFIXES or path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                real = os.path.realpath(path)
                if real in seen or size > MAX_TEXT_BYTES:
                    continue
                seen.add(real)
                yield Path(real)


def reference_categories(path: str) -> list[str]:
    lower = path.lower().replace("\\", "/")
    categories: list[str] = []
    if "final_summary" in lower:
        categories.append("FINAL_SUMMARY")
    if "handoff" in lower:
        categories.append("HANDOFF")
    if "checkpoint" in lower and "registry" in lower:
        categories.append("CHECKPOINT_REGISTRY")
    if "figure" in lower or "paper_draft/figures" in lower:
        categories.append("FIGURE_BANK")
    if any(token in lower for token in ("metric", "qualitative", "source_data", "docs/paper", "paper_draft")):
        categories.append("PAPER_QUALITATIVE_OR_METRIC_SOURCE")
    if any(token in lower for token in ("execution_seal", "result_seal", "reporting_seal", "verification")):
        categories.append("SEAL_OR_VERIFICATION")
    if "manifest" in lower:
        categories.append("MANIFEST")
    if "artifact" in lower and "registry" in lower:
        categories.append("ARTIFACT_REGISTRY")
    if not categories:
        categories.append("GENERAL_GIT_JSON_MARKDOWN_REFERENCE")
    return categories


class ByteMultiPatternMatcher:
    """Linear-time byte matcher with failure links."""

    def __init__(self, patterns: Iterable[bytes]) -> None:
        self.transitions: list[dict[int, int]] = [{}]
        self.failures: list[int] = [0]
        self.outputs: list[list[bytes]] = [[]]
        for pattern in patterns:
            state = 0
            for value in pattern:
                next_state = self.transitions[state].get(value)
                if next_state is None:
                    next_state = len(self.transitions)
                    self.transitions[state][value] = next_state
                    self.transitions.append({})
                    self.failures.append(0)
                    self.outputs.append([])
                state = next_state
            self.outputs[state].append(pattern)

        queue: deque[int] = deque()
        for state in self.transitions[0].values():
            queue.append(state)
        while queue:
            state = queue.popleft()
            for value, next_state in self.transitions[state].items():
                queue.append(next_state)
                fallback = self.failures[state]
                while fallback and value not in self.transitions[fallback]:
                    fallback = self.failures[fallback]
                self.failures[next_state] = self.transitions[fallback].get(value, 0)
                self.outputs[next_state].extend(self.outputs[self.failures[next_state]])

    def find(self, content: bytes) -> set[bytes]:
        state = 0
        found: set[bytes] = set()
        for value in content:
            while state and value not in self.transitions[state]:
                state = self.failures[state]
            state = self.transitions[state].get(value, 0)
            if self.outputs[state]:
                found.update(self.outputs[state])
        return found


def scan_references(entries: list[dict[str, Any]], repo_roots: list[str], scan_roots: list[str]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    token_map: dict[bytes, list[tuple[int, str]]] = defaultdict(list)
    basename_counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        basename_counts[entry["basename"]] += 1
    for index, entry in enumerate(entries):
        token_map[entry["path"].encode()].append((index, "ABSOLUTE_PATH"))
        token_map[entry["sha256"].encode()].append((index, "SHA256"))
        if basename_counts[entry["basename"]] == 1:
            token_map[entry["basename"].encode()].append((index, "UNIQUE_BASENAME"))

    matcher = ByteMultiPatternMatcher(token_map)

    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files_audited: list[dict[str, Any]] = []
    for file_index, path in enumerate(iter_reference_files(repo_roots, scan_roots), start=1):
        if file_index % 500 == 0:
            print(f"REFERENCE {file_index}: {path}", file=sys.stderr, flush=True)
        try:
            content = path.read_bytes()
        except OSError:
            continue
        matched_indices: set[int] = set()
        matched_tokens = matcher.find(content)
        for token in matched_tokens:
            targets = token_map[token]
            for index, matched_by in targets:
                record = {
                    "reference_path": str(path),
                    "matched_by": matched_by,
                    "categories": reference_categories(str(path)),
                }
                if record not in references[entries[index]["path"]]:
                    references[entries[index]["path"]].append(record)
                matched_indices.add(index)
        if matched_indices:
            files_audited.append({"path": str(path), "matched_checkpoint_count": len(matched_indices)})
    return files_audited, references


def seal_evidence(attempt_root: str, references: list[dict[str, Any]]) -> dict[str, Any]:
    evidence: list[str] = []
    if os.path.isdir(attempt_root):
        for current, _, files in os.walk(attempt_root):
            for name in files:
                lower = name.lower()
                if any(token in lower for token in ("seal", "final_summary", "final_verification", "handoff", "closure")):
                    evidence.append(os.path.join(current, name))
    for reference in references:
        categories = set(reference["categories"])
        if categories.intersection({"FINAL_SUMMARY", "HANDOFF", "SEAL_OR_VERIFICATION", "MANIFEST"}):
            evidence.append(reference["reference_path"])
    evidence = sorted(set(evidence))
    return {
        "sealed_attempt": bool(evidence),
        "seal_evidence": evidence,
        "byte_exact_archive_copy": "NOT_PROVEN",
    }


def classify_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_sha[entry["sha256"]].append(entry)
        by_experiment[entry["logical_experiment"]].append(entry)

    for group in by_experiment.values():
        known_steps = [item["step"] for item in group if item["step"] is not None]
        highest = max(known_steps) if known_steps else None
        for entry in group:
            lower = entry["basename"].lower()
            entry["is_initialization"] = entry["step"] == 0 or any(token in lower for token in ("init", "initial", "base"))
            entry["is_final_checkpoint"] = any(token in lower for token in ("final", "best")) or (
                highest is not None and entry["step"] == highest and len(set(known_steps)) > 1
            )
            entry["is_intermediate_milestone"] = (
                entry["step"] is not None and highest is not None and 0 < entry["step"] < highest
            )

    duplicate_groups: list[dict[str, Any]] = []
    for sha256, group in sorted(by_sha.items()):
        for entry in group:
            entry["duplicate_copy_count"] = len(group)
            entry["duplicate_paths"] = sorted(item["path"] for item in group if item is not entry)
            entry["is_unique_copy"] = len(group) == 1
        if len(group) <= 1:
            continue
        ranked = sorted(
            group,
            key=lambda item: (
                item["path"] != EXPECTED_SHORT_CANARY_PATH,
                not item["reference_flags"]["paper_or_figure"],
                not item["is_final_checkpoint"],
                len(item["path"]),
                item["path"],
            ),
        )
        keep_path = ranked[0]["path"]
        delete_paths: list[str] = []
        release_bytes = 0
        blockers: dict[str, list[str]] = {}
        for item in ranked[1:]:
            reasons: list[str] = []
            if item["sealed_attempt"]:
                reasons.append("SEALED_TREE_WITHOUT_BYTE_EXACT_ARCHIVE")
            if item["open_by_processes"]:
                reasons.append("OPEN_BY_PROCESS")
            absolute_refs = [ref for ref in item["references"] if ref["matched_by"] == "ABSOLUTE_PATH"]
            if absolute_refs:
                reasons.append("ABSOLUTE_PATH_REFERENCE")
            if reasons:
                blockers[item["path"]] = reasons
            else:
                delete_paths.append(item["path"])
                release_bytes += item["bytes"]
        duplicate_groups.append({
            "sha256": sha256,
            "copy_count": len(group),
            "bytes_per_copy": group[0]["bytes"],
            "paths": [item["path"] for item in ranked],
            "logical_experiments": sorted(set(item["logical_experiment"] for item in group)),
            "recommended_keep_path": keep_path,
            "safe_delete_candidate_paths": delete_paths,
            "blocked_candidate_paths": blockers,
            "projected_reclaimed_bytes": release_bytes,
        })

    duplicate_delete_paths = {
        path
        for group in duplicate_groups
        for path in group["safe_delete_candidate_paths"]
    }
    for entry in entries:
        if entry["path"] == EXPECTED_SHORT_CANARY_PATH:
            classification = "CRITICAL_ACTIVE_KEEP"
            reason = "FORMAL_BASE_INITIALIZATION"
        elif entry["open_by_processes"]:
            classification = "CRITICAL_ACTIVE_KEEP"
            reason = "OPEN_BY_ACTIVE_PROCESS"
        elif entry["reference_flags"]["paper_or_figure"]:
            classification = "CRITICAL_ACTIVE_KEEP"
            reason = "CURRENT_PAPER_OR_FIGURE_REFERENCE"
        elif entry["sealed_attempt"]:
            classification = "SEALED_PROVENANCE_KEEP"
            reason = "SEALED_ATTEMPT_WITHOUT_PROVEN_BYTE_EXACT_ARCHIVE"
        elif entry["is_final_checkpoint"] and entry["is_unique_copy"]:
            classification = "UNIQUE_FINAL_KEEP"
            reason = "UNIQUE_FINAL_OR_HIGHEST_STEP_CHECKPOINT"
        elif entry["path"] in duplicate_delete_paths:
            classification = "DUPLICATE_SAFE_DELETE_CANDIDATE"
            reason = "BYTE_IDENTICAL_COPY_WITH_STABLE_KEEP_PATH_AND_NO_BLOCKER"
        elif (
            "optimizer" in entry["basename"].lower()
            and entry["state_probe"]["model_state"] == "NOT_DETECTED"
            and not entry["references"]
        ):
            classification = "TEMPORARY_OPTIMIZER_STATE_CANDIDATE"
            reason = "OPTIMIZER_ONLY_FILE_FOR_NONSEALED_UNREFERENCED_RUN"
        else:
            classification = "UNKNOWN_KEEP"
            reason = "INSUFFICIENT_EVIDENCE_FOR_SAFE_DELETION_OR_REGENERATION"
        entry["classification"] = classification
        entry["classification_reason"] = reason
        entry["deletion_breaks_attempt_aggregate_sha"] = bool(entry["sealed_attempt"])
        entry["regenerability"] = (
            "POSSIBLE_BUT_NOT_PROVEN" if entry["is_intermediate_milestone"] and not entry["sealed_attempt"]
            else "NOT_PROVEN"
        )
    return duplicate_groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--hash-cache")
    args = parser.parse_args()

    roots, root_audit = discover_roots()
    hash_cache = load_hash_cache(args.hash_cache)
    entries: list[dict[str, Any]] = []
    print(f"Scanning {len(roots)} research roots", file=sys.stderr, flush=True)
    for index, (root, path) in enumerate(iter_checkpoint_paths(roots), start=1):
        stat = path.stat()
        cache_entry = hash_cache.get(str(path), {})
        if cache_entry.get("bytes") == stat.st_size and cache_entry.get("mtime_ns") == stat.st_mtime_ns and cache_entry.get("sha256"):
            checkpoint_sha256 = cache_entry["sha256"]
            print(f"HASH_CACHE {index}: {path} ({stat.st_size} bytes)", file=sys.stderr, flush=True)
        else:
            print(f"HASH {index}: {path} ({stat.st_size} bytes)", file=sys.stderr, flush=True)
            checkpoint_sha256 = sha256_file(path)
            hash_cache[str(path)] = {
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": checkpoint_sha256,
            }
            save_hash_cache(args.hash_cache, hash_cache)
        logical_experiment, attempt, attempt_root = path_roles(root, str(path))
        entries.append({
            "path": str(path),
            "basename": path.name,
            "scan_root": root,
            "logical_experiment": logical_experiment,
            "attempt": attempt,
            "attempt_root": attempt_root,
            "step": parse_step(str(path)),
            "bytes": stat.st_size,
            "sha256": checkpoint_sha256,
            "modified_time_utc": utc_iso(stat.st_mtime),
            "filesystem": filesystem_info(str(path)),
            "state_probe": probe_checkpoint(path),
        })

    print(f"Discovered and hashed {len(entries)} checkpoints", file=sys.stderr, flush=True)
    checkpoint_paths = {entry["path"] for entry in entries}
    open_files = collect_open_files(checkpoint_paths)
    repo_roots = worktree_paths_for_reference_scan(args.repo_root)
    reference_file_audit, references = scan_references(entries, repo_roots, roots)
    seal_cache: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry["open_by_processes"] = open_files.get(entry["path"], [])
        entry["references"] = references.get(entry["path"], [])
        categories = {category for ref in entry["references"] for category in ref["categories"]}
        entry["reference_flags"] = {
            "git_json_or_markdown": bool(entry["references"]),
            "final_summary": "FINAL_SUMMARY" in categories,
            "handoff": "HANDOFF" in categories,
            "checkpoint_registry": "CHECKPOINT_REGISTRY" in categories,
            "figure_bank": "FIGURE_BANK" in categories,
            "paper_qualitative_or_metric_source": "PAPER_QUALITATIVE_OR_METRIC_SOURCE" in categories,
            "paper_or_figure": bool(categories.intersection({"FIGURE_BANK", "PAPER_QUALITATIVE_OR_METRIC_SOURCE"})),
            "seal_or_manifest": bool(categories.intersection({"SEAL_OR_VERIFICATION", "MANIFEST", "ARTIFACT_REGISTRY"})),
        }
        attempt_root = entry["attempt_root"]
        if attempt_root not in seal_cache:
            seal_cache[attempt_root] = seal_evidence(attempt_root, [])
        seal = {
            "sealed_attempt": seal_cache[attempt_root]["sealed_attempt"],
            "seal_evidence": list(seal_cache[attempt_root]["seal_evidence"]),
            "byte_exact_archive_copy": "NOT_PROVEN",
        }
        for reference in entry["references"]:
            categories = set(reference["categories"])
            if categories.intersection({"FINAL_SUMMARY", "HANDOFF", "SEAL_OR_VERIFICATION", "MANIFEST"}):
                seal["seal_evidence"].append(reference["reference_path"])
        seal["seal_evidence"] = sorted(set(seal["seal_evidence"]))
        seal["sealed_attempt"] = bool(seal["seal_evidence"])
        entry.update(seal)

    duplicate_groups = classify_entries(entries)
    sealed_attempts: dict[str, dict[str, Any]] = {}
    for entry in entries:
        root = entry["attempt_root"]
        if root not in sealed_attempts:
            sealed_attempts[root] = {
                "attempt_root": root,
                "checkpoint_paths": [],
                "checkpoint_bytes": 0,
                "sealed_attempt": False,
                "seal_evidence": [],
                "byte_exact_archive_copy": "NOT_PROVEN",
                "recommendation": "UNKNOWN_KEEP",
            }
        item = sealed_attempts[root]
        item["checkpoint_paths"].append(entry["path"])
        item["checkpoint_bytes"] += entry["bytes"]
        item["sealed_attempt"] = item["sealed_attempt"] or entry["sealed_attempt"]
        item["seal_evidence"] = sorted(set(item["seal_evidence"] + entry["seal_evidence"]))
        if item["sealed_attempt"]:
            item["recommendation"] = "SEALED_PROVENANCE_KEEP_OR_ARCHIVE_WHOLE_ATTEMPT"

    critical = next((entry for entry in entries if entry["path"] == EXPECTED_SHORT_CANARY_PATH), None)
    result = {
        "schema_version": "canondressgs.subject00.checkpoint_cloud_scan.v1",
        "task_id": "AAAI27-SUBJECT00-CHECKPOINT-RECLAMATION-ADJUDICATION-001",
        "audit_timestamp_utc": utc_iso(),
        "mutation_count": 0,
        "audit_scratch_hash_cache": args.hash_cache,
        "scan_roots": root_audit,
        "reference_repo_roots": repo_roots,
        "reference_files_with_matches": reference_file_audit,
        "cloud_filesystem": filesystem_info("/root/autodl-tmp"),
        "checkpoint_count": len(entries),
        "checkpoint_total_bytes": sum(entry["bytes"] for entry in entries),
        "checkpoints": sorted(entries, key=lambda item: item["path"]),
        "duplicate_groups": duplicate_groups,
        "sealed_attempts": sorted(sealed_attempts.values(), key=lambda item: item["attempt_root"]),
        "short_canary_step0": {
            "expected_path": EXPECTED_SHORT_CANARY_PATH,
            "expected_sha256": EXPECTED_SHORT_CANARY_SHA256,
            "exists": critical is not None,
            "observed_sha256": critical["sha256"] if critical else None,
            "sha256_matches": bool(critical and critical["sha256"] == EXPECTED_SHORT_CANARY_SHA256),
        },
    }
    json.dump(result, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
