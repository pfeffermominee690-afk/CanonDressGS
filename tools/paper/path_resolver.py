from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = {
    "repo_root": "CANONDRESSGS_REPO_ROOT",
    "asset_root": "CANONDRESSGS_ASSET_ROOT",
    "output_root": "CANONDRESSGS_OUTPUT_ROOT",
}
PLACEHOLDER = re.compile(r"\$\{(CANONDRESSGS_(?:REPO|ASSET|OUTPUT)_ROOT)\}")


@dataclass(frozen=True)
class PaperPaths:
    repo_root: Path
    asset_root: Path | None
    output_root: Path | None

    @property
    def protocol_root(self) -> Path:
        return self.repo_root / "paper_protocol"

    @property
    def formal_output_root(self) -> Path:
        if self.output_root is None:
            raise ValueError("CANONDRESSGS_OUTPUT_ROOT or --output-root is required")
        return self.output_root / "AAAI27-SEEN-OUTFIT-PAPER"

    def experiment_root(self, experiment_id: str, seed: int | None) -> Path:
        seed_name = "fixed" if seed is None else str(seed)
        return self.formal_output_root / experiment_id / f"seed_{seed_name}"


def _resolve(value: str | Path | None, env_name: str, fallback: Path | None) -> Path | None:
    selected = value if value is not None else os.environ.get(env_name)
    if selected is None:
        return fallback
    return Path(selected).expanduser().resolve()


def resolve_paths(
    *,
    repo_root: str | Path | None = None,
    asset_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> PaperPaths:
    repo = _resolve(repo_root, ENV_KEYS["repo_root"], PROJECT_ROOT)
    assert repo is not None
    return PaperPaths(
        repo_root=repo,
        asset_root=_resolve(asset_root, ENV_KEYS["asset_root"], None),
        output_root=_resolve(output_root, ENV_KEYS["output_root"], None),
    )


def expand_protocol_path(value: str, paths: PaperPaths, *, require_resolved: bool = True) -> Path:
    mapping: Mapping[str, Path | None] = {
        ENV_KEYS["repo_root"]: paths.repo_root,
        ENV_KEYS["asset_root"]: paths.asset_root,
        ENV_KEYS["output_root"]: paths.output_root,
    }

    def replacement(match: re.Match[str]) -> str:
        resolved = mapping[match.group(1)]
        if resolved is None:
            if require_resolved:
                raise ValueError(f"unresolved path root: {match.group(1)}")
            return match.group(0)
        return str(resolved)

    expanded = PLACEHOLDER.sub(replacement, value)
    candidate = Path(expanded)
    if not candidate.is_absolute() and not PLACEHOLDER.search(expanded):
        candidate = paths.repo_root / candidate
    return candidate


def contains_user_specific_path(text: str) -> bool:
    normalized = text.replace("\\", "/").lower()
    forbidden = ("/" + "users/", "/" + "model_train/", "/root/" + "autodl-tmp/")
    return any(token in normalized for token in forbidden)
