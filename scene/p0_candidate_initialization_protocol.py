from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


DETERMINISTIC_ZERO_INITIALIZATION = "DETERMINISTIC_ZERO_INITIALIZATION"
RANDOM_SEEDED_INITIALIZATION = "RANDOM_SEEDED_INITIALIZATION"
PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD = (
    "PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD"
)


@dataclass(frozen=True)
class CandidateOutput:
    pre_transform_output: torch.Tensor
    standardized_coefficients: torch.Tensor
    set_feature: torch.Tensor


@dataclass(frozen=True)
class ClassifierOutput:
    logits: torch.Tensor


@dataclass(frozen=True)
class CandidateBuild:
    identity: str
    module: nn.Module
    initialization_policy: str
    candidate_optimizer_created: bool = False
    candidate_optimizer_step_count: int = 0

    def manifest(self) -> dict[str, Any]:
        names = [name for name, parameter in self.module.named_parameters() if parameter.requires_grad]
        return {
            "identity": self.identity,
            "initialization_policy": self.initialization_policy,
            "trainable_parameter_names": names,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.module.parameters()
                if parameter.requires_grad
            ),
            "initialization_sha256": tensor_mapping_sha256(self.module.state_dict()),
            "candidate_optimizer_created": self.candidate_optimizer_created,
            "candidate_optimizer_step_count": self.candidate_optimizer_step_count,
        }


class OursV2DeterministicZeroCandidate(nn.Module):
    """Protocol-only Ours-v2 identity: LayerNorm -> zero-initialized Linear(4)."""

    initialization_policy = DETERMINISTIC_ZERO_INITIALIZATION

    def __init__(self, input_dim: int = 512, rank: int = 4) -> None:
        super().__init__()
        if input_dim <= 0 or rank <= 0:
            raise ValueError("input_dim and rank must be positive")
        self.input_dim = int(input_dim)
        self.rank = int(rank)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.rank)
        nn.init.ones_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, frozen_f2_set_feature: torch.Tensor) -> CandidateOutput:
        value = _one_row(frozen_f2_set_feature, self.input_dim, "Ours-v2")
        pre_transform = self.linear(self.normalization(value)).reshape(self.rank)
        _require_finite("Ours-v2 output", pre_transform)
        return CandidateOutput(pre_transform, pre_transform, value.reshape(-1))


class B6ReferenceClassifierCandidate(nn.Module):
    """Protocol-only B6 reference classifier with seeded random Linear(5)."""

    initialization_policy = RANDOM_SEEDED_INITIALIZATION

    def __init__(self, input_dim: int = 512, class_count: int = 5) -> None:
        super().__init__()
        if input_dim <= 0 or class_count <= 1:
            raise ValueError("invalid B6 dimensions")
        self.input_dim = int(input_dim)
        self.class_count = int(class_count)
        self.normalization = nn.LayerNorm(self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.class_count)

    def forward(self, frozen_f2_set_feature: torch.Tensor) -> ClassifierOutput:
        value = _one_row(frozen_f2_set_feature, self.input_dim, "B6")
        logits = self.linear(self.normalization(value)).reshape(self.class_count)
        _require_finite("B6 logits", logits)
        return ClassifierOutput(logits)


class ProtocolComplexTrunk(nn.Module):
    """Protocol-only validity-aware complex reference-set trunk.

    This class does not replace or modify historical LegacyRFFRank4/B5.
    """

    def __init__(self, raw_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if raw_dim <= 0 or hidden_dim <= 0:
            raise ValueError("invalid complex trunk dimensions")
        self.raw_dim = int(raw_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(self.raw_dim),
            nn.Linear(self.raw_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.reference_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.attention_score = nn.Linear(self.hidden_dim, 1)
        self.pooled_projection = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

    def forward(self, packed_reference_rows: torch.Tensor) -> torch.Tensor:
        if packed_reference_rows.ndim == 2 and packed_reference_rows.shape[0] == 1:
            packed_reference_rows = packed_reference_rows[0]
        expected = 3 * self.raw_dim + 3
        if packed_reference_rows.ndim != 1 or packed_reference_rows.numel() != expected:
            raise ValueError("complex packed reference-token shape mismatch")
        raw = packed_reference_rows[: 3 * self.raw_dim].reshape(3, self.raw_dim)
        valid = packed_reference_rows[3 * self.raw_dim :].reshape(3, 1)
        if valid.sum().item() <= 0:
            raise ValueError("complex reference set is empty")
        binary_valid = valid > 0
        features = self.reference_mlp(self.token_adapter(raw)) * valid
        denominator = valid.sum(dim=0, keepdim=True).clamp_min(1e-8)
        mean_feature = (features * valid).sum(dim=0, keepdim=True) / denominator
        lowest = torch.finfo(features.dtype).min
        max_feature = features.masked_fill(~binary_valid, lowest).amax(dim=0, keepdim=True)
        logits = self.attention_score(features).masked_fill(~binary_valid, lowest)
        attention = torch.softmax(logits, dim=0)
        attention_feature = (features * attention).sum(dim=0, keepdim=True)
        pooled = self.pooled_projection(
            torch.cat((mean_feature, max_feature, attention_feature), dim=-1)
        )
        _require_finite("complex pooled feature", pooled)
        return pooled


class ProtocolComplexCandidate(nn.Module):
    """M3/M4 candidate with random trunk and a deterministic zero output head."""

    initialization_policy = PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD

    def __init__(
        self, trunk: ProtocolComplexTrunk, *, identity: str, legacy_tanh: bool
    ) -> None:
        super().__init__()
        if identity not in {"M3", "M4"}:
            raise ValueError("complex identity must be M3 or M4")
        self.identity = identity
        self.legacy_tanh = bool(legacy_tanh)
        self.trunk = trunk
        self.output_head = nn.Linear(trunk.hidden_dim, 4)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, packed_reference_rows: torch.Tensor) -> CandidateOutput:
        set_feature = self.trunk(packed_reference_rows)
        pre_transform = self.output_head(set_feature).reshape(4)
        standardized = torch.tanh(pre_transform) if self.legacy_tanh else pre_transform
        _require_finite(f"{self.identity} output", standardized)
        return CandidateOutput(pre_transform, standardized, set_feature.reshape(-1))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ours_v2_candidate(
    *, input_dim: int = 512, seed: int = 0, device: torch.device | str = "cpu"
) -> CandidateBuild:
    seed_all(seed)
    module = OursV2DeterministicZeroCandidate(input_dim=input_dim, rank=4).to(device)
    return CandidateBuild(
        identity="Ours-v2",
        module=module,
        initialization_policy=DETERMINISTIC_ZERO_INITIALIZATION,
    )


def build_b6_candidate(
    *, input_dim: int = 512, seed: int, device: torch.device | str = "cpu"
) -> CandidateBuild:
    seed_all(seed)
    module = B6ReferenceClassifierCandidate(input_dim=input_dim, class_count=5).to(device)
    return CandidateBuild(
        identity="B6",
        module=module,
        initialization_policy=RANDOM_SEEDED_INITIALIZATION,
    )


def build_m3_m4_paired_candidates(
    *, raw_dim: int, seed: int, device: torch.device | str = "cpu"
) -> tuple[CandidateBuild, CandidateBuild]:
    seed_all(seed)
    shared_initial_trunk = ProtocolComplexTrunk(raw_dim=raw_dim)
    m3 = ProtocolComplexCandidate(
        copy.deepcopy(shared_initial_trunk), identity="M3", legacy_tanh=False
    ).to(device)
    m4 = ProtocolComplexCandidate(
        copy.deepcopy(shared_initial_trunk), identity="M4", legacy_tanh=True
    ).to(device)
    return (
        CandidateBuild("M3", m3, PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD),
        CandidateBuild("M4", m4, PAIRED_RANDOM_TRUNK_WITH_ZERO_OUTPUT_HEAD),
    )


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def selected_state_sha256(module: nn.Module, prefix: str) -> str:
    selected = {
        name: value
        for name, value in module.state_dict().items()
        if name == prefix or name.startswith(prefix + ".")
    }
    return "NOT_PRESENT" if not selected else tensor_mapping_sha256(selected)


def _one_row(value: torch.Tensor, width: int, label: str) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.shape != (1, width):
        raise ValueError(f"{label} input must have shape [1,{width}]")
    _require_finite(f"{label} input", value)
    return value


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
