from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from scene.gaussian_clothing_residuals import CHANNELS, GaussianClothingResiduals


CHANNEL_TO_BOUND = {
    "delta_xyz": "xyz",
    "delta_log_scaling": "log_scaling",
    "delta_rotvec": "rotation",
    "delta_opacity_logit": "opacity_logit",
    "delta_sh0": "sh0",
    "delta_shN": "shN",
}


@dataclass(frozen=True)
class BasisDecomposition:
    basis: "ExplicitGaussianResidualBasis"
    teacher_coefficients: dict[str, torch.Tensor]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReferenceCoefficientOutput:
    coefficients: torch.Tensor
    pooled_local_feature: torch.Tensor
    fused_feature: torch.Tensor


def _validate_bounds(channel_bounds: Mapping[str, float]) -> dict[str, float]:
    if set(channel_bounds) != set(CHANNEL_TO_BOUND.values()):
        raise ValueError("channel_bounds must contain the exact six residual bounds")
    result = {name: float(value) for name, value in channel_bounds.items()}
    if any(value <= 0 or not torch.isfinite(torch.tensor(value)) for value in result.values()):
        raise ValueError("all residual bounds must be finite and positive")
    return result


def normalized_residual_dict(
    residuals: GaussianClothingResiduals,
    channel_bounds: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    bounds = _validate_bounds(channel_bounds)
    values = {}
    for name in CHANNELS:
        value = getattr(residuals, name)
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be a finite floating-point tensor")
        values[name] = value / bounds[CHANNEL_TO_BOUND[name]]
    return values


def residual_from_normalized_dict(
    values: Mapping[str, torch.Tensor],
    channel_bounds: Mapping[str, float],
) -> GaussianClothingResiduals:
    bounds = _validate_bounds(channel_bounds)
    if set(values) != set(CHANNELS):
        raise ValueError("normalized residual values must contain exactly six channels")
    physical = {}
    for name in CHANNELS:
        value = values[name]
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"normalized {name} must be finite floating point")
        physical[name] = value * bounds[CHANNEL_TO_BOUND[name]]
    return GaussianClothingResiduals(**physical)


def tensor_mapping_fingerprint(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ExplicitGaussianResidualBasis(nn.Module):
    """A frozen, explicit per-Gaussian field in bound-normalized residual space."""

    representation_type = "explicit_low_rank_per_gaussian_residual_basis"

    def __init__(
        self,
        mean_normalized: Mapping[str, torch.Tensor],
        basis_normalized: Mapping[str, torch.Tensor],
        channel_bounds: Mapping[str, float],
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.channel_bounds = _validate_bounds(channel_bounds)
        if set(mean_normalized) != set(CHANNELS) or set(basis_normalized) != set(CHANNELS):
            raise ValueError("mean and basis must contain the exact six residual channels")
        ranks = {int(basis_normalized[name].shape[0]) for name in CHANNELS}
        if len(ranks) != 1 or next(iter(ranks)) <= 0:
            raise ValueError("all basis channels must share one positive rank")
        self.rank = ranks.pop()
        gaussian_counts = set()
        self._shapes: dict[str, tuple[int, ...]] = {}
        for name in CHANNELS:
            mean = mean_normalized[name].detach().clone()
            basis = basis_normalized[name].detach().clone()
            if basis.shape[1:] != mean.shape or mean.ndim < 1:
                raise ValueError(f"basis {name} must have shape [K,*mean.shape]")
            if not torch.is_floating_point(mean) or not torch.isfinite(mean).all():
                raise ValueError(f"mean {name} must be finite floating point")
            if basis.device != mean.device or basis.dtype != mean.dtype or not torch.isfinite(basis).all():
                raise ValueError(f"basis {name} must match the finite mean tensor")
            gaussian_counts.add(int(mean.shape[0]))
            self._shapes[name] = tuple(int(value) for value in mean.shape)
            if trainable:
                self.register_parameter(f"mean__{name}", nn.Parameter(mean))
                self.register_parameter(f"basis__{name}", nn.Parameter(basis))
            else:
                self.register_buffer(f"mean__{name}", mean)
                self.register_buffer(f"basis__{name}", basis)
        if len(gaussian_counts) != 1:
            raise ValueError("all six channels must share one Gaussian count")
        self.gaussian_count = gaussian_counts.pop()

    @property
    def explicit_scalar_count(self) -> int:
        return sum(
            getattr(self, f"mean__{name}").numel() + getattr(self, f"basis__{name}").numel()
            for name in CHANNELS
        )

    @property
    def mean_scalar_count(self) -> int:
        return sum(getattr(self, f"mean__{name}").numel() for name in CHANNELS)

    @property
    def basis_scalar_count(self) -> int:
        return sum(getattr(self, f"basis__{name}").numel() for name in CHANNELS)

    def normalized_fields(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            {name: getattr(self, f"mean__{name}") for name in CHANNELS},
            {name: getattr(self, f"basis__{name}") for name in CHANNELS},
        )

    def fingerprint(self) -> str:
        mean, basis = self.normalized_fields()
        return tensor_mapping_fingerprint({
            **{f"mean/{name}": value for name, value in mean.items()},
            **{f"basis/{name}": value for name, value in basis.items()},
        })

    def compose_normalized(
        self,
        coefficients: torch.Tensor,
        *,
        gaussian_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        reference = getattr(self, f"mean__{CHANNELS[0]}")
        if coefficients.ndim == 2 and coefficients.shape[0] == 1:
            coefficients = coefficients[0]
        if coefficients.shape != (self.rank,):
            raise ValueError(f"coefficients must have shape [{self.rank}] or [1,{self.rank}]")
        if coefficients.device != reference.device or coefficients.dtype != reference.dtype:
            raise ValueError("coefficients must share basis device and dtype")
        if not torch.isfinite(coefficients).all():
            raise ValueError("coefficients contain NaN or Inf")
        if gaussian_indices is None:
            indices = torch.arange(self.gaussian_count, device=reference.device)
        else:
            indices = gaussian_indices
            if indices.ndim != 1 or indices.dtype != torch.long or indices.device != reference.device:
                raise ValueError("gaussian_indices must be one-dimensional long on the basis device")
            if indices.numel() and (indices.min() < 0 or indices.max() >= self.gaussian_count):
                raise IndexError("gaussian_indices are out of range")
        actual_chunk = int(chunk_size or max(1, indices.numel()))
        if actual_chunk <= 0:
            raise ValueError("chunk_size must be positive")
        parts: dict[str, list[torch.Tensor]] = {name: [] for name in CHANNELS}
        for start in range(0, int(indices.numel()), actual_chunk):
            selected = indices[start:start + actual_chunk]
            for name in CHANNELS:
                mean = getattr(self, f"mean__{name}").index_select(0, selected)
                basis = getattr(self, f"basis__{name}").index_select(1, selected)
                parts[name].append(mean + torch.tensordot(coefficients, basis, dims=([0], [0])))
        return {
            name: torch.cat(values, dim=0) if values else reference.new_empty((0, *self._shapes[name][1:]))
            for name, values in parts.items()
        }

    def forward(
        self,
        coefficients: torch.Tensor,
        *,
        gaussian_indices: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> GaussianClothingResiduals:
        normalized = self.compose_normalized(
            coefficients, gaussian_indices=gaussian_indices, chunk_size=chunk_size,
        )
        return residual_from_normalized_dict(normalized, self.channel_bounds)


def build_centered_difference_basis(
    teacher_residuals: Mapping[str, GaussianClothingResiduals],
    channel_bounds: Mapping[str, float],
    outfit_order: Sequence[str],
) -> BasisDecomposition:
    order = tuple(outfit_order)
    if len(order) != 2 or set(teacher_residuals) != set(order) or order[0] == order[1]:
        raise ValueError("centered-difference decomposition requires exactly two ordered teachers")
    first = normalized_residual_dict(teacher_residuals[order[0]], channel_bounds)
    second = normalized_residual_dict(teacher_residuals[order[1]], channel_bounds)
    for name in CHANNELS:
        if first[name].shape != second[name].shape:
            raise ValueError(f"teacher shape mismatch for {name}")
        if first[name].device != second[name].device or first[name].dtype != second[name].dtype:
            raise ValueError(f"teacher device/dtype mismatch for {name}")
    mean = {name: (first[name] + second[name]) * 0.5 for name in CHANNELS}
    difference = {name: ((second[name] - first[name]) * 0.5).unsqueeze(0) for name in CHANNELS}
    basis = ExplicitGaussianResidualBasis(mean, difference, channel_bounds)
    reference = first[CHANNELS[0]]
    coefficients = {
        order[0]: reference.new_tensor([-1.0]),
        order[1]: reference.new_tensor([1.0]),
    }
    metadata = {
        "method": "ordered_two_outfit_centered_half_difference",
        "rank": 1,
        "outfit_order": list(order),
        "teacher_coefficients": {name: value.detach().cpu().tolist() for name, value in coefficients.items()},
        "teacher_fingerprints": {
            name: tensor_mapping_fingerprint(normalized_residual_dict(teacher_residuals[name], channel_bounds))
            for name in order
        },
        "basis_fingerprint": basis.fingerprint(),
        "space": "bound_normalized_six_channel_gaussian_residual",
    }
    return BasisDecomposition(basis, coefficients, metadata)


def _flatten_normalized_fields(values: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
    shapes = {name: tuple(values[name].shape) for name in CHANNELS}
    return torch.cat([values[name].reshape(-1) for name in CHANNELS]), shapes


def _unflatten_normalized_fields(flat: torch.Tensor, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    values, start = {}, 0
    for name in CHANNELS:
        count = int(torch.tensor(shapes[name]).prod().item())
        values[name] = flat[start:start + count].reshape(shapes[name])
        start += count
    if start != flat.numel():
        raise ValueError("flat residual size does not match the six-channel shapes")
    return values


def build_svd_basis(
    teacher_residuals: Mapping[str, GaussianClothingResiduals],
    channel_bounds: Mapping[str, float],
    outfit_order: Sequence[str],
    rank: int,
) -> BasisDecomposition:
    """General M-outfit deterministic SVD path retained for future expansion."""

    order = tuple(outfit_order)
    if len(order) < 2 or set(teacher_residuals) != set(order) or len(set(order)) != len(order):
        raise ValueError("SVD decomposition requires at least two ordered teachers")
    normalized = {name: normalized_residual_dict(teacher_residuals[name], channel_bounds) for name in order}
    flattened = []; shapes = None
    for name in order:
        flat, current_shapes = _flatten_normalized_fields(normalized[name])
        if shapes is None:
            shapes = current_shapes
        elif current_shapes != shapes:
            raise ValueError("all SVD teachers must share six-channel shapes")
        flattened.append(flat)
    matrix = torch.stack(flattened)
    mean_flat = matrix.mean(0); centered = matrix - mean_flat
    maximum_rank = min(len(order) - 1, int(centered.shape[1]))
    if rank <= 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [1,{maximum_rank}]")
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis_flat = vh[:rank].clone()
    for index in range(rank):
        pivot = int(basis_flat[index].abs().argmax())
        if basis_flat[index, pivot] < 0:
            basis_flat[index].neg_()
    coefficient_matrix = centered @ basis_flat.t()
    assert shapes is not None
    mean = _unflatten_normalized_fields(mean_flat, shapes)
    basis_fields = {name: [] for name in CHANNELS}
    for component in basis_flat:
        values = _unflatten_normalized_fields(component, shapes)
        for name in CHANNELS:
            basis_fields[name].append(values[name])
    basis_tensors = {name: torch.stack(values) for name, values in basis_fields.items()}
    basis = ExplicitGaussianResidualBasis(mean, basis_tensors, channel_bounds)
    coefficients = {name: coefficient_matrix[index] for index, name in enumerate(order)}
    metadata = {
        "method": "deterministic_centered_svd", "rank": rank, "outfit_order": list(order),
        "teacher_coefficients": {name: value.detach().cpu().tolist() for name, value in coefficients.items()},
        "teacher_fingerprints": {name: tensor_mapping_fingerprint(normalized[name]) for name in order},
        "basis_fingerprint": basis.fingerprint(), "space": "bound_normalized_six_channel_gaussian_residual",
    }
    return BasisDecomposition(basis, coefficients, metadata)


def project_residual_onto_basis(
    basis: ExplicitGaussianResidualBasis,
    residual: GaussianClothingResiduals,
) -> torch.Tensor:
    """Project a held-out residual onto a frozen explicit basis in normalized space."""

    normalized = normalized_residual_dict(residual, basis.channel_bounds)
    mean, components = basis.normalized_fields()
    target_flat, shapes = _flatten_normalized_fields(normalized)
    mean_flat, mean_shapes = _flatten_normalized_fields(mean)
    if shapes != mean_shapes:
        raise ValueError("held-out residual shape does not match the basis")
    component_flat = torch.stack([_flatten_normalized_fields({name: components[name][index] for name in CHANNELS})[0] for index in range(basis.rank)])
    gram = component_flat @ component_flat.t()
    return torch.linalg.solve(gram, component_flat @ (target_flat - mean_flat))


class DiagnosticCoefficientPredictor(nn.Module):
    """Stage-B tool: maps a diagnostic latent to only K coefficients."""

    def __init__(self, latent_dim: int, rank: int, hidden_dim: int) -> None:
        super().__init__()
        if min(latent_dim, rank, hidden_dim) <= 0:
            raise ValueError("diagnostic coefficient dimensions must be positive")
        self.rank = int(rank)
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, rank),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, diagnostic_latent: torch.Tensor) -> torch.Tensor:
        if diagnostic_latent.ndim == 1:
            diagnostic_latent = diagnostic_latent.unsqueeze(0)
        output = self.network(diagnostic_latent)
        if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] != self.rank:
            raise ValueError("diagnostic coefficient predictor requires one latent row")
        if not torch.isfinite(output).all():
            raise FloatingPointError("diagnostic coefficient predictor produced NaN or Inf")
        return output[0]


class ReferenceBasisCoefficientPredictor(nn.Module):
    """Stage-C head: global reference feature + confidence-pooled local feature -> K coefficients."""

    def __init__(self, global_dim: int, local_dim: int, rank: int, hidden_dim: int) -> None:
        super().__init__()
        if min(global_dim, local_dim, rank, hidden_dim) <= 0:
            raise ValueError("reference coefficient dimensions must be positive")
        self.global_dim = int(global_dim)
        self.local_dim = int(local_dim)
        self.rank = int(rank)
        self.network = nn.Sequential(
            nn.LayerNorm(global_dim + local_dim),
            nn.Linear(global_dim + local_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, rank),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        global_clothing_embedding: torch.Tensor,
        completed_anchor_features: torch.Tensor,
        completion_confidence: torch.Tensor,
    ) -> ReferenceCoefficientOutput:
        if global_clothing_embedding.shape != (1, self.global_dim):
            raise ValueError("global clothing embedding has an invalid shape")
        if completed_anchor_features.ndim != 2 or completed_anchor_features.shape[1] != self.local_dim:
            raise ValueError("completed anchor features have an invalid shape")
        if completion_confidence.shape != (completed_anchor_features.shape[0], 1):
            raise ValueError("completion confidence has an invalid shape")
        tensors = (global_clothing_embedding, completed_anchor_features, completion_confidence)
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("reference coefficient inputs contain NaN or Inf")
        weights = completion_confidence.clamp(0, 1)
        pooled = (completed_anchor_features * weights).sum(0, keepdim=True) / weights.sum().clamp_min(1e-6)
        fused = torch.cat((global_clothing_embedding, pooled), dim=1)
        coefficients = self.network(fused)
        if coefficients.shape != (1, self.rank) or not torch.isfinite(coefficients).all():
            raise FloatingPointError("reference coefficient predictor produced invalid coefficients")
        return ReferenceCoefficientOutput(coefficients[0], pooled, fused)
