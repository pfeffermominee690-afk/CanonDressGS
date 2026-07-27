from __future__ import annotations

from dataclasses import dataclass
import torch

from scene.gaussian_clothing_residuals import AnchorClothingResiduals


@dataclass(frozen=True)
class ClothingGateBundle:
    geometry_gate: torch.Tensor
    appearance_gate: torch.Tensor
    confidence: torch.Tensor | None = None
    gate_source: str = "unspecified"

    def validate(self, anchor_count: int, reference: torch.Tensor) -> "ClothingGateBundle":
        for name in ("geometry_gate", "appearance_gate", "confidence"):
            value = getattr(self, name)
            if value is None and name == "confidence": continue
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (anchor_count, 1):
                raise ValueError(f"{name} must have shape [{anchor_count},1]")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match residual dtype/device")
            if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError(f"{name} must be finite in [0,1]")
        return self

    def apply(self, residuals: AnchorClothingResiduals) -> AnchorClothingResiduals:
        values = [x for x in residuals.as_dict().values() if x is not None]
        if not values: raise ValueError("residual bundle is empty")
        self.validate(values[0].shape[0], values[0])
        opacity_gate = torch.maximum(self.geometry_gate, self.appearance_gate)
        return AnchorClothingResiduals(
            delta_xyz=None if residuals.delta_xyz is None else residuals.delta_xyz * self.geometry_gate,
            delta_log_scaling=None if residuals.delta_log_scaling is None else residuals.delta_log_scaling * self.geometry_gate,
            delta_rotvec=None if residuals.delta_rotvec is None else residuals.delta_rotvec * self.geometry_gate,
            delta_opacity_logit=None if residuals.delta_opacity_logit is None else residuals.delta_opacity_logit * opacity_gate,
            delta_sh0=None if residuals.delta_sh0 is None else residuals.delta_sh0 * self.appearance_gate,
            delta_shN=None if residuals.delta_shN is None else residuals.delta_shN * self.appearance_gate,
        )


class TemporaryReferenceGateAdapter:
    gate_source = "temporary_precomputed_reference_only"

    @classmethod
    def from_reference_gate(cls, gate: torch.Tensor, confidence: torch.Tensor | None = None) -> ClothingGateBundle:
        return ClothingGateBundle(gate, gate, confidence, cls.gate_source)
