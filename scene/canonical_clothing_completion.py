from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from scene.clothing_gate_bundle import ClothingGateBundle
from utils.anchor_graph_utils import validate_anchor_graph


@dataclass(frozen=True)
class CanonicalClothingCompletionOutput:
    completed_anchor_features: torch.Tensor
    geometry_gate: torch.Tensor
    appearance_gate: torch.Tensor
    confidence: torch.Tensor
    observed_clothing_probability: torch.Tensor
    observation_coverage: torch.Tensor
    completion_mask: torch.Tensor
    observed_anchor_features: torch.Tensor
    diagnostics: dict[str, Any]

    def gate_bundle(self) -> ClothingGateBundle:
        return ClothingGateBundle(
            self.geometry_gate, self.appearance_gate, self.confidence,
            gate_source="online_reference_mask_completion",
        )


class GraphResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, hidden, indices, weights):
        neighbor = (hidden[indices] * weights.unsqueeze(-1)).sum(dim=1)
        output = hidden + self.mlp(self.norm(torch.cat((hidden, neighbor), dim=-1)))
        if not torch.isfinite(output).all():
            raise FloatingPointError("graph residual block produced NaN or Inf")
        return output


class CanonicalClothingCompleter(nn.Module):
    completion_version = 1
    gate_source = "online_reference_mask_completion"

    def __init__(self, local_dim: int, global_dim: int, position_dim: int,
                 hidden_dim: int = 128, num_blocks: int = 4) -> None:
        super().__init__()
        for name,value in {"local_dim":local_dim,"global_dim":global_dim,"position_dim":position_dim,"hidden_dim":hidden_dim,"num_blocks":num_blocks}.items():
            if not isinstance(value,int) or isinstance(value,bool) or value<=0:
                raise ValueError(f"{name} must be a positive integer")
        self.local_dim,self.global_dim,self.position_dim=local_dim,global_dim,position_dim
        input_dim=local_dim*2+global_dim+position_dim+2
        self.input_projection=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.SiLU())
        self.blocks=nn.ModuleList(GraphResidualBlock(hidden_dim) for _ in range(num_blocks))
        self.completed_feature_head=nn.Linear(hidden_dim,local_dim)
        self.geometry_gate_head=nn.Linear(hidden_dim,1)
        self.appearance_gate_head=nn.Linear(hidden_dim,1)
        self.confidence_head=nn.Linear(hidden_dim,1)

    def forward(self, global_clothing_embedding, observed_surface_feature,
                observed_clothing_feature, observed_clothing_probability,
                observation_coverage, canonical_anchor_position_features,
                anchor_graph_indices, anchor_graph_weights):
        if global_clothing_embedding.shape != (1,self.global_dim):
            raise ValueError("global_clothing_embedding has invalid shape")
        A=observed_surface_feature.shape[0]
        expected={"observed_surface_feature":(A,self.local_dim),"observed_clothing_feature":(A,self.local_dim),"observed_clothing_probability":(A,1),"observation_coverage":(A,1),"canonical_anchor_position_features":(A,self.position_dim)}
        values=locals()
        for name,shape in expected.items():
            value=values[name]
            if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        for name in ("observed_clothing_probability","observation_coverage"):
            value=values[name]
            if ((value<0)|(value>1)).any(): raise ValueError(f"{name} must be in [0,1]")
        validate_anchor_graph(anchor_graph_indices,anchor_graph_weights,A,anchor_graph_indices.shape[1])
        inputs=torch.cat((observed_surface_feature,observed_clothing_feature,global_clothing_embedding.expand(A,-1),canonical_anchor_position_features,observed_clothing_probability,observation_coverage),dim=-1)
        hidden=self.input_projection(inputs)
        for block in self.blocks: hidden=block(hidden,anchor_graph_indices,anchor_graph_weights)
        predicted_missing=self.completed_feature_head(hidden)
        coverage=observation_coverage
        completed=coverage*observed_clothing_feature+(1-coverage)*predicted_missing
        geometry_missing=torch.sigmoid(self.geometry_gate_head(hidden))
        appearance_missing=torch.sigmoid(self.appearance_gate_head(hidden))
        geometry=coverage*observed_clothing_probability+(1-coverage)*geometry_missing
        appearance=coverage*observed_clothing_probability+(1-coverage)*appearance_missing
        confidence=(coverage+(1-coverage)*torch.sigmoid(self.confidence_head(hidden))).clamp(0,1)
        output=CanonicalClothingCompletionOutput(
            completed,geometry.clamp(0,1),appearance.clamp(0,1),confidence,
            observed_clothing_probability,coverage,(coverage<1-1e-6).to(coverage.dtype),
            observed_clothing_feature,
            {"gate_source":self.gate_source,"anchor_count":A,"hidden_dim":hidden.shape[1],"blocks":len(self.blocks)},
        )
        output.gate_bundle().validate(A,completed)
        return output
