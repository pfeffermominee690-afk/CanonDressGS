from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from scene.gaussian_clothing_residuals import AnchorClothingResiduals


class AnchorClothingMLP(nn.Module):
    """Predict raw canonical offsets at shared anchors.

    SiLU hidden activations provide a smooth mapping from anchor and clothing
    features to continuous geometric and appearance offsets.
    """

    output_dim = 7

    def __init__(
        self,
        anchor_feature_dim: int,
        clothing_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        dimensions = {
            "anchor_feature_dim": anchor_feature_dim,
            "clothing_dim": clothing_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value)!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.anchor_feature_dim = anchor_feature_dim
        self.clothing_dim = clothing_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.local_feature_dim: int | None = None
        self.local_feature_adapter: nn.Linear | None = None

        input_dim = anchor_feature_dim + clothing_dim
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.hidden_layers.extend(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        )
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        self.scaling_head = self._zero_head(3)
        self.rotation_head = self._zero_head(3)
        self.opacity_head = self._zero_head(1)
        self.sh0_head = self._zero_head(3)
        self.shN_head: nn.Linear | None = None
        self.channel_bounds = {
            "xyz": 0.05, "log_scaling": 0.35, "rotation": 0.2617993878,
            "opacity_logit": 2.0, "sh0": 0.25, "shN": 0.10,
        }
        self.enabled_residual_channels = {"delta_xyz"}

    @property
    def xyz_head(self) -> nn.Linear:
        """Legacy-compatible seven-row layer; rows 0:3 are the xyz head."""

        return self.output_layer

    def initialize_six_channel_heads(self, shN_flat_dim: int) -> None:
        if not isinstance(shN_flat_dim, int) or isinstance(shN_flat_dim, bool) or shN_flat_dim <= 0:
            raise ValueError("shN_flat_dim must be a positive integer")
        if self.shN_head is not None:
            if self.shN_head.out_features != shN_flat_dim:
                raise ValueError("shN head has already been initialized with another width")
            return
        self.shN_head = self._zero_head(shN_flat_dim).to(
            device=self.output_layer.weight.device,
            dtype=self.output_layer.weight.dtype,
        )

    def configure_six_channel_decoder(self, config: dict, shN_flat_dim: int) -> None:
        self.initialize_six_channel_heads(shN_flat_dim)
        names = {
            "xyz": ("delta_xyz", "max_abs"), "log_scaling": ("delta_log_scaling", "max_abs"),
            "rotation": ("delta_rotvec", "max_angle_rad"), "opacity_logit": ("delta_opacity_logit", "max_abs"),
            "sh0": ("delta_sh0", "max_abs"), "shN": ("delta_shN", "max_abs"),
        }
        enabled = set()
        for key, (field, bound_key) in names.items():
            value = config.get(key)
            if not isinstance(value, dict) or not isinstance(value.get("enabled"), bool):
                raise ValueError(f"dressable_channels.{key} requires boolean enabled")
            bound = float(value.get(bound_key, 0))
            if not torch.isfinite(torch.tensor(bound)) or bound <= 0:
                raise ValueError(f"dressable_channels.{key}.{bound_key} must be positive and finite")
            self.channel_bounds[key] = bound
            if value["enabled"]: enabled.add(field)
        self.enabled_residual_channels = enabled

    def forward_film_six_channel(
        self, anchor_features: torch.Tensor, film_gamma: list[torch.Tensor],
        film_beta: list[torch.Tensor], anchor_clothing_features: torch.Tensor | None = None,
    ) -> AnchorClothingResiduals:
        if self.shN_head is None:
            raise RuntimeError("initialize/configure six-channel heads before forward")
        hidden = self._film_hidden(anchor_features, film_gamma, film_beta, anchor_clothing_features)
        legacy = self.output_layer(hidden)
        raw = {
            "delta_xyz": legacy[:, :3], "delta_log_scaling": self.scaling_head(hidden),
            "delta_rotvec": self.rotation_head(hidden), "delta_opacity_logit": self.opacity_head(hidden),
            "delta_sh0": self.sh0_head(hidden), "delta_shN": self.shN_head(hidden),
        }
        bounded = {
            "delta_xyz": self.channel_bounds["xyz"] * torch.tanh(raw["delta_xyz"]),
            "delta_log_scaling": self.channel_bounds["log_scaling"] * torch.tanh(raw["delta_log_scaling"]),
            "delta_opacity_logit": self.channel_bounds["opacity_logit"] * torch.tanh(raw["delta_opacity_logit"]),
            "delta_sh0": self.channel_bounds["sh0"] * torch.tanh(raw["delta_sh0"]),
            "delta_shN": self.channel_bounds["shN"] * torch.tanh(raw["delta_shN"]),
        }
        norm = torch.linalg.vector_norm(raw["delta_rotvec"], dim=-1, keepdim=True)
        unit_scale = torch.where(
            norm > 1e-6,
            torch.tanh(norm) / torch.clamp(norm, min=1e-6),
            torch.ones_like(norm),
        )
        scale = self.channel_bounds["rotation"] * unit_scale
        bounded["delta_rotvec"] = raw["delta_rotvec"] * scale
        return AnchorClothingResiduals(**{
            name: value if name in self.enabled_residual_channels else torch.zeros_like(value)
            for name, value in bounded.items()
        })

    def forward(
        self,
        anchor_features: torch.Tensor,
        clothing_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict xyz, scaling, and opacity offsets for all anchors."""

        if anchor_features.ndim != 2 or anchor_features.shape[1] != self.anchor_feature_dim:
            raise ValueError(
                "anchor_features must have shape [A, "
                f"{self.anchor_feature_dim}], got {tuple(anchor_features.shape)}"
            )
        if clothing_embedding.shape != (1, self.clothing_dim):
            raise ValueError(
                "clothing_embedding must have shape "
                f"(1, {self.clothing_dim}), got {tuple(clothing_embedding.shape)}"
            )
        if anchor_features.device != clothing_embedding.device:
            raise ValueError("anchor_features and clothing_embedding must be on the same device")
        if anchor_features.dtype != clothing_embedding.dtype:
            raise ValueError("anchor_features and clothing_embedding must have the same dtype")

        num_anchors = anchor_features.shape[0]
        clothing_features = clothing_embedding.expand(num_anchors, -1)
        features = torch.cat((anchor_features, clothing_features), dim=-1)
        output = self.output_layer(self._forward_hidden(features))
        return self._split_output(output)

    def forward_anchor_only(
        self,
        anchor_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the shared MLP without clothing concatenation or modulation."""

        self._validate_anchor_features(anchor_features)
        hidden = self._anchor_input_linear(anchor_features)
        hidden = self.activation(hidden)
        for layer in self.hidden_layers[1:]:
            hidden = self.activation(layer(hidden))
        return self._split_output(self.output_layer(hidden))

    def forward_film(
        self,
        anchor_features: torch.Tensor,
        film_gamma: list[torch.Tensor],
        film_beta: list[torch.Tensor],
        anchor_clothing_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run anchor-only inference with per-hidden-layer FiLM modulation."""

        hidden = self._film_hidden(anchor_features, film_gamma, film_beta, anchor_clothing_features)
        return self._split_output(self.output_layer(hidden))

    def _film_hidden(self, anchor_features, film_gamma, film_beta, anchor_clothing_features):
        self._validate_anchor_features(anchor_features)
        self._validate_film_parameters(film_gamma, film_beta, anchor_features)
        fused = self._fuse_local_features(anchor_features, anchor_clothing_features)
        hidden = self.activation((1 + film_gamma[0]) * self._anchor_input_linear(fused) + film_beta[0])
        for index, layer in enumerate(self.hidden_layers[1:], start=1):
            hidden = self.activation((1 + film_gamma[index]) * layer(hidden) + film_beta[index])
        return hidden

    def _zero_head(self, output_dim: int) -> nn.Linear:
        head = nn.Linear(self.hidden_dim, output_dim)
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        return head

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Deterministic migration: legacy Gate 4 checkpoints predate these heads.
        for name in ("scaling_head", "rotation_head", "opacity_head", "sh0_head", "shN_head"):
            module = getattr(self, name)
            if module is None: continue
            for suffix, tensor in (("weight", module.weight), ("bias", module.bias)):
                key = f"{prefix}{name}.{suffix}"
                if key not in state_dict: state_dict[key] = torch.zeros_like(tensor)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def initialize_local_feature_adapter(self, local_feature_dim: int) -> None:
        """Add a zero-initialized residual adapter without changing old layers."""

        if not isinstance(local_feature_dim, int) or isinstance(local_feature_dim, bool):
            raise TypeError("local_feature_dim must be an int")
        if local_feature_dim <= 0:
            raise ValueError("local_feature_dim must be positive")
        if self.local_feature_adapter is not None:
            if self.local_feature_dim != local_feature_dim:
                raise ValueError(
                    "local feature adapter is already initialized with dimension "
                    f"{self.local_feature_dim}"
                )
            return
        reference = self.hidden_layers[0].weight
        adapter = nn.Linear(local_feature_dim, self.anchor_feature_dim).to(
            device=reference.device, dtype=reference.dtype
        )
        nn.init.zeros_(adapter.weight)
        nn.init.zeros_(adapter.bias)
        self.local_feature_dim = local_feature_dim
        self.local_feature_adapter = adapter

    @property
    def film_hidden_dims(self) -> list[int]:
        """Widths of all hidden layers that receive FiLM modulation."""

        return [layer.out_features for layer in self.hidden_layers]

    def _forward_hidden(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        return hidden

    def _anchor_input_linear(self, anchor_features: torch.Tensor) -> torch.Tensor:
        first_layer = self.hidden_layers[0]
        anchor_weight = first_layer.weight[:, : self.anchor_feature_dim]
        return F.linear(anchor_features, anchor_weight, first_layer.bias)

    def _fuse_local_features(
        self,
        anchor_features: torch.Tensor,
        anchor_clothing_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if anchor_clothing_features is None:
            return anchor_features
        if self.local_feature_adapter is None or self.local_feature_dim is None:
            raise RuntimeError(
                "initialize_local_feature_adapter() before passing local clothing features"
            )
        expected_shape = (anchor_features.shape[0], self.local_feature_dim)
        if not isinstance(anchor_clothing_features, torch.Tensor):
            raise TypeError("anchor_clothing_features must be a torch.Tensor")
        if tuple(anchor_clothing_features.shape) != expected_shape:
            raise ValueError(
                f"anchor_clothing_features must have shape {expected_shape}, got "
                f"{tuple(anchor_clothing_features.shape)}"
            )
        if (
            anchor_clothing_features.device != anchor_features.device
            or anchor_clothing_features.dtype != anchor_features.dtype
        ):
            raise ValueError(
                "anchor_clothing_features must match anchor feature device and dtype"
            )
        if not torch.isfinite(anchor_clothing_features).all():
            raise ValueError("anchor_clothing_features contains NaN or Inf")
        return anchor_features + self.local_feature_adapter(anchor_clothing_features)

    def _validate_anchor_features(self, anchor_features: torch.Tensor) -> None:
        if anchor_features.ndim != 2 or anchor_features.shape[1] != self.anchor_feature_dim:
            raise ValueError(
                "anchor_features must have shape [A, "
                f"{self.anchor_feature_dim}], got {tuple(anchor_features.shape)}"
            )

    def _validate_film_parameters(
        self,
        film_gamma: list[torch.Tensor],
        film_beta: list[torch.Tensor],
        anchor_features: torch.Tensor,
    ) -> None:
        if not isinstance(film_gamma, list) or not isinstance(film_beta, list):
            raise TypeError("film_gamma and film_beta must be lists of tensors")
        expected_count = len(self.hidden_layers)
        if len(film_gamma) != expected_count or len(film_beta) != expected_count:
            raise ValueError(
                f"expected {expected_count} gamma/beta tensors, got "
                f"{len(film_gamma)} gamma and {len(film_beta)} beta"
            )
        for index, (gamma, beta, width) in enumerate(
            zip(film_gamma, film_beta, self.film_hidden_dims)
        ):
            expected_shape = (1, width)
            for name, value in (("gamma", gamma), ("beta", beta)):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"film {name}[{index}] must be a torch.Tensor")
                if tuple(value.shape) != expected_shape:
                    raise ValueError(
                        f"film {name}[{index}] has shape {tuple(value.shape)}, "
                        f"expected {expected_shape}"
                    )
                if value.device != anchor_features.device or value.dtype != anchor_features.dtype:
                    raise ValueError(
                        f"film {name}[{index}] must match anchor feature device and dtype"
                    )

    @staticmethod
    def _split_output(output: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "delta_xyz": output[:, :3],
            "delta_scaling": output[:, 3:6],
            "delta_opacity": output[:, 6:7],
        }
