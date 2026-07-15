from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from scene.gaussian_clothing_residuals import (
    AnchorClothingResiduals,
    CanonicalGaussianOverrides,
    GaussianClothingResiduals,
    compose_canonical_gaussian_overrides,
    interpolate_anchor_clothing_residuals,
)

from scene.anchor_clothing_mlp import AnchorClothingMLP
from scene.clothing_embedding import ClothingEmbedding
from scene.clothing_hypernetwork import ClothingFiLMGenerator
from scene.clothing_gate_bundle import ClothingGateBundle

if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel


class DressableGaussianModel(nn.Module):
    """Compose clothing offsets with an MMLPHuman canonical Gaussian model.

    Direct per-cloth parameters and optional manual debug offsets are kept in
    raw canonical parameter space. The wrapped model remains responsible for
    activation, pose deformation, and rendering.
    """

    def __init__(
        self,
        base_model: GaussianModel | None = None,
        num_clothes: int = 0,
        learnable_xyz: bool = True,
        learnable_scaling: bool = True,
        learnable_opacity: bool = True,
    ) -> None:
        super().__init__()
        self.base_model: GaussianModel | None = None
        self.offset_mode = "direct"
        self.num_clothes = 0
        self._direct_num_clothes = 0
        self._anchor_num_clothes = 0
        self._film_num_clothes = 0
        self._direct_initialized = False
        self.learnable_xyz = learnable_xyz
        self.learnable_scaling = learnable_scaling
        self.learnable_opacity = learnable_opacity
        self.enable_delta_xyz = True
        self.enable_delta_scaling = True
        self.enable_delta_opacity = True
        self.register_parameter("cloth_delta_xyz", None)
        self.register_parameter("cloth_delta_scaling", None)
        self.register_parameter("cloth_delta_opacity", None)
        self.clothing_embedding: ClothingEmbedding | None = None
        self.anchor_clothing_mlp: AnchorClothingMLP | None = None
        self.clothing_film_generator: ClothingFiLMGenerator | None = None
        self.register_buffer("_manual_mask", torch.empty(0), persistent=False)
        self.register_buffer("_manual_delta_xyz", torch.empty(0), persistent=False)
        self.register_buffer("_manual_delta_scaling", torch.empty(0), persistent=False)
        self.register_buffer("_manual_delta_opacity", torch.empty(0), persistent=False)
        self.register_buffer("anchor_features", torch.empty(0))
        self.register_buffer(
            "gaussian_anchor_indices",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer("gaussian_anchor_weights", torch.empty(0))
        if base_model is not None:
            self.set_base_model(base_model)
        if num_clothes < 0:
            raise ValueError(f"num_clothes must be non-negative, got {num_clothes}")
        if num_clothes > 0:
            self.initialize_learnable_clothing_offsets(num_clothes)

    def set_base_model(self, base_model: GaussianModel) -> None:
        """Attach an existing MMLPHuman GaussianModel without taking ownership."""

        self._validate_base_model(base_model)
        self._validate_base_against_learnable_offsets(base_model)
        self.base_model = base_model
        self.clear_manual_clothing_offset()

    def initialize_learnable_clothing_offsets(self, num_clothes: int) -> None:
        """Initialize one direct raw canonical offset table per clothing item.

        Calling this method again intentionally replaces the clothing parameters,
        so any optimizer created from :meth:`clothing_parameters` must be rebuilt.
        """

        if not isinstance(num_clothes, int) or isinstance(num_clothes, bool):
            raise TypeError(f"num_clothes must be an int, got {type(num_clothes)!r}")
        if num_clothes <= 0:
            raise ValueError(f"num_clothes must be positive, got {num_clothes}")

        base_model = self._require_base_model()
        self._direct_num_clothes = num_clothes
        self._direct_initialized = True
        self.cloth_delta_xyz = self._make_clothing_parameter(
            num_clothes,
            base_model._xyz,
            self.learnable_xyz,
        )
        self.cloth_delta_scaling = self._make_clothing_parameter(
            num_clothes,
            base_model._scaling,
            self.learnable_scaling,
        )
        self.cloth_delta_opacity = self._make_clothing_parameter(
            num_clothes,
            base_model._opacity,
            self.learnable_opacity,
        )
        self.offset_mode = "direct"
        self.num_clothes = num_clothes

    def initialize_anchor_clothing_generator(
        self,
        num_clothes: int,
        anchor_features: torch.Tensor,
        gaussian_anchor_indices: torch.Tensor,
        gaussian_anchor_weights: torch.Tensor,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        """Initialize shared anchor conditioning and Gaussian interpolation."""

        if not isinstance(num_clothes, int) or isinstance(num_clothes, bool):
            raise TypeError(f"num_clothes must be an int, got {type(num_clothes)!r}")
        if num_clothes <= 0:
            raise ValueError(f"num_clothes must be positive, got {num_clothes}")

        base_model = self._require_base_model()
        features, indices, weights = self._validate_and_normalize_anchor_inputs(
            anchor_features,
            gaussian_anchor_indices,
            gaussian_anchor_weights,
            base_model,
        )

        clothing_embedding = ClothingEmbedding(num_clothes, embedding_dim).to(
            device=base_model._xyz.device,
            dtype=base_model._xyz.dtype,
        )
        anchor_clothing_mlp = AnchorClothingMLP(
            anchor_feature_dim=features.shape[1],
            clothing_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device=base_model._xyz.device, dtype=base_model._xyz.dtype)

        self.anchor_features = features
        self.gaussian_anchor_indices = indices
        self.gaussian_anchor_weights = weights
        self.clothing_embedding = clothing_embedding
        self.anchor_clothing_mlp = anchor_clothing_mlp
        self.clothing_film_generator = None
        self._anchor_num_clothes = num_clothes
        self._film_num_clothes = 0
        self.offset_mode = "anchor_mlp"
        self.num_clothes = num_clothes

    def initialize_film_clothing_generator(
        self,
        num_clothes: int,
        anchor_features: torch.Tensor,
        gaussian_anchor_indices: torch.Tensor,
        gaussian_anchor_weights: torch.Tensor,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        hyper_hidden_dim: int = 128,
    ) -> None:
        """Initialize embedding, shared anchor MLP, and lightweight FiLM generator."""

        if not isinstance(num_clothes, int) or isinstance(num_clothes, bool):
            raise TypeError(f"num_clothes must be an int, got {type(num_clothes)!r}")
        if num_clothes <= 0:
            raise ValueError(f"num_clothes must be positive, got {num_clothes}")

        base_model = self._require_base_model()
        features, indices, weights = self._validate_and_normalize_anchor_inputs(
            anchor_features,
            gaussian_anchor_indices,
            gaussian_anchor_weights,
            base_model,
        )
        clothing_embedding = ClothingEmbedding(num_clothes, embedding_dim).to(
            device=base_model._xyz.device,
            dtype=base_model._xyz.dtype,
        )
        anchor_clothing_mlp = AnchorClothingMLP(
            anchor_feature_dim=features.shape[1],
            clothing_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device=base_model._xyz.device, dtype=base_model._xyz.dtype)
        clothing_film_generator = ClothingFiLMGenerator(
            clothing_dim=embedding_dim,
            hidden_dims=anchor_clothing_mlp.film_hidden_dims,
            hyper_hidden_dim=hyper_hidden_dim,
        ).to(device=base_model._xyz.device, dtype=base_model._xyz.dtype)

        self.anchor_features = features
        self.gaussian_anchor_indices = indices
        self.gaussian_anchor_weights = weights
        self.clothing_embedding = clothing_embedding
        self.anchor_clothing_mlp = anchor_clothing_mlp
        self.clothing_film_generator = clothing_film_generator
        self._anchor_num_clothes = num_clothes
        self._film_num_clothes = num_clothes
        self.offset_mode = "anchor_film"
        self.num_clothes = num_clothes

    def set_offset_mode(self, mode: str) -> None:
        """Select exactly one learnable clothing-offset parameterization."""

        if mode not in {"direct", "anchor_mlp", "anchor_film"}:
            raise ValueError("offset_mode must be 'direct', 'anchor_mlp', or 'anchor_film'")
        if mode == "direct":
            if not self._direct_initialized:
                raise RuntimeError("direct clothing offsets have not been initialized")
            self.num_clothes = self._direct_num_clothes
        elif mode == "anchor_mlp":
            if self.clothing_embedding is None or self.anchor_clothing_mlp is None:
                raise RuntimeError("anchor clothing generator has not been initialized")
            self.num_clothes = self._anchor_num_clothes
        else:
            if (
                self.clothing_embedding is None
                or self.anchor_clothing_mlp is None
                or self.clothing_film_generator is None
            ):
                raise RuntimeError("FiLM clothing generator has not been initialized")
            self.num_clothes = self._film_num_clothes
        self.offset_mode = mode

    def set_manual_clothing_offset(
        self,
        gaussian_mask: torch.Tensor,
        xyz_offset: torch.Tensor | float | None = None,
        scaling_offset: torch.Tensor | float | None = None,
        opacity_offset: torch.Tensor | float | None = None,
    ) -> None:
        """Set a hand-authored clothing offset over a Gaussian subset.

        The mask is stored as a ``[N, 1]`` floating-point buffer on the same
        device and dtype as the base xyz tensor. Offsets are stored as full-size
        tensors matching their target raw base parameters, so later offset
        composition stays simple and shape-safe.
        """

        base_model = self._require_base_model()
        mask = self._normalize_mask(gaussian_mask, base_model)

        self._manual_mask = mask
        self._manual_delta_xyz = self._normalize_vector_offset(
            xyz_offset,
            base_model._xyz,
            "xyz_offset",
        )
        self._manual_delta_scaling = self._normalize_vector_offset(
            scaling_offset,
            base_model._scaling,
            "scaling_offset",
        )
        self._manual_delta_opacity = self._normalize_opacity_offset(
            opacity_offset,
            base_model._opacity,
            "opacity_offset",
        )

    def clear_manual_clothing_offset(self) -> None:
        """Remove manual clothing offsets and restore zero-offset behavior."""

        if self.base_model is None:
            self._manual_mask = torch.empty(0)
            self._manual_delta_xyz = torch.empty(0)
            self._manual_delta_scaling = torch.empty(0)
            self._manual_delta_opacity = torch.empty(0)
            return

        self._manual_mask = torch.zeros(
            (self.base_model._xyz.shape[0], 1),
            device=self.base_model._xyz.device,
            dtype=self.base_model._xyz.dtype,
        )
        self._manual_delta_xyz = torch.zeros_like(self.base_model._xyz)
        self._manual_delta_scaling = torch.zeros_like(self.base_model._scaling)
        self._manual_delta_opacity = torch.zeros_like(self.base_model._opacity)

    def compute_clothing_offsets(
        self,
        cloth_id: torch.Tensor | int | None = None,
        clothing_embedding: torch.Tensor | None = None,
        anchor_clothing_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compose the active learnable mode with masked manual offsets."""

        base_model = self._require_base_model()
        if self.offset_mode == "direct":
            self._require_id_only_condition(
                cloth_id,
                clothing_embedding,
                anchor_clothing_features,
                "direct",
            )
            learnable_offsets = self._compute_direct_clothing_offsets(cloth_id)
        elif self.offset_mode == "anchor_mlp":
            self._require_id_only_condition(
                cloth_id,
                clothing_embedding,
                anchor_clothing_features,
                "anchor_mlp",
            )
            anchor_offsets = self.compute_anchor_clothing_offsets(cloth_id)
            learnable_offsets = self.interpolate_anchor_offsets_to_gaussians(anchor_offsets)
        elif self.offset_mode == "anchor_film":
            anchor_offsets = self.compute_film_anchor_offsets(
                cloth_id=cloth_id,
                clothing_embedding=clothing_embedding,
                anchor_clothing_features=anchor_clothing_features,
            )
            learnable_offsets = self.interpolate_anchor_offsets_to_gaussians(anchor_offsets)
        else:
            raise RuntimeError(f"unsupported offset_mode: {self.offset_mode!r}")

        mask = self._manual_mask
        if mask.numel() == 0:
            mask = torch.zeros(
                (base_model._xyz.shape[0], 1),
                device=base_model._xyz.device,
                dtype=base_model._xyz.dtype,
            )

        offsets = {
            "delta_xyz": learnable_offsets["delta_xyz"] + self._manual_delta_xyz * mask,
            "delta_scaling": (
                learnable_offsets["delta_scaling"] + self._manual_delta_scaling * mask
            ),
            "delta_opacity": (
                learnable_offsets["delta_opacity"]
                + self._manual_delta_opacity * mask.squeeze(-1)
            ),
        }
        return self._apply_offset_channel_gate(offsets)

    def configure_offset_channels(
        self,
        *,
        enable_delta_xyz: bool = True,
        enable_delta_scaling: bool = True,
        enable_delta_opacity: bool = True,
    ) -> None:
        """Enable output channels without changing the three-head architecture."""

        values = {
            "enable_delta_xyz": enable_delta_xyz,
            "enable_delta_scaling": enable_delta_scaling,
            "enable_delta_opacity": enable_delta_opacity,
        }
        for name, value in values.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
            setattr(self, name, value)

    def compute_anchor_clothing_offsets(
        self,
        cloth_id: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        """Predict raw offsets for every shared canonical anchor."""

        if self.clothing_embedding is None or self.anchor_clothing_mlp is None:
            raise RuntimeError("anchor clothing generator has not been initialized")
        embedding = self.clothing_embedding(cloth_id)
        return self.anchor_clothing_mlp(self.anchor_features, embedding)

    def compute_film_anchor_offsets(
        self,
        cloth_id: torch.Tensor | int | None = None,
        clothing_embedding: torch.Tensor | None = None,
        anchor_clothing_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict FiLM anchor offsets from exactly one clothing condition.

        ``cloth_id`` preserves the ID-conditioned baseline. An external
        ``clothing_embedding`` enables image conditioning without storing the
        embedding on this module. Per-anchor image features are intentionally
        deferred to the next implementation stage.
        """

        if (
            self.clothing_embedding is None
            or self.anchor_clothing_mlp is None
            or self.clothing_film_generator is None
        ):
            raise RuntimeError("FiLM clothing generator has not been initialized")
        if anchor_clothing_features is not None:
            self._validate_anchor_clothing_features(anchor_clothing_features)
        film_parameters = self._compute_film_parameters(
            cloth_id=cloth_id,
            clothing_embedding=clothing_embedding,
        )
        offsets = self.anchor_clothing_mlp.forward_film(
            self.anchor_features,
            film_parameters["film_gamma"],
            film_parameters["film_beta"],
            anchor_clothing_features=anchor_clothing_features,
        )
        return self._apply_offset_channel_gate(offsets)

    def configure_six_channel_decoder(self, channel_config: dict) -> None:
        base = self._require_base_model()
        if self.anchor_clothing_mlp is None:
            raise RuntimeError("anchor clothing generator has not been initialized")
        shN_flat_dim = int(torch.tensor(base._shN.shape[1:]).prod().item())
        shn = channel_config.get("shN", {})
        if shn.get("enabled"):
            if "sh_degree" not in shn:
                raise ValueError("enabled shN requires explicit sh_degree")
            degree = int(shn["sh_degree"])
            capacity = int(round((base._shN.shape[1] + 1) ** 0.5 - 1))
            if degree < 1 or degree > capacity:
                raise ValueError("configured sh_degree exceeds _shN capacity")
        self.anchor_clothing_mlp.configure_six_channel_decoder(channel_config, shN_flat_dim)

    def compute_film_anchor_residuals(
        self, gate_bundle: ClothingGateBundle, *, cloth_id=None,
        clothing_embedding=None, anchor_clothing_features=None,
    ) -> tuple[AnchorClothingResiduals, AnchorClothingResiduals]:
        if self.anchor_clothing_mlp is None or self.clothing_film_generator is None:
            raise RuntimeError("FiLM clothing generator has not been initialized")
        film = self._compute_film_parameters(cloth_id=cloth_id, clothing_embedding=clothing_embedding)
        bounded = self.anchor_clothing_mlp.forward_film_six_channel(
            self.anchor_features, film["film_gamma"], film["film_beta"], anchor_clothing_features
        )
        return bounded, gate_bundle.apply(bounded)

    def interpolate_anchor_offsets_to_gaussians(
        self,
        anchor_offsets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Interpolate anchor offsets to each Gaussian using normalized weights."""

        base_model = self._require_base_model()
        if self.gaussian_anchor_indices.numel() == 0:
            raise RuntimeError("anchor-to-Gaussian interpolation has not been initialized")

        num_anchors = self.anchor_features.shape[0]
        expected_shapes = {
            "delta_xyz": (num_anchors, 3),
            "delta_scaling": (num_anchors, 3),
            "delta_opacity": (num_anchors, 1),
        }
        for key, expected_shape in expected_shapes.items():
            if key not in anchor_offsets:
                raise KeyError(f"anchor_offsets is missing required key: {key}")
            value = anchor_offsets[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"anchor_offsets[{key!r}] must be a torch.Tensor")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"anchor_offsets[{key!r}] has shape {tuple(value.shape)}, "
                    f"expected {expected_shape}"
                )

        weights = self.gaussian_anchor_weights.unsqueeze(-1)

        def interpolate(value: torch.Tensor) -> torch.Tensor:
            gathered = value[self.gaussian_anchor_indices]
            return (gathered * weights).sum(dim=1)

        delta_xyz = interpolate(anchor_offsets["delta_xyz"])
        delta_scaling = interpolate(anchor_offsets["delta_scaling"])
        delta_opacity = interpolate(anchor_offsets["delta_opacity"])
        return {
            "delta_xyz": delta_xyz,
            "delta_scaling": delta_scaling,
            "delta_opacity": delta_opacity.reshape(base_model._opacity.shape),
        }

    def interpolate_anchor_clothing_residuals(
        self,
        anchor_residuals: AnchorClothingResiduals,
    ) -> GaussianClothingResiduals:
        """Interpolate the version-1 full attribute contract to Gaussian space."""

        return interpolate_anchor_clothing_residuals(
            anchor_residuals,
            self.gaussian_anchor_indices,
            self.gaussian_anchor_weights,
            self._require_base_model(),
        )

    def compose_canonical_gaussian_overrides(
        self,
        gaussian_residuals: GaussianClothingResiduals,
        enabled_channels,
    ) -> CanonicalGaussianOverrides:
        """Compose raw canonical attributes without bypassing base pose bases."""

        return compose_canonical_gaussian_overrides(
            self._require_base_model(), gaussian_residuals, enabled_channels
        )

    def clothing_offset_regularization(
        self,
        cloth_id: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        """Return unweighted L2 terms for the selected total clothing offsets."""

        offsets = self.compute_clothing_offsets(cloth_id)
        return {
            "xyz_l2": offsets["delta_xyz"].square().mean(),
            "scaling_l2": offsets["delta_scaling"].square().mean(),
            "opacity_l2": offsets["delta_opacity"].square().mean(),
        }

    def clothing_parameters(self) -> Iterator[nn.Parameter]:
        """Yield parameters from only the active clothing-offset mode."""

        if self.offset_mode == "direct":
            for parameter in (
                self.cloth_delta_xyz,
                self.cloth_delta_scaling,
                self.cloth_delta_opacity,
            ):
                if parameter is not None:
                    yield parameter
            return

        if self.offset_mode == "anchor_mlp":
            if self.clothing_embedding is None or self.anchor_clothing_mlp is None:
                raise RuntimeError("anchor clothing generator has not been initialized")
            yield from self.clothing_embedding.parameters()
            yield from self.anchor_clothing_mlp.parameters()
            return

        if self.offset_mode == "anchor_film":
            if (
                self.clothing_embedding is None
                or self.anchor_clothing_mlp is None
                or self.clothing_film_generator is None
            ):
                raise RuntimeError("FiLM clothing generator has not been initialized")
            yield from self.clothing_embedding.parameters()
            yield from self.anchor_clothing_mlp.parameters()
            yield from self.clothing_film_generator.parameters()
            return

        raise RuntimeError(f"unsupported offset_mode: {self.offset_mode!r}")

    def film_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only FiLM generator parameters for a dedicated optimizer group."""

        if self.clothing_film_generator is None:
            raise RuntimeError("FiLM clothing generator has not been initialized")
        yield from self.clothing_film_generator.parameters()

    def film_regularization(
        self,
        cloth_id: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        """Return unweighted L2 penalties for generated FiLM parameters."""

        film_parameters = self._compute_film_parameters(cloth_id)
        gamma = torch.cat(film_parameters["film_gamma"], dim=-1)
        beta = torch.cat(film_parameters["film_beta"], dim=-1)
        return {
            "gamma_l2": gamma.square().mean(),
            "beta_l2": beta.square().mean(),
        }

    @torch.no_grad()
    def get_film_statistics(
        self,
        cloth_id: torch.Tensor | int,
    ) -> dict[str, float]:
        """Return detached FiLM magnitude diagnostics for logging."""

        film_parameters = self._compute_film_parameters(cloth_id)
        gamma = torch.cat(film_parameters["film_gamma"], dim=-1).detach().abs()
        beta = torch.cat(film_parameters["film_beta"], dim=-1).detach().abs()
        return {
            "gamma_abs_mean": gamma.mean().item(),
            "gamma_abs_max": gamma.max().item(),
            "beta_abs_mean": beta.mean().item(),
            "beta_abs_max": beta.max().item(),
        }

    def _compute_film_parameters(
        self,
        cloth_id: torch.Tensor | int | None = None,
        clothing_embedding: torch.Tensor | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        if self.clothing_embedding is None or self.clothing_film_generator is None:
            raise RuntimeError("FiLM clothing generator has not been initialized")
        if cloth_id is not None and clothing_embedding is not None:
            raise ValueError("cloth_id and clothing_embedding are mutually exclusive")
        if cloth_id is None and clothing_embedding is None:
            raise ValueError("provide exactly one of cloth_id or clothing_embedding")
        if clothing_embedding is None:
            embedding = self.clothing_embedding(cloth_id)
        else:
            embedding = self._validate_external_clothing_embedding(clothing_embedding)
        return self.clothing_film_generator(embedding)

    def anchor_offset_smoothness(
        self,
        cloth_id: torch.Tensor | int,
        anchor_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean xyz-offset difference across an anchor edge graph."""

        if self.offset_mode not in {"anchor_mlp", "anchor_film"}:
            raise RuntimeError(
                "anchor_offset_smoothness is only valid in anchor_mlp or anchor_film mode"
            )
        if not isinstance(anchor_edges, torch.Tensor):
            raise TypeError(f"anchor_edges must be a torch.Tensor, got {type(anchor_edges)!r}")
        if anchor_edges.ndim != 2 or anchor_edges.shape[1] != 2 or anchor_edges.shape[0] == 0:
            raise ValueError(
                f"anchor_edges must have non-empty shape [E, 2], got {tuple(anchor_edges.shape)}"
            )
        if anchor_edges.dtype != torch.long:
            raise TypeError("anchor_edges must have dtype torch.int64")

        edges = anchor_edges.to(device=self.anchor_features.device)
        num_anchors = self.anchor_features.shape[0]
        if torch.any(edges < 0) or torch.any(edges >= num_anchors):
            raise IndexError(f"anchor_edges indices must be in [0, {num_anchors})")
        if self.offset_mode == "anchor_film":
            delta_xyz = self.compute_film_anchor_offsets(cloth_id)["delta_xyz"]
        else:
            delta_xyz = self.compute_anchor_clothing_offsets(cloth_id)["delta_xyz"]
        differences = delta_xyz[edges[:, 0]] - delta_xyz[edges[:, 1]]
        return torch.linalg.vector_norm(differences, dim=-1).mean()

    def get_dressed_canonical_params(
        self,
        cloth_id: torch.Tensor | int | None = None,
        clothing_embedding: torch.Tensor | None = None,
        anchor_clothing_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compose raw canonical Gaussian parameters with clothing offsets.

        Returned ``xyz``, ``scaling``, and ``opacity`` stay in the raw canonical
        parameter space. Activation functions, pose-dependent basis offsets, LBS
        deformation, and rasterization remain the responsibility of the original
        MMLPHuman ``GaussianModel`` path.
        """

        base_model = self._require_base_model()
        offsets = self.compute_clothing_offsets(
            cloth_id=cloth_id,
            clothing_embedding=clothing_embedding,
            anchor_clothing_features=anchor_clothing_features,
        )

        return {
            "xyz": base_model._xyz + offsets["delta_xyz"],
            "scaling": base_model._scaling + offsets["delta_scaling"],
            "opacity": base_model._opacity + offsets["delta_opacity"],
            "rotation": base_model._rotation,
            "sh0": base_model._sh0,
            "shN": base_model._shN,
        }

    def forward(
        self,
        cloth_id: torch.Tensor | int | None = None,
        clothing_embedding: torch.Tensor | None = None,
        anchor_clothing_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return dressed raw canonical parameters for one clothing item."""

        return self.get_dressed_canonical_params(
            cloth_id=cloth_id,
            clothing_embedding=clothing_embedding,
            anchor_clothing_features=anchor_clothing_features,
        )

    def render(
        self,
        viewpoint_camera: dict[str, torch.Tensor],
        cloth_id: torch.Tensor | int | None = None,
        clothing_embedding: torch.Tensor | None = None,
        anchor_clothing_features: torch.Tensor | None = None,
        **render_kwargs: Any,
    ):
        """Render through the wrapped GaussianModel using dressed raw params."""

        base_model = self._require_base_model()
        dressed_params = self.get_dressed_canonical_params(
            cloth_id=cloth_id,
            clothing_embedding=clothing_embedding,
            anchor_clothing_features=anchor_clothing_features,
        )
        return base_model.render(
            viewpoint_camera,
            canonical_overrides=dressed_params,
            **render_kwargs,
        )

    def _require_base_model(self) -> Any:
        if self.base_model is None:
            raise RuntimeError("DressableGaussianModel requires set_base_model() before use.")
        return self.base_model

    def _apply_offset_channel_gate(
        self,
        offsets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        enabled = {
            "delta_xyz": self.enable_delta_xyz,
            "delta_scaling": self.enable_delta_scaling,
            "delta_opacity": self.enable_delta_opacity,
        }
        return {
            key: value if enabled[key] else torch.zeros_like(value)
            for key, value in offsets.items()
        }

    def _validate_external_clothing_embedding(
        self,
        clothing_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(clothing_embedding, torch.Tensor):
            raise TypeError("clothing_embedding must be a torch.Tensor")
        if self.clothing_film_generator is None:
            raise RuntimeError("FiLM clothing generator has not been initialized")
        expected_shape = (1, self.clothing_film_generator.clothing_dim)
        if tuple(clothing_embedding.shape) != expected_shape:
            raise ValueError(
                f"clothing_embedding must have shape {expected_shape}, got "
                f"{tuple(clothing_embedding.shape)}"
            )
        if not torch.is_floating_point(clothing_embedding):
            raise TypeError("clothing_embedding must have a floating-point dtype")
        if clothing_embedding.device != self.anchor_features.device:
            raise ValueError(
                "clothing_embedding must be on the same device as anchor_features"
            )
        if clothing_embedding.dtype != self.anchor_features.dtype:
            raise ValueError(
                "clothing_embedding must have the same dtype as anchor_features"
            )
        if not torch.isfinite(clothing_embedding).all():
            raise ValueError("clothing_embedding contains NaN or Inf")
        return clothing_embedding

    def _validate_anchor_clothing_features(
        self,
        anchor_clothing_features: torch.Tensor,
    ) -> None:
        if not isinstance(anchor_clothing_features, torch.Tensor):
            raise TypeError("anchor_clothing_features must be a torch.Tensor")
        if self.anchor_clothing_mlp is None:
            raise RuntimeError("anchor clothing generator has not been initialized")
        local_dim = self.anchor_clothing_mlp.local_feature_dim
        if local_dim is None or self.anchor_clothing_mlp.local_feature_adapter is None:
            raise RuntimeError(
                "AnchorClothingMLP local feature adapter has not been initialized"
            )
        expected_shape = (self.anchor_features.shape[0], local_dim)
        if tuple(anchor_clothing_features.shape) != expected_shape:
            raise ValueError(
                f"anchor_clothing_features must have shape {expected_shape}, got "
                f"{tuple(anchor_clothing_features.shape)}"
            )
        if (
            anchor_clothing_features.device != self.anchor_features.device
            or anchor_clothing_features.dtype != self.anchor_features.dtype
        ):
            raise ValueError(
                "anchor_clothing_features must match anchor_features device and dtype"
            )
        if not torch.isfinite(anchor_clothing_features).all():
            raise ValueError("anchor_clothing_features contains NaN or Inf")

    @staticmethod
    def _require_id_only_condition(
        cloth_id: torch.Tensor | int | None,
        clothing_embedding: torch.Tensor | None,
        anchor_clothing_features: torch.Tensor | None,
        mode: str,
    ) -> None:
        if clothing_embedding is not None or anchor_clothing_features is not None:
            raise ValueError(f"{mode} mode supports only the cloth_id condition")
        if cloth_id is None:
            raise ValueError(f"{mode} mode requires cloth_id")

    def _compute_direct_clothing_offsets(
        self,
        cloth_id: torch.Tensor | int,
    ) -> dict[str, torch.Tensor]:
        base_model = self._require_base_model()
        resolved_cloth_id = self._resolve_cloth_id(
            cloth_id,
            self._direct_num_clothes if self._direct_initialized else 0,
        )
        return {
            "delta_xyz": self._select_learnable_offset(
                self.cloth_delta_xyz,
                resolved_cloth_id,
                base_model._xyz,
            ),
            "delta_scaling": self._select_learnable_offset(
                self.cloth_delta_scaling,
                resolved_cloth_id,
                base_model._scaling,
            ),
            "delta_opacity": self._select_learnable_offset(
                self.cloth_delta_opacity,
                resolved_cloth_id,
                base_model._opacity,
            ),
        }

    def _resolve_cloth_id(
        self,
        cloth_id: torch.Tensor | int,
        num_clothes: int | None = None,
    ) -> int:
        if isinstance(cloth_id, bool):
            raise TypeError("cloth_id must be a Python int or scalar torch.Tensor, not bool")
        if isinstance(cloth_id, int):
            resolved = cloth_id
        elif isinstance(cloth_id, torch.Tensor):
            if cloth_id.ndim == 0:
                if cloth_id.dtype == torch.bool or torch.is_floating_point(cloth_id):
                    raise TypeError("scalar cloth_id tensor must have an integer dtype")
                resolved = int(cloth_id.item())
            else:
                raise NotImplementedError(
                    "batched cloth_id tensors are not supported; pass a Python int "
                    "or scalar integer tensor"
                )
        else:
            raise TypeError(
                f"cloth_id must be a Python int or scalar torch.Tensor, got {type(cloth_id)!r}"
            )

        range_size = self.num_clothes if num_clothes is None else num_clothes
        if range_size > 0 and not 0 <= resolved < range_size:
            raise IndexError(
                f"cloth_id {resolved} is out of range for {range_size} clothes"
            )
        return resolved

    @staticmethod
    def _validate_and_normalize_anchor_inputs(
        anchor_features: torch.Tensor,
        gaussian_anchor_indices: torch.Tensor,
        gaussian_anchor_weights: torch.Tensor,
        base_model: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(anchor_features, torch.Tensor):
            raise TypeError("anchor_features must be a torch.Tensor")
        if anchor_features.ndim != 2 or min(anchor_features.shape) <= 0:
            raise ValueError(
                f"anchor_features must have non-empty shape [A, F], got {tuple(anchor_features.shape)}"
            )
        if not torch.is_floating_point(anchor_features):
            raise TypeError("anchor_features must have a floating-point dtype")

        if not isinstance(gaussian_anchor_indices, torch.Tensor):
            raise TypeError("gaussian_anchor_indices must be a torch.Tensor")
        if gaussian_anchor_indices.dtype != torch.long:
            raise TypeError("gaussian_anchor_indices must have dtype torch.int64")
        if gaussian_anchor_indices.ndim != 2 or gaussian_anchor_indices.shape[1] <= 0:
            raise ValueError(
                "gaussian_anchor_indices must have non-empty shape [N, K], got "
                f"{tuple(gaussian_anchor_indices.shape)}"
            )

        num_gaussians = base_model._xyz.shape[0]
        if gaussian_anchor_indices.shape[0] != num_gaussians:
            raise ValueError(
                f"gaussian_anchor_indices has N={gaussian_anchor_indices.shape[0]}, "
                f"but base_model has N={num_gaussians} Gaussians"
            )
        num_anchors = anchor_features.shape[0]
        if torch.any(gaussian_anchor_indices < 0) or torch.any(
            gaussian_anchor_indices >= num_anchors
        ):
            raise IndexError(f"gaussian_anchor_indices values must be in [0, {num_anchors})")

        if not isinstance(gaussian_anchor_weights, torch.Tensor):
            raise TypeError("gaussian_anchor_weights must be a torch.Tensor")
        if not torch.is_floating_point(gaussian_anchor_weights):
            raise TypeError("gaussian_anchor_weights must have a floating-point dtype")
        if gaussian_anchor_weights.shape != gaussian_anchor_indices.shape:
            raise ValueError(
                "gaussian_anchor_weights must match gaussian_anchor_indices shape "
                f"{tuple(gaussian_anchor_indices.shape)}, got "
                f"{tuple(gaussian_anchor_weights.shape)}"
            )
        if not torch.isfinite(gaussian_anchor_weights).all():
            raise ValueError("gaussian_anchor_weights must contain only finite values")
        if torch.any(gaussian_anchor_weights < 0):
            raise ValueError("gaussian_anchor_weights must be non-negative")
        row_sums = gaussian_anchor_weights.sum(dim=1, keepdim=True)
        if torch.any(row_sums <= 0):
            raise ValueError("each gaussian_anchor_weights row must have a positive sum")

        device = base_model._xyz.device
        dtype = base_model._xyz.dtype
        features = anchor_features.to(device=device, dtype=dtype)
        indices = gaussian_anchor_indices.to(device=device)
        weights = gaussian_anchor_weights.to(device=device, dtype=dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return features, indices, weights

    @staticmethod
    def _make_clothing_parameter(
        num_clothes: int,
        base_parameter: torch.Tensor,
        enabled: bool,
    ) -> nn.Parameter | None:
        if not enabled:
            return None
        value = base_parameter.new_zeros((num_clothes, *base_parameter.shape))
        return nn.Parameter(value)

    @staticmethod
    def _select_learnable_offset(
        parameter: nn.Parameter | None,
        cloth_id: int,
        base_parameter: torch.Tensor,
    ) -> torch.Tensor:
        if parameter is None:
            return torch.zeros_like(base_parameter)
        return parameter[cloth_id]

    def _validate_base_against_learnable_offsets(self, base_model: Any) -> None:
        parameter_pairs = (
            ("cloth_delta_xyz", self.cloth_delta_xyz, base_model._xyz),
            ("cloth_delta_scaling", self.cloth_delta_scaling, base_model._scaling),
            ("cloth_delta_opacity", self.cloth_delta_opacity, base_model._opacity),
        )
        for name, parameter, base_parameter in parameter_pairs:
            if parameter is None:
                continue
            if tuple(parameter.shape) != (self._direct_num_clothes, *base_parameter.shape):
                raise ValueError(
                    f"new base_model is incompatible with initialized {name}: "
                    f"expected shape {tuple(parameter.shape[1:])} per cloth, got "
                    f"{tuple(base_parameter.shape)}; call "
                    "initialize_learnable_clothing_offsets() after attaching a compatible base"
                )
            if parameter.device != base_parameter.device or parameter.dtype != base_parameter.dtype:
                raise ValueError(
                    f"new base_model is incompatible with initialized {name}: device/dtype "
                    f"changed from {parameter.device}/{parameter.dtype} to "
                    f"{base_parameter.device}/{base_parameter.dtype}; move the wrapper and base "
                    "together or reinitialize clothing offsets"
                )

        if self.gaussian_anchor_indices.numel() == 0:
            return
        if self.gaussian_anchor_indices.shape[0] != base_model._xyz.shape[0]:
            raise ValueError(
                "new base_model Gaussian count is incompatible with initialized "
                "anchor-to-Gaussian interpolation; reinitialize the anchor generator"
            )
        if (
            self.anchor_features.device != base_model._xyz.device
            or self.anchor_features.dtype != base_model._xyz.dtype
            or self.gaussian_anchor_weights.device != base_model._xyz.device
            or self.gaussian_anchor_weights.dtype != base_model._xyz.dtype
        ):
            raise ValueError(
                "new base_model device/dtype is incompatible with the initialized anchor "
                "generator; move both together or reinitialize the anchor generator"
            )

    @staticmethod
    def _normalize_mask(gaussian_mask: torch.Tensor, base_model: Any) -> torch.Tensor:
        if not isinstance(gaussian_mask, torch.Tensor):
            raise TypeError(f"gaussian_mask must be a torch.Tensor, got {type(gaussian_mask)!r}")

        num_gaussians = base_model._xyz.shape[0]
        if gaussian_mask.shape == (num_gaussians,):
            mask = gaussian_mask.reshape(num_gaussians, 1)
        elif gaussian_mask.shape == (num_gaussians, 1):
            mask = gaussian_mask
        else:
            raise ValueError(
                f"gaussian_mask must have shape ({num_gaussians},) or "
                f"({num_gaussians}, 1), got {tuple(gaussian_mask.shape)}"
            )

        if mask.dtype == torch.bool:
            mask_bool = mask.to(device=base_model._xyz.device)
        else:
            mask_bool = mask.to(device=base_model._xyz.device) != 0
        return mask_bool.to(dtype=base_model._xyz.dtype)

    @staticmethod
    def _normalize_vector_offset(
        offset: torch.Tensor | float | None,
        target: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if offset is None:
            return torch.zeros_like(target)

        if isinstance(offset, float):
            return torch.full_like(target, offset)

        if not isinstance(offset, torch.Tensor):
            raise TypeError(f"{name} must be None, float, or torch.Tensor, got {type(offset)!r}")

        offset = offset.to(device=target.device, dtype=target.dtype)
        if offset.ndim == 0:
            return torch.ones_like(target) * offset
        if tuple(offset.shape) == (3,):
            return offset.reshape(1, 3).expand_as(target)
        if tuple(offset.shape) == tuple(target.shape):
            return offset

        raise ValueError(
            f"{name} must be a float, Tensor[3], or Tensor{tuple(target.shape)}, "
            f"got Tensor{tuple(offset.shape)}"
        )

    @staticmethod
    def _normalize_opacity_offset(
        offset: torch.Tensor | float | None,
        target: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if offset is None:
            return torch.zeros_like(target)

        if isinstance(offset, float):
            return torch.full_like(target, offset)

        if not isinstance(offset, torch.Tensor):
            raise TypeError(f"{name} must be None, float, or torch.Tensor, got {type(offset)!r}")

        offset = offset.to(device=target.device, dtype=target.dtype)
        if offset.ndim == 0:
            return torch.ones_like(target) * offset
        if tuple(offset.shape) == tuple(target.shape):
            return offset

        raise ValueError(
            f"{name} must be a float or Tensor{tuple(target.shape)}, "
            f"got Tensor{tuple(offset.shape)}"
        )

    @staticmethod
    def _validate_base_model(base_model: Any) -> None:
        required_attrs = ("_xyz", "_scaling", "_opacity", "_rotation", "_sh0", "_shN")
        missing = [name for name in required_attrs if not hasattr(base_model, name)]
        if missing:
            raise AttributeError(f"base_model is missing required Gaussian parameters: {missing}")

        for name in required_attrs:
            value = getattr(base_model, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"base_model.{name} must be a torch.Tensor, got {type(value)!r}")

        num_gaussians = base_model._xyz.shape[0]
        expected_shapes = {
            "_xyz": (num_gaussians, 3),
            "_scaling": (num_gaussians, 3),
            "_opacity": (num_gaussians,),
            "_rotation": (num_gaussians, 4),
            "_sh0": (num_gaussians, 1, 3),
            "_shN": (num_gaussians, 3, 3),
        }

        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(getattr(base_model, name).shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"base_model.{name} has shape {actual_shape}, expected {expected_shape}"
                )
