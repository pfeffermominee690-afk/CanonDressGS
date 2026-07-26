from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


MILESTONES = (0, 20, 50, 100, 200, 300)
FORBIDDEN_FORWARD_KEYS = {
    "outfit_id", "cloth_id", "target_rgb", "target_image", "target_mask",
    "teacher_residual", "teacher_coefficient",
}


@dataclass(frozen=True)
class ModelBuildSpec:
    family: str
    trainable_parameter_contract: str
    frozen_components: tuple[str, ...]
    output: str
    basis_rank: int | None = None


class PaperMethodAdapter:
    prefixes: tuple[str, ...] = ()
    trainable = True
    training_steps = 300
    role = "inference_method"
    seen_only = False
    model_family = "reference_coefficient_predictor"
    parameter_scope = "coefficient_predictor_only"

    def __init__(self, experiment: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.experiment = dict(experiment)
        self.config = config

    @classmethod
    def supports(cls, method: str) -> bool:
        return method.startswith(cls.prefixes)

    def validate_contract(self) -> dict[str, Any]:
        if self.experiment.get("executable") is not True:
            raise ValueError("adapter requires executable experiment")
        seed = self.experiment.get("seed")
        if self.trainable and seed not in self.config["training"]["seeds"]:
            raise ValueError("trainable adapter seed is outside frozen protocol")
        if not self.trainable and seed is not None:
            raise ValueError("fixed adapter must not carry a seed")
        if self.training_steps not in (0, self.config["training"]["steps"]):
            raise ValueError("adapter training budget differs from frozen protocol")
        if self.seen_only and self.config["data"]["held_out_diagnostic"] == "O07":
            allowed_held_out = False
        else:
            allowed_held_out = self.role != "optimization_upper_bound"
        return {
            "adapter": type(self).__name__,
            "method": self.experiment["method"],
            "role": self.role,
            "seen_only": self.seen_only,
            "allowed_in_held_out": allowed_held_out,
            "trainable": self.trainable,
            "training_steps": self.training_steps,
            "seed": seed,
            "target_forward_input": False,
            "outfit_id_in_model": False,
        }

    def build_model(self) -> ModelBuildSpec:
        return ModelBuildSpec(
            family=self.model_family,
            trainable_parameter_contract=self.parameter_scope if self.trainable else "none",
            frozen_components=("basis", "base_gaussians", "image_backbone", "mmlp_human", "renderer"),
            output="basis_coefficients" if self.trainable else "fixed_residual",
            basis_rank=self.config["basis"]["rank"],
        )

    def build_training_plan(self) -> dict[str, Any]:
        return {
            "optimizer_required": self.trainable,
            "optimizer_parameter_scope": self.parameter_scope if self.trainable else [],
            "steps": self.training_steps,
            "milestones": list(MILESTONES if self.trainable else (0,)),
            "seed": self.experiment.get("seed"),
            "balanced_outfits": list(self.config["data"]["seen_outfits"]),
            "target_view_schedule": "round_robin",
            "early_stop": False,
            "best_seed_selection": False,
            "checkpoint_state": [
                "model", "optimizer", "rng", "scheduler", "global_step",
                "condition_position", "fixed_output_parity",
            ] if self.trainable else [],
        }

    def run_forward(self, payload: Mapping[str, Any], forward: Callable[..., Any]) -> Any:
        overlap = FORBIDDEN_FORWARD_KEYS.intersection(payload)
        if overlap:
            raise ValueError(f"forbidden prediction-forward fields: {sorted(overlap)}")
        return forward(**dict(payload))

    def collect_metrics(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [dict(record) for record in records]
    def export_visual_inputs(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [dict(record) for record in records]
