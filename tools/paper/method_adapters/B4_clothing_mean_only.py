from .base import PaperMethodAdapter


class B4ClothingMeanOnlyAdapter(PaperMethodAdapter):
    prefixes = ("B4_",)
    model_family = "frozen_f2_clothing_weighted_mean_set_mean_layernorm_linear"
    parameter_scope = "layernorm_and_linear_coefficients"
