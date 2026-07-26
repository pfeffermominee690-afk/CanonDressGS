from .base import PaperMethodAdapter


class B3GlobalReferenceAdapter(PaperMethodAdapter):
    prefixes = ("B3_",)
    model_family = "frozen_backbone_global_average_layernorm_linear"
    parameter_scope = "layernorm_and_linear_coefficients"
