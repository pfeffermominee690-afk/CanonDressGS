from .base import PaperMethodAdapter


class OursExplicitBasisV1Adapter(PaperMethodAdapter):
    prefixes = ("Ours_",)
    model_family = "frozen_f2_mean_max_set_mean_max_layernorm_linear_explicit_basis"
    parameter_scope = "layernorm_and_linear_4_only"

    def validate_contract(self):
        report = super().validate_contract()
        if self.config["basis"]["rank"] != 4:
            raise ValueError("Ours requires frozen rank-4 basis")
        report["trainable_parameter_count"] = self.config["predictor"]["trainable_parameters"]
        return report
