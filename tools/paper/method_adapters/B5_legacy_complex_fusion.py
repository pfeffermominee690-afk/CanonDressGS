from .base import PaperMethodAdapter


class B5LegacyComplexFusionAdapter(PaperMethodAdapter):
    prefixes = ("B5_",)
    model_family = "mask_aware_tokens_plus_reference_set_coefficient_fusion_v1"
    parameter_scope = "rf_f_predictor_only"

    def validate_contract(self):
        report = super().validate_contract()
        report["historical_numbers_reusable"] = False
        report["unified_rerun_required"] = True
        return report
