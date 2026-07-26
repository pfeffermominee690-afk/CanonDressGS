from .base import PaperMethodAdapter


class B2OutfitIdLookupAdapter(PaperMethodAdapter):
    prefixes = ("B2_",)
    trainable = False
    training_steps = 0
    role = "seen_memorization_upper_bound"
    seen_only = True
    model_family = "frozen_teacher_coefficient_lookup"
    parameter_scope = "none"

    def validate_contract(self):
        report = super().validate_contract()
        report["outfit_id_in_model"] = False
        report["lookup_key_used_only_by_evaluator"] = True
        return report
