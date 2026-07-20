from .base import PaperMethodAdapter


class B1DirectTeacherAdapter(PaperMethodAdapter):
    prefixes = ("B1_",)
    trainable = False
    training_steps = 0
    role = "optimization_upper_bound"
    model_family = "frozen_shared_canonical_teacher_residual"
    parameter_scope = "none"
