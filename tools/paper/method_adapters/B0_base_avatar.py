from .base import PaperMethodAdapter


class B0BaseAvatarAdapter(PaperMethodAdapter):
    prefixes = ("B0_",)
    trainable = False
    training_steps = 0
    model_family = "frozen_base_avatar"
    parameter_scope = "none"
