# Controller Multitask Gradient Analysis

Diagnostic autograd was run at seed 0, steps 0/60/150, cycle batches 0/8/16/24, and all four rotations. No optimizer was created or stepped and no checkpoint was written.

Garment versus consistency mean shared-path cosine: `-0.135875`; conflicting fraction `0.583333`. Independent heads without common active parameters are explicitly recorded as `NO_SHARED_GRADIENT_PATH` rather than cosine zero.
