# Matched V1 training contract

Matched V1 is freshly initialized for every rotation and seed and is never
loaded from the historical formal V1 checkpoint. It has 3,589 trainable
controller parameters, frozen F2, the same 160 training records, identical
five-record clean batches, 150 optimizer steps, 750 clean exposures, and the
same six-checkpoint schedule as V2.

Both families use Adam with lr=0.02, weight_decay=0, betas=(0.9,0.999),
epsilon=1e-8, constant LambdaLR, no warmup, FP32, accumulation=1, gradient-norm
clipping at 5.0, and `zero_grad(set_to_none=True)`. Matched V1 uses only the
original mean soft-target garment cross entropy, its probability-ratio weight,
and historical 0.90/0.10 fallback. It does not use compatibility,
HARD_GEOMETRY_SOFT_VA, V2 mixedness/weight supervision, or consistency.

The historical formal V1 is retained only as
`HISTORICAL_PROTOCOL_MISMATCHED_REFERENCE`.
