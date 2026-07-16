# Module 3 Resume Audit

## Actual controlled-training state

- Optimizer: `torch.optim.Adam`
- Trainable scope: `CanonicalClothingCompleter` only. Every pre-existing image encoder, aggregator, HyperNetwork, six-channel decoder and frozen MMLP-Human base parameter is frozen during the 100-step completion fixture.
- Parameter groups: one group containing all completer parameters in `named_parameters()` order.
- Learning rate: `0.001`
- Adam betas / eps / weight decay: PyTorch defaults `(0.9, 0.999) / 1e-8 / 0`
- Scheduler: not instantiated.
- AMP / GradScaler: not used.
- Gradient clipping: not used.
- Gradient accumulation: not used; one optimizer step per controlled iteration.

## Determinism and data state

- Fixed protocol: S1=`f000_c018`, S2=`f1000_c000`, S12=both, target=`f2000_c009`.
- Graph: deterministic `base.nbr_vt` topology plus two-hop expansion, K=8.
- Fixture and dressed references: fixed Module 2 synthetic teacher assets.
- Reference sampling and batch sampling: no sampling inside the controlled loop.
- Feature holdout: deterministic 30% selection generated once from the S12 observed mask with seed `20260716`.
- Model mode: pre-existing modules in eval/frozen state; completer optimized directly.
- SH degree: explicitly controlled by the Module 2 metadata and restored after rendering.

## Legacy checkpoint boundary

`GATE6-ONLINE-COMPLETION-001/checkpoint_final.pth` stores model weights, graph tensors and method metadata. It does not store optimizer moments, scheduler/AMP state, global step or RNG state. Its resume capability is therefore:

`model_weights_only`

Module 3.1 initializes from those model weights, creates the real optimizer, performs one warm-up step to establish Adam moments, and only then writes a versioned fully resumable checkpoint.
