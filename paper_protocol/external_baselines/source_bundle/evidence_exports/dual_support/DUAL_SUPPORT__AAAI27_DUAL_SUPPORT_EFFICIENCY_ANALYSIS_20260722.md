# AAAI27 Dual-Support Efficiency Analysis

**RESEARCH EVALUATION — NOT PAPER FINAL**

Dual support retains both immutable endpoint Gaussian branches at runtime; it does not copy or rewrite checkpoints.

| measure | value |
|---|---|
| HARD active Gaussians | 199999.250 |
| DUAL active Gaussians | 399961.900 |
| active ratio DUAL/HARD | 1.999817 |
| render-time ratio DUAL/HARD | 1.625864 |
| peak-VRAM ratio DUAL/HARD | 1.007490 |
| peak-VRAM increase bytes | 44997805 |
| five endpoint checkpoints bytes | 193038452 |
| basis bytes | 88007427 |
| base-avatar checkpoint bytes | 1706610302 |
| evaluation output bytes before finalize | 797281780 |
| checkpoint copies created | 0 |

Onboarding requires endpoint identities, normalized top-2 weights, confidence, compatibility/entropy handling, and a single-endpoint fallback. Runtime cost cannot be omitted from method selection.
