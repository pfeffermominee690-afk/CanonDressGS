# Controller perturbation failure chain

All 480 perturbation records were traced through frozen feature, logit, probability, pair, weight, mode and reused formal render outputs. No new perturbation render was generated.

| Perturbation | Primary stage | Pair flip | Mode flip | Secondary-weight drift | Render LPIPS drift |
|---|---|---:|---:|---:|---:|
| assignment_permutation | FEATURE_EXTRACTION | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| blur | MULTIPLE_STAGES | 0.6500 | 0.5500 | 0.2114 | 0.0974 |
| grayscale | FALLBACK_GATE | 0.0000 | 0.2000 | 0.0463 | 0.0190 |
| hue | RENDER_ONLY | 0.0167 | 0.1833 | 0.0348 | 0.0160 |
| mask_dilation | MULTIPLE_STAGES | 0.5167 | 0.2833 | 0.1063 | 0.0474 |
| mask_erosion | MULTIPLE_STAGES | 0.2333 | 0.3667 | 0.1738 | 0.0505 |
| reference_dropout | FALLBACK_GATE | 0.0333 | 0.4833 | 0.1541 | 0.0656 |
| single_reference | MULTIPLE_STAGES | 0.7500 | 0.5667 | 0.2001 | 0.1176 |

Blur and single-reference perturbations are multi-stage failures: blur pair/mode flip rates are 0.65/0.55 with mean LPIPS drift 0.09736; single-reference rates are 0.75/0.5667 with drift 0.11765. Mask erosion/dilation also propagate across multiple stages. Reference dropout is dominated by fallback-gate changes (mode flip 0.4833), while assignment permutation is effectively invariant. This is an independent robustness failure, not evidence that clean-input pair compatibility is caused only by feature drift.
