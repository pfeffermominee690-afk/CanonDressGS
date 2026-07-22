# Dual-Support pair compatibility analysis

| Pair | Pair accuracy | Actual activation | Weight MAE | Exact-oracle max grade | Exact-oracle severe count |
|---|---:|---:|---:|---:|---:|
| O01_O02 | 0.9583 | 0.4028 | 0.1461 | 2 | 0 |
| O01_O03 | 1.0000 | 0.6806 | 0.1620 | 3 | 8 |
| O01_O04 | 1.0000 | 0.3750 | 0.1953 | 2 | 0 |
| O01_O08 | 0.8889 | 0.4722 | 0.1980 | 1 | 0 |
| O02_O03 | 1.0000 | 0.4861 | 0.2009 | 3 | 8 |
| O02_O04 | 1.0000 | 0.3611 | 0.2274 | 2 | 0 |
| O02_O08 | 0.9167 | 0.3611 | 0.2109 | 1 | 0 |
| O03_O04 | 1.0000 | 0.8750 | 0.1532 | 2 | 0 |
| O03_O08 | 0.8611 | 0.5694 | 0.1697 | 2 | 0 |
| O04_O08 | 1.0000 | 0.5000 | 0.2126 | 2 | 0 |

O01_O03 and O02_O03 each have pair prediction accuracy 1.0, yet each retains exact-oracle maximum grade 3 and 8 archived severe observations. The 60 opened causal sheets reproduce the same residual: correct pair and 2/3-1/3 oracle weight do not remove full-body patch/cloud/mottle, edge scatter or silhouette discontinuity. Their failure is consistent with pair-specific support/coverage, opacity-overlap and silhouette incompatibility centered on O03 rather than reference confusion or pair-selection error alone.

Other pairs retain minor-to-moderate residual mottling, but no other pair has archived exact-oracle grade-3 cases. Identity contamination remains 0, double outline remains 0, and ghosting remains at most 1. No compatibility gate or pair blacklist was implemented.
