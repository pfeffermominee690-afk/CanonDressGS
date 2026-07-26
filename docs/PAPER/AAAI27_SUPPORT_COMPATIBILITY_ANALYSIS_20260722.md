# AAAI27 Support Compatibility Analysis

**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**

| pair | label | top10 | weighted Jaccard | exclusive | xyz conflict | opacity conflict |
|---|---|---|---|---|---|---|
| O01_O02 | stable | 0.2901 | 0.5270 | 0.0006 | 0.3076 | 0.4229 |
| O01_O03 | unstable | 0.2240 | 0.4915 | 0.0006 | 0.4024 | 0.4824 |
| O01_O04 | stable | 0.2566 | 0.4604 | 0.0006 | 0.3940 | 0.5072 |
| O01_O08 | unstable | 0.2949 | 0.5099 | 0.0005 | 0.3065 | 0.4588 |
| O02_O03 | unstable | 0.2257 | 0.5235 | 0.0002 | 0.3847 | 0.4627 |
| O02_O04 | unstable | 0.2784 | 0.5149 | 0.0002 | 0.3726 | 0.4525 |
| O02_O08 | unstable | 0.3000 | 0.5498 | 0.0003 | 0.2841 | 0.4219 |
| O03_O04 | stable | 0.3163 | 0.5553 | 0.0001 | 0.2486 | 0.3460 |
| O03_O08 | unstable | 0.2190 | 0.5126 | 0.0002 | 0.3323 | 0.4392 |
| O04_O08 | unstable | 0.4178 | 0.5740 | 0.0002 | 0.2922 | 0.3506 |

## Preregistered stable/unstable analysis

| feature | stable median | unstable median | rho | exact p | strong |
|---|---|---|---|---|---|
| top_10_support_overlap | 0.2901 | 0.27845 | 0.1899 | 1.0000 | False |
| top_20_support_overlap | 0.42485 | 0.4118 | 0.2659 | 0.7333 | False |
| weighted_jaccard | 0.52698 | 0.51487 | 0.0380 | 0.6000 | False |
| exclusive_fraction_sum | 0.00057031 | 0.00024442 | 0.2659 | 0.2000 | False |
| overlap_residual_cosine | 0.27877 | 0.21414 | 0.1899 | 0.6000 | False |
| xyz_direction_conflict | 0.30756 | 0.33235 | -0.1140 | 1.0000 | False |
| opacity_activation_conflict_fraction | 0.42294 | 0.45247 | -0.1140 | 0.5333 | False |
| sh_disagreement | 0.64473 | 0.64324 | -0.0380 | 1.0000 | False |
| teacher_silhouette_iou | 0.90143 | 0.88032 | 0.4938 | 0.1667 | False |
| boundary_distance | 6.6574 | 7.2004 | -0.2659 | 0.4667 | False |

## Per-Gaussian role conflict

| pair | magnitude median | magnitude p95 | xyz conflict | opacity conflict | SH0 conflict |
|---|---|---|---|---|---|
| O01_O02 | 1.4303 | 4.3165 | 0.3076 | 0.4229 | 0.3392 |
| O01_O03 | 1.4861 | 5.0452 | 0.4024 | 0.4824 | 0.5501 |
| O01_O04 | 1.5870 | 5.6657 | 0.3940 | 0.5072 | 0.5106 |
| O01_O08 | 1.4548 | 4.8507 | 0.3065 | 0.4588 | 0.3343 |
| O02_O03 | 1.4124 | 4.6522 | 0.3847 | 0.4627 | 0.5701 |
| O02_O04 | 1.4629 | 4.5938 | 0.3726 | 0.4525 | 0.5307 |
| O02_O08 | 1.3825 | 4.0499 | 0.2841 | 0.4219 | 0.3566 |
| O03_O04 | 1.3705 | 4.0134 | 0.2486 | 0.3460 | 0.2512 |
| O03_O08 | 1.4591 | 4.5591 | 0.3323 | 0.4392 | 0.4476 |
| O04_O08 | 1.3693 | 4.0635 | 0.2922 | 0.3506 | 0.2965 |

No sleeves/trousers Gaussian mapping was invented. The per-Gaussian report uses only the frozen index order and protected membership.
