# Reference-Conditioned Dual-Support Controller: Efficiency

Task: `AAAI27-REFERENCE-CONDITIONED-DUAL-SUPPORT-CONTROLLER-FORMAL-002`

PAPER_FINAL: `0`

The Controller has 3,589 trainable parameters. Each final checkpoint is 285,756 bytes (857,268 bytes across three seeds), with zero checkpoint copies. Total formal output storage at sealing is 4,024,681,191 bytes; this includes append-only failed attempts, valid paired renders, metrics, visuals, and audits.

| Mode | Rendered records | Mean active Gaussians | Mean render time (s) | Max peak VRAM (bytes) |
|---|---:|---:|---:|---:|
| SINGLE_ENDPOINT | 654 | 199999.26 | 0.008602 | 5852499968 |
| DUAL_SUPPORT | 366 | 399949.56 | 0.013540 | 5898653696 |

Effective average active-Gaussian count is 271746.13. SINGLE_ENDPOINT builds one immutable endpoint support; DUAL_SUPPORT builds two weighted immutable supports at runtime. It performs no geometry interpolation and stores no composed checkpoint.

Mean Controller forward time is 0.001550 s. Endpoint selection is included in this timer. F2 cache assembly is recorded per seed, while branch-construction and atomic total-inference timers were not separately instrumented; they are reported as unavailable instead of inferred. Rendering time is reported directly by mode above.

The archived Ours-v2, B6, B7, FULL_LINEAR, HARD, and Oracle provenance remains in the machine summary. Raw timing/metric comparisons are retained only where their original instrumentation and information boundary are explicit; no missing baseline timer is invented. PAPER_FINAL remains 0.
