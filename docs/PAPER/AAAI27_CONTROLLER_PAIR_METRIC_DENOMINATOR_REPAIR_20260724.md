# AAAI27 Controller Pair Metric Denominator Repair

Status: `PASS`

The historical all-record pair metric mixed 20 pure records and 60 mixed records per rotation. Pure exact top-2 depended on an input-invisible template label and is not a legal identification target.

`0.812500` is derived from the actual protocol counts as `240/320 * 1.0 + 80/320 * 0.25`. Uniform, majority, and Bayes-optimal calculations all produce the same value. It is a latent-label chance ceiling for a perfect mixed classifier, not a representation ceiling.

| Family | Historical all-record | Corrected mixed-only | R0/R1/R2/R3 |
|---|---:|---:|---|
| V2 | 0.601042 | 0.718056 | 0.727778/0.688889/0.716667/0.738889 |
| Matched V1 | 0.631250 | 0.758333 | 0.733333/0.688889/0.844444/0.766667 |

The corrected V2 mixed-only protocol/unique-query/rotation/seed/global pair aggregates are `0.718056` / `0.718056` / `0.718056` / `0.718056` / `0.718056`. The corresponding matched V1 values are `0.758333` / `0.758333` / `0.758333` / `0.758333` / `0.758333`.

Pure results:

| Family | top-1 | SINGLE | false-mixed |
|---|---:|---:|---:|
| V2 | 1.000000 | 1.000000 | 0.050000 |
| Matched V1 | 1.000000 | 0.900000 | 0.100000 |

Formal-pure secondary safety is V2 top-1/SINGLE `1.000000` / `1.000000` and matched V1 `1.000000` / `0.933333`. This split is secondary endpoint safety only.

| Pair | V2 pair | V1 pair | V2 order | V1 order |
|---|---:|---:|---:|---:|
| O01_O02 | 0.986111 | 0.986111 | 0.569444 | 0.638889 |
| O01_O03 | 0.222222 | 0.347222 | 0.597222 | 0.597222 |
| O01_O04 | 0.791667 | 0.347222 | 0.638889 | 0.597222 |
| O01_O08 | 0.513889 | 0.486111 | 0.763889 | 0.819444 |
| O02_O03 | 0.500000 | 0.944444 | 0.916667 | 0.902778 |
| O02_O04 | 0.902778 | 1.000000 | 0.916667 | 0.986111 |
| O02_O08 | 0.958333 | 0.958333 | 0.819444 | 0.888889 |
| O03_O04 | 1.000000 | 1.000000 | 0.930556 | 0.888889 |
| O03_O08 | 0.472222 | 0.680556 | 0.930556 | 1.000000 |
| O04_O08 | 0.833333 | 0.833333 | 0.875000 | 0.833333 |

| Family | Correct-pair MAE | Correct-pair RMSE | Correct-order MAE | Correct-order RMSE | Compatible DUAL | Incompatible HARD | Incompatible DUAL | Wrong-pair DUAL | Mixed false-SINGLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0.158058 | 0.198143 | 0.135928 | 0.171065 | 0.189378 | 0.313889 | 0.000000 | 0.072824 | 0.830556 |
| Matched V1 | 0.203023 | 0.236070 | 0.169369 | 0.193918 | 0.523007 | 0.000000 | 0.501533 | 0.131513 | 0.569444 |

Routing denominators were already mixed-only or prediction-conditioned mixed subsets: `ROUTING_DENOMINATOR_ALREADY_VALID`.

`TASK_CONDITIONAL_IDENTIFICATION_ACCURACY` is `0.788542` for V2 and `0.818750` for matched V1. It is a secondary summary selected by offline GT cardinality and does not replace the split primary metrics.
