# Pure Endpoint Perturbation Robustness

| Variant | Canon top-1 | Endpoint flip rate |
|---|---|---|
| mild_blur | 0.3500 | 0.6500 |
| mask_erosion | 1.0000 | 0.0000 |
| mask_dilation | 1.0000 | 0.0000 |
| assignment_permutation | 1.0000 | 0.0000 |
| reference_dropout | 1.0000 | 0.0000 |
| single_reference | 0.9500 | 0.0500 |

Complete dropout passes safe abstention and emits Base Avatar. It is audited
outside the frozen six-variant denominator.
