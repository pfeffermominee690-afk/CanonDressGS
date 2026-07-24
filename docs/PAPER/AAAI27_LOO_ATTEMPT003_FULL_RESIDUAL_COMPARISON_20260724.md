# AAAI27 LOO Full-Residual Comparison

| Garment | LPIPS recovery | RGB MAE recovery | Pass |
|---|---:|---:|---|
| O01 | null | 0.006784 | False |
| O02 | null | 0.000549 | False |
| O03 | null | -0.002332 | False |
| O04 | null | 0.006922 | False |
| O08 | null | 0.012149 | False |

Nonpositive denominators are retained as `null` with the registered explicit reason. Recovery gate: `False`.
Low/full wall-time fraction: 1.166070.

## Equal-Wall-Time Secondary Diagnostic

| Garment | Mean completed step | LPIPS | RGB MAE | Full wall time (s) |
|---|---:|---:|---:|---:|
| O01 | 300.000000 | 0.140151 | 0.182826 | 20.422200 |
| O02 | 298.500000 | 0.127915 | 0.187157 | 20.103866 |
| O03 | 300.000000 | 0.152128 | 0.162503 | 19.797782 |
| O04 | 300.000000 | 0.136865 | 0.180835 | 19.967326 |
| O08 | 300.000000 | 0.124571 | 0.171764 | 19.787540 |

The state is the actual last completed full-residual step within each paired low-dimensional measured wall time; it does not change the primary 300-step result.
