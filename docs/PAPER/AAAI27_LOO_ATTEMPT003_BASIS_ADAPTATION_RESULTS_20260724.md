# AAAI27 LOO Basis Adaptation Results

Classification: `LOO_BASIS_CAPACITY_LIMITED`.

This execution is a same-identity leave-one-garment-out simulation inside the subject02 five-garment study set. It does not establish arbitrary garments, open-world adaptation, unseen identities, cross-identity transfer, or real captured multi-outfit performance.


| Garment | Successful rotations | Garment gate | LPIPS reduction | RGB MAE reduction |
|---|---:|---|---:|---:|
| O01 | 0/4 | False | -0.000211 | 0.000561 |
| O02 | 0/4 | False | -0.000078 | 0.000014 |
| O03 | 0/4 | False | -0.000331 | -0.000053 |
| O04 | 0/4 | False | -0.000086 | 0.000273 |
| O08 | 0/4 | False | 0.000133 | 0.000608 |

- Hard-lookup gate: `False` (0/5 garments).
- View scaling: `VIEW_BUDGET_SCALING_NEGATIVE`.
- Recovery gate: `False`.
- Safety gate: `False`.
- PAPER_FINAL: `false`.
