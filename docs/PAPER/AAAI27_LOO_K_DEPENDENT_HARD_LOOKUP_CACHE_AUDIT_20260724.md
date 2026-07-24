# AAAI27 K-Dependent Hard-Lookup Cache Audit

The six static tracks were recovered from the runner's `STATIC_METHODS` constant and the amended baseline registry.

| Track | Logical | Guaranteed K-invariant | Same keys | Different keys |
|---|---:|---|---:|---:|
| BASE_AVATAR | 40 | True | 20 | 0 |
| REFERENCE_NEAREST_HARD_LOOKUP | 40 | False | 15 | 5 |
| RESIDUAL_NEAREST_ORACLE | 40 | True | 20 | 0 |
| ORACLE_PROJECTION_LOO_BASIS | 40 | True | 20 | 0 |
| HELD_OUT_TEACHER_ENDPOINT | 40 | True | 20 | 0 |
| CONVEX_COMBINATION_ORACLE | 40 | True | 20 | 0 |

Five tracks contribute 100 guaranteed K-shared hits. Reference Nearest Hard Lookup contributes 15 observed deterministic hits and 5 distinct physical keys. All 240 logical static cells remain in the denominator. Unaudited static tracks: 0.
