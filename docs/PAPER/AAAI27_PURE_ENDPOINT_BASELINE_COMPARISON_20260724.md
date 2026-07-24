# Pure Endpoint Baseline Comparison

| Method | Top-1 | RGB MAE | LPIPS |
|---|---|---|---|
| Teacher Endpoint | 1.0000 | 0.000000 | 0.000000 |
| Outfit-ID Oracle | 1.0000 | 0.000001 | 0.000000 |
| Reference Classifier Lookup | 1.0000 | 0.000001 | 0.000000 |
| Nearest-Centroid Lookup | 1.0000 | 0.000001 | 0.000000 |
| Linear Coefficient Predictor | 1.0000 | 0.053907 | 0.067913 |
| CanonDressGS-Endpoint | 1.0000 | 0.000001 | 0.000000 |

Teacher Endpoint is the frozen residual target, not an upper bound. Outfit-ID
Oracle uses ground-truth outfit identity and is not deployable.
