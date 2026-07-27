# Seen-outfit main comparison

> SMOKE ONLY — NOT PAPER RESULTS

| method | garment_rgb_mae | edit_reduction | target_closer_fraction | protected_rgb_mae | background_rgb_mae | nearest_teacher_accuracy | correct_vs_swapped_wins | trainable_parameter_count |
|---|---|---|---|---|---|---|---|---|
| Base Avatar | 0.144819 | 0 | 0 | 0 | 0 | N/A | 80 | 0 |
| B1 — Optimization Upper Bound | 1.87498e-17 | 1 | 0.844415 | 0 | 0.000432079 | 1 | 80 | 0 |
| B2 — Seen-only Lookup | 1.87498e-17 | 1 | 0.844415 | 0 | 0.000432079 | 1 | 80 | 0 |
| Global Reference Feature | 0.182999 | -0.216184 | 0.380585 | 0 | 0.000365981 | 0.2 | 39 | 52 |
| Clothing Mean Only | 0.182999 | -0.216184 | 0.380585 | 0 | 0.000365981 | 0.2 | 39 | 52 |
| Legacy Complex Fusion | 0.182999 | -0.216184 | 0.380585 | 0 | 0.000365981 | 0.2 | 39 | 52 |
| Ours | 0.0926534 | 0.368234 | 0.844415 | 0 | 0.000289404 | 0.7 | 75 | 52 |
