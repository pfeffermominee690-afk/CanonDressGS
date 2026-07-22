# Gaussian spatial artifact analysis

- Ours-v2: `SPATIAL_ARTIFACT_SEVERE`.
- M3: `SPATIAL_ARTIFACT_SEVERE`.
- M4: `SPATIAL_ARTIFACT_SEVERE`.
- Intrinsic normal SHA-256: `0719ced1c3bc8cd97a957152d099fcc98f692b8f8697b3761e3b86fcb0fc06df`.
- B6/B7 cite the shared B1 teacher residual and are not duplicated as independent spatial evidence.

## Frozen-threshold results

- Ours-v2 has displacement P95/P99 `0.0752417954 / 0.0995277118` and frozen `tau_1` exceedance fraction `0.2375459551`; it is therefore `SPATIAL_ARTIFACT_SEVERE` even though protected displacement and garment-exterior outlier counts are zero.
- M3 has displacement P95/P99 `0.0473049014 / 0.0589697346` and `tau_1` exceedance fraction `0.0351023472`. Its three reviewed source sheets additionally show severe whole-body cloud/mottle contamination, edge scatter, and silhouette discontinuity.
- M4 has displacement P95/P99 `0.0786796279 / 0.1025507790` and `tau_1` exceedance fraction `0.2782380615`. Its three reviewed source sheets retain the same severe whole-body contamination failure.
- B1 has `tau_1` exceedance fraction `0.2383952811`; historical A6 and historical old Ours also exceed the frozen threshold and remain classified severe. Historical B4 is not a separate residual row here, but its three formal visual sheets retain severe whole-body contamination.

Zero protected displacement and zero garment-exterior outliers are reported as real positives, but they do not override the preregistered trust-radius failures or the manual visual evidence. All observed cloud, mottle, edge scatter, full-body contamination, and silhouette discontinuity remain in the item-level 57/57 review. No spatial threshold was changed and no failed scientific result was rerun or tuned.
