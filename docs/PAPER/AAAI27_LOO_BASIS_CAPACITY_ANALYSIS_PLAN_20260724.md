# AAAI27 LOO Basis Capacity Analysis Plan

## Purpose

This analysis separates representation capacity from few-view optimization.
It is offline-only and may read the held-out Teacher residual after each LOO
basis has been constructed without that garment.

## Oracle Projection

For split i:

`c_oracle = B_-i^T flatten(bound_normalized(R_i^T - mu_-i))`

and:

`R_oracle = mu_-i + B_-i c_oracle`.

The runtime must verify that `B_-i` is the newly recomputed centered rank<=3
basis and that its input-contract hash matches the split manifest. The source
five-garment rank-4 basis is forbidden.

## Required Statistics

For each held-out garment, report:

- all centered-matrix singular values;
- component and cumulative explained variance;
- numerical rank under the frozen tolerance;
- rank-3 reconstruction error for every basis garment;
- exact four-garment residual and render parity;
- repeated-build hash and subspace principal-angle stability;
- held-out oracle residual RMSE;
- oracle RGB MAE, PSNR, SSIM, LPIPS, silhouette IoU, boundary F,
  protected LPIPS, and identity;
- span distance and projection ratio;
- residual-space nearest and convex-combination oracle diagnostics.

No oracle value may enter adaptation initialization, loss, regularization,
checkpoint selection, or test selection.

## Capacity Gate

The preregistered oracle capacity thresholds are projection ratio >=0.70,
bound-normalized residual RMSE <=0.15, and oracle projection LPIPS <=0.05.
If at least three garments fail, the classification is
`LOO_BASIS_CAPACITY_LIMITED`.

If capacity passes but deployable few-view adaptation cannot reach the oracle
or recover 75 percent of full-residual gains, the classification is
`LOO_OPTIMIZATION_LIMITED`. If adaptation cannot beat hard lookup, it is
`LOO_HARD_LOOKUP_NOT_OUTPERFORMED`.

The current protocol cannot reach those classifications because the view-fold
contract is incomplete. `LOO_PROTOCOL_INCOMPLETE` takes precedence.

## Safety and Claim Boundary

All oracle renders use the same frozen protected guard, MMLP-Human, renderer,
pose/camera, and metrics. This analysis does not establish arbitrary garment
generation, open-world adaptation, cross-identity transfer, unseen identity,
real captured multi-outfit performance, or unbounded continuous editing.
