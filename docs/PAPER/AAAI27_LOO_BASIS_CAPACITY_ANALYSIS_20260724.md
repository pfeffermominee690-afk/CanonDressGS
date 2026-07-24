# AAAI27 LOO Basis Capacity Analysis

## Available Basis Evidence

Only the first split, `LOO-O01`, produced a basis artifact before the attempt stopped. Its training bank was `O02/O03/O04/O08`; held-out Teacher use in basis and normalization was zero. Numerical rank and selected rank were both 3. The basis fingerprint was `a7a16acf8c0920c5d4bba80c76be87f343e708cba7a8836d4bba3fee176410b9`, the basis artifact SHA-256 was `71144cafcdfd070942dc0a808b65216539d0320c3a801a1cde7fb0b75ecd40c6`, and the normalization SHA-256 was `91a4121a58fcf8b839e60284290022b9b2963e4b8a26c339f0dd5e7dab7f1d19`.

Residual reconstruction was numerically close and passed the frozen residual threshold, but the rendered reconstructions failed the independent max-absolute parity gate. This is an execution-contract failure at basis validation, not evidence that the held-out garment lies inside or outside the learned span.

## Unavailable Capacity Denominators

Oracle Projection, Residual Nearest Oracle, Convex Combination Oracle, projection render metrics, projection residual error, span distance, oracle coefficients, and adaptation-to-oracle gaps were not computed. Capacity versus optimization diagnosis is therefore `null` with reason `SCIENTIFIC_EXECUTION_INVALID_BEFORE_DIAGNOSTIC_DENOMINATORS`.

No `LOO_BASIS_CAPACITY_LIMITED` or `LOO_OPTIMIZATION_LIMITED` scientific classification may be assigned from this attempt. The only valid classification is `LOO_ADAPTATION_EXECUTION_INVALID`.
