# LOO Basis Numerical Closure

The repaired construction casts Teacher physical residual tensors to float64 before bound normalization, computes the arithmetic mean and centered matrix in float64, sets the final centered row to the negative sum of the prior rows, runs rank-3 SVD, projects with the single orthonormal transpose solver, reconstructs in float64, and casts once at shared canonical composition.

For O01, N0/N1/N2 fourth singular values were `8.549413905711845e-05`, `2.541724320322229e-13`, and `2.514491513080209e-13`. Anchor-difference rank was `3`, maximum principal angle was `3.078324659199975e-06` degrees, and anchor reconstruction RMSE was `1.9722429683226479e-16`. Solver selection was mathematical and global, not garment- or image-dependent.

All five repaired splits have numerical rank 3, selected rank 3, strict centered-sum zero, repeat fingerprints, and four endpoint renders under the original gate. No rank-4 basis, held-out Teacher, threshold relaxation, optimizer, or attempt_002 was used. `PAPER_FINAL=false`.
