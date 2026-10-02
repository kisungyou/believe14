# PCA validation ledger

- Status: validated; implementation: `believe14.linear.PCA`.
- Authority: Pearson (1901); computation is the equivalent modern centered SVD.
- Specification: `Xc = X - mean(X)` and `Xc = U diag(s) V^T`; retain the first
  `k` rows of `V^T`. Sample explained variances are `s_j^2 / (n - 1)`.
- Whitening: scores are multiplied by `sqrt(n - 1) / s_j`; zero retained
  singular values are rejected. Reconstruction reverses that scaling.
  Inverse transformation restores the fitted reference and mean offset with
  compensated addition, preserving representable variation when their combined
  public `mean_` would round it away. Truly unrepresentable outputs fail.
- Equivalence: compare loading projectors, not signs or bases inside repeated
  singular-value blocks.
- Evidence: literal NumPy SVD, covariance identity under whitening,
  reconstruction, rank-deficiency, and finite-input tests in `test_linear.py`.
- Rdimtools: not used as an oracle. Out-of-sample map is the fitted affine
  projection; complexity is economy-SVD complexity.

- **Complexity:** `O(min(np^2,n^2p)) time; O(np) memory`; symbols follow the
  [method catalog](../../methods.md).
