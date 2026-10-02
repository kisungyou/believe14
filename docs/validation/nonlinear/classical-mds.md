# ClassicalMDS validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Torgerson (1952),
  [doi:10.1007/BF02288916](https://doi.org/10.1007/BF02288916).
- **Definition:** for validated dissimilarities `D`, form
  `B = -J (D**2) J / 2`, where `J = I - 11^T/n`. Coordinates are the leading
  positive eigenvectors multiplied by square roots of their eigenvalues.
- **Conventions:** Euclidean feature and precomputed-distance inputs are supported.
  The complete spectrum is retained. Significant negative eigenvalues are reported,
  not silently clipped; a requested axis without a positive eigenvalue is rejected.
- **Equivalence and evidence:** centered Gram matrices, rather than eigenvector signs,
  are compared with an independently centered Euclidean fixture. Symmetry, hollowness,
  numerical rank, and eigen-residual failures are tested. Gram entries and spectrum
  are rescaled by exponent arithmetic without forming an intermediate squared
  distance scale. Analytic two-point tests cover representable values near both
  the largest finite value and the subnormal range.
- **Contract:** transductive; no `transform`. Rdimtools is not an oracle.

- **Complexity:** `O(n^2 p + n^3) time; O(np + n^2) memory`; symbols follow the
  [method catalog](../../methods.md).
