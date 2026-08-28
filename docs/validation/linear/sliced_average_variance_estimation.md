# SlicedAverageVarianceEstimation validation ledger

- Status: validated; implementation follows Cook (2000),
  DOI 10.1080/03610920008832598.
- Preprocessing and slicing are identical to SIR: ML covariance whitening on its
  numerical support and deterministic tie-preserving quantile slices.
- Kernel: `M=sum_h p_h (I-Cov[Z|slice h])^2`; within-slice covariance uses
  divisor `n_h`, matching the conditional expectation. Each slice must contain
  at least two observations.
- Directions are the leading eigenvectors mapped back through the predictor
  whitener. No response-based prediction API is implied.
- Evidence: symmetric quadratic central-subspace recovery (where SIR has zero
  inverse mean), covariance and spectral residuals, slice-size failures, and
  affine feature-transform checks. Compare direction projectors, not signs.
- Complexity and legacy: dense covariance formation and feature eigendecomposition
  dominate the calculation. Rdimtools is not an oracle.
