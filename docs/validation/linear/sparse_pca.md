# SparsePCA validation ledger

- Status: validated; implementation: `believe14.linear.SparsePCA`.
- Authority: Zou, Hastie, and Tibshirani (2006),
  DOI 10.1198/106186006X113430.
- Frozen objective:
  `||Xc-Xc B A^T||_F^2/(2n) + alpha*||B||_1 + ridge_alpha*||B||_F^2/2`,
  subject to `A^T A=I`. This scaling defines the public penalty parameters.
- Updates: each column of `B` is solved by cyclic elastic-net coordinate descent;
  `A` is the exact orthogonal Procrustes solution from the SVD of `Xc^T Xc B/n`.
- Following Algorithm 1 step 5, public sparse loading `j` is
  `B_j / ||B_j||`. Raw penalized coefficients remain in `raw_components_`, and
  the reconstruction map absorbs their norms so it still represents `Xc B A^T`.
- Initialization: centered PCA by default or a local seeded random orthobasis.
  Both outer and coordinate stopping conditions must pass for convergence.
- Transform: `(X-mean)B`. `A` is retained separately as
  `reconstruction_components_`; no inverse API is claimed.
- Evidence: declared reconstruction identity, unit loading norms, sparsity,
  monotone objective, KKT residual, zero-variance handling, seeded behavior, and
  forced non-convergence tests. Compare reconstructed matrices, objectives, and
  supports rather than signed columns.
- Complexity and legacy: work scales with outer iterations, coordinate iterations,
  samples, features, and retained components. Rdimtools is not an oracle.
