# FastICA validation ledger

- Status: validated; implementation: `believe14.linear.FastICA`.
- Authority: Hyvärinen (1999), DOI 10.1109/72.761722.
- Preprocessing: center, retain a rank-valid SVD subspace, and whiten with the
  ML covariance convention so `Xw^T Xw/n = I`.
- Symmetric update: for every row of `W`, use
  `E[x g(w^T x)] - E[g'(w^T x)]w`, with
  `g(u)=tanh(alpha*u)`, followed by symmetric decorrelation
  `(W W^T)^(-1/2) W`.
- Initialization/randomness: local Gaussian generator; integer seeds replay and
  global RNG state is untouched. Stop on maximum sign-invariant row alignment.
- Transform/inverse: fitted unmixing and its Moore--Penrose mixing matrix.
- Evidence: whitening covariance, exact full-rank inverse, non-Gaussian mixture,
  seeded replay, and forced non-convergence tests. Source order and sign are
  non-identifiable and are never raw-coordinate acceptance criteria.
- Complexity and legacy: the dense path combines a thin SVD with iterative
  symmetric fixed-point updates. Rdimtools output is not an acceptance target.
