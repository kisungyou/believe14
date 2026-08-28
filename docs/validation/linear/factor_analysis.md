# FactorAnalysis validation ledger

- Status: validated; implementation: `believe14.linear.FactorAnalysis`.
- Authority: Rubin and Thayer (1982), DOI 10.1007/BF02293851.
- Model: centered observations follow `N(0, L L^T + diag(psi))`, with latent
  factors `N(0, I)`. The sample covariance uses divisor `n`, as required by ML.
- EM: `B=(I+L^T Psi^-1 L)^-1 L^T Psi^-1`, `C_xz=S B^T`,
  `C_zz=(I+L^T Psi^-1 L)^-1+B S B^T`, then
  `L=C_xz C_zz^-1` and `psi=diag(S-L C_xz^T)`.
- Constraint: uniquenesses have the explicit lower bound
  `min_noise_variance`; activation is recorded in diagnostics. Convergence uses
  relative observed-data log-likelihood change and max-iteration exhaustion is
  reported as non-convergence.
- Transform/reconstruction: posterior factor mean `B x`; conditional mean
  reconstruction `L z + mean`.
- Evidence: covariance recovery, posterior-mean identity, positive covariance,
  convergence, and forced non-convergence tests. Factor rotations/signs are not
  identifiable; covariance matrices are the primary comparison object.
- Complexity and legacy: fitting forms the dense feature covariance and performs
  rank-`k` EM updates until convergence. Rdimtools is not an oracle.
