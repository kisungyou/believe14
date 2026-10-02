# ProbabilisticPCA validation ledger

- Status: validated; implementation: `believe14.linear.ProbabilisticPCA`.
- Authority: Tipping and Bishop (1999), DOI 10.1111/1467-9868.00196.
- Model: `x = W z + mu + epsilon`, `z ~ N(0,I)`, and
  `epsilon ~ N(0, sigma^2 I)`; the covariance eigenspectrum uses divisor `n`.
- Closed-form ML: `sigma^2` is the mean of the `p-k` discarded eigenvalues and
  `W=U_k (Lambda_k-sigma^2 I)^(1/2)`, with the arbitrary latent rotation fixed
  to identity and deterministic eigenvector signs.
- Transform: posterior mean `(W^T W+sigma^2 I)^-1 W^T (x-mu)`;
  reconstruction is `W z + mu`.
  Reconstruction restores the mean through the fitted reference and offset
  with compensated addition, preserving representable variation around a large
  offset. Truly unrepresentable reconstructed outputs fail explicitly.
- Failure rule: a requested component not separated from the isotropic-noise
  eigenspace, or an ML noise estimate on the scale-relative singular zero
  boundary, is rejected rather than perturbed.
- Evidence: literal covariance eigenspectrum, posterior identity, likelihood,
  reconstruction-shape, and invalid-rank tests. Compare model covariance or
  loading projectors under repeated eigenvalues. Covariance accumulation uses
  `Xc/sqrt(n)` before multiplication and the discarded-eigenvalue average avoids
  an overflowing sum. Extreme-scale tests check the exact Gaussian likelihood
  scaling identity. Non-finite likelihood or diagnostics fail explicitly.
- **Complexity:** `O(np^2 + p^3)` time and `O(np + p^2)` peak memory;
  symbols follow the [method catalog](../../methods.md). Rdimtools is not an oracle.
