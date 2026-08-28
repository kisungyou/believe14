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
- Failure rule: a requested component not separated from the isotropic-noise
  eigenspace, or an ML noise estimate on the scale-relative singular zero
  boundary, is rejected rather than perturbed.
- Evidence: literal covariance eigenspectrum, posterior identity, likelihood,
  reconstruction-shape, and invalid-rank tests. Compare model covariance or
  loading projectors under repeated eigenvalues.
- Complexity and legacy: the dense path is dominated by the centered thin SVD.
  Rdimtools output does not override the closed-form paper likelihood.
