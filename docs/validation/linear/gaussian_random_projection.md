# GaussianRandomProjection validation ledger

- Status: validated; implementation: `believe14.linear.GaussianRandomProjection`.
- Authority: Bingham and Mannila (2001), DOI 10.1145/502512.502546.
- Specification: draw `R_ji` independently from `N(0, 1/k)` and return `X R^T`.
  No centering, fitting to the observations, or orthogonalization is performed.
- Randomness: only a local NumPy `Generator` is used; an integer seed replays
  exactly and the legacy global RNG is untouched.
- Numerical contract: invalid component counts and non-finite inputs are
  rejected; no normalization, fallback, or post-hoc repair is performed.
- Equivalence: fixed seeds compare matrices exactly; distributional audits check
  zero mean, variance `1/k`, and Johnson--Lindenstrauss distance concentration.
- Evidence: literal matrix-product, replay, scaling, and global-state tests.
- Rdimtools: not used as an oracle. The fitted matrix defines the native
  out-of-sample map; cost is `O(npk)`.
