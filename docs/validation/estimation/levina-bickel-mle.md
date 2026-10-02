# LevinaBickelMLE validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Levina and Bickel, *Maximum Likelihood Estimation of
  Intrinsic Dimension*, Advances in NIPS 17 (2004), 777–784,
  [paper](https://proceedings.neurips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html),
  Equations 8–10.
- **Local statistic:** for ordered positive neighbor radii `T_j(x)`,
  `m_k(x) = a_k / sum_{j=1}^{k-1} log[T_k(x)/T_j(x)]`. Here `a_k=k-2` by
  default, the paper's first-order unbiased correction; setting
  `bias_correction=False` selects the literal likelihood numerator `k-1`.
- **Aggregation:** arithmetic mean over observations at each integer
  `k=k_min,...,k_max`, followed by an arithmetic mean over those `k` values,
  exactly as Equation 9. Defaults are the paper's fixed range 10–20.
- **Output:** global real `dimension_`, per-observation mean
  `local_dimensions_`, estimates by `k`, and their across-scale variation.
- **Invariances:** translation, orthogonal transformation, uniform scaling, and
  row permutation. Exact distance ties use stable row-index ordering.
- **Failure policy:** reject duplicates/zero radii and any local denominator made
  zero by neighbor-radius ties; never jitter distances or drop affected rows.
  Log ratios are evaluated without dividing widely separated radii, with `log1p`
  for close radii. Local estimates must be finite and positive.
- **Advertised regime:** locally homogeneous Poisson approximation on one
  manifold, with `k` small relative to sample size. Estimates are scale-dependent
  when the data have noise or multiple dimensional regimes.
- **Complexity:** `O(n^2(p + log n) + n h^2)` time and `O(np + n^2)` peak memory;
  symbols follow the [method catalog](../../methods.md). Rdimtools is not an oracle.
- **Independent evidence:** both numerator conventions have literal local-formula
  fixtures; invariance, neighbor-range, duplicate/tie, and synthetic recovery
  tests pass. No Rdimtools behavior is inherited.
