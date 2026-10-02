# TSNE validation ledger

- **Status:** validated exact-dense implementation for believe14 0.1.0.
- **Primary source:** van der Maaten and Hinton (2008),
  [JMLR 9:2579-2605](https://www.jmlr.org/papers/v9/vandermaaten08a.html).
- **Probabilities:** each Gaussian conditional distribution is matched to entropy
  `log(perplexity)` by bisection in log precision, then
  `P_ij = (p_j|i + p_i|j)/(2n)`. The bracket comes from the row's smallest and
  largest positive squared-distance gaps: its endpoints approach the uniform
  and tied-nearest distributions within float64 precision. Raw distances are
  converted to log squared-distance gaps using the factored difference of
  squares, avoiding overflow and loss of small gaps under row normalization.
  Bandwidth search stops on its entropy criterion or explicitly reports
  floating-point stagnation; its work is independent of the embedding optimizer's
  `max_iter`. Low-dimensional
  probabilities use the one-degree-of-freedom Student kernel over all ordered pairs.
  Perplexity is restricted to `[1, n_samples - 1]`; every row's achieved entropy is
  checked to absolute tolerance `1e-8`. An unattainable entropy caused by tied nearest
  distances is rejected rather than accepted after an exhausted binary search.
- **Optimization:** believe14 evaluates the exact symmetric objective and analytic
  gradient with dense L-BFGS-B. Early exaggeration is an explicit first objective
  phase; PCA or a local-Generator random start is explicit. Barnes-Hut, FFT, and
  nearest-neighbor probability approximations are absent. Before the standard
  phase, a contracted start is restored to standard deviation `1e-4`, preserving
  its geometry. An exactly collapsed start reuses the original initialization.
  This explicit initialization convention prevents a near-zero early-exaggeration
  solution from passing an absolute gradient test at an uninformative stationary
  configuration. The rescaling is exposed in `early_exaggeration_rescaled_`.
- **Convergence:** success requires the standard phase and successful L-BFGS-B
  termination. The gradient threshold `tol=1e-7` and relative function-change
  threshold `function_tol=1e-12` are independent. `stopping_reason_` always
  records the solver message, and `gradient_converged_` reports whether the
  maximum absolute gradient entry satisfies `tol`. Function-change success without
  gradient convergence carries a warning; it is not a stationarity certificate.
  Final non-exaggerated KL, gradient norm, and iterations are retained. No claim
  of global optimality is made. See the [SciPy stopping criteria](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
- **Evidence:** tests check conditional entropy residuals, `P`
  symmetry/normalization/zero diagonal, the analytic objective gradient against
  central differences, seed replay, legacy-global-RNG isolation, invalid and
  tied-infeasible perplexity, finite KL, and the absence of an out-of-sample API.
  Independent scalar roots check the bandwidths of a dense cluster beside a
  distant outlier; further checks cover exact uniform scaling, squared-distance
  gaps at the subnormal boundary, and raw distances whose squares cannot be
  represented. Three independent Gaussian seeds are also checked for small final gradients
  and negligible improvement under a tighter optimization restart.
  Rdimtools is not an oracle.

- **Complexity:** `O(n^2 p + min(np^2,n^2p) + E n^2 k) time; O(np + n^2) memory`; symbols follow the
  [method catalog](../../methods.md).
