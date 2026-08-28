# TSNE validation ledger

- **Status:** validated exact-dense implementation for believe14 0.1.0.
- **Primary source:** van der Maaten and Hinton (2008),
  [JMLR 9:2579-2605](https://www.jmlr.org/papers/v9/vandermaaten08a.html).
- **Probabilities:** each Gaussian conditional distribution is binary-searched to
  entropy `log(perplexity)`, then `P_ij = (p_j|i + p_i|j)/(2n)`. Low-dimensional
  probabilities use the one-degree-of-freedom Student kernel over all ordered pairs.
  Perplexity is restricted to `[1, n_samples - 1]`; every row's achieved entropy is
  checked to absolute tolerance `1e-8`. An unattainable entropy caused by tied nearest
  distances is rejected rather than accepted after an exhausted binary search.
- **Optimization:** believe14 evaluates the exact symmetric objective and analytic
  gradient with dense L-BFGS-B. Early exaggeration is an explicit first objective
  phase; PCA or a local-Generator random start is explicit. Barnes-Hut, FFT, and
  nearest-neighbor probability approximations are absent.
- **Convergence:** exhausting either budget never becomes success implicitly. The final
  non-exaggerated KL divergence, gradient norm, iterations, and solver message are
  recorded. No claim of global optimality is made.
- **Evidence:** tests check conditional entropy residuals, `P`
  symmetry/normalization/zero diagonal, the analytic objective gradient against
  central differences, seed replay, legacy-global-RNG isolation, invalid and
  tied-infeasible perplexity, finite KL, and the absence of an out-of-sample API.
  Complexity is `O(t n^2 k)` time and `O(n^2)` memory. Rdimtools is not an oracle.
