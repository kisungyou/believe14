# MetricMDS validation ledger

- **Status:** validated for believe14 0.1.0.
- **Authority:** Kruskal's metric raw-stress criterion and de Leeuw's SMACOF
  majorization.
- **Objective:** `sum_{i<j} (D_ij - ||z_i-z_j||)^2`, with all off-diagonal
  weights equal to one. Each Guttman update is centered and must not increase stress
  beyond a scale-aware roundoff allowance.
- **Conventions:** initialization is explicitly `classical` or local-Generator
  `random`; default is classical. Convergence is relative raw-stress change. Exhausting
  `max_iter` is recorded as non-convergence. Coincident fitted points paired with a
  positive target distance are rejected rather than divided by an epsilon.
- **Evidence:** the reported objective is recomputed literally from pair distances;
  tests check descent from the classical start, precomputed validation, RNG isolation,
  and truthful diagnostics.
- **Contract:** transductive; no `transform`. Dense complexity is `O(t n^2 k)` time
  and `O(n^2)` memory. Rdimtools is not an oracle.
