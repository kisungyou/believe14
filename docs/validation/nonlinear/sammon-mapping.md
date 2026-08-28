# SammonMapping validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Sammon (1969),
  [doi:10.1109/T-C.1969.222678](https://doi.org/10.1109/T-C.1969.222678).
- **Objective:** `E = sum_{i<j} (D_ij-d_ij)^2/D_ij / sum_{i<j} D_ij`.
  believe14 minimizes this exact criterion with an analytic gradient and L-BFGS-B.
- **Conventions:** classical or local-random initialization is explicit. Every
  off-diagonal input dissimilarity must be positive, because zero makes the published
  weighting undefined. Coincident embedded points with positive targets are likewise
  rejected at gradient evaluation.
- **Evidence:** a test recomputes the paper-normalized criterion independently;
  finite-difference development checks cover the analytic gradient, and tests cover
  duplicate rejection, determinism, and truthful solver status.
- **Contract:** transductive; no `transform`. Complexity is `O(t n^2 k)` time and
  `O(n^2)` memory. Rdimtools is not an oracle.
