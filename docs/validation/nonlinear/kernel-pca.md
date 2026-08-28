# KernelPCA validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Schölkopf, Smola, and Müller (1998),
  [doi:10.1162/089976698300017467](https://doi.org/10.1162/089976698300017467).
- **Definition:** center the training Gram matrix as `Kc = J K J`; retain positive
  eigenpairs. Training coordinates are `V sqrt(lambda)` and query coordinates are the
  centered cross-kernel multiplied by `V/sqrt(lambda)`.
- **Linear-kernel stability:** training features are centered before their Gram matrix
  is formed, and query features use the stored training mean. This is algebraically
  identical to Gram double-centering while avoiding catastrophic cancellation under a
  large common feature offset.
- **Kernels:** linear, RBF, integer-degree polynomial, and precomputed. RBF/poly
  `gamma` defaults to `1/n_features`. A significantly indefinite centered kernel or
  insufficient positive numerical rank is rejected, not repaired.
- **Out of sample:** feature inputs use the fitted kernel; precomputed queries supply
  exactly one kernel column per training observation. Both use training centering
  statistics, implementing the Nyström projection.
- **Evidence:** the linear-kernel training Gram equals the literal centered feature
  Gram, remains stable under a large common translation, and training cross-kernel
  projection reproduces fitted coordinates. Symmetry, shape, definiteness, and rank
  failures are tested. Rdimtools is not an oracle.
- **Complexity:** the exact dense path costs `O(n^2 p + n^3)` time and `O(n^2)`
  memory, with kernel-evaluation cost replacing `O(n^2 p)` where appropriate.
