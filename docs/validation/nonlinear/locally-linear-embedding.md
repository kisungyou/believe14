# LocallyLinearEmbedding validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Roweis and Saul (2000),
  [doi:10.1126/science.290.5500.2323](https://doi.org/10.1126/science.290.5500.2323).
- **Definition:** exact neighbors minimize each local affine reconstruction under
  `sum_j w_ij = 1`; the embedding uses the nonconstant bottom eigenvectors of
  `(I-W)^T(I-W)`.
- **Regularization:** local covariance receives the declared
  `regularization * trace(C) * I` (default `1e-3`). Zero-scatter neighborhoods and
  singular normalizations are rejected instead of jittered. The undirected union graph
  must be connected.
- **Equivalence and evidence:** tests independently verify every affine row sum,
  reconstruction objective, eigensystem residual, graph failure, and translation
  behavior. Signs and repeated-eigenspace bases are not treated as identifiable.
- **Contract:** transductive; no `transform`. Complexity is
  `O(n^2 p + n k^3 + n^3)` in the dense reference path. Rdimtools is not an oracle.
