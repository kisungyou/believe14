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
- **Contract:** transductive; no `transform`. Rdimtools is not an oracle.

- **Constant-mode constraint:** the global symmetric eigenproblem is solved on
  the orthogonal complement of the known constant vector using Helmert contrasts.
  This preserves mean-zero orthonormal coordinates even when the zero eigenvalue
  is repeated; deleting an arbitrary first eigenvector would not. Noiseless flat
  LTSA tests compare the complete coordinate projector and row-permuted geometry.

- **Complexity:** `O(n^2 p + n h^2 p + n h^3 + n^3) time; O(np + n^2) memory`; symbols follow the
  [method catalog](../../methods.md).
