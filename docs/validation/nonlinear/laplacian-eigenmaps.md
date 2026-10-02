# LaplacianEigenmaps validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Belkin and Niyogi (2003),
  [doi:10.1162/089976603321780317](https://doi.org/10.1162/089976603321780317).
- **Definition:** on the exact symmetric-union neighbor affinity `W`, solve
  `(D-W) f = lambda D f` subject to `f.T D 1 = 0`. The symmetric normalized
  operator is restricted to the complement of the known square-root degree
  vector before diagonalization. The resulting coordinates are `D`-orthonormal;
  weak connectivity and repeated numerical zero eigenvalues cannot mix the
  constant mode into retained coordinates.
- **Affinity:** either binary or the declared heat kernel
  `exp(-gamma ||x_i-x_j||^2)`. Neighbor ties are stable. The graph must be connected
  and have positive degrees; neither graph repair nor diagonal regularization occurs.
- **Equivalence and evidence:** output axes are checked for `D`-orthonormality,
  weighted orthogonality to the constant vector, and normalized generalized-eigen
  residual. Tests cover graph disconnection, weakly connected positive affinities,
  repeated numerical modes, rotations, permutations, and parameter failures.
  Well-conditioned cases are compared with a direct generalized eigensolve.
  Numerical rank describes the unnormalized graph Laplacian.
- **Contract:** transductive; no `transform`. Dense reference complexity is cubic in
  `n` after exact graph construction. Rdimtools is not an oracle.

- **Complexity:** `O(n^2 p + n^3) time; O(np + n^2) memory`; symbols follow the
  [method catalog](../../methods.md).
