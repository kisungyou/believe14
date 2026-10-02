# LocalTangentSpaceAlignment validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Zhang and Zha (2004),
  [doi:10.1137/S1064827502419154](https://doi.org/10.1137/S1064827502419154).
- **Definition:** each neighborhood includes its center and exact neighbors. Its
  centered left singular vectors define the local tangent basis; local matrices
  `I - [1/sqrt(k+1), Theta][...]^T` are accumulated and globally diagonalized.
- **Failure policy:** the symmetric-union graph must be connected, `n_neighbors` must
  exceed output dimension, and every neighborhood must have sufficient numerical
  tangent rank. No rank-completing noise is introduced.
- **Equivalence and evidence:** tests check normalized alignment-eigen residuals,
  translation-invariant pairwise embedding geometry, deterministic neighbors, and
  disconnected/rank-deficient failures. Eigenspace geometry, not signs, is identified.
- **Contract:** transductive; no `transform`. Dense reference complexity includes a
  cubic global eigendecomposition. Rdimtools is not an oracle.

- **Constant-mode constraint:** the global symmetric eigenproblem is solved on
  the orthogonal complement of the known constant vector using Helmert contrasts.
  This preserves mean-zero orthonormal coordinates even when the zero eigenvalue
  is repeated; deleting an arbitrary first eigenvector would not. Noiseless flat
  LTSA tests compare the complete coordinate projector and row-permuted geometry.

- **Complexity:** `O(n^2 p + n min(hp^2,h^2p) + n h^2 k + n^3) time; O(np + n^2) memory`; symbols follow the
  [method catalog](../../methods.md).
