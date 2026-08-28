# LaplacianEigenmaps validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Belkin and Niyogi (2003),
  [doi:10.1162/089976603321780317](https://doi.org/10.1162/089976603321780317).
- **Definition:** on the exact symmetric-union neighbor affinity `W`, solve
  `(D-W) f = lambda D f`; omit the constant eigenvector and return the next axes.
- **Affinity:** either binary or the declared heat kernel
  `exp(-gamma ||x_i-x_j||^2)`. Neighbor ties are stable. The graph must be connected
  and have positive degrees; neither graph repair nor diagonal regularization occurs.
- **Equivalence and evidence:** output axes are checked for `D`-orthonormality and
  normalized generalized-eigen residual. Tests cover graph disconnection, rotations,
  permutations, and parameter failures.
- **Contract:** transductive; no `transform`. Dense reference complexity is cubic in
  `n` after exact graph construction. Rdimtools is not an oracle.
