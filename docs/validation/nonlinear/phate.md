# PHATE validation ledger

- **Status:** validated fixed-time implementation for believe14 0.1.0.
- **Primary source:** Moon et al. (2019),
  [doi:10.1038/s41587-019-0336-3](https://doi.org/10.1038/s41587-019-0336-3).
- **Kernel:** row bandwidth is the exact `k`th nonself distance. Directed alpha-decay
  values `exp(-(D_ij/bandwidth_i)^decay)` are averaged with their transpose and
  row-normalized to the diffusion operator.
- **Potential:** the declared integer diffusion power is transformed by `-log`; values
  below the explicit `potential_floor` are regularized and reported in diagnostics.
  Euclidean distances between potential rows are embedded by Metric MDS/SMACOF.
- **Fixed conventions:** 0.1.0 requires an explicit fixed diffusion time and does not
  implement the optional von-Neumann-entropy time selector. Zero bandwidth,
  disconnected numerical affinity, zero degree, and invalid log parameters fail.
- **Evidence:** tests independently reconstruct pairwise distances, row bandwidths,
  directed alpha-decay values, the symmetrized affinity, row normalization, and
  potential distances; they also check symmetry/hollowness, zero-bandwidth rejection,
  literal MDS stress, and truthful convergence. The method is transductive; Rdimtools
  is not an oracle.
- **Complexity:** exact pairwise distances and diffusion storage use `O(n^2)`
  memory; dense diffusion powers and the final eigensolver are cubic in `n`.
