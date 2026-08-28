# Isomap validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Tenenbaum, de Silva, and Langford (2000),
  [doi:10.1126/science.290.5500.2319](https://doi.org/10.1126/science.290.5500.2319).
- **Definition:** construct the exact Euclidean `k`-nearest-neighbor graph, take the
  undirected union, compute exact all-pairs shortest paths, then apply Classical MDS.
- **Graph policy:** self-neighbors are excluded; stable row index resolves equal
  distances. Duplicate observations and other zero-length selected edges are rejected
  because sparse shortest-path storage cannot represent them as ordinary weighted
  edges. A disconnected requested graph is rejected. believe14 never adds edges,
  substitutes infinite paths, or increases `n_neighbors` silently.
- **Equivalence and evidence:** an unevenly sampled line has independently computed
  literal path distances and recovers their one-dimensional Euclidean geometry;
  duplicate-edge, disconnected-cluster, and negative-spectrum failures are tested.
- **Contract:** transductive in 0.1.0; no uncited interpolation or `transform`.
  Complexity is `O(n^2 p + n^3)` time and `O(n^2)` memory. Rdimtools is not an oracle.
