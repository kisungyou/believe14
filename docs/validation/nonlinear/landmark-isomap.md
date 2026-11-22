# LandmarkIsomap validation ledger

- **Status:** validated.
- **Primary sources.** de Silva and Tenenbaum introduce landmark Isomap and
  landmark triangulation. Bengio et al. derive the fixed-training-graph
  extension for a query joined to its nearest neighbors.
  [Landmark Isomap](https://proceedings.neurips.cc/paper_files/paper/2002/file/5d6646aad9bcc0be55b2c82f69750387-Paper.pdf).
  [Out-of-sample extensions](https://proceedings.neurips.cc/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html).
- **Definition.** Form the undirected union of exact Euclidean neighbor edges,
  run Dijkstra's algorithm from each chosen landmark, and perform landmark
  classical scaling using those distances. Given a query $x$ with neighbors
  $N(x)$, its distance to landmark $l$ is
  $\min_{j\in N(x)}(\lVert x-x_j\rVert_2 + d_G(j,l))$. Project this distance
  vector using the fitted landmark scaling basis.
- **Boundaries.** Ties in a neighborhood use ascending training index. Reject
  disconnected graphs, zero-length edges and insufficient positive landmark
  rank. A reusable exact graph must match the ordered training data and cover
  the requested neighbor count.
- **Equivalence and evidence.** An independently constructed shortest-path graph
  verifies the training and query formulas. Line geometry, training replay and
  all-landmark equivalence to full Isomap are tested up to orthogonal
  orientation. Rdimtools is not an oracle.
- **Complexity.** With $m$ landmarks, $k$ neighbors and $d$ embedding dimensions,
  training costs $O(n^2(p + \log n) + mn(k+1)\log n + m^3 + nmd)$ time.
  Reusing a graph skips the exact search. Working memory is
  $O(np + nk + nm + m^2)$, including the stored $n$ by $m$ path table.
  Each query costs $O(np + n\log n + km + md)$ time.
