# LandmarkMDS validation ledger

- **Status:** validated.
- **Primary source.** de Silva and Tenenbaum, *Sparse Multidimensional Scaling
  Using Landmark Points* (2004), equations for landmark triangulation.
  [Manuscript](https://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf).
- **Definition.** Center the squared distances among the selected landmarks
  and retain the largest positive eigenpairs of the resulting Gram matrix.
  Let $V$ and $\Lambda$ contain these eigenvectors and eigenvalues, $c$ the
  landmark column means of squared distances and $d^2$ the squared distances
  from a new point to the landmarks. Its coordinates are
  $(c-d^2)^T V\Lambda^{-1/2}/2$. For features this reduces to projecting
  the features centered at the landmark centroid onto the landmark principal
  directions.
- **Boundaries.** Select at least one more landmark than the requested dimension.
  Landmarks must have adequate positive numerical rank. For general
  dissimilarities, negative spectrum is recorded in the fitted diagnostics.
  With precomputed input, transform expects all original training columns.
- **Evidence and equivalence.** Plane recovery and held-out distances agree with
  Euclidean geometry. Independent landmark PCA verifies truncated embeddings.
  Training replay, signed non-Euclidean spectra and all-landmark equivalence to
  classical scaling are tested. Embeddings are equivalent up to orthogonal
  orientation within retained eigenspaces. Rdimtools is not an oracle.
- **Complexity.** With $m$ landmarks, $p$ features and $d$ output dimensions,
  the feature path costs $O(m^2p + m^3 + mpd + npd)$ time and
  $O(np + m^2 + pd + nd)$ memory. Validating a supplied square dissimilarity
  matrix costs $O(n^2)$ time and memory. Distance-based extension costs
  $O(nmd)$ time and $O(nm + nd)$ working memory.
