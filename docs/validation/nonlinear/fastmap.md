# FastMap validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Faloutsos and Lin (1995),
  [doi:10.1145/223784.223812](https://doi.org/10.1145/223784.223812).
- **Definition:** each coordinate uses pivot objects `a,b` and
  `(D(a,x)^2 + D(a,b)^2 - D(b,x)^2)/(2 D(a,b))`; squared residual distances subtract
  the squared coordinate difference before the next axis.
- **Conventions:** farthest-pivot sweeps begin at row zero, and stable first-index
  `argmax` resolves ties. Significant negative residual squared distances fail; only
  roundoff-scale negatives are clamped to zero. Residual rank uses the standard
  `n * eps` tolerance after normalizing the largest original dissimilarity to one.
  Exhausted residual rank yields identical zero-distance pivots, explicit zero
  trailing coordinates, and a diagnostic warning.
- **Out of sample:** feature-input fits retain pivot feature vectors and apply the same
  residual-distance recursion. For a precomputed fit, query input is the rectangular
  matrix of dissimilarities from each query to every fitted observation; its pivot
  columns drive the identical cited recursion.
- **Evidence:** rank-two Euclidean distances and fitted pivot extension are recovered
  to roundoff. Complexity is `O(k n^2)` time/memory for the current exact dense path.
  Rdimtools is not an oracle.
