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
- **Out of sample:** feature-input fits retain orthonormal residual pivot directions.
  Both fitting and queries use direct shifted projections, algebraically equal to
  the cited distance formula, preserving coordinates under large orthogonal offsets.
  For a precomputed fit, queries supply distances to all fitted observations and
  use residual-distance recursion. If estimated roundoff in the squared distances
  exceeds `1e-6` times the squared residual pivot separation, projection resolution
  is inadequate and the query fails explicitly.
- **Evidence:** rank-two Euclidean distances and fitted pivot extension are recovered
  to roundoff. Tests also exercise full-rank orthonormality, training replay, very
  distant off-axis queries, and rejection of ill-conditioned precomputed queries.
  Rdimtools is not an oracle.

- **Complexity:** `O(n^2 p + k n^2 + Rkn) time; O(np + n^2) memory`; symbols follow the
  [method catalog](../../methods.md).
