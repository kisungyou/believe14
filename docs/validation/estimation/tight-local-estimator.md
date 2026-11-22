# TightLocalEstimator validation ledger

- **Status:** validated for the recorded local likelihood, geometric identities,
  and synthetic examples below.
- **Primary source:** Amsaleg et al., *Intrinsic Dimensionality Estimation
  within Tight Localities: A Theoretical and Experimental Analysis*, sections
  4.1–4.3 ([author manuscript](https://arxiv.org/abs/2209.14475)).
- **Statistic:** use the extended paper's TLE variant with reflected measurements
  and without measurements originating at the query itself. Let the query be
  the origin and divide all neighbor offsets by the outer neighbor radius.
  For each ordered pair of different neighbors `(u,v)`, compute the first
  positive contact time `t` satisfying `||(1-t)u-v||=t` and repeat with
  `-u` as the starting point. The local estimate is the negative reciprocal
  of the mean of the `2k(k-1)` log contact times. The reported global summary
  is the arithmetic mean over fitted observations.
- **Numerical conventions:** solve the quadratic from Theorem 2 using
  a cancellation-resistant choice of positive root. A target exactly on the
  outer surface has unit contact time for a distinct start. Reject duplicates,
  exact reflected coincidences (zero log argument), and a neighborhood with
  zero likelihood denominator. Evaluate pair separations before dividing by
  the outer radius. For radii tied within floating-point precision, resolve
  the surface among the selected neighbors using exact binary input coordinates.
  Near the surface, compute the complement of the contact time to retain its
  small negative logarithm. When the outer radius is subnormal, recompute
  nearby offsets in power-of-two-scaled units so that rounding to subnormal
  distance units cannot change the neighborhood. Accumulate the global mean
  without overflowing an intermediate sum.
- **Input and output:** finite dense Euclidean row observations with at least
  three nonself neighbors. Equal input distances are ordered by original row
  index. `local_dimensions_` has one estimate per observation and
  `local_dimensions_by_k_` stores these in a single column. Larger precomputed
  exact neighbor lists may be reused on the same ordered dataset. The estimator
  describes neighborhoods of the fitted samples and provides no new-sample
  transform. Nearby manifold boundaries, intersections, curvature, and density
  changes can affect local estimates.
- **Complexity:** with `n` observations, `p` features and `k` neighbors,
  `O(n²(p+log n) + nk²p)` time and `O(np + n² + k²p)` peak memory with
  the default exact neighbor search. A supplied graph removes repeated search
  for ordinary radii. A subnormal neighborhood can require a new candidate
  scan, including when a graph is supplied.
- **Evidence:** a scalar root finder uses the unsquared moving-ball equation
  for every ordered pair in a small fixture. Further checks cover rigid
  motions, scaling, independent spatial dimension regimes, singular
  neighborhoods, and identical results using exact precomputed neighbors.
  An independent rational interval-contact formula tests one-dimensional
  data with neighbor separations that span more than 450 orders of magnitude.
  Rdimtools output is not used as a mathematical oracle.
