# PHATE validation ledger

- **Status:** validated within the domains and checks below.
- **Primary source:** Moon et al. (2019),
  [doi:10.1038/s41587-019-0336-3](https://doi.org/10.1038/s41587-019-0336-3).
- **Kernel:** row bandwidth is the positive `k`th nonself distance, regardless of the global diameter. Directed alpha-decay
  values `exp(-(D_ij/bandwidth_i)^decay)` are averaged with their transpose and
  row-normalized to the diffusion operator.
- **Potential:** the declared integer diffusion power is transformed by `-log`; values
  below the explicit `potential_floor` are regularized and reported in diagnostics.
  Euclidean distances between potential rows are embedded by Metric MDS/SMACOF.
- **Time selection.** The default uses 20 diffusion steps. With `diffusion_time="auto"`,
  normalize the singular values raised to each integer power from 1 through
  `max_diffusion_time`. Fit two lines to the resulting entropy curve at each
  interior split. Choose the earliest split with minimum total absolute residual.
  The fitted value and curve are available as `diffusion_time_` and `diffusion_entropy_`.
- **Landmarks.** For fewer landmarks than observations, select the first row
  and then the row farthest from the selected set. Assign samples to their
  closest landmark, breaking ties by selection order. For affinity matrix
  $K$ and membership matrix $H$, row-normalize $KH$ to obtain $P$ and
  row-normalize its transpose to obtain $Q$. Diffuse and fit metric MDS on
  $QP$, then interpolate the landmark embedding with $P$. The dense sample
  affinity is retained. Specifying every sample recovers the full method.
- **Failure domain.** Zero adaptive bandwidth, disconnected affinity, insufficient
  landmark rank or count, and invalid potential floors are rejected.
- **Evidence:** tests independently reconstruct pairwise distances, row bandwidths,
  directed alpha-decay values, the symmetrized affinity, row normalization, and
  potential distances; they also check symmetry/hollowness, zero-bandwidth rejection,
  literal MDS stress, and truthful convergence. The method is transductive; Rdimtools
  is not an oracle.
- **Independent extension checks.** Tests reconstruct the transition from an
  explicit membership matrix, verify the entropy and knee independently, and
  compare the full and all-landmark paths.
- **Complexity.** For $m$ diffusion states (samples or landmarks), $A$ time
  candidates (zero for fixed time), and $T$ MDS iterations, an upper bound is
  $O(n^2(p+\log n) + nm^2 + m^3\log(t+1) + Tm^2k + A(m+A))$ time and
  $O(np + n^2 + A)$ memory. Symbols follow the [method catalog](../../methods.md).
