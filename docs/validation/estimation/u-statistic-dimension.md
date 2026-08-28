# UStatisticDimension validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Hein and Audibert, *Intrinsic Dimensionality Estimation of
  Submanifolds in Euclidean Space*, ICML 2005, 289–296,
  [doi:10.1145/1102351.1102388](https://doi.org/10.1145/1102351.1102388),
  Sections 3.1–3.2.
- **Kernel statistic:** `k(x)=(1-x)_+` and
  `U_{n,h,l}=mean[k(||X_i-X_j||^2/h^2)]/h^l`, using the correct one-sample
  unordered-pair or two-sample Cartesian-pair normalization.
- **Bandwidths:** `h_l(N)` is the mean full-sample nearest-neighbor radius and
  `h_l(n)=h_l(N)[(N/n)(log n/log N)]^(1/l)`.
- **Subsamples:** sizes are `floor(N/r)`, `r=1,...,5`. For each `r`, all
  `r(r+1)/2` one-/two-sample statistics from interleaved equal-size partitions
  are averaged. A local random permutation before partitioning prevents external
  row ordering from determining the partitions while retaining their i.i.d.
  interpretation.
- **Decision rule:** for every integer `l` from 1 through `max_dimension`, fit
  `log U` on `log h` by weighted least squares with weights `1/r`; choose the
  smallest absolute slope. The default upper bound is `min(n_features,15)`.
- **Output:** integer-valued float `dimension_`, base bandwidth, dimensionless
  bandwidth factors, log bandwidths, compact-kernel means, log U-statistics,
  slopes, and the sampled order. The logarithmic representation preserves the
  paper statistic when an absolute bandwidth or its raw `h^{-l}` factor would
  overflow or underflow float64. The absolute winning slope is the numerical
  residual; a boundary winner is reported in diagnostics.
- **Invariances:** translation, orthogonal transformation, and uniform scaling.
  Repeated fits are reproducible for an integer seed and never touch global RNG.
- **Failure policy:** reject duplicates, zero base bandwidth, a zero compact-kernel
  statistic, or collapsed log bandwidths. No bandwidth inflation is performed.
- **Advertised regime:** smooth sampled submanifold under the paper's regularity
  conditions; at least ten samples are required for five nontrivial scales.
- **Complexity:** `O(n^2 p + l_max n^2)` time and `O(n^2)` memory.
- **Independent evidence:** literal full-sample U-statistic fixture, five-scale
  synthetic recovery, geometric/RNG tests, and explicit degeneracy tests. No
  Rdimtools values are acceptance criteria.
