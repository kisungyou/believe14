# UStatisticDimension validation ledger

- **Status:** experimental for believe14 0.1.0. The reference equations and
  numerical implementation are checked; finite-sample accuracy has not met the
  package's full-inventory release requirements.
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
- **Theoretical regime:** smooth sampled submanifold under the paper's regularity
  conditions; at least ten samples are required for five nontrivial scales.
  Ten is a computational minimum, not an accuracy guarantee. The package does
  not advertise a validated finite-sample accuracy regime for this estimator.
- **Observed accuracy limitation:** the independent seeds 5701--5705 on the
  uniform dimension-3 flat with 240 observations yield `[2, 3, 3, 3, 4]`.
  RMSE `0.63246` exceeds the prespecified `0.5` release threshold. Both extreme
  results agree with an independent all-partition kernel and weighted-regression
  calculation. This is a finite-sample accuracy limitation, and the release gate
  remains failed. Valid arithmetic and convergence do not guarantee recovery.
- **Release policy:** experimental status is visible in registry metadata and
  every fit's diagnostic warnings. The full-inventory scientific release gate
  retains the original failed result. A separate informational supported-scope
  result excludes only experimental accuracy requirements; evidence integrity,
  convergence, and finite-output requirements still apply to this estimator.
- **Independent partition check:** across 50 alternative partition seeds, the
  first failing dataset estimates 2 in 38 fits and 3 in 12; the last failing
  dataset estimates 4 in all 50. Replacing partition means by their conditional
  expectation over uniform permutations (the full unordered-pair kernel mean at
  each bandwidth) also gives `[2, 3, 3, 3, 4]`. Changing the partition seed does
  not remove the underlying finite-sample limitation.
- **Complexity:** `O(n^2(p + D))` time and `O(np + n^2)` peak memory;
  symbols follow the [method catalog](../../methods.md). Rdimtools is not an oracle.
- **Independent evidence:** literal full-sample U-statistic fixture, five-scale
  synthetic recovery, geometric/RNG tests, and explicit degeneracy tests. No
  Rdimtools values are acceptance criteria.

## Independent finite-sample characterization

The frozen September 2026 study uses 50 fresh replicates per scenario, independent
data and partition seeds, eight ambient coordinates, and `max_dimension=8`.
No parameter was selected using these outcomes. All 800 fits completed with
finite outputs. Uniform-flat RMSE against the generating dimension was:

| Intrinsic dimension | 120 samples | 240 samples | 480 samples | 960 samples |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.1414 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.3742 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.7071 | 0.6481 | 0.5099 | 0.3742 |

For dimension 5, pointwise bootstrap 95% RMSE intervals were `[0.600, 0.800]`,
`[0.529, 0.748]`, `[0.374, 0.616]`, and `[0.245, 0.490]`, respectively. The
smaller samples clearly miss the unchanged `0.5` reference. These observations
do not establish a universal minimum sample size or simultaneous coverage.

At 480 samples, Gaussian dimension-3, sphere dimension-2, and Swiss-roll
dimension-2 scenarios each had zero observed error. Adding Gaussian noise with
standard deviation `0.01` to a dimension-3 flat gave RMSE `0.1414` against the
latent dimension; its support dimension is actually 8, so this is a finite-scale
robustness result. Zero errors among 50 replicates still permits an error rate
of about 7.1% at the upper Wilson 95% bound. In particular, the fresh
dimension-3 results do not invalidate the original failed holdout.

See the [reproduction commands](../../development/index.md). Designs and raw
evidence are saved under ignored `build/`, including the original source hashes.
The original study's recorded source hashes match the implementation with its
experimental-status metadata. Reusing those scenarios and seeds is a
reproducibility check, not an additional independent validation sample.
