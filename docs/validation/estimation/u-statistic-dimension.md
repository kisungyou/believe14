# UStatisticDimension validation ledger

- **Status:** validated for believe14 0.1.0 within the documented scope. The
  reference equations and numerical implementation are checked, and the
  prospective 4,500-fit protocol below passed for nine specified low-dimensional
  configurations. Historical failures and higher-dimensional limitations remain.
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
  Ten is a computational minimum, not an accuracy guarantee. Empirical accuracy
  validation covers only the nine exact configurations of the protocol below.
- **Observed accuracy limitation:** the independent seeds 5701--5705 on the
  uniform dimension-3 flat with 240 observations yield `[2, 3, 3, 3, 4]`.
  RMSE `0.63246` exceeds the prespecified `0.5` release threshold. Both extreme
  results agree with an independent all-partition kernel and weighted-regression
  calculation. This finite-sample accuracy limitation remains a failed historical
  panel. Valid arithmetic and convergence do not guarantee recovery.
- **Release policy:** the original failed panel remains in `historical_panel_gate`.
  The prospective protocol below requires a separate simultaneous accuracy
  certificate. Evidence integrity, convergence, and finite-output requirements
  remain mandatory. A permanent finite-sample warning is recorded in fit
  diagnostics; registry metadata and this ledger state the validation status.
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

## Prospective accuracy protocol

This is an explicit revision of the validation protocol for the unchanged
reference estimator. It distinguishes a fixed five-dataset panel from population
RMSE on nine specified distributions. The original panel continues to fail.
No bandwidth, partition rule, estimator formula, or numerical limit is changed.

Before inspecting new outcomes, the protocol fixes 500 independent data/estimator
seed pairs for each original scenario, beginning at seed `926400000`, and the
exact original sample sizes, ambient dimensions, and candidate ceilings. The
scope is uniform flats of dimension 1--3 at 240 and 360 samples, the original
lightly noisy dimension-2 flat, sphere, and Swiss roll at 300 samples. It does not
cover the dimension-5 limitations documented above or arbitrary unseen data.

For integer absolute estimation error `e` bounded by `J`, the exact identity is
`MSE = sum(j=1..J) (2*j-1) P(e >= j)`. The nine configurations have bounds
`J = [4,4,3,3,3,3,3,1,1]`, totaling 25 binomial tails. Each tail receives a
one-sided [Clopper--Pearson upper confidence bound](https://www.barestatistics.nl/uploads/1/1/7/9/11797954/clopper__pearson_1934.pdf)
with error allocation `0.05/25`. A union bound therefore gives at least 95%
simultaneous coverage for all nine population MSE upper bounds, assuming the
specified independent trials and fixed estimator. Dependence between nested
tail counts does not invalidate this bound. Uncertainty remains positive when
zero errors are observed.

To define risk even when computation fails, a failed or invalid fit is assigned
loss `J`; a valid fit has loss `e`. The certificate bounds the root mean square of
this bounded loss. When fits succeed this is ordinary RMSE, and the bound also
controls RMSE conditional on a valid fit because failure receives the maximum
loss. It does not assert that future failure probability is zero.

Every upper bound must be at most `0.5`; this also bounds absolute bias and
standard deviation among valid fits by `0.5`. Any observed failed, missing,
noninteger, nonfinite, or inconsistent fit prevents certification regardless of
the numerical bound. The fixed design and
source fingerprints are stored before the first prospective run. Later CI
invocations replay those same seeds and are not new independent evidence.
The implementation, tests, and frozen constants are in
`tools/ustatistic_certification.py`; reproduction commands are in the development
guide. Its decision must be computed from trusted execution and raw fitted
records, not a caller-supplied pass flag.

## Prospective results

The protocol was committed as
[`07ab8d4`](https://github.com/kisungyou/believe14/commit/07ab8d4)
before the new fits. The first study completed on September 25, 2026 using
Python 3.13.11, NumPy 2.5.2, SciPy 1.18.0, and scikit-learn 1.9.0. All 4,500
fits produced valid finite outputs. Independent verification regenerated each
dataset fingerprint and recomputed every decision from the saved raw records.

| Configuration | Observed RMSE | Simultaneous 95% upper bound |
| --- | ---: | ---: |
| Uniform flat, dimension 1, 240 samples | 0.00000 | 0.44456 |
| Uniform flat, dimension 1, 360 samples | 0.00000 | 0.44456 |
| Uniform flat, dimension 2, 240 samples | 0.04472 | 0.34002 |
| Uniform flat, dimension 2, 360 samples | 0.00000 | 0.33342 |
| Uniform flat, dimension 3, 240 samples | 0.18439 | 0.40385 |
| Uniform flat, dimension 3, 360 samples | 0.04472 | 0.34002 |
| Noisy flat, latent dimension 2, 300 samples | 0.00000 | 0.33342 |
| Sphere, dimension 2, 300 samples | 0.00000 | 0.11114 |
| Swiss roll, dimension 2, 300 samples | 0.00000 | 0.11114 |

Every bound is below `0.5`, so this prospective protocol passes. The original
five-run panel still fails. The noisy scenario assesses recovery of latent
dimension at the specified noise scale; its support dimension is the ambient
dimension. These results do not extend certification to the dimension-5
configurations above or establish a universal sample-size rule.

The initial raw evidence and frozen design are retained under
`build/ustatistic-certification/`. The frozen-design SHA-256 is
`4007efb7da011ef64bfd44b72a2bee98ca3aaef1a7195ab6dd58b8c06dbd50b9`.
Its evidence SHA-256 is
`84b7307272a5c58aab360fdd5baa6c622471d90bde3648f0965dacf009ea8186`.
The scientific CI artifact also contains the full reproducible evidence.
