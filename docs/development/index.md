# Development

Scientific changes to `believe14` advance as complete vertical slices: a paper
specification, an independent literal oracle, an implementation, numerical and
API tests, an executable method card, and validation evidence.

## Local checks

The repository exposes the same small command interface used by continuous
integration:

```console
make quality
make test
make examples
make package-check
make release-check
```

`make examples` builds the documentation against the packaged wheel rather
than the source checkout. `make release-check` adds research checks, clean-tree
and repository-provenance verification, installed-artifact tests, and the
complete scientific release audit.

## Adding or changing a method

Read {doc}`Contributing scientific methods <../contributing>` before changing
an estimator. It defines the required ledger fields, numerical implementation
policy, and the only accepted promotion sequence. Public status is withheld
when a formula is ambiguous, convergence is false, calibration fails, or an
out-of-sample rule lacks a normative source.

## Validation evidence

The {doc}`validation ledgers <../validation/index>` record each public method's
normative reference, formulation, numerical conventions, failure domain, and
independent evidence. The machine-readable ledger and Python registry must
agree exactly.

The scientific audit uses five prespecified calibration seeds and five separate
holdout seeds. Its low-dimensional accuracy thresholds are unchanged. An extended
characterization exercises uniform flats of dimension 5, 10, and 15, Gaussian
sampling of dimension 2 and 8, and anisotropic dimension-3 data at 400 and 800
samples. Candidate bounds exceed truth except the U-statistic's published maximum
of 15. These additional scenarios require complete finite converged fits and
report accuracy; they do not assert universal recovery at those dimensions.

Every scenario archives raw estimates, fit parameters, numeric-output checks,
diagnostics, and exact sample hashes. The gate obtains required seeds and allowed
DANCo one-dimensional exclusions from trusted specifications, checks sample hashes,
and recomputes every summary. Missing/duplicate runs, nonconvergence, nonfinite
outputs, unapproved exemptions, or inconsistent metadata fail closed.

Mean/bias intervals use Student's t distribution; RMSE intervals use a fixed-seed
percentile bootstrap, and failure-rate intervals use Wilson's score method.
With only five replicates these intervals are descriptive and can be wide; zero
observed failures does not demonstrate a zero failure probability. The noisy
scenario labels its target as latent manifold dimension and its support dimension
as the full ambient dimension. Fixed positive-radius correlation slopes and
finite-sample high-dimensional bias must be interpreted at their sampled scales.

The September 2026 holdout check blocks full-inventory release: UStatisticDimension on
the dimension-3 uniform flat at 240 samples has RMSE `0.63246`, above the existing
`0.5` limit. All five fits are finite and converged, and independent literal
calculations confirm the estimates. The original thresholds and holdout seeds
are retained so this limitation remains visible.

`UStatisticDimension` now has experimental accuracy status in the public registry,
its validation ledger, and fit diagnostics. Its implementation remains the
published reference algorithm. A successful computation does not certify accurate
dimension recovery, and no sample-size cutoff is advertised as sufficient.

The audit also reports an informational `supported_scope_gate`. This separates
the accuracy of the other estimators from the experimental method while still
requiring complete evidence, finite outputs, convergence, and numerical checks
for all 30 methods. It lists experimental accuracy failures explicitly. It cannot
override the full-inventory `release_gate`, which remains the publication gate;
the existing accuracy thresholds and holdout datasets have not changed.

Reproduce the larger, independent U-statistic study with:

```console
python -m tools.ustatistic_characterization freeze
python -m tools.ustatistic_characterization run
```

The tool freezes its design before collecting outcomes, keeps data generation and
partition randomness separate, and archives sample hashes, fit parameters, raw
estimates, failures, and uncertainty intervals under
`build/ustatistic-characterization/`. These observations characterize
finite-sample behavior; they do not substitute a new passing benchmark for the
original failed holdout.

## Releasing

The repository's [release guide](https://github.com/kisungyou/believe14/blob/main/RELEASING.md)
documents clean annotated tags, immutable wheel and source artifacts, TestPyPI
rehearsal, PyPI publication, GitHub Releases, and documentation deployment.

```{toctree}
:hidden:
:maxdepth: 2

Contributing scientific methods <../contributing>
Validation evidence <../validation/index>
```
