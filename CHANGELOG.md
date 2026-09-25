# Changelog

## Unreleased

- Add an explicitly prospective U-statistic accuracy protocol with 500 fixed
  independent trials per original scenario and simultaneous 95% RMSE upper
  bounds. Preserve the original failed panels as historical evidence; keep the
  reference estimator, accuracy limits, and all other required checks unchanged.
- Mark UStatisticDimension accuracy experimental in registry metadata, diagnostics,
  and documentation; retain the reference algorithm and original failed release
  gate. Add a separate informational supported-scope result and a reproducible
  independent finite-sample characterization.
- Correct DANCo KL cancellation, LTSA/LLE constant-mode selection, FastMap feature
  queries, PPCA covariance accumulation, neighbor log ratios, and MDS rescaling.
- Reject DANCo neighborhoods with fewer than three neighbors and unrepresentable
  exposed statistics; add independent regression and numerical-range coverage.
- Separate t-SNE stopping tolerances and expose the stopping reason and gradient
  convergence; validate optimization quality with tighter restarts.
- Make scientific gates verify complete raw evidence against trusted scenarios,
  add holdout/extended distribution studies and uncertainty intervals, and correct
  dense time/memory costs in the registry and validation ledgers.
- Run the regular test suite after both wheel and source-distribution installation
  in the existing supported-platform workflows.

## [0.1.0]

- Initial release with 12 linear reducers/selectors, 12 nonlinear embeddings,
  and 6 intrinsic-dimension estimators.
- Added paper-audit ledgers, independent numerical validation, and a public
  capability registry.
- Added an executable method card for every public estimator and six
  task-oriented guides built against the installed wheel.
- Added installed wheel and sdist checks across Python 3.12–3.14 on Linux,
  macOS, and Windows.
- Added an immutable-artifact release path through TestPyPI, PyPI, GitHub
  Releases, and GitHub Pages using trusted publishing.

[0.1.0]: https://github.com/kisungyou/believe14/releases/tag/v0.1.0
