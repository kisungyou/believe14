# Contributing scientific methods

A method is added as one vertical slice: a specification ledger, independent
literal oracle, production estimator, contract and numerical tests, API
documentation, and registry record. Candidate code remains outside public
exports until all evidence passes.

## Required ledger fields

Every ledger states:

- the exact primary paper, version, pages, and equations;
- preprocessing, objectives, normalizations, denominators, and defaults;
- graph construction, tie handling, symmetrization, and connectivity policy;
- solver initialization, convergence test, and explicit failure behavior;
- stochastic stream semantics and allowed output equivalence;
- input domain, degeneracies, complexity, and out-of-sample status;
- independent formula fixtures, comparators, metamorphic properties, and
  empirical validation scenarios;
- known errata and deliberate differences from Rdimtools.

## Numerical implementation

Keep the audited reference formulation in Python. Use shared private kernels
only after their conventions have been matched explicitly. Do not move a
formula into native code merely for presumed speed; first record a repeatable
profile showing a stable shared bottleneck. A future native kernel must retain
the Python reference path and pass value, error, and diagnostic parity tests.

## Promotion

The only valid promotion sequence is `proposed`, `specified`, `implemented`,
`validated`, then `public`. Unresolved formula ambiguity, false convergence,
failed calibration, or an uncited out-of-sample rule blocks promotion. A
roadmap may name blocked or proposed work, but source distributions and wheels
must not expose placeholder estimators.
