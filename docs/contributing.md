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
- validation status matching the registry and manifest, including unresolved
  accuracy limitations;
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

Public availability does not freeze validation status. New adverse evidence
requires a public method's status to become `experimental` in its registry
record, manifest, and ledger, with a visible fit-diagnostic warning and documented
limitations. Restoration requires independent evidence under a protocol fixed
before new outcomes; document any protocol revision and preserve prior failures.
`UStatisticDimension` met this requirement in its prospective 4,500-fit study.
`validated` describes recorded evidence
within the documented scope, not a general accuracy or maturity guarantee.
Experimental status does not excuse arithmetic or evidence-integrity failures,
erase failed scenarios, or override the full-inventory publication gate.
