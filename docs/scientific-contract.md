# Scientific contract

## Normative sources

Each method is specified from its primary paper and published errata. Author
code and independent implementations may clarify conventions but do not
silently override the paper. Rdimtools is used only for inventory and forensic
comparison; matching legacy output is not a correctness criterion.

## Inputs and failures

Observations occupy rows. Version 0.1.0 accepts finite, dense, real-valued
`float64` data and runs on CPU. Invalid parameter domains, undefined
statistics, disconnected graphs, material indefiniteness, and failed solver
convergence are reported explicitly. Implementations do not repair scientific
inputs or switch algorithms silently.

## Embedding equivalence

Coordinates are compared only up to transformations permitted by the method:
projectors or principal angles for linear subspaces, complete invariant
subspaces for repeated eigenvalues, and centered Gram matrices or distance
matrices for Euclidean embeddings. Raw eigenvector signs and arbitrary basis
rotations are not scientific targets.

## Randomness

Stochastic methods use a local `numpy.random.Generator`. An integer seed
replays a fit; a supplied generator advances normally; the NumPy global random
state is never read or mutated.

## Out-of-sample behavior

An estimator exposes `transform` only for its native linear map or a cited
Nyström or pivot extension. Transductive estimators intentionally have no
`transform` attribute.

## Release evidence

Release audits run against the installed wheel. Their JSON records artifact
hashes, commit and dirty state, the complete installed dependency environment,
platform and thread-pool/BLAS configuration, estimator parameters, seeds, input
hashes, diagnostics, and finite-output checks. Dimension-estimation gates span
prespecified flat dimensions and sample sizes, additive noise, a sphere, and a
curved manifold; missing scenarios, methods, or replicates fail closed.
