# Validation evidence

Every public method has a ledger in the matching family directory. A ledger
records the normative reference, exact formulation, numerical conventions,
failure domain, output equivalence, and evidence used for promotion.

The machine-readable catalog is stored in `methods.toml`. CI requires exact
agreement between this catalog, the Python registry, public family exports,
validation ledgers, and installed-wheel inventory. Ledger paths must be unique;
each ledger must record matching validation status, a normative source, frozen formulation,
numerical and output contracts, complexity, independent evidence, and the
Rdimtools divergence policy. Placeholders fail the release gate.

Public availability and validation status are separate. All 30 entries are
`public`; 29 have `validation_status="validated"`, meaning their recorded tests
pass within the documented scope. This label is not a general accuracy or
maturity guarantee. `UStatisticDimension` has `validation_status="experimental"`
because its reference implementation fails the prespecified accuracy threshold.
No supported accuracy regime is established for it. The Python registry,
manifest, and ledger must agree on this status.

The full-inventory scientific release gate retains all methods and the original
thresholds. Its accuracy failure remains visible and blocks publication.
The separate supported-scope result is informational: it excludes experimental
accuracy requirements while retaining computational and evidence-integrity
checks for every method. It does not override the release gate.

```{toctree}
:maxdepth: 1
:glob:

linear/*
nonlinear/*
estimation/*
```
