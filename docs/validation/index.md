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
`public` and have `validation_status="validated"`, meaning their recorded tests
pass within the documented scope. This label is not a general accuracy or
maturity guarantee. `UStatisticDimension` passed the prospective protocol on nine
specified low-dimensional configurations. Its original failed panel and
higher-dimensional limitations remain documented. The Python registry, manifest,
and ledger must agree on validation status.

The scientific release gate retains all methods and the original accuracy limits.
The U-statistic validation protocol now requires 500 fixed independent replicates
for each original scenario and simultaneous 95% upper RMSE bounds at most `0.5`.
The other accuracy requirements and every computational/integrity requirement
remain mandatory. The original failed panel is retained as historical evidence;
the new protocol does not turn that panel into a passing result. See the
[development protocol](../development/index.md).

```{toctree}
:maxdepth: 1
:glob:

linear/*
nonlinear/*
estimation/*
```
