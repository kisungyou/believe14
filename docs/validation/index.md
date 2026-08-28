# Validation evidence

Every public method has a ledger in the matching family directory. A ledger
records the normative reference, exact formulation, numerical conventions,
failure domain, output equivalence, and evidence used for promotion.

The machine-readable catalog is stored in `methods.toml`. CI requires exact
agreement between this catalog, the Python registry, public family exports,
validation ledgers, and installed-wheel inventory. Ledger paths must be unique;
each ledger must record validated status, a normative source, frozen formulation,
numerical and output contracts, complexity, independent evidence, and the
Rdimtools divergence policy. Placeholders fail the release gate.

Promotion states are `proposed`, `specified`, `implemented`, `validated`, and
`public`. Version 0.1.0 ships only entries in the final `public` state.

```{toctree}
:maxdepth: 1
:glob:

linear/*
nonlinear/*
estimation/*
```
