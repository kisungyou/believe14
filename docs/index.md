---
html_theme.sidebar_secondary.remove: true
---

# believe14

`believe14` is a paper-first Python library for dimensionality reduction and
intrinsic-dimension estimation. Its 30 public estimators share explicit
numerical diagnostics, deterministic examples, and validation ledgers that
freeze the mathematical conventions used by version 0.1.0.

```python
from believe14.linear import PCA

embedding = PCA(n_components=2).fit_transform(X)
```

<div class="landing-grid">
  <a class="landing-card" href="guides/choosing-a-method.html">
    <h2>Choose a method</h2>
    <p>Filter the registry by family, supervision, input, and out-of-sample support.</p>
  </a>
  <a class="landing-card" href="methods.html">
    <h2>Method catalog</h2>
    <p>Browse all 30 public estimators and their exact computational contracts.</p>
  </a>
  <a class="landing-card" href="guides/linear-latent-representations.html">
    <h2>Executable guides</h2>
    <p>Learn workflows through small, deterministic examples executed during every build.</p>
  </a>
  <a class="landing-card" href="api.html">
    <h2>API reference</h2>
    <p>Inspect constructors, fitted attributes, diagnostics, and public protocols.</p>
  </a>
</div>

```{admonition} Scientific scope
:class: note
Version 0.1.0 accepts finite dense real `float64` data on CPU. Methods reject
undefined statistics, disconnected graphs, and false convergence rather than
silently repairing scientific inputs or changing algorithms.
```

```{toctree}
:hidden:
:maxdepth: 2

guides/choosing-a-method
guides/linear-latent-representations
guides/supervised-and-paired-reductions
guides/distance-and-graph-embeddings
guides/stress-and-stochastic-embeddings
guides/intrinsic-dimension
methods
api
scientific-contract
references
validation/index
contributing
```

```{toctree}
:hidden:
:glob:

examples/*
```
