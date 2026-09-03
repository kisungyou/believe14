---
html_theme.sidebar_secondary.remove: true
---

# believe14

`believe14` is a paper-first Python library for dimensionality reduction and
intrinsic-dimension estimation. Its 30 public estimators share explicit
numerical diagnostics, deterministic examples, and validation ledgers that
freeze the mathematical conventions used by version 0.1.0.

<div class="landing-grid">
  <a class="landing-card" href="getting_started/index.html">
    <h2>Getting started</h2>
    <p>Install believe14, run a complete PCA example, and read its numerical diagnostics.</p>
  </a>
  <a class="landing-card" href="tutorials/index.html">
    <h2>Tutorials</h2>
    <p>Learn workflows through six deterministic guides executed during every build.</p>
  </a>
  <a class="landing-card" href="methods.html">
    <h2>Methods</h2>
    <p>Browse all 30 estimators by family and open their executable method cards.</p>
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
:maxdepth: 1

Getting started <getting_started/index>
Tutorials <tutorials/index>
Methods <methods>
API reference <api>
Development <development/index>
```
