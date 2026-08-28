---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Intrinsic dimension

Dimension estimators summarize local or multiscale geometry without producing
an embedding. Correlation Dimension fits correlation-integral scaling; TwoNN,
Levina--Bickel MLE, and MiND-ML use nearest-neighbor laws; U-Statistic Dimension
uses convergence rates; DANCo combines norm and angle concentration
{cite:p}`grassberger1983,facco2017,levina2004,hein2005,lombardi2011,ceruti2014`.

Correlation-integral methods identify a scaling regime where

$$
C(r) \propto r^d.
$$

```{code-cell} ipython3
import numpy as np

from believe14.estimation import (
    DANCo,
    CorrelationDimension,
    LevinaBickelMLE,
    MiNDML,
    TwoNN,
    UStatisticDimension,
)

rng = np.random.default_rng(14)
flat = rng.uniform(-1.0, 1.0, size=(120, 2))
X = np.column_stack(
    (flat, flat[:, 0] + 0.3 * flat[:, 1], np.zeros((120, 2)))
)

models = [
    CorrelationDimension(),
    TwoNN(),
    LevinaBickelMLE(k_min=5, k_max=10),
    UStatisticDimension(max_dimension=3, random_state=14),
    MiNDML(n_neighbors=6, max_dimension=3),
    DANCo(n_neighbors=6, max_dimension=3, random_state=14),
]

for model in models:
    model.fit(X)
    assert np.isfinite(model.dimension_)
    assert model.diagnostics_.converged
    print(f"{type(model).__name__:24s} dimension={model.dimension_:.3f}")
```

The continuous estimates need not be integers; U-Statistic Dimension and DANCo
select from discrete candidates. Estimates are meaningful only inside each
method's sample-size, neighbor, tie, and support assumptions. `fit` stores
`dimension_`, optional `local_dimensions_`, and `diagnostics_`; these classes do
not invent `predict`, `score`, or `transform`.

```{admonition} DANCo boundary
:class: warning
DANCo's angular statistic is degenerate for intrinsically one-dimensional or
collinear data. Version 0.1.0 rejects that regime explicitly and searches
candidate dimensions starting at two.
```

See [Correlation Dimension](../examples/correlation_dimension.md),
[TwoNN](../examples/two_nn.md),
[Levina--Bickel MLE](../examples/levina_bickel_mle.md),
[U-Statistic Dimension](../examples/u_statistic_dimension.md),
[MiND-ML](../examples/mi_ndml.md), and [DANCo](../examples/danco.md).
