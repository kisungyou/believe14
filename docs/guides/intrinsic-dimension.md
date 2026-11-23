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
Levina--Bickel MLE, and MiND-ML use nearest-neighbor laws. Tight Local Estimation
uses distances from moving centers in each neighborhood. U-Statistic Dimension
uses convergence rates and DANCo combines norm and angle concentration
{cite:p}`grassberger1983,facco2017,levina2004,hein2005,lombardi2011,ceruti2014,amsaleg2022`.

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
    TightLocalEstimator,
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
    TightLocalEstimator(n_neighbors=12),
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
collinear data. DANCo rejects that regime explicitly and searches
candidate dimensions starting at two.
```

## Dimension across observations and scales

Rows of `local_dimensions_by_k_` correspond to the fitted observations. Columns
follow `k_values_`. Levina--Bickel reports the entire requested range and Tight
Local Estimation reports one column for its chosen neighborhood size. On a
mixture of manifolds, compare observations within each region. A global mean
can fall between the dimensions of the two regions. A neighborhood crossing
their boundary can mix their geometries.

```{code-cell} ipython3
from believe14.neighbors import build_neighbor_graph

neighbors = build_neighbor_graph(X, n_neighbors=15)
local = LevinaBickelMLE(k_min=5, k_max=15).fit(X, neighbors=neighbors)
assert local.local_dimensions_by_k_.shape == (len(X), 11)
print(local.k_values_)
print(np.median(local.local_dimensions_by_k_, axis=0).round(2))
```

TwoNN needs two neighbors in the graph. Levina--Bickel needs `k_max`, Tight
Local Estimation needs `n_neighbors`, and MiND-ML and DANCo need
`n_neighbors + 1`. DANCo computes separate neighborhoods for its simulated
calibration samples.

See [Tight Local Estimation](../examples/tight_local_estimator.md),
[Correlation Dimension](../examples/correlation_dimension.md),
[TwoNN](../examples/two_nn.md),
[Levina--Bickel MLE](../examples/levina_bickel_mle.md),
[U-Statistic Dimension](../examples/u_statistic_dimension.md),
[MiND-ML](../examples/mi_ndml.md), and [DANCo](../examples/danco.md).
