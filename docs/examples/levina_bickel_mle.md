---
believe14_estimator: LevinaBickelMLE
believe14_family: estimation
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LevinaBickelMLE

Use the Levina--Bickel estimator for local Poisson-process kNN likelihoods averaged
over a declared neighbor range. The range must remain local relative to curvature
and density variation.

```{code-cell} ipython3
import numpy as np
from believe14.estimation import LevinaBickelMLE
from _example_data import sphere

X = sphere()
model = LevinaBickelMLE(k_min=6, k_max=10, bias_correction=True).fit(X)
assert np.isfinite(model.dimension_) and model.local_dimensions_.shape == (72,)
assert model.local_dimensions_by_k_.shape == (len(X), 5)
assert np.allclose(model.local_dimensions_by_k_.mean(axis=1), model.local_dimensions_)
assert not hasattr(model, "transform") and not hasattr(model, "predict")
{
    "dimension": round(model.dimension_, 3),
    "dimensions_by_k": np.round(model.dimensions_by_k_, 3).tolist(),
    "local_range": tuple(np.round([model.local_dimensions_.min(), model.local_dimensions_.max()], 3)),
    "across_k_residual": model.diagnostics_.residual_norm,
}
```

In `local_dimensions_by_k_[i, j]`, row `i` identifies an observation and
column `j` gives its estimate at neighbor count `k_values_[j]`. Averaging across
columns yields `local_dimensions_`. Averaging down rows gives `dimensions_by_k_`. Diagnostics summarize variation across neighbor orders. Zero neighbor
distances and a range beyond sample size are invalid; curvature, boundaries, and
density nonuniformity can dominate an overly large `k`.

Primary reference: {cite:p}`levina2004`.
