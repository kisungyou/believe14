---
believe14_estimator: TwoNN
believe14_family: estimation
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# TwoNN

Use TwoNN for a global intrinsic-dimension estimate from first/second neighbor
distance ratios under the locally uniform Poisson approximation. It needs distinct
observations and enough ratio variation.

```{code-cell} ipython3
import numpy as np
from believe14.estimation import TwoNN
from _example_data import flat_data

X = flat_data(intrinsic_dimension=3)
model = TwoNN(discard_fraction=0.1).fit(X)
assert np.isfinite(model.dimension_)
assert not hasattr(model, "transform") and not hasattr(model, "predict")
{
    "dimension": round(model.dimension_, 3),
    "ratios_used": model.ratios_.size,
    "fit_coordinates": (model.fit_x_.shape, model.fit_y_.shape),
    "origin_ols_residual": model.diagnostics_.residual_norm,
}
```

`ratios_`, `fit_x_`, and `fit_y_` expose the origin-constrained linear fit.
Diagnostics report its residual. Ties, duplicates, boundary effects, strong
density variation, and the subjective discard fraction can bias the result; this
dimension estimator intentionally implements only `fit`.

Primary reference: {cite:p}`facco2017`.
