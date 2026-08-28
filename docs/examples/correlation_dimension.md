---
believe14_estimator: CorrelationDimension
believe14_family: estimation
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# CorrelationDimension

Use correlation dimension when a scale interval exhibits an approximately linear
log correlation-integral curve. The estimator describes sampled geometry over the
chosen radii, not automatically a global manifold dimension.

```{code-cell} ipython3
import numpy as np
from believe14.estimation import CorrelationDimension
from _example_data import sphere

X = sphere()
model = CorrelationDimension(n_radii=12, quantile_range=(0.08, 0.35)).fit(X)
assert np.isfinite(model.dimension_)
assert not hasattr(model, "transform") and not hasattr(model, "predict")
{
    "dimension": round(model.dimension_, 3),
    "radii": np.round(model.radii_[[0, -1]], 3).tolist(),
    "correlation_integral": np.round(model.correlation_integral_[[0, -1]], 3).tolist(),
    "log_fit_residual": model.diagnostics_.residual_norm,
}
```

`radii_`, `correlation_integral_`, and `intercept_` expose the fitted scaling
line. Diagnostics report log-OLS residual and rank. Duplicate-heavy data, empty
counts, or no admissible scaling interval fail explicitly; inspect the curve
rather than treating its slope as scale-free truth.

Primary reference: {cite:p}`grassberger1983`.
