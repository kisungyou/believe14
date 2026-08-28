---
believe14_estimator: MiNDML
believe14_family: estimation
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# MiNDML

Use MiND-ML for a continuous maximum-likelihood dimension estimate from normalized
minimum-neighbor distances. Its locally uniform ball approximation is most
credible away from boundaries and severe density gradients.

```{code-cell} ipython3
import numpy as np
from believe14.estimation import MiNDML
from _example_data import curved_data

X = curved_data()
model = MiNDML(n_neighbors=8, max_dimension=3).fit(X)
assert 1.0 <= model.dimension_ <= 3.0
assert not hasattr(model, "transform") and not hasattr(model, "predict")
{
    "dimension": round(model.dimension_, 3),
    "normalized_distance_range": tuple(np.round([model.normalized_distances_.min(), model.normalized_distances_.max()], 3)),
    "log_likelihood": round(model.log_likelihood_, 3),
    "optimizer_residual": model.diagnostics_.residual_norm,
}
```

`normalized_distances_` and `log_likelihood_` expose the likelihood data and
optimum. Diagnostics report bounded scalar optimization success. Duplicates,
invalid normalized distances, and a boundary optimum are explicit concerns. The
continuous estimate is deliberately not silently rounded to an integer.

Primary reference: {cite:p}`lombardi2011`.
