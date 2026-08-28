---
believe14_estimator: SlicedAverageVarianceEstimation
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# SlicedAverageVarianceEstimation

Use SAVE when response slices may change conditional covariance as well as their
means. It assumes sufficient samples per tie-preserving slice to estimate local
covariance.

```{code-cell} ipython3
import numpy as np
from believe14.linear import SlicedAverageVarianceEstimation
from _example_data import single_index_data

X, response = single_index_data()
model = SlicedAverageVarianceEstimation(n_components=2, n_slices=6)
embedding = model.fit_transform(X, response)
queries = model.transform(X[:3])
assert embedding.shape == (60, 2) and np.isfinite(queries).all()
{
    "shape": embedding.shape,
    "realized_slices": model.n_slices_,
    "eigenvalues": np.round(model.eigenvalues_, 4).tolist(),
    "spectral_residual": model.diagnostics_.residual_norm,
}
```

`components_`, `eigenvalues_`, and `slice_labels_` expose the SAVE eigensystem.
The diagnostic residual measures its normalized equation error. Very small slices
and singular global predictor covariance are explicit failure regimes; results can
depend materially on `n_slices`.

Primary reference: {cite:p}`cook2000`.
