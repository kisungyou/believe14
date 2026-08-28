---
believe14_estimator: SlicedInverseRegression
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# SlicedInverseRegression

Use SIR for sufficient dimension reduction when response-slice means carry the
signal and the predictor distribution is compatible with its linearity condition.

```{code-cell} ipython3
import numpy as np
from believe14.linear import SlicedInverseRegression
from _example_data import single_index_data

X, response = single_index_data()
model = SlicedInverseRegression(n_components=2, n_slices=6)
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

`components_`, `eigenvalues_`, and tie-preserving `slice_labels_` define the
estimated central subspace. Diagnostics check the standardized eigensystem.
SIR can miss symmetric dependencies such as a purely even response and requires
enough distinct responses to form nondegenerate slices.

Primary reference: {cite:p}`li1991`.
