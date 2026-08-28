---
believe14_estimator: LinearDiscriminantAnalysis
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LinearDiscriminantAnalysis

Use this Fisher/Rao projection to find linear directions separating labeled
classes. It is a reducer, not a classifier, and allows at most `classes - 1`
directions.

```{code-cell} ipython3
import numpy as np
from believe14.linear import LinearDiscriminantAnalysis
from _example_data import multiclass_data

X, labels = multiclass_data()
model = LinearDiscriminantAnalysis(n_components=2, regularization=1e-8)
embedding = model.fit_transform(X, labels)
queries = model.transform(X[:3])
assert embedding.shape == (48, 2) and np.isfinite(queries).all()
{
    "shape": embedding.shape,
    "classes": model.classes_.tolist(),
    "fisher_eigenvalues": np.round(model.eigenvalues_, 3).tolist(),
    "generalized_eigen_residual": model.diagnostics_.residual_norm,
}
```

`components_`, `within_scatter_`, and `between_scatter_` expose the generalized
eigensystem. The diagnostic residual checks that system. Singular within-class
scatter can require explicit regularization; infinite Fisher directions are
rejected rather than silently regularized.

Primary reference: {cite:p}`fisher1936`.
