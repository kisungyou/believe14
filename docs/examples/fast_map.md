---
believe14_estimator: FastMap
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# FastMap

Use FastMap for a deterministic pivot approximation to metric coordinates. Input
dissimilarities must obey a sufficiently Euclidean residual geometry; feature
input permits a cited pivot-distance extension for new rows.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import FastMap
from _example_data import swiss_roll

X, _ = swiss_roll()
model = FastMap(n_components=2, pivot_iterations=5)
embedding = model.fit_transform(X)
queries = model.transform(X[:3])  # distances to stored pivots define this extension
assert embedding.shape == (54, 2) and np.allclose(queries, embedding[:3])
{
    "shape": embedding.shape,
    "pivots": model.pivot_indices_.tolist(),
    "pivot_distances": np.round(model.pivot_distances_, 3).tolist(),
    "active_rank": model.n_active_components_,
}
```

`pivot_indices_`, `pivot_distances_`, and `n_active_components_` reveal the whole
construction. Diagnostics warn when residual metric rank is exhausted. Pivot
choices are heuristic, and significantly negative residual squared distances are
rejected instead of repaired.

Primary reference: {cite:p}`faloutsos1995`.
