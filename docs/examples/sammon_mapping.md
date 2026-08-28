---
believe14_estimator: SammonMapping
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# SammonMapping

Use Sammon Mapping when preserving small positive pairwise distances is especially
important. Distinct observations and strictly positive off-diagonal
dissimilarities are required.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import SammonMapping
from _example_data import swiss_roll

X, _ = swiss_roll()
model = SammonMapping(
    n_components=2, init="random", max_iter=150, tol=1e-6, random_state=14
)
embedding = model.fit_transform(X)
replay = SammonMapping(
    n_components=2, init="random", max_iter=150, tol=1e-6, random_state=14
).fit_transform(X)
assert embedding.shape == (54, 2)
assert np.allclose(embedding, replay)
assert not hasattr(model, "transform")  # the optimized configuration is transductive
{
    "shape": embedding.shape,
    "normalized_sammon_stress": round(model.stress_, 5),
    "iterations": model.n_iter_,
    "gradient_norm": model.diagnostics_.residual_norm,
    "converged": model.diagnostics_.converged,
}
```

`stress_`, `dissimilarity_matrix_`, and `n_iter_` expose the nonlinear optimum.
Diagnostics report optimizer success and the analytic-gradient norm. Duplicate
points are inadmissible, local minima remain possible, and the fitted training
configuration has no mathematically specified `transform`.

Primary reference: {cite:p}`sammon1969`.
