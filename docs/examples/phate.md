---
believe14_estimator: PHATE
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# PHATE

Use PHATE for a small branching or trajectory dataset where diffusion-potential
distances should emphasize progression. Neighbor count, decay, diffusion time,
and the explicit probability floor define the result.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import PHATE
from _example_data import branching

X, branches = branching()
model = PHATE(
    n_components=2, n_neighbors=6, decay=5, diffusion_time=5,
    mds_max_iter=150, mds_tol=1e-5,
)
embedding = model.fit_transform(X)
assert embedding.shape == (42, 2)
assert not hasattr(model, "transform")  # no PHATE extension is claimed in 0.1.0
{
    "shape": embedding.shape,
    "potential_distances": model.potential_distances_.shape,
    "stress": round(model.stress_, 4),
    "converged": model.diagnostics_.converged,
    "warnings": model.diagnostics_.warnings,
}
```

`affinity_matrix_`, `diffusion_potential_`, `potential_distances_`, and `stress_`
expose each composite stage. Diagnostics report the final metric-MDS stopping rule
and any explicit floor regularization. Zero adaptive bandwidth or disconnected
numerical affinity fails, and `transform` is intentionally absent.

Primary reference: {cite:p}`moon2019`.
