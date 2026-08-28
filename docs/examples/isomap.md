---
believe14_estimator: Isomap
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# Isomap

Use Isomap when geodesic distance on a connected sampled manifold should unfold
into Euclidean coordinates. Exact neighbor count controls the geometry and must
connect the graph without creating excessive shortcuts.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import Isomap
from _example_data import swiss_roll

X, roll_coordinate = swiss_roll()
model = Isomap(n_components=2, n_neighbors=8)
embedding = model.fit_transform(X)
assert embedding.shape == (54, 2)
assert not hasattr(model, "transform")  # no interpolation is added to paper Isomap
{
    "shape": embedding.shape,
    "geodesic_matrix": model.geodesic_distances_.shape,
    "leading_eigenvalues": np.round(model.eigenvalues_[:2], 3).tolist(),
    "diagnostic_warnings": model.diagnostics_.warnings,
}
```

`geodesic_distances_`, `eigenvalues_`, and `embedding_` expose both algorithm
stages. Diagnostics report the classical-scaling residual and warn about
non-Euclidean geodesics. Duplicate observations or a disconnected exact graph are
rejected; this transductive class intentionally has no `transform`.

Primary reference: {cite:p}`tenenbaum2000`.
