---
believe14_estimator: LocalTangentSpaceAlignment
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LocalTangentSpaceAlignment

Use LTSA when local tangent-plane coordinates can be aligned into a global
manifold chart. Each exact neighborhood must contain enough independent local
variation for the requested dimension.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import LocalTangentSpaceAlignment
from _example_data import swiss_roll

X, _ = swiss_roll()
model = LocalTangentSpaceAlignment(n_components=2, n_neighbors=8)
embedding = model.fit_transform(X)
assert embedding.shape == (54, 2)
assert not hasattr(model, "transform")  # alignment was solved only for fitted vertices
{
    "shape": embedding.shape,
    "alignment_matrix": model.alignment_matrix_.shape,
    "eigenvalues": np.round(model.eigenvalues_, 6).tolist(),
    "alignment_residual": model.diagnostics_.residual_norm,
}
```

`alignment_matrix_`, `eigenvalues_`, and `embedding_` expose the global solve.
Diagnostics check its eigensystem. Rank-deficient tangent neighborhoods and a
disconnected graph are explicit failures. There is no cited LTSA extension in
0.1.0, so `transform` is intentionally absent.

Primary reference: {cite:p}`zhang2004`.
