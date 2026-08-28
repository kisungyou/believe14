---
believe14_estimator: LocallyLinearEmbedding
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LocallyLinearEmbedding

Use LLE when local barycentric reconstruction weights are expected to transfer to
a low-dimensional manifold. Neighborhoods must be connected and locally
nondegenerate.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import LocallyLinearEmbedding
from _example_data import swiss_roll

X, _ = swiss_roll()
model = LocallyLinearEmbedding(n_components=2, n_neighbors=8, regularization=1e-3)
embedding = model.fit_transform(X)
assert embedding.shape == (54, 2)
assert not hasattr(model, "transform")  # only training reconstruction weights were derived
{
    "shape": embedding.shape,
    "weights": model.reconstruction_weights_.shape,
    "reconstruction_error": model.reconstruction_error_,
    "spectral_residual": model.diagnostics_.residual_norm,
}
```

`reconstruction_weights_`, `reconstruction_error_`, and `eigenvalues_` expose the
local and global stages. Diagnostics check the alignment eigensystem. Degenerate
neighborhoods and disconnected graphs fail explicitly; no nearest-neighbor
interpolation is invented, so `transform` is absent.

Primary reference: {cite:p}`roweis2000`.
