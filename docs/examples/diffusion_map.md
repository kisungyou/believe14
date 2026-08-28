---
believe14_estimator: DiffusionMap
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# DiffusionMap

Use Diffusion Map for multiscale coordinates from a density-normalized Markov
operator. RBF scale, density exponent, and diffusion time define the geometry.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import DiffusionMap
from _example_data import swiss_roll

X, _ = swiss_roll()
model = DiffusionMap(n_components=2, gamma=0.2, alpha=1.0, diffusion_time=1)
embedding = model.fit_transform(X)
queries = model.transform(X[:3])  # Nyström extension of diffusion eigenfunctions
assert embedding.shape == (54, 2) and np.allclose(queries, embedding[:3])
{
    "shape": embedding.shape,
    "diffusion_eigenvalues": np.round(model.eigenvalues_, 4).tolist(),
    "operator": model.diffusion_operator_.shape,
    "spectral_residual": model.diagnostics_.residual_norm,
}
```

`diffusion_operator_`, `eigenvalues_`, and `kernel_density_` expose normalization.
Diagnostics check the symmetric-conjugate eigensystem. Numerically disconnected
kernel support and zero-density queries fail explicitly; extrapolation far from
training support can be unstable despite the justified Nyström rule.

Primary reference: {cite:p}`coifman2006`.
