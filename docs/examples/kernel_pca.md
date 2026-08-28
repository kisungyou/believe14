---
believe14_estimator: KernelPCA
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# KernelPCA

Use Kernel PCA for nonlinear spectral coordinates induced by a positive-semidefinite
kernel. Kernel scale, especially RBF `gamma`, is part of the model rather than an
innocent plotting choice.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import KernelPCA
from _example_data import circles

X, rings = circles()
model = KernelPCA(n_components=2, kernel="rbf", gamma=0.7)
embedding = model.fit_transform(X)
queries = model.transform(X[:3])  # centered-kernel Nyström projection
assert embedding.shape == (48, 2) and np.allclose(queries, embedding[:3])
{
    "shape": embedding.shape,
    "eigenvalues": np.round(model.eigenvalues_, 3).tolist(),
    "dual_coefficients": model.dual_coef_.shape,
    "spectral_residual": model.diagnostics_.residual_norm,
}
```

`eigenvalues_`, `eigenvectors_`, and `dual_coef_` expose the centered-kernel
eigensystem. Diagnostics check its residual and positive rank. Significantly
indefinite precomputed kernels and requests above positive numerical rank fail
explicitly; inverse transformation is not defined.

Primary reference: {cite:p}`scholkopf1998`.
