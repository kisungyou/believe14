---
believe14_estimator: LaplacianEigenmaps
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LaplacianEigenmaps

Use Laplacian Eigenmaps for coordinates that vary smoothly over an exact neighbor
graph. Heat-kernel scale and neighbor count jointly determine graph geometry.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import LaplacianEigenmaps
from _example_data import swiss_roll

X, _ = swiss_roll()
model = LaplacianEigenmaps(n_components=2, n_neighbors=8, weighting="heat", gamma=0.2)
embedding = model.fit_transform(X)
assert embedding.shape == (54, 2)
degree_weights = model.degree_ / model.degree_.sum()
assert np.allclose(degree_weights @ embedding, 0.0, atol=1e-12)
assert np.allclose(embedding.T @ (model.degree_[:, None] * embedding), np.eye(2))
assert not hasattr(model, "transform")  # eigenvectors exist only on fitted graph vertices
{
    "shape": embedding.shape,
    "affinity": model.affinity_matrix_.shape,
    "eigenvalues": np.round(model.eigenvalues_, 5).tolist(),
    "generalized_eigen_residual": model.diagnostics_.residual_norm,
}
```

`affinity_matrix_`, `degree_`, and `eigenvalues_` expose the generalized
eigensystem. Diagnostics measure its residual after removing the constant mode.
Coordinates have zero degree-weighted mean and unit degree-weighted norm. The
constant mode is excluded explicitly before solving, including when weak graph
connections make multiple eigenvalues numerically indistinguishable from zero.
A disconnected graph is inadmissible, and there is no paper-defined vertex
extension here; the estimator therefore has no `transform`.

Primary reference: {cite:p}`belkin2003`.
