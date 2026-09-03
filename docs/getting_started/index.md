---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
html_theme.sidebar_secondary.remove: true
---

# Getting started

`believe14` provides paper-audited dimensionality-reduction and
intrinsic-dimension estimators with scikit-learn-style interfaces. Version
0.1.0 requires Python 3.12 or newer and operates on finite, dense, real
`float64` arrays with observations in rows.

## Installation

Install the published package from PyPI:

```console
python -m pip install believe14
```

## A complete PCA workflow

This offline example constructs a small rank-two dataset, fits a centered-SVD
PCA model, and applies its justified linear out-of-sample transform
{cite:p}`pearson1901`.

```{code-cell} ipython3
import numpy as np

from believe14.linear import PCA

latent = np.array(
    [
        [-2.0, -1.0],
        [-1.5, 0.5],
        [-0.5, -1.5],
        [0.0, 1.0],
        [0.5, -0.5],
        [1.0, 1.5],
        [1.5, -1.0],
        [2.0, 1.0],
    ],
    dtype=np.float64,
)
mixing = np.array(
    [[1.0, 0.4, -0.8, 0.3], [0.2, 1.1, 0.5, -0.7]],
    dtype=np.float64,
)
offset = np.array([3.0, -2.0, 0.5, 1.0], dtype=np.float64)
X = latent @ mixing + offset

pca = PCA(n_components=2)
embedding = pca.fit_transform(X)

assert embedding.shape == (8, 2)
assert pca.components_.shape == (2, 4)
assert np.all(np.isfinite(embedding))
(embedding.shape, np.round(pca.explained_variance_ratio_, 3))
```

`PCA` is inductive: the fitted centering and component matrix define a
paper-justified map for new observations with the same four input features.

```{code-cell} ipython3
new_latent = np.array([[0.25, 0.75], [-0.75, 0.25]], dtype=np.float64)
new_X = new_latent @ mixing + offset
new_embedding = pca.transform(new_X)

assert new_embedding.shape == (2, 2)
assert np.all(np.isfinite(new_embedding))
np.round(new_embedding, 3)
```

## Reading fit diagnostics

Every successful estimator fit stores an immutable `diagnostics_` record. For
this direct factorization, `solver` names the centered SVD, `converged` is true
without an iteration count, `residual_norm` measures the relative discarded
reconstruction, and `numerical_rank` and `condition_estimate` describe the
centered sample matrix.

```{code-cell} ipython3
diagnostics = pca.diagnostics_
assert diagnostics.converged
assert diagnostics.numerical_rank == 2
assert diagnostics.residual_norm is not None
assert np.isfinite(diagnostics.residual_norm)
{
    "solver": diagnostics.solver,
    "converged": diagnostics.converged,
    "n_iter": diagnostics.n_iter,
    "residual_norm": round(diagnostics.residual_norm, 12),
    "numerical_rank": diagnostics.numerical_rank,
}
```

## Inductive and transductive methods

Only estimators with a mathematical out-of-sample rule expose `transform`.
Many nonlinear embeddings are transductive: their coordinates are defined only
for the fitted observations, so they expose `fit` and `fit_transform` but no
placeholder interpolation. Dimension estimators instead report `dimension_`.
Use {doc}`Choosing a method <../guides/choosing-a-method>` to filter these
capabilities before selecting an estimator.

The {doc}`scientific contract <../scientific-contract>` defines the source
hierarchy, numerical failure policy, random-state behavior, and equivalence
rules applied throughout the library.

```{toctree}
:hidden:
:maxdepth: 1

Scientific contract <../scientific-contract>
```
