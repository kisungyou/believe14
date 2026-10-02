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

This website includes the unreleased changes on `main`. To use those changes,
install the current development version from the repository:

```console
python -m pip install "believe14 @ git+https://github.com/kisungyou/believe14.git@main"
```

See the [changelog](../changelog.md) for the distinction between unreleased
corrections and published versions.

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

### Compare with the ground truth

The left panel shows the known two-dimensional coordinates in `latent`; the
right shows the PCA embedding computed from the four observed features in `X`.
Matching colors and numbers identify the same eight observations. Both panels
use equal aspect ratios and the same axis limits.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    alt: >-
      Two scatter plots showing the eight ground-truth latent coordinates on the
      left and their PCA embedding on the right, with matching colors and
      observation numbers.
    width: "100%"
---
%matplotlib inline
import matplotlib.pyplot as plt

colors = plt.get_cmap("tab10")(np.arange(len(latent)))
coordinates = np.vstack([latent, embedding])
lower = coordinates.min(axis=0) - 0.6
upper = coordinates.max(axis=0) + 0.6
variance = 100 * pca.explained_variance_ratio_

fig, axes = plt.subplots(
    1, 2, figsize=(9.6, 4.3), dpi=150, sharex=True, sharey=True,
    layout="constrained",
)
panels = [
    (
        latent, "Ground truth: latent coordinates",
        "Latent coordinate 1", "Latent coordinate 2",
    ),
    (
        embedding, "PCA embedding",
        f"PC 1 ({variance[0]:.1f}% variance)",
        f"PC 2 ({variance[1]:.1f}% variance)",
    ),
]
for ax, (points, title, xlabel, ylabel) in zip(axes, panels):
    ax.scatter(
        points[:, 0], points[:, 1], c=colors, s=85,
        edgecolors="white", linewidths=0.8, zorder=3,
    )
    for number, point in enumerate(points, start=1):
        ax.annotate(
            str(number), point, xytext=(7, 7),
            textcoords="offset points", fontsize=10,
        )
    ax.set(
        title=title, xlabel=xlabel, ylabel=ylabel,
        xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
plt.show()
```

The mixing matrix changes lengths and angles in the latent coordinates. PCA
then centers the observed data and chooses axes of greatest variance, so its
coordinates need not match `latent`. Both components together retain all
variation in this rank-two dataset: reconstructing the original observations
recovers `X` up to floating-point error.

```{code-cell} ipython3
assert np.allclose(pca.inverse_transform(embedding), X, rtol=1e-12, atol=1e-12)
```

### Transform new observations

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
