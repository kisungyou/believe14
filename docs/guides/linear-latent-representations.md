---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Linear latent representations

The six unsupervised linear reducers answer different questions. PCA preserves
sample variance, Gaussian projection offers a data-independent randomized map,
FA and PPCA specify Gaussian latent-variable models, FastICA searches for
independent sources, and Sparse PCA trades reconstruction against sparse
loadings {cite:p}`pearson1901,bingham2001,rubin1982,tipping1999,hyvarinen1999,zou2006`.

For PCA, the fitted coordinates have the centered linear form

$$
Z = (X - \mu) V_k^{\mathsf{T}}.
$$

We use one deterministic two-factor dataset for a compact comparison.

```{code-cell} ipython3
import numpy as np

from believe14.linear import (
    PCA,
    FactorAnalysis,
    FastICA,
    GaussianRandomProjection,
    ProbabilisticPCA,
    SparsePCA,
)

rng = np.random.default_rng(14)
latent = rng.normal(size=(100, 2))
mixing = np.array(
    [[1.2, -0.4, 0.9, 0.0, 0.2, 0.4], [0.0, 0.7, 0.2, 1.1, -0.5, 0.3]]
)
X = latent @ mixing + 0.08 * rng.normal(size=(100, 6))

models = [
    PCA(2),
    GaussianRandomProjection(2, random_state=14),
    FactorAnalysis(2, max_iter=1000),
    ProbabilisticPCA(2),
    FastICA(2, random_state=14, max_iter=1000),
    SparsePCA(2, alpha=0.02, max_iter=1000),
]

for model in models:
    scores = model.fit_transform(X)
    assert scores.shape == (100, 2)
    assert np.isfinite(scores).all()
    print(
        f"{type(model).__name__:28s}",
        f"solver={model.diagnostics_.solver:24s}",
        f"converged={model.diagnostics_.converged}",
    )
```

All six expose `transform` because their fitted model defines a global linear
map. PCA, FA, PPCA, and FastICA additionally expose a mathematically defined
`inverse_transform`; projection and sparse-component models do not promise an
exact inverse. Components are identifiable only up to the equivalence stated
in each method's validation ledger.

Use the cards for [PCA](../examples/pca.md),
[Gaussian projection](../examples/gaussian_random_projection.md),
[Factor Analysis](../examples/factor_analysis.md),
[PPCA](../examples/probabilistic_pca.md),
[FastICA](../examples/fast_ica.md), and
[Sparse PCA](../examples/sparse_pca.md) to inspect fitted attributes and
failure boundaries.
