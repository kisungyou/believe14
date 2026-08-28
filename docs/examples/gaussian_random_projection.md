---
believe14_estimator: GaussianRandomProjection
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# GaussianRandomProjection

Use this fast, data-independent map when approximate distance preservation matters
more than interpretable axes. Features must be finite and use compatible scales.

```{code-cell} ipython3
import numpy as np
from believe14.linear import GaussianRandomProjection
from _example_data import latent_data

X, _ = latent_data()
model = GaussianRandomProjection(n_components=3, random_state=14)
embedding = model.fit_transform(X)
queries = model.transform(X[:4])
replay = GaussianRandomProjection(n_components=3, random_state=14).fit_transform(X)
assert embedding.shape == (48, 3) and np.allclose(queries, embedding[:4])
assert np.allclose(embedding, replay)
{
    "shape": embedding.shape,
    "projection_matrix": model.components_.shape,
    "solver": model.diagnostics_.solver,
    "rank": model.diagnostics_.numerical_rank,
}
```

`components_` is the sampled Gaussian matrix scaled by
`1 / sqrt(n_components)`. Diagnostics report its numerical rank and condition;
there is no iterative convergence claim. Very aggressive compression can distort
individual distances, and results require an explicit seed for replay.

Primary reference: {cite:p}`bingham2001`.
