---
believe14_estimator: PCA
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# PCA

Use PCA for a centered linear subspace whose variance-ordered axes are meaningful.
It assumes finite real features in rows; centering is fitted from the training set.

```{code-cell} ipython3
import numpy as np
from believe14.linear import PCA
from _example_data import latent_data

X, _ = latent_data()
model = PCA(n_components=2)
embedding = model.fit_transform(X)
queries = model.transform(X[:3])       # the fitted linear map is inductive
reconstructed = model.inverse_transform(embedding)
assert embedding.shape == (48, 2) and queries.shape == (3, 2)
{
    "shape": embedding.shape,
    "explained_fraction": round(float(model.explained_variance_ratio_.sum()), 3),
    "components": model.components_.shape,
    "relative_residual": model.diagnostics_.residual_norm,
}
```

`components_`, `singular_values_`, and `explained_variance_` describe the fitted
subspace. A small diagnostic residual checks reconstruction by that subspace; it
is not a guarantee that two components explain all variation. PCA rejects a
requested dimension above the centered rank and is sensitive to feature scale.

Primary reference: {cite:p}`pearson1901`.
