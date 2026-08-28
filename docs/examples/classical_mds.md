---
believe14_estimator: ClassicalMDS
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# ClassicalMDS

Use Classical MDS when Euclidean coordinates should reproduce a matrix of pairwise
dissimilarities through squared-distance double centering. This example computes
dissimilarities from features, but `dissimilarity="precomputed"` is also supported.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import ClassicalMDS
from _example_data import swiss_roll

X, _ = swiss_roll()
model = ClassicalMDS(n_components=2)
embedding = model.fit_transform(X)
assert embedding.shape == (54, 2)
assert not hasattr(model, "transform")  # training-distance geometry is transductive
{
    "shape": embedding.shape,
    "leading_eigenvalues": np.round(model.eigenvalues_[:2], 3).tolist(),
    "gram_shape": model.gram_matrix_.shape,
    "relative_residual": model.diagnostics_.residual_norm,
}
```

`gram_matrix_`, `eigenvalues_`, and `dissimilarity_matrix_` expose the full
spectral calculation. Diagnostics report retained rank and reconstruction error.
Negative Gram eigenvalues diagnose non-Euclidean dissimilarities; there is no
uncited out-of-sample interpolation or `transform` method.

Primary reference: {cite:p}`torgerson1952`.
