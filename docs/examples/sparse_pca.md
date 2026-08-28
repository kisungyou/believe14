---
believe14_estimator: SparsePCA
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# SparsePCA

Use Sparse PCA when a linear representation with elastic-net-sparse loadings is
preferred to orthogonal PCA axes. The sparsity penalties depend on feature scale.

```{code-cell} ipython3
import numpy as np
from believe14.linear import SparsePCA
from _example_data import latent_data

X, _ = latent_data()
model = SparsePCA(
    n_components=2, alpha=0.005, init="random", tol=1e-4,
    max_iter=300, random_state=14,
)
embedding = model.fit_transform(X)
queries = model.transform(X[:3])
replay = SparsePCA(
    n_components=2, alpha=0.005, init="random", tol=1e-4,
    max_iter=300, random_state=14,
).fit_transform(X)
assert embedding.shape == (48, 2) and np.isfinite(queries).all()
assert np.allclose(embedding, replay)
{
    "shape": embedding.shape,
    "zero_loadings": int(np.count_nonzero(model.components_ == 0.0)),
    "objective": model.objective_,
    "converged": model.diagnostics_.converged,
    "kkt_residual": model.kkt_residual_,
}
```

`components_`, `raw_components_`, and `objective_history_` expose the elastic-net
solution. Diagnostics require both optimization and KKT tolerances. Large
`alpha` can collapse a component to zero; the class then fails explicitly rather
than returning a lower-rank result.

Primary reference: {cite:p}`zou2006`.
