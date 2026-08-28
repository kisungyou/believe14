---
believe14_estimator: FastICA
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# FastICA

Use FastICA to separate non-Gaussian independent sources from a linear mixture.
The centered data must have sufficient rank; component order and signs are not
scientifically identifiable.

```{code-cell} ipython3
import numpy as np
from believe14.linear import FastICA
from _example_data import source_data

X, _ = source_data()
model = FastICA(n_components=3, random_state=14, max_iter=500)
sources = model.fit_transform(X)
queries = model.transform(X[:4])
reconstruction = model.inverse_transform(sources)
replay = FastICA(n_components=3, random_state=14, max_iter=500).fit_transform(X)
assert sources.shape == (72, 3) and np.allclose(queries, sources[:4])
assert np.allclose(sources, replay)
{
    "shape": sources.shape,
    "unmixing": model.components_.shape,
    "mixing": model.mixing_.shape,
    "converged": model.diagnostics_.converged,
    "fixed_point_residual": model.diagnostics_.residual_norm,
}
```

`components_`, `mixing_`, and `sources_` store the symmetric fixed-point solution.
The residual measures change modulo component signs. Gaussian sources cannot be
uniquely separated, and a false diagnostic convergence flag requires changing the
model or iteration budget rather than silently accepting the result.

Primary reference: {cite:p}`hyvarinen1999`.
