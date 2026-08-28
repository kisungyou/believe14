---
believe14_estimator: MetricMDS
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# MetricMDS

Use Metric MDS to minimize unweighted raw stress between input dissimilarities and
embedding distances. Dissimilarities must be finite, symmetric, nonnegative, and
zero on the diagonal.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import MetricMDS
from _example_data import swiss_roll

X, _ = swiss_roll()
model = MetricMDS(
    n_components=2, init="random", max_iter=300, tol=1e-5, random_state=14
)
embedding = model.fit_transform(X)
replay = MetricMDS(
    n_components=2, init="random", max_iter=300, tol=1e-5, random_state=14
).fit_transform(X)
assert embedding.shape == (54, 2) and model.diagnostics_.converged
assert np.allclose(embedding, replay)
assert not hasattr(model, "transform")  # SMACOF locates only fitted objects
{
    "shape": embedding.shape,
    "raw_stress": round(model.stress_, 3),
    "iterations": model.n_iter_,
    "normalized_step": model.diagnostics_.residual_norm,
}
```

`embedding_`, `stress_`, and `n_iter_` describe the SMACOF solution. The
diagnostic convergence flag means its normalized stopping rule was met; stress is
not comparable across arbitrary distance scales. Local minima depend on
initialization, and no out-of-sample `transform` is defined.

Primary reference: {cite:p}`deleeeuw1977`.
