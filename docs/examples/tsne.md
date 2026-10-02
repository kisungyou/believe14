---
believe14_estimator: TSNE
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# TSNE

Use exact dense t-SNE to visualize local probability neighborhoods in a small
dataset. Global distances and cluster sizes are not preserved; `perplexity` must
be attainable for every row.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import TSNE
from _example_data import digits_subset

X, labels = digits_subset()
model = TSNE(
    n_components=2, perplexity=8, early_exaggeration_iter=50,
    max_iter=300, tol=1e-5, random_state=14,
)
embedding = model.fit_transform(X)
replay = TSNE(
    n_components=2, perplexity=8, early_exaggeration_iter=50,
    max_iter=300, tol=1e-5, random_state=14,
).fit_transform(X)
assert embedding.shape == (40, 2) and np.allclose(embedding, replay)
entropy_error = float(np.max(np.abs(model.perplexity_entropy_residuals_)))
assert entropy_error <= 1e-8
assert not hasattr(model, "transform")  # t-SNE optimizes only training coordinates
{
    "shape": embedding.shape,
    "kl_divergence": round(model.kl_divergence_, 4),
    "maximum_entropy_error": entropy_error,
    "iterations": model.n_iter_,
    "converged": model.diagnostics_.converged,
}
```

`joint_probabilities_`, `perplexity_entropy_residuals_`, and `kl_divergence_`
make the exact objective auditable. Each conditional entropy matches
`log(perplexity)` within `1e-8`. Bandwidth search brackets and solves in log
precision so widely separated distance scales can be handled without squaring
large distances. This search is independent of `max_iter`, which controls the
embedding optimizer. Tied nearest distances, including duplicates, can still
make a requested perplexity unattainable; such requests fail explicitly.

Diagnostics report L-BFGS convergence and gradient norm. Different seeds can
produce different valid layouts, and the method has no `transform`.

Primary reference: {cite:p}`vandermaaten2008`.
