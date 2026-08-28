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
assert not hasattr(model, "transform")  # t-SNE optimizes only training coordinates
{
    "shape": embedding.shape,
    "kl_divergence": round(model.kl_divergence_, 4),
    "iterations": model.n_iter_,
    "converged": model.diagnostics_.converged,
}
```

`joint_probabilities_`, `perplexity_entropy_residuals_`, and `kl_divergence_`
make the exact objective auditable. Diagnostics report L-BFGS convergence and
gradient norm. Different seeds can produce different valid layouts, duplicates
can make perplexity unattainable, and the method has no `transform`.

Primary reference: {cite:p}`vandermaaten2008`.
