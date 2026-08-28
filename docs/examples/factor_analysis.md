---
believe14_estimator: FactorAnalysis
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# FactorAnalysis

Use factor analysis for a Gaussian latent model with feature-specific noise.
Observations must support the requested latent rank and the model assumes diagonal
noise covariance.

```{code-cell} ipython3
import numpy as np
from believe14.linear import FactorAnalysis
from _example_data import latent_data

X, _ = latent_data()
model = FactorAnalysis(n_components=2, tol=1e-4, max_iter=500)
embedding = model.fit_transform(X)
queries = model.transform(X[:3])
reconstruction = model.inverse_transform(embedding)
assert embedding.shape == (48, 2) and np.isfinite(reconstruction).all()
{
    "shape": embedding.shape,
    "loadings": model.loadings_.shape,
    "noise_range": tuple(np.round([model.noise_variance_.min(), model.noise_variance_.max()], 5)),
    "converged": model.diagnostics_.converged,
    "iterations": model.diagnostics_.n_iter,
}
```

`loadings_`, `noise_variance_`, and `posterior_covariance_` define the fitted
Gaussian model. The diagnostic convergence flag records the Rubin--Thayer EM
stopping test; a false flag must not be treated as convergence. Ill-conditioned,
rank-deficient, or nearly zero-noise samples may be inadmissible.

Primary reference: {cite:p}`rubin1982`.
