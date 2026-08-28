---
believe14_estimator: ProbabilisticPCA
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# ProbabilisticPCA

Use PPCA when a linear Gaussian latent model with one isotropic noise variance is
appropriate. It requires at least one discarded covariance direction to estimate
that variance.

```{code-cell} ipython3
import numpy as np
from believe14.linear import ProbabilisticPCA
from _example_data import latent_data

X, _ = latent_data()
model = ProbabilisticPCA(n_components=2)
embedding = model.fit_transform(X)
queries = model.transform(X[:3])
reconstruction = model.inverse_transform(embedding)
assert embedding.shape == (48, 2) and np.isfinite(queries).all()
{
    "shape": embedding.shape,
    "loadings": model.loadings_.shape,
    "noise_variance": round(model.noise_variance_, 6),
    "negative_log_likelihood": model.diagnostics_.objective_value,
}
```

`loadings_`, `noise_variance_`, and `log_likelihood_` are the closed-form ML
solution. The diagnostic objective is negative log likelihood and its residual
compares fitted and empirical covariance. PPCA rejects a singular zero-noise
boundary or a latent dimension not separated from the noise eigenspace.

Primary reference: {cite:p}`tipping1999`.
