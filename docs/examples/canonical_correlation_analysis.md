---
believe14_estimator: CanonicalCorrelationAnalysis
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# CanonicalCorrelationAnalysis

Use Hotelling CCA for two paired feature views measured on the same rows. It
assumes pairing is exact and estimates only directions supported by both centered
view ranks.

```{code-cell} ipython3
import numpy as np
from believe14.linear import CanonicalCorrelationAnalysis
from _example_data import paired_data

X, Y = paired_data()
model = CanonicalCorrelationAnalysis(n_components=2)
X_scores, Y_scores = model.fit_transform(X, Y)
X_queries, Y_queries = model.transform(X[:3], Y[:3])
assert X_scores.shape == Y_scores.shape == (52, 2)
{
    "shapes": (X_scores.shape, Y_scores.shape),
    "correlations": np.round(model.canonical_correlations_, 3).tolist(),
    "x_weights": model.x_weights_.shape,
    "whitening_residual": model.diagnostics_.residual_norm,
}
```

`x_weights_`, `y_weights_`, and `canonical_correlations_` describe the paired
solution. Diagnostics check score whitening and view conditioning. CCA fails when
the requested component count exceeds shared covariance support; unpaired rows
have no defined interpretation.

Primary reference: {cite:p}`hotelling1936`.
