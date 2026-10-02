---
believe14_estimator: PLSRegression
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# PLSRegression

Use NIPALS PLS2 when a low-dimensional X representation should covary with one
or more continuous targets. Rows must be paired and constant X columns are not
admissible when scaling is enabled.

```{code-cell} ipython3
import numpy as np
from believe14.linear import PLSRegression
from _example_data import paired_data

X, Y = paired_data()
model = PLSRegression(n_components=2, scale=True)
X_scores, Y_scores = model.fit_transform(X, Y)
queries = model.transform(X[:3])
predictions = model.predict(X[:3])
assert X_scores.shape == Y_scores.shape == (52, 2)
{
    "score_shapes": (X_scores.shape, Y_scores.shape),
    "predictions": predictions.shape,
    "coefficients": model.coef_.shape,
    "converged": model.diagnostics_.converged,
}
```

`x_weights_`, `x_loadings_`, `y_loadings_`, and `coef_` record the frozen PLS2
deflation convention. Each component starts from the leading singular direction
of the residual X–Y cross-covariance, then uses the NIPALS updates and stopping
rule. This avoids selecting a weaker covariance direction merely because of
target-column order. The initialization adds a thin SVD of a matrix with one row
per feature and one column per target; include this cost when working with many
features and targets.

Use `predict` to apply the fitted regression. It restores target offsets with
compensated arithmetic, reducing loss of small predicted differences around a
large target mean. Diagnostics summarize component-wise NIPALS convergence and
relative prediction residual. Do not interpret PLS coefficients causally, and do
not accept a false convergence flag as a fitted optimum.

Primary reference: {cite:p}`wold1984`.
