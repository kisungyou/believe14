---
believe14_estimator: NeighborhoodComponentsAnalysis
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# NeighborhoodComponentsAnalysis

Use labeled training observations to learn a linear map for nearest-neighbor classification. This example holds out every fourth row and evaluates how often its closest training neighbor shares its label.

```{code-cell} ipython3
import numpy as np
from scipy.spatial.distance import cdist
from believe14.linear import NeighborhoodComponentsAnalysis
from _example_data import multiclass_data

X, labels = multiclass_data()
held_out = np.arange(len(X)) % 4 == 0
model = NeighborhoodComponentsAnalysis(2).fit(X[~held_out], labels[~held_out])
train, test = model.transform(X[~held_out]), model.transform(X[held_out])
closest = cdist(test, train).argmin(axis=1)
accuracy = np.mean(labels[~held_out][closest] == labels[held_out])
assert test.shape == (12, 2)
{
    "held_out_accuracy": float(accuracy),
    "training_expected_accuracy": round(model.expected_accuracy_, 4),
    "solver": model.diagnostics_.solver,
}
```

`expected_accuracy_` averages same-class neighbor probabilities across the training observations, excluding each observation from its own neighbors. The held-out accuracy above uses the single closest training observation. The optimization objective is nonconvex, so its value can depend on `init`. Inspect `diagnostics_.converged` and the gradient residual when comparing fits.

Primary reference: {cite:p}`goldberger2004nca`.
