---
believe14_estimator: FisherScore
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# FisherScore

Use Fisher Score to select individual features with high sample-count-weighted
between-class variation relative to within-class variation. It is a univariate
selector and does not model feature interactions.

```{code-cell} ipython3
import numpy as np
from believe14.linear import FisherScore
from _example_data import multiclass_data

X, labels = multiclass_data()
model = FisherScore(n_features_to_select=2)
selected = model.fit_transform(X, labels)
queries = model.transform(X[:3])
restored = model.inverse_transform(selected)
assert selected.shape == (48, 2) and queries.shape == (3, 2)
{
    "shape": selected.shape,
    "selected_indices": np.flatnonzero(model.get_support()).tolist(),
    "top_scores": np.round(model.scores_[model.ranking_[:2]], 3).tolist(),
    "solver": model.diagnostics_.solver,
}
```

`scores_`, `ranking_`, and `get_support()` expose the selection. Diagnostics are
closed-form and warn when zero within-class variance makes a score infinite.
Feature-wise scoring can discard jointly discriminative variables and requires at
least two classes.

Primary reference: {cite:p}`duda2001`.
