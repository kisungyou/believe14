---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Supervised and paired reductions

These methods use information beyond one feature matrix. LDA and Fisher Score
use class labels; PLS, SIR, and SAVE use response variables; CCA uses two
aligned views {cite:p}`fisher1936,hotelling1936,wold1984,li1991,cook2000,duda2001`.
They are projections or selectors, not replacement classifiers.

```{code-cell} ipython3
import numpy as np

from believe14.linear import (
    CanonicalCorrelationAnalysis,
    FisherScore,
    LinearDiscriminantAnalysis,
    PLSRegression,
    SlicedAverageVarianceEstimation,
    SlicedInverseRegression,
)

rng = np.random.default_rng(14)
labels = np.repeat(np.arange(3), 40)
centers = np.array(
    [[-1.5, 0.0, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0, 0.0]]
)
X = centers[labels] + 0.55 * rng.normal(size=(120, 5))
target = 1.8 * X[:, 0] - 0.7 * X[:, 1] + 0.15 * rng.normal(size=120)
Y = np.column_stack(
    (
        X[:, 0] + 0.2 * rng.normal(size=120),
        X[:, 1] - 0.2 * rng.normal(size=120),
        rng.normal(size=120),
    )
)

models = [
    LinearDiscriminantAnalysis(2, regularization=1e-8).fit(X, labels),
    CanonicalCorrelationAnalysis(2).fit(X, Y),
    PLSRegression(2).fit(X, target),
    SlicedInverseRegression(2, n_slices=6).fit(X, target),
    SlicedAverageVarianceEstimation(2, n_slices=6).fit(X, target),
    FisherScore(2).fit(X, labels),
]

for model in models:
    assert model.diagnostics_.converged
    print(type(model).__name__, model.diagnostics_.solver)
```

CCA validates row alignment and can transform one view or both views. PLS
provides `predict` under its frozen NIPALS PLS2 convention. Fisher Score is a
feature selector: `get_support()` identifies original columns and `transform`
returns those columns without rotating them. The remaining estimators return
supervised subspace coordinates for new feature rows.

See the cards for [LDA](../examples/linear_discriminant_analysis.md),
[CCA](../examples/canonical_correlation_analysis.md),
[PLS2](../examples/pls_regression.md),
[SIR](../examples/sliced_inverse_regression.md),
[SAVE](../examples/sliced_average_variance_estimation.md), and
[Fisher Score](../examples/fisher_score.md).
