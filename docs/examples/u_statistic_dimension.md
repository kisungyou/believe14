---
believe14_estimator: UStatisticDimension
believe14_family: estimation
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# UStatisticDimension

Use this estimator to explore compact-kernel means across
prescribed bandwidth and sample-size scales. It is a global, finite-candidate
estimate and uses a seeded sample ordering.

```{warning}
Accuracy validation covers nine specified low-dimensional benchmark configurations.
Individual fits can still be inaccurate, and higher-dimensional limitations remain.
Successful computation and reproducible output do not certify dimension recovery.
See the [validation ledger](../validation/estimation/u-statistic-dimension.md),
including the original failed panel and the prospective accuracy study.
```

```{code-cell} ipython3
import numpy as np
from believe14.estimation import UStatisticDimension
from _example_data import flat_data

X = flat_data(intrinsic_dimension=3)
model = UStatisticDimension(max_dimension=5, random_state=14).fit(X)
replay = UStatisticDimension(max_dimension=5, random_state=14).fit(X)
assert model.dimension_ == replay.dimension_
assert np.array_equal(model.sample_order_, replay.sample_order_)
assert not hasattr(model, "transform") and not hasattr(model, "predict")
{
    "dimension": model.dimension_,
    "candidate_dimensions": model.candidate_dimensions_.tolist(),
    "bandwidth_factors": np.round(model.bandwidth_factors_, 3).tolist(),
    "weighted_slope_residual": model.diagnostics_.residual_norm,
    "warnings": model.diagnostics_.warnings,
}
```

`kernel_means_`, `log_u_statistics_`, `slopes_`, and `sample_order_` make the
search reproducible. Diagnostics report the winning slope-fit mismatch. Results
can be sensitive to sample size, boundaries, and candidate ceiling; the class
does not manufacture prediction or transformation semantics.

Primary reference: {cite:p}`hein2005`.
