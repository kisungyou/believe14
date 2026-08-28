---
believe14_estimator: DANCo
believe14_family: estimation
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# DANCo

Use DANCo when both nearest-neighbor norm ratios and local angle concentration are
informative, and a seeded synthetic calibration over integer candidate dimensions
is acceptable. The admissible candidate range starts at dimension two.

```{code-cell} ipython3
import numpy as np
from believe14.estimation import DANCo
from _example_data import flat_data

X = flat_data(intrinsic_dimension=3)
model = DANCo(n_neighbors=6, max_dimension=5, random_state=14).fit(X)
replay = DANCo(n_neighbors=6, max_dimension=5, random_state=14).fit(X)
assert model.dimension_ == replay.dimension_
assert np.allclose(model.divergences_, replay.divergences_)
assert not hasattr(model, "transform") and not hasattr(model, "predict")
{
    "dimension": model.dimension_,
    "candidates": model.candidate_dimensions_.tolist(),
    "divergences": np.round(model.divergences_, 4).tolist(),
    "warnings": model.diagnostics_.warnings,
}
```

`normalized_distances_`, angle summaries, calibration arrays, and `divergences_`
expose both concentration terms. Diagnostics warn when the optimum is on the
candidate boundary. One-dimensional or collinear angular geometry, duplicates,
and insufficient neighbors are explicitly rejected rather than misreported.

Primary reference: {cite:p}`ceruti2014`.
