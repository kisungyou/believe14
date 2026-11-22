---
believe14_estimator: TightLocalEstimator
believe14_family: estimation
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# TightLocalEstimator

Use Tight Local Estimation to compare dimensions across regions of a dataset.
It fits a local likelihood from moving and reflected centers in each observed
neighborhood. Here, samples from a line and a plane sit far
enough apart that a 12-neighbor search stays within each region.

```{code-cell} ipython3
import numpy as np

from believe14.estimation import TightLocalEstimator
from believe14.neighbors import build_neighbor_graph

rng = np.random.default_rng(751)
line = np.column_stack([rng.uniform(size=100), np.zeros(100)])
plane = 3.0 + rng.uniform(size=(120, 2))
X = np.vstack([line, plane])
neighbors = build_neighbor_graph(X, n_neighbors=16)
model = TightLocalEstimator(n_neighbors=12).fit(X, neighbors=neighbors)
line_median = np.median(model.local_dimensions_[:len(line)])
plane_median = np.median(model.local_dimensions_[len(line):])
assert line_median < plane_median and model.diagnostics_.converged
{
    "line median": round(line_median, 2),
    "plane median": round(plane_median, 2),
    "global mean": round(model.dimension_, 2),
}
```

The estimator averages over all ordered pairs of distinct neighbors and their
reflected starting points. The single value in `dimension_` averages these local
estimates across the two regions. Choose the neighbor count for the spatial scale
of interest. Refit with another count using the same graph to inspect sensitivity.
Exact reflected coincidences and duplicate observations can give zero adjusted
distances, in which case fitting raises an error.

Primary reference: {cite:p}`amsaleg2022`.
