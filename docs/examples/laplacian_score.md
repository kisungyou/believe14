---
believe14_estimator: LaplacianScore
believe14_family: linear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LaplacianScore

Use Laplacian Score to select features that vary smoothly between neighboring observations. On the path below, the first coordinate changes gradually and the second reverses sign at each step. The selector retains the first coordinate.

```{code-cell} ipython3
import numpy as np
from believe14.linear import LaplacianScore
from believe14.neighbors import build_neighbor_graph

X = np.column_stack((
    [0.0, 1.0, 2.1, 3.3, 4.6],
    [0.05, -0.05, 0.05, -0.05, 0.05],
))
graph = build_neighbor_graph(X, n_neighbors=2)
model = LaplacianScore(1, n_neighbors=1).fit(X, neighbors=graph)
queries = np.array([[5.3, -0.05], [6.1, 0.05]])
assert model.get_support(indices=True).tolist() == [0]
selected = model.transform(queries)
assert selected.shape == (2, 1) and np.array_equal(selected, queries[:, :1])
{"scores": model.scores_.round(3).tolist(), "solver": model.diagnostics_.solver}
```

Feature units affect which observations are neighbors. Binary weights count each edge equally. With `weighting="heat"`, a pair at distance $d$ receives weight $\exp(-\gamma d^2)$. A disconnected graph raises an error. Increase `n_neighbors` or adjust the heat bandwidth to connect the sample. Constant columns are ineligible.

Primary reference: {cite:p}`he2005laplacian`.
