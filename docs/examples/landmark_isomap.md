---
believe14_estimator: LandmarkIsomap
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LandmarkIsomap

Use graph distances when chords across a curved dataset would distort the
geometry. Landmark Isomap computes paths from a subset of observations and
extends the resulting coordinates to the remaining observations.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import LandmarkIsomap

angle = np.linspace(0, 1.5 * np.pi, 41)
X = np.column_stack((np.cos(angle), np.sin(angle)))
model = LandmarkIsomap(1, n_neighbors=2, landmark_indices=[0, 10, 20, 30, 40])
embedding = model.fit_transform(X)
new_angle = (angle[15] + angle[16]) / 2
query = [[np.cos(new_angle), np.sin(new_angle)]]
position = model.transform(query)[0, 0]
assert min(embedding[15:17, 0]) < position < max(embedding[15:17, 0])
print(model.landmark_distances_.shape, position)
```

`landmark_distances_[i, j]` is the shortest path from training point `i` to
`landmark_indices_[j]`. A query attaches to its nearest training points, with
no edges between queries. Choose `n_neighbors` so the training graph is
connected. The fitted diagnostics flag non-Euclidean landmark geodesics.
Reuse exact neighborhoods with `fit(X, neighbors=graph)`.

Primary reference: {cite:p}`desilva2002global,desilva2004landmark,bengio2003extensions`.
