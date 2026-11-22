---
believe14_estimator: LandmarkMDS
believe14_family: nonlinear
jupytext:
  text_representation: {extension: .md, format_name: myst}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# LandmarkMDS

Use representative training points when you need to fit a classical-scaling
coordinate system once and locate additional observations in it. On a plane,
three noncollinear landmarks suffice to recover Euclidean distances.

```{code-cell} ipython3
import numpy as np
from believe14.nonlinear import LandmarkMDS

X = np.array([[-1., -1.], [-1., 1.], [0., 0.], [1., -1.], [1., 1.]])
model = LandmarkMDS(2, landmark_indices=[0, 1, 3])
embedding = model.fit_transform(X)
new_point = np.array([[0.3, 0.4]])
projected = model.transform(new_point)
np.testing.assert_allclose(
    np.linalg.norm(embedding - projected, axis=1),
    np.linalg.norm(X - new_point, axis=1),
)
print(model.landmark_indices_, embedding.shape)
```

The coordinates use the landmark centroid as their origin. Features and
precomputed distances are both supported. With `dissimilarity="precomputed"`,
fit takes a square training distance matrix and transform takes one column per
original training observation. The model uses the columns for its selected
landmarks. `eigenvalues_` contains the signed spectrum of the landmark Gram
matrix. The fitted diagnostics report its positive numerical rank. A dimension above this rank raises an error.

Primary reference: {cite:p}`desilva2004landmark`.
