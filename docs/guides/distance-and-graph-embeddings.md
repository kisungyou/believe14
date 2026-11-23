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

# Distance and graph embeddings

Distance methods preserve pairwise dissimilarities; kernel methods diagonalize
a centered similarity; graph methods first define local adjacency and then
solve a geodesic, reconstruction, Laplacian, diffusion, or alignment problem
{cite:p}`torgerson1952,faloutsos1995,scholkopf1998,tenenbaum2000,roweis2000,belkin2003,coifman2006,zhang2004`.

The following deterministic helix keeps the exact-neighbor graphs connected.

```{code-cell} ipython3
import numpy as np

from believe14.nonlinear import (
    ClassicalMDS,
    DiffusionMap,
    FastMap,
    Isomap,
    KernelPCA,
    LaplacianEigenmaps,
    LocallyLinearEmbedding,
    LocalTangentSpaceAlignment,
)

parameter = np.linspace(0.0, 3.0 * np.pi, 48)
X = np.column_stack(
    (np.cos(parameter), np.sin(parameter), parameter / (3.0 * np.pi))
)

models = [
    ClassicalMDS(2),
    FastMap(2),
    KernelPCA(2, gamma=2.0),
    Isomap(2, n_neighbors=6),
    LocallyLinearEmbedding(2, n_neighbors=7),
    LaplacianEigenmaps(2, n_neighbors=6, gamma=2.0),
    DiffusionMap(2, gamma=2.0),
    LocalTangentSpaceAlignment(2, n_neighbors=7),
]

for model in models:
    embedding = model.fit_transform(X)
    assert embedding.shape == (48, 2)
    assert np.isfinite(embedding).all()
    print(
        f"{type(model).__name__:30s}",
        f"out_of_sample={hasattr(model, 'transform')}",
    )
```

The missing `transform` attributes above are intentional. Classical MDS,
Isomap, LLE, Laplacian Eigenmaps, and LTSA are transductive. FastMap
uses its pivots, Kernel PCA uses Nyström kernel centering, and Diffusion Map
uses its cited Nyström transition rule. Supplying precomputed distances or a
kernel does not create a new-data rule by itself.

Read the cards for [Classical MDS](../examples/classical_mds.md),
[FastMap](../examples/fast_map.md), [Kernel PCA](../examples/kernel_pca.md),
[Isomap](../examples/isomap.md), [LLE](../examples/locally_linear_embedding.md),
[Laplacian Eigenmaps](../examples/laplacian_eigenmaps.md),
[Diffusion Map](../examples/diffusion_map.md), and
[LTSA](../examples/local_tangent_space_alignment.md). Metric and nonmetric
stress objectives are treated in the next guide.

## Fitting a coordinate system on landmarks

[Landmark MDS](../examples/landmark_mds.md) scales distances among a subset of
training points. [Landmark Isomap](../examples/landmark_isomap.md) uses
shortest paths to those landmarks on the fitted neighbor graph. Both can embed
a new point in the fitted coordinates. Representative landmarks matter because
their geometry determines the retained directions
{cite:p}`desilva2004landmark,desilva2002global,bengio2003extensions`.

```{code-cell} ipython3
from believe14.neighbors import build_neighbor_graph
from believe14.nonlinear import LandmarkIsomap, LandmarkMDS

landmarks = np.linspace(0, len(X) - 1, 12, dtype=int)
graph = build_neighbor_graph(X, n_neighbors=6)
direct = LandmarkMDS(2, landmark_indices=landmarks).fit(X)
geodesic = LandmarkIsomap(
    2, n_neighbors=6, landmark_indices=landmarks
).fit(X, neighbors=graph)

query = (X[13:14] + X[14:15]) / 2
print(direct.transform(query), geodesic.transform(query))
assert geodesic.landmark_distances_.shape == (len(X), len(landmarks))
```

Specify `landmark_indices` to use a designed sample. Alternatively,
`n_landmarks` selects that many evenly spaced row indices. With neither supplied,
the default count is the ceiling of the square root of the training size,
raised to at least one more than the requested dimension. This deterministic
selection depends on row order. For Landmark Isomap each query attaches to
training points independently, so embedding a batch preserves the coordinates
obtained for each query alone.
