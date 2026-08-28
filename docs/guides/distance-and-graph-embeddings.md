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
Isomap, LLE, Laplacian Eigenmaps, and LTSA are transductive in 0.1.0. FastMap
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
