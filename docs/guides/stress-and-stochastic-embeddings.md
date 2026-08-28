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

# Stress and stochastic embeddings

Metric MDS minimizes raw stress by SMACOF; Sammon Mapping reweights short
distances; exact t-SNE minimizes a symmetric neighborhood divergence; PHATE
constructs diffusion-potential distances and embeds them with metric MDS
{cite:p}`deleeeuw1977,sammon1969,vandermaaten2008,moon2019`.

The unweighted raw-stress objective used by Metric MDS is

$$
\sigma(Z) = \sum_{i < j} \left(d_{ij} - \lVert z_i - z_j \rVert_2\right)^2.
$$

```{code-cell} ipython3
import numpy as np

from believe14.nonlinear import MetricMDS, PHATE, TSNE, SammonMapping

parameter = np.linspace(0.0, 4.0 * np.pi, 36)
X = np.column_stack(
    (np.cos(parameter), np.sin(parameter), parameter / (4.0 * np.pi))
)

models = [
    MetricMDS(2, max_iter=200, tol=1e-7, random_state=14),
    SammonMapping(2, max_iter=200, tol=1e-7, random_state=14),
    TSNE(
        2,
        perplexity=6.0,
        early_exaggeration_iter=50,
        max_iter=300,
        tol=1e-6,
        random_state=14,
    ),
    PHATE(2, n_neighbors=5, diffusion_time=5, mds_max_iter=200),
]

for model in models:
    embedding = model.fit_transform(X)
    diagnostics = model.diagnostics_
    assert embedding.shape == (36, 2)
    assert np.isfinite(embedding).all()
    print(
        f"{type(model).__name__:14s}",
        f"iterations={diagnostics.n_iter!s:>3s}",
        f"objective={diagnostics.objective_value:.6g}",
        f"converged={diagnostics.converged}",
    )
```

All four methods are transductive and therefore expose no `transform`.
`random_state` controls random initialization where selected and makes t-SNE
replay exactly without reading NumPy's global random state. PHATE is
deterministic in this release; its `FitDiagnostics` reports the composite
diffusion-potential and SMACOF solver. An exhausted iteration budget raises or
records non-convergence according to the method contract—it is never relabeled
as success.

See [Metric MDS](../examples/metric_mds.md),
[Sammon Mapping](../examples/sammon_mapping.md), [t-SNE](../examples/tsne.md),
and [PHATE](../examples/phate.md).
