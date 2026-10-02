(method-catalog)=
# Methods

These tables are generated directly from the public registry. Approach and
supervision are metadata rather than additional package namespaces. Select a
family for its complete set of executable method cards, or follow an estimator
name in the catalog directly to its card.

Validation status refers to the recorded evidence and documented scope, not a
general accuracy or maturity guarantee. All 30 entries have validated status.
`UStatisticDimension` passed the prospective accuracy protocol on nine specified
low-dimensional configurations; its historical failed panel and higher-dimensional
limitations remain documented. Successful numerical diagnostics alone do not
certify dimension recovery. Filter this metadata with
`list_estimators(validation_status="validated")` or
`list_estimators(validation_status="experimental")`.

The costs describe the current dense **fit**, including diagnostics and peak
working storage, with conservative upper bounds. Let `n` be samples, `p` input
features, `q` response/paired features, `k` output components, `h` neighbors, and
`D` candidate dimension bound. `T` counts solver iterations (per component for
PLS), `C` inner coordinate sweeps, `E` objective/gradient evaluations including
line searches, `R` pivot sweeps, `S` slices, `L` classes, `B` radii and `Q` scalar quadrature
evaluations per candidate. PHATE's `t` is diffusion time. Fixed entropy-search
iterations are absorbed in t-SNE's bound. Precomputed inputs omit feature-distance
construction. Transform costs are separate from fit; these are operation/storage
bounds, not measured runtimes. In particular, dense neighbor searches fully sort
each row and dense covariance diagnostics can require cubic feature-space work.

<div class="landing-grid family-grid">
  <a class="landing-card" href="methods/linear.html">
    <h2>Linear</h2>
    <p>Twelve projections, latent-variable models, supervised reductions, and a feature selector.</p>
  </a>
  <a class="landing-card" href="methods/nonlinear.html">
    <h2>Nonlinear</h2>
    <p>Twelve distance, kernel, graph, stress, diffusion, and stochastic embeddings.</p>
  </a>
  <a class="landing-card" href="methods/estimation.html">
    <h2>Dimension estimation</h2>
    <p>Six estimators for intrinsic dimension from distances, neighborhoods, and concentration.</p>
  </a>
</div>

## Linear

```{believe14-catalog} linear
```

## Nonlinear

```{believe14-catalog} nonlinear
```

## Estimation

```{believe14-catalog} estimation
```

```{toctree}
:hidden:
:maxdepth: 2

Linear methods <methods/linear>
Nonlinear methods <methods/nonlinear>
Dimension estimation <methods/estimation>
References <references>
```
