# Dimension estimation

The estimation family reports intrinsic dimension rather than embedding
coordinates. Every successful fit exposes `dimension_` and immutable
`diagnostics_`; methods expose `local_dimensions_` only when the audited
estimator defines them.

<div class="method-grid">
  <a class="method-card" href="../examples/correlation_dimension.html"><span class="method-card-title">CorrelationDimension</span><span class="method-card-description">Correlation-integral scaling over a selected distance regime.</span></a>
  <a class="method-card" href="../examples/two_nn.html"><span class="method-card-title">TwoNN</span><span class="method-card-description">A nearest-neighbor distance-ratio estimate.</span></a>
  <a class="method-card" href="../examples/levina_bickel_mle.html"><span class="method-card-title">LevinaBickelMLE</span><span class="method-card-description">Local Poisson-process likelihood from exact neighbors.</span></a>
  <a class="method-card" href="../examples/u_statistic_dimension.html"><span class="method-card-title">UStatisticDimension</span><span class="method-card-description">A convergence-rate estimate built from U-statistics.</span></a>
  <a class="method-card" href="../examples/mi_ndml.html"><span class="method-card-title">MiNDML</span><span class="method-card-description">Minimum-neighbor-distance maximum likelihood.</span></a>
  <a class="method-card" href="../examples/danco.html"><span class="method-card-title">DANCo</span><span class="method-card-description">Angle-and-norm concentration with an explicit one-dimensional limit.</span></a>
</div>

```{toctree}
:hidden:
:maxdepth: 1

../examples/correlation_dimension
../examples/two_nn
../examples/levina_bickel_mle
../examples/u_statistic_dimension
../examples/mi_ndml
../examples/danco
```
