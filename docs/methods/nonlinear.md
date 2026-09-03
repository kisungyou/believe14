# Nonlinear methods

The nonlinear family contains twelve distance, kernel, graph, diffusion,
stress, and stochastic embeddings. Check each card's out-of-sample status:
transductive methods intentionally do not invent a `transform` operation.

<div class="method-grid">
  <a class="method-card" href="../examples/classical_mds.html"><span class="method-card-title">ClassicalMDS</span><span class="method-card-description">Double-centered squared dissimilarities and a spectral embedding.</span></a>
  <a class="method-card" href="../examples/metric_mds.html"><span class="method-card-title">MetricMDS</span><span class="method-card-description">Unweighted raw-stress minimization by SMACOF majorization.</span></a>
  <a class="method-card" href="../examples/sammon_mapping.html"><span class="method-card-title">SammonMapping</span><span class="method-card-description">Normalized stress emphasizing preservation of short distances.</span></a>
  <a class="method-card" href="../examples/fast_map.html"><span class="method-card-title">FastMap</span><span class="method-card-description">Pivot-based metric coordinates with a cited extension rule.</span></a>
  <a class="method-card" href="../examples/kernel_pca.html"><span class="method-card-title">KernelPCA</span><span class="method-card-description">Centered-kernel eigenvectors with Nyström out-of-sample extension.</span></a>
  <a class="method-card" href="../examples/isomap.html"><span class="method-card-title">Isomap</span><span class="method-card-description">Exact neighbors, graph geodesics, and classical scaling.</span></a>
  <a class="method-card" href="../examples/locally_linear_embedding.html"><span class="method-card-title">LocallyLinearEmbedding</span><span class="method-card-description">Local barycentric reconstruction followed by global alignment.</span></a>
  <a class="method-card" href="../examples/laplacian_eigenmaps.html"><span class="method-card-title">LaplacianEigenmaps</span><span class="method-card-description">A graph-Laplacian generalized eigensystem.</span></a>
  <a class="method-card" href="../examples/diffusion_map.html"><span class="method-card-title">DiffusionMap</span><span class="method-card-description">Density-normalized diffusion coordinates with Nyström extension.</span></a>
  <a class="method-card" href="../examples/local_tangent_space_alignment.html"><span class="method-card-title">LocalTangentSpaceAlignment</span><span class="method-card-description">Local tangent SVDs assembled into a global alignment.</span></a>
  <a class="method-card" href="../examples/tsne.html"><span class="method-card-title">TSNE</span><span class="method-card-description">Exact dense symmetric t-SNE with explicit convergence diagnostics.</span></a>
  <a class="method-card" href="../examples/phate.html"><span class="method-card-title">PHATE</span><span class="method-card-description">Diffusion-potential distances followed by metric MDS.</span></a>
</div>

```{toctree}
:hidden:
:maxdepth: 1

../examples/classical_mds
../examples/metric_mds
../examples/sammon_mapping
../examples/fast_map
../examples/kernel_pca
../examples/isomap
../examples/locally_linear_embedding
../examples/laplacian_eigenmaps
../examples/diffusion_map
../examples/local_tangent_space_alignment
../examples/tsne
../examples/phate
```
