# CanonicalCorrelationAnalysis validation ledger

- Status: validated; paired-view implementation of Hotelling CCA.
- Authority: Hotelling (1936), DOI 10.1093/biomet/28.3-4.321.
- Preprocessing: center each aligned view independently. Thin SVDs define the
  exact numerical covariance supports; no implicit ridge is added.
- Solver: singular values of `Qx^T Qy` are canonical correlations. Weights map
  those singular vectors back through `V diag(sqrt(n-1)/s)`, giving canonical
  scores with sample covariance identity.
- API: `fit(X,Y)`, `transform(X)` for the first view, and `transform(X,Y)` or
  `fit_transform(X,Y)` for paired scores. Row counts must agree.
- Evidence: unit within-view score covariance, diagonal cross-view covariance,
  rank-deficient support, paired-shape validation, and feature-permutation tests.
  Canonical subspaces are the comparison object for repeated correlations.
- Complexity and legacy: the dense path uses two thin SVDs plus the support
  cross-SVD. Rdimtools results are not acceptance criteria.
