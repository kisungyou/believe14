# SlicedInverseRegression validation ledger

- Status: validated; implementation follows Li (1991),
  DOI 10.1080/01621459.1991.10475035.
- Preprocessing: center predictors and whiten on the numerical support of their
  ML covariance `Xc^T Xc/n`; directions are mapped back to original coordinates.
- Slicing: numeric responses use deterministic equal-frequency quantile edges;
  duplicate edges are removed and ties always remain together. If unique values
  do not exceed `n_slices`, each unique value forms one slice.
- Kernel: `M=sum_h p_h m_h m_h^T`, where `m_h=E[Z|slice h]`; retain its largest
  eigenvectors. The limit is `min(rank(X), number_of_slices-1)`.
- Evidence: linear central-subspace recovery, strict-monotone response
  invariance, covariance whitening, spectral residual, and tied-response tests.
  Compare subspace projectors; the fitted affine map is the out-of-sample rule.
- Complexity and legacy: dense covariance formation and feature eigendecomposition
  dominate the calculation. Rdimtools values are not acceptance criteria.
