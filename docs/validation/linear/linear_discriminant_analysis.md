# LinearDiscriminantAnalysis validation ledger

- Status: validated; transformer only; no classifier API.
- Authority: Fisher (1936) and Rao's multiclass canonical extension.
- Scatter definitions: `Sw=sum_c sum_i(x_ci-mu_c)(x_ci-mu_c)^T` and
  `Sb=sum_c n_c(mu_c-mu)(mu_c-mu)^T`, with no denominator because the common
  factor cancels in the generalized eigenproblem.
- Solver: solve `Sb v=lambda Sw v` on the numerical range of `Sw`. The optional
  documented ridge is `regularization*trace(Sw)/p`; when `trace(Sw)=0`, its
  explicit fallback scale is `trace(Sw+Sb)/p`. Zero means exactly zero. If `Sb`
  has energy in `null(Sw)` at zero regularization, the Fisher quotient has
  infinite directions and fitting fails rather than silently discarding them.
- Dimensional limit: at most `min(number_of_classes-1, p)` and no more than the
  solved denominator rank. Singular unsupported requests fail explicitly.
- Evidence: generalized-eigen residual, isotropic two-class direction, label
  recoding invariance, class/shape errors, and rank diagnostics. Compare
  discriminant projectors, never eigenvector signs.
- Complexity and legacy: scatter construction is dense and the feature-space
  generalized eigensystem is cubic in feature count. Rdimtools is not an oracle.
