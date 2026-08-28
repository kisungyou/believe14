# PLSRegression validation ledger

- Status: validated; NIPALS PLS2 regression.
- Authority: Wold et al. (1984), DOI 10.1137/0905052.
- Frozen convention: center X and Y; optionally scale each column by sample
  standard deviation (`ddof=1`). At each component, alternate X weights and Y
  scores, normalize the X weight to unit norm, then use regression-mode
  deflation `X <- X-t p^T` and `Y <- Y-t q^T`.
- Mapping: X rotations are `R=W(P^T W)^-1`; predictions in standardized space
  are `X R Q^T`, then target scaling and the fitted intercept are restored.
  `predict` evaluates this affine map through the fitted reference-shifted
  centering state, which is algebraically identical to `X @ coef_ + intercept_`
  but avoids catastrophic cancellation at large common offsets. The coefficient
  attributes record that algebraic affine form; direct raw evaluation is not the
  numerical prediction contract on ill-conditioned offsets.
  Passing targets to `transform` returns both X scores and target projection
  scores; `fit_transform(X, y)` follows that paired convention.
- Stopping: sign-invariant X-weight change; every component must converge.
  Degenerate X weights, Y weights, scores, or prematurely exhausted Y residuals
  are explicit failures. Diagnostics report the relative residual sum of squares,
  avoiding an unrepresentable raw SSE without changing the fitted NIPALS problem.
- Evidence: exact low-rank multiresponse prediction, transform dimensions,
  coefficient/intercept identity at ordinary scale, reference-shifted prediction,
  scaling, clone, and non-convergence tests.
- Complexity and legacy: work scales with the declared component count, NIPALS
  iterations, samples, and view widths. Rdimtools is not an acceptance oracle.
