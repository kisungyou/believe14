# DiffusionMap validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Coifman and Lafon (2006),
  [doi:10.1016/j.acha.2006.04.006](https://doi.org/10.1016/j.acha.2006.04.006).
- **Definition:** start with the full RBF kernel, divide by
  `q_i^alpha q_j^alpha`, row-normalize, and diagonalize its symmetric conjugate.
  Omit the stationary eigenfunction and scale each retained right eigenfunction by
  `lambda**diffusion_time`.
- **Conventions:** `alpha` is in `[0,1]`, default `1`; `gamma` defaults to
  `1/n_features`; diffusion time is a nonnegative integer. Nonpositive density/degree
  and selected numerical-zero eigenvalues are rejected.
- **Out of sample:** query densities and degrees are recomputed against training data;
  right eigenfunctions use `psi(x) = sum_j p(x,j) psi_j / lambda`, the stated Nyström
  eigen-equation.
- **Evidence:** the fitted operator is row-stochastic, the training Nyström extension
  reproduces coordinates, and normalized symmetric-eigen residuals are tested.
  Complexity is `O(n^2 p + n^3)` time and `O(n^2)` memory. Rdimtools is not an oracle.
