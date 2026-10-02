# DiffusionMap validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Coifman and Lafon (2006),
  [doi:10.1016/j.acha.2006.04.006](https://doi.org/10.1016/j.acha.2006.04.006).
- **Definition:** start with the full RBF kernel, divide by
  `q_i^alpha q_j^alpha`, row-normalize, and diagonalize its symmetric conjugate.
  Restrict the symmetric operator to the orthogonal complement of the known
  square-root degree vector before diagonalization, then scale each retained
  right eigenfunction by `lambda**diffusion_time`. This excludes the stationary
  mode even when weak connections make several eigenvalues numerically equal to
  one. The right eigenfunctions have zero stationary mean and are orthonormal
  under the stationary distribution.
- **Conventions:** `alpha` is in `[0,1]`, default `1`; `gamma` defaults to
  `1/n_features`; diffusion time is a nonnegative integer. Nonpositive density/degree
  and selected numerical-zero eigenvalues are rejected.
- **Out of sample:** query densities and degrees are recomputed against training data;
  right eigenfunctions use `psi(x) = sum_j p(x,j) psi_j / lambda`, the stated Nyström
  eigen-equation.
- **Evidence:** the fitted operator is row-stochastic, the training Nyström extension
  reproduces coordinates, and normalized symmetric-eigen residuals are tested.
  Tests include weakly connected positive kernels with repeated numerical modes,
  stationary orthogonality, and the identity between all-coordinate Euclidean
  distances and diffusion distances computed directly from transition rows.
  Numerical rank counts the nonstationary restricted spectrum.
  Rdimtools is not an oracle.

- **Complexity:** `O(n^2 p + n^3) time; O(np + n^2) memory`; symbols follow the
  [method catalog](../../methods.md).
