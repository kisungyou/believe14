# DANCo validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Ceruti et al., *DANCo: An intrinsic dimensionality
  estimator exploiting angle and norm concentration*, Pattern Recognition 47
  (2014), 2569–2581,
  [doi:10.1016/j.patcog.2014.02.013](https://doi.org/10.1016/j.patcog.2014.02.013),
  Equations 2–13.
- **Norm component:** compute `rho_i=T_1(i)/T_{k+1}(i)`, maximize the MiND
  likelihood, and compare it with calibrated estimates using the paper's
  closed-form Equation 3 KL divergence.
- **Angle component:** form every pair among the `k` normalized neighbor
  directions at each observation. Estimate local von Mises mean directions and
  concentrations by Equations 6–8, average those parameters over observations,
  and use the stable scaled-Bessel form of Equation 9's KL divergence.
- **Calibration:** for each integer `d=2,...,max_dimension`, draw one sample of
  `N` points uniformly from the unit `d`-ball using Gaussian directions and
  radial law `U^(1/d)`. Recompute both statistics and select the candidate
  minimizing their summed KL divergence (Equation 13). Default `k=10`; default
  upper bound is ambient feature count and must be at least two.
- **Numerical convention:** the exact `eta=1` angular boundary has infinite von
  Mises concentration and is rejected; it is never clipped to a large finite
  surrogate. The high-concentration expression is algebraically factored to
  avoid cancellation for admissible `eta<1`, and scaled Bessel functions avoid
  overflow.
- **Output:** integer-valued float `dimension_`, all data and calibration
  parameters, per-candidate divergences, optimizer evaluations, and a boundary
  warning where applicable.
- **Invariances/randomness:** translation, orthogonal transformation, uniform
  scaling, and row permutation in the absence of ambiguous neighbor identities.
  Calibration uses only a local Generator, is seed-reproducible, and leaves the
  global RNG untouched.
- **Failure policy:** reject one-dimensional or embedded-collinear data,
  duplicate/zero radii, first-to-`(k+1)` neighbor ties, angular resultants equal
  to one, invalid candidate bounds, and any non-finite KL. Other distance ties
  use stable row-index order; no perturbation or fallback estimator is used.
- **Advertised regime:** intrinsic dimension at least two, with smooth local
  sampling approximated by uniform balls. Stochastic calibration uncertainty
  should be assessed with multiple seeds for release-scale empirical studies.
- **Complexity:** exact dense `O((D-1) n^2 p + (D-1) n k^2)` time and `O(n^2)`
  peak memory for candidates `2,...,D`, where `D=max_dimension`.
- **Independent evidence:** identity checks for both closed-form KL terms,
  seeded calibration replay, global-RNG isolation, geometric/permutation and
  singular-tie tests, explicit one-dimensional failure, and two- and
  three-dimensional uniform-ball recovery below a loose upper bound across
  multiple seeds. Rdimtools is not an oracle.
