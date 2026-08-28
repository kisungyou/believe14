# MiNDML validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Lombardi et al., *Minimum Neighbor Distance Estimators of
  Intrinsic Dimension*, ECML PKDD 2011, LNCS 6912, 374–389,
  [doi:10.1007/978-3-642-23783-6_24](https://doi.org/10.1007/978-3-642-23783-6_24).
- **Statistic:** `rho_i=T_1(i)/T_{k+1}(i)` and
  `g(r;k,d)=k d r^(d-1)(1-r^d)^(k-1)`. The summed log likelihood is evaluated
  directly with `log1p`, then maximized continuously on the closed interval
  `[1,max_dimension]`, including explicit endpoint comparisons.
- **Convention:** default `k=10`; default upper bound is ambient feature count.
  The continuous maximum is returned without the integer rounding used by some
  secondary implementations.
- **Output:** one real `dimension_`, normalized distances, maximized log
  likelihood, normalized score residual, optimizer evaluations, and an explicit
  boundary warning where applicable.
- **Invariances:** translation, orthogonal transformation, uniform scaling, and
  row permutation. Stable row index resolves equal-distance neighbor identity.
- **Failure policy:** reject zero radii and a tie between first and `(k+1)`-st
  radii because it places `rho` at the singular likelihood boundary. No clipping,
  jitter, or observation removal is permitted.
- **Advertised regime:** locally uniform sampling within the `(k+1)`-neighbor
  ball; finite-sample boundary and high-dimension bias should be inspected via
  the retained likelihood and diagnostics.
- **Complexity:** `O(n^2 p)` time and `O(n^2)` memory.
- **Independent evidence:** literal ratio/likelihood/score fixture, closed-bound
  checks, invariance and tie tests, and two-dimensional uniform recovery. The
  implementation was derived independently of Rdimtools.
