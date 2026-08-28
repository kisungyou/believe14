# CorrelationDimension validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Grassberger and Procaccia, *Measuring the strangeness of
  strange attractors*, Physica D 9 (1983), 189–208,
  [doi:10.1016/0167-2789(83)90298-1](https://doi.org/10.1016/0167-2789(83)90298-1).
- **Statistic:** for `n` observations,
  `C_n(r) = 2/[n(n-1)] sum_{i<j} 1{||x_i-x_j|| < r}`. The reported dimension is
  the ordinary-least-squares slope of `log C_n(r)` on `log r` over the declared
  radii. The inequality is strict and unordered pairs exclude the diagonal.
- **Scale convention:** explicit `radii` are used verbatim after validation. If
  absent, 20 log-spaced radii span the 0.05 and 0.20 positive pair-distance
  quantiles. This automatic interval is a documented finite-sample convention,
  not a claim that the paper identifies a universal scaling region.
- **Output:** one positive real `dimension_`; slope intercept, radii, empirical
  integrals, and normalized regression residual are retained.
- **Invariances:** translation and orthogonal transformation; uniform scaling
  when automatic radii are used, or when explicit radii are scaled with the data;
  row permutation.
- **Failure policy:** reject duplicates/numerically zero pair distances,
  non-increasing radii, saturated scale intervals, or a nonpositive fitted slope.
  Pair-distance ties otherwise remain in the strict empirical count.
- **Advertised regime:** a scale interval displaying approximately power-law
  correlation-integral growth. Multiple scaling regimes require separate fits.
- **Complexity:** `O(n^2 p)` time and `O(n^2)` memory for exact dense distances.
- **Independent evidence:** literal six-pair count/slope fixture, Euclidean
  invariance tests, duplicate and scale failures, and two-dimensional uniform
  recovery. No Rdimtools output or implementation behavior is an oracle.
