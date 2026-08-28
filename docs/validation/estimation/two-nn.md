# TwoNN validation ledger

- **Status:** validated for believe14 0.1.0.
- **Primary source:** Facco et al., *Estimating the intrinsic dimension of
  datasets by a minimal neighborhood information*, Scientific Reports 7 (2017),
  12140, [doi:10.1038/s41598-017-11873-y](https://doi.org/10.1038/s41598-017-11873-y),
  Equations 5–7 and algorithm steps 1–5.
- **Statistic:** `mu_i = r_2(i)/r_1(i)`. After stable ascending sorting,
  `F_emp(mu_(i)) = i/n`; the reported dimension is the least-squares slope
  through the origin of `-log(1-F_emp)` on `log(mu)`.
- **Tail convention:** the largest 10% of ratios are discarded by default, as
  adopted for the paper's reported experiments. The empirical CDF denominator
  remains the original `n` after trimming.
- **Output:** one positive real `dimension_`, with all sorted ratios and retained
  regression coordinates available for audit.
- **Invariances:** translation, orthogonal transformation, uniform scaling, and
  row permutation. Positive neighbor-distance ties are retained; stable row index
  decides neighbor identity, while only radii enter the statistic.
- **Failure policy:** reject duplicates/zero radii, a trim retaining fewer than
  two values, or retained ratios all equal to one. No jitter is added.
- **Advertised regime:** density is approximately constant on the scale of the
  second neighbor; heavy tails rely on the declared trimming convention.
- **Complexity:** `O(n^2 p)` time and `O(n^2)` memory for exact dense neighbors.
- **Independent evidence:** literal empirical-CDF regression fixture,
  transformation/permutation tests, RNG-independent behavior, singular-tie
  rejection, and two-dimensional uniform recovery. Rdimtools is not an oracle.
