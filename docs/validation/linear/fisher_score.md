# FisherScore validation ledger

- Status: validated; supervised feature selector.
- Authority: the frozen 0.1.0 contract is the multiclass, sample-count-weighted
  Fisher between/within ratio below; no legacy implementation is normative.
- Specification: for feature `j`, score
  `sum_c n_c(mu_cj-mu_j)^2 / sum_c sum_{i in c}(x_ij-mu_cj)^2`.
  The class-count weighting is mandatory; neither scatter is averaged.
- Degeneracy: zero within-class and positive between-class scatter yields an
  infinite score; a feature constant globally receives zero. Stable sorting by
  original column order resolves equal scores.
- API: `fit(X,y)`, `get_support`, inherited selector feature names, and dense
  float64 `transform`; exactly `n_features_to_select` columns are returned.
- Evidence: literal hand-computed multiclass ratio, selected-column identity,
  label recoding, ties, constant columns, cloning, and finite-input tests.
- Rdimtools: not used as an oracle. The selector is an inductive column map with
  `O(np)` fitting complexity.
