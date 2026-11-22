# LaplacianScore validation ledger

- **Status:** validated unsupervised selector of original columns.
- **Authority**. He, Cai and Niyogi define the degree-weighted feature quotient {cite:p}`he2005laplacian`. The estimator uses an undirected union of exact Euclidean nearest-neighbor lists, with binary weights or heat weights `exp(-gamma * distance²)`.
- **Specification**. Write `D = diag(W @ 1)`, `L = D - W` and center each feature around its degree-weighted mean to obtain `g`. Increasing values of `g.T @ L @ g / (g.T @ D @ g)` measure increasing variation across edges. The smallest quotients are selected. Computing the numerator as a sum of squared edge differences avoids cancellation for smooth features.
- **Domain**. Supply finite dense rows and a connected graph. Constant columns have zero denominator and are ineligible. Their stored score is the upper bound 2, and `eligible_mask_` distinguishes them from valid maximizers. The fit fails when the request exceeds the number of eligible columns.
- **API**. `fit(X, neighbors=graph)` optionally reuses a compatible exact graph. Ties follow input column order. `get_support()` identifies the selected columns, and `transform` extracts those columns from new observations in their original order.
- **Diagnostics**. The closed-form solver records a warning for constant columns. The quotient is invariant to constant shifts and independent nonzero rescalings of a feature when the graph is held fixed.
- **Evidence**. Independent quadratic forms on a hand-constructed weighted-degree path verify the scores of a smooth and an alternating feature. Additional checks cover constants, equal scores, disconnected graphs, translations and reuse of an exact graph.
- **Rdimtools**. The published objective and explicit quadratic forms provide the reference.
- **Complexity:** `O(n^2(p + log n) + p log p) time; O(np + n^2) memory` including fresh exact neighbor search, dense graph storage, feature scoring and ranking.
