# NeighborhoodComponentsAnalysis validation ledger

- **Status:** validated supervised linear transformer.
- **Authority**. Goldberger, Roweis, Hinton and Salakhutdinov maximize expected stochastic leave-one-out classification accuracy {cite:p}`goldberger2004nca`.
- **Specification**. For each observation, set its self probability to zero and distribute probability over the remaining observations in proportion to `exp(-||A(x_i-x_j)||²)`. Sum each row's probabilities for neighbors with the same label and average those sums. The solver minimizes the negative of this mean.
- **Solver**. L-BFGS uses the analytic derivative with respect to every map entry. A per-row subtraction of the smallest eligible squared distance stabilizes the softmax. Optimization uses centered observations divided by their largest absolute entry. The reported map is in original input units.
- **Domain**. Supply dense finite features with variation, at least two classes, at least one within-class pair, and a requested dimension at most `min(p, n-1)`. A singleton class contributes zero to its own success probability. PCA and identity initializations are deterministic. The objective can have multiple stationary points.
- **API and diagnostics**. `transform` applies the learned linear map after training-mean subtraction to any row. The estimator reports the components, class labels, expected accuracy, iteration count, and maximum absolute gradient in optimization units. A fit that stops before reaching the gradient tolerance records a warning.
- **Equivalence**. Orthogonal transformations of output coordinates preserve distances, probabilities and the objective. Label recoding and a common input translation also preserve the objective.
- **Evidence**. Central differences of a separately implemented scalar objective verify every gradient entry. Held-out nearest-neighbor predictions test the learned geometry. An independent distance calculation verifies expected accuracy. Further checks cover initializations, recoding, translation, degeneracy and iteration limits.
- **Rdimtools**. The published objective and its numerical derivative provide the reference.
- **Complexity:** `O(min(np^2,n^2p) + E(n^2(p+k) + npk)) time; O(n^2 + np + kp) memory` for PCA initialization and E objective/gradient evaluations with k output coordinates.
