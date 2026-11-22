# mypy: disallow-subclassing-any=False
"""Laplacian Score feature selection of He, Cai and Niyogi (2005)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.graphs import neighbor_graph, require_connected
from believe14._core.linalg import stable_center
from believe14._core.validation import (
    validate_features,
    validate_n_components,
    validate_n_neighbors,
    validate_positive_real,
)

if TYPE_CHECKING:
    from believe14.neighbors import ExactNeighborGraph


class LaplacianScore(SelectorMixin, BaseEstimator):
    """Select original features with the smallest graph Laplacian quotient.

    Scores measure squared variation across graph edges relative to
    degree-weighted variance. Constant features are ineligible and assigned the
    maximum possible quotient (two). Ties follow original column order.
    """

    def __init__(
        self,
        n_features_to_select: int = 1,
        *,
        n_neighbors: int = 5,
        weighting: Literal["binary", "heat"] = "binary",
        gamma: float = 1.0,
    ) -> None:
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.gamma = gamma

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        neighbors: ExactNeighborGraph | None = None,
    ) -> Self:
        del y
        features = validate_features(self, X, reset=True)
        count = validate_n_components(
            self.n_features_to_select,
            maximum=features.shape[1],
            name="n_features_to_select",
        )
        k = validate_n_neighbors(self.n_neighbors, n_samples=len(features))
        if self.weighting not in ("heat", "binary"):
            raise ValueError("weighting must be 'binary' or 'heat'.")
        gamma = validate_positive_real(self.gamma, name="gamma")
        graph = neighbor_graph(
            features,
            n_neighbors=k,
            weighting=self.weighting,
            gamma=gamma,
            neighbors=neighbors,
        )
        require_connected(graph)
        # Multiplying the graph by a common constant preserves the quotient.
        weights = graph.weights / graph.weights.max()
        degree = weights.sum(axis=1)
        rows, cols = np.nonzero(np.triu(weights, k=1))
        edge_weights = weights[rows, cols]
        _, centered = stable_center(features)
        scales = np.max(np.abs(centered), axis=0)
        eligible = scales > 0
        if np.count_nonzero(eligible) < count:
            raise ValueError("Too few nonconstant features for n_features_to_select.")
        scores = np.full(features.shape[1], 2.0)
        for column in np.flatnonzero(eligible):
            values = centered[:, column] / scales[column]
            values -= (degree @ values) / degree.sum()
            denominator = degree @ (values * values)
            if denominator <= 0:
                raise FloatingPointError(
                    "A nonconstant feature has zero numerical graph variance."
                )
            differences = values[rows] - values[cols]
            scores[column] = edge_weights @ (differences * differences) / denominator
        if not np.isfinite(scores).all():
            raise FloatingPointError("A Laplacian feature score is nonfinite.")
        # lexsort is stable; eligibility is primary, then ascending score.
        order = np.lexsort((scores, ~eligible))
        support = np.zeros(features.shape[1], dtype=bool)
        support[order[:count]] = True
        self.n_features_to_select_ = count
        self.eligible_mask_ = eligible
        self.scores_ = scores
        self.ranking_ = order
        self._support_mask = support
        self.diagnostics_ = diagnostics(
            "laplacian_score",
            warnings=()
            if eligible.all()
            else ("Constant features are ineligible for selection.",),
        )
        return self

    def _get_support_mask(self) -> NDArray[np.bool_]:
        check_is_fitted(self, "_support_mask")
        return self._support_mask

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "_support_mask")
        features = validate_features(self, X, reset=False, min_samples=1)
        return np.asarray(features[:, self._support_mask])
