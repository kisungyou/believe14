# mypy: disallow-subclassing-any=False
"""Supervised linear feature-selection estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.linalg import stable_center
from believe14._core.validation import validate_features, validate_n_components

from ._common import validate_vector


class FisherScore(SelectorMixin, BaseEstimator):
    """Select features using the multiclass Fisher between/within ratio."""

    def __init__(self, n_features_to_select: int = 10) -> None:
        self.n_features_to_select = n_features_to_select

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> FisherScore:
        if y is None:
            raise ValueError("y is required for FisherScore.")
        Xv = validate_features(self, X, reset=True)
        labels = validate_vector(y, n_samples=Xv.shape[0])
        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("y must contain at least two classes.")
        k = validate_n_components(
            self.n_features_to_select,
            maximum=Xv.shape[1],
            name="n_features_to_select",
        )
        _, centered = stable_center(Xv)
        feature_scale = np.max(np.abs(centered), axis=0)
        normalized = np.divide(
            centered,
            feature_scale,
            out=np.zeros_like(centered),
            where=feature_scale > 0.0,
        )
        numerator = np.zeros(Xv.shape[1], dtype=np.float64)
        denominator = np.zeros_like(numerator)
        for index in range(classes.size):
            group = normalized[encoded == index]
            group_mean, deviations = stable_center(group)
            numerator += group.shape[0] * group_mean**2
            denominator += np.sum(deviations**2, axis=0)
        scores = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.inf),
            where=denominator > 0.0,
        )
        scores[(denominator == 0.0) & (numerator == 0.0)] = 0.0
        order = np.argsort(-scores, kind="stable")
        support = np.zeros(Xv.shape[1], dtype=bool)
        support[order[:k]] = True
        self.n_features_to_select_ = k
        self.classes_ = classes
        self.scores_ = scores
        self.ranking_ = order
        self._support_mask = support
        self.diagnostics_ = diagnostics(
            "fisher_between_within_ratio",
            numerical_rank=int(np.count_nonzero(denominator > 0.0)),
            warnings=(
                ("features with zero within-class variance have infinite score",)
                if np.any(np.isinf(scores))
                else ()
            ),
        )
        return self

    def _get_support_mask(self) -> NDArray[np.bool_]:
        check_is_fitted(self, "_support_mask")
        return self._support_mask

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "_support_mask")
        Xv = validate_features(self, X, reset=False, min_samples=1)
        return np.asarray(Xv[:, self._support_mask], dtype=np.float64)
