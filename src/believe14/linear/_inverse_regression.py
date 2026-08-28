# mypy: disallow-subclassing-any=False
"""Sliced inverse-regression estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.linalg import (
    apply_centering,
    canonicalize_columns,
    centering_state,
    stable_center,
    stable_mean,
)
from believe14._core.validation import validate_features, validate_n_components

from ._common import component_names, covariance_support, response_slices


def _inverse_regression_basis(
    X: NDArray[np.float64],
    y: ArrayLike,
    *,
    n_components: int,
    n_slices: int,
    method: str,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    int,
    float,
    float,
]:
    """Compute SIR or SAVE directions in standardized predictor coordinates."""

    n_samples = X.shape[0]
    _, centered = stable_center(X)
    whitening, _, rank, condition = covariance_support(centered)
    slices = response_slices(y, n_samples=n_samples, n_slices=n_slices)
    groups = np.unique(slices)
    maximum = min(rank, groups.size - 1 if method == "sir" else rank)
    k = validate_n_components(n_components, maximum=maximum)
    standardized = centered @ whitening
    matrix = np.zeros((rank, rank), dtype=np.float64)
    identity = np.eye(rank)
    for group in groups:
        subset = standardized[slices == group]
        probability = subset.shape[0] / float(n_samples)
        if method == "sir":
            group_mean = stable_mean(subset)
            matrix += probability * np.outer(group_mean, group_mean)
        else:
            if subset.shape[0] < 2:
                raise ValueError(
                    "SAVE requires at least two observations in every slice."
                )
            _, deviations = stable_center(subset)
            group_covariance = deviations.T @ deviations / float(subset.shape[0])
            difference = identity - group_covariance
            matrix += probability * (difference @ difference)
    eigenvalues, eigenvectors = linalg.eigh(
        (matrix + matrix.T) * 0.5, check_finite=False
    )
    order = np.argsort(eigenvalues)[::-1][:k]
    reduced_directions = eigenvectors[:, order]
    directions = canonicalize_columns(whitening @ reduced_directions)
    selected = np.asarray(eigenvalues[order], dtype=np.float64)
    residual = matrix @ reduced_directions - reduced_directions * selected
    residual_norm = float(
        np.linalg.norm(residual) / max(1.0, float(np.linalg.norm(matrix)))
    )
    return directions, selected, slices, rank, condition, residual_norm


class SlicedInverseRegression(TransformerMixin, BaseEstimator):
    """Li's sliced inverse regression with tie-preserving quantile slices."""

    def __init__(self, n_components: int = 2, *, n_slices: int = 10) -> None:
        self.n_components = n_components
        self.n_slices = n_slices

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> SlicedInverseRegression:
        if y is None:
            raise ValueError("y is required for SlicedInverseRegression.")
        Xv = validate_features(self, X, reset=True)
        directions, values, slices, rank, condition, residual = (
            _inverse_regression_basis(
                Xv,
                y,
                n_components=self.n_components,
                n_slices=self.n_slices,
                method="sir",
            )
        )
        self.n_components_ = directions.shape[1]
        self.mean_, centered = stable_center(Xv)
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.components_ = directions.T
        self.eigenvalues_ = values
        self.slice_labels_ = slices
        self.n_slices_ = int(np.unique(slices).size)
        self.diagnostics_ = diagnostics(
            "standardized_sir_eigh",
            residual_norm=residual,
            numerical_rank=rank,
            condition_estimate=condition,
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("components_", "mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        centered = apply_centering(
            Xv, self._center_reference_, self._center_offset_mean_
        )
        return np.asarray(centered @ self.components_.T)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "components_")
        return component_names(self, input_features)


class SlicedAverageVarianceEstimation(TransformerMixin, BaseEstimator):
    """Cook's SAVE using maximum-likelihood covariance within each slice."""

    def __init__(self, n_components: int = 2, *, n_slices: int = 5) -> None:
        self.n_components = n_components
        self.n_slices = n_slices

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> SlicedAverageVarianceEstimation:
        if y is None:
            raise ValueError("y is required for SlicedAverageVarianceEstimation.")
        Xv = validate_features(self, X, reset=True)
        directions, values, slices, rank, condition, residual = (
            _inverse_regression_basis(
                Xv,
                y,
                n_components=self.n_components,
                n_slices=self.n_slices,
                method="save",
            )
        )
        self.n_components_ = directions.shape[1]
        self.mean_, centered = stable_center(Xv)
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.components_ = directions.T
        self.eigenvalues_ = values
        self.slice_labels_ = slices
        self.n_slices_ = int(np.unique(slices).size)
        self.diagnostics_ = diagnostics(
            "standardized_save_eigh",
            residual_norm=residual,
            numerical_rank=rank,
            condition_estimate=condition,
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("components_", "mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        centered = apply_centering(
            Xv, self._center_reference_, self._center_offset_mean_
        )
        return np.asarray(centered @ self.components_.T)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "components_")
        return component_names(self, input_features)
