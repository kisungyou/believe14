# mypy: disallow-subclassing-any=False
"""Principal-component and random-projection estimators."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.linalg import (
    apply_centering,
    centered_svd,
    centering_state,
    numerical_rank,
    stable_mean,
)
from believe14._core.validation import (
    make_rng,
    validate_features,
    validate_n_components,
)

from ._common import component_names


class PCA(TransformerMixin, BaseEstimator):
    """Principal component analysis by an economy SVD of centered data."""

    def __init__(self, n_components: int = 2, *, whiten: bool = False) -> None:
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> PCA:
        del y
        Xv = validate_features(self, X, reset=True)
        n_samples, n_features = Xv.shape
        if not isinstance(self.whiten, bool):
            raise TypeError("whiten must be a boolean.")
        k = validate_n_components(self.n_components, maximum=min(n_samples, n_features))
        centered, _, singular_values, vectors_t = centered_svd(Xv)
        rank = numerical_rank(singular_values, shape=Xv.shape)
        if self.whiten and (k > rank or n_samples < 2):
            raise ValueError(
                "whitening requires all retained components to be nonzero."
            )
        self.n_components_ = k
        self.n_samples_ = n_samples
        self.mean_ = stable_mean(Xv)
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.components_ = np.asarray(vectors_t[:k], dtype=np.float64)
        self.singular_values_ = np.asarray(singular_values[:k], dtype=np.float64)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                scaled_singular_values = singular_values / np.sqrt(float(n_samples - 1))
                all_variance = scaled_singular_values**2
            except FloatingPointError as error:
                raise FloatingPointError(
                    "The PCA variances are not representable in float64."
                ) from error
        self.explained_variance_ = np.asarray(all_variance[:k], dtype=np.float64)
        if np.any((singular_values[:k] > 0.0) & (self.explained_variance_ == 0.0)):
            raise FloatingPointError(
                "The PCA variances underflow float64; rescale the input data."
            )
        with np.errstate(over="raise", invalid="raise"):
            try:
                total = float(np.sum(all_variance, dtype=np.float64))
            except FloatingPointError as error:
                raise FloatingPointError(
                    "The total PCA variance is not representable in float64."
                ) from error
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total else np.zeros(k)
        )
        residual = centered - (centered @ self.components_.T) @ self.components_
        centered_scale = float(np.max(np.abs(centered), initial=0.0))
        if centered_scale == 0.0:
            normalized_residual = 0.0
        else:
            denominator = float(np.linalg.norm(centered / centered_scale))
            normalized_residual = float(
                np.linalg.norm(residual / centered_scale) / denominator
            )
        self.diagnostics_ = diagnostics(
            "centered_svd",
            residual_norm=normalized_residual,
            numerical_rank=rank,
            condition_estimate=(
                float(singular_values[0] / singular_values[rank - 1]) if rank else None
            ),
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("components_", "mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        scores = (
            apply_centering(Xv, self._center_reference_, self._center_offset_mean_)
            @ self.components_.T
        )
        if self.whiten:
            scores *= np.sqrt(float(self.n_samples_ - 1)) / self.singular_values_
        return np.asarray(scores, dtype=np.float64)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("components_", "mean_"))
        scores = np.asarray(X, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != self.n_components_:
            raise ValueError(f"X must have exactly {self.n_components_} columns.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("X must contain only finite values.")
        if self.whiten:
            scores = (
                scores * self.singular_values_ / np.sqrt(float(self.n_samples_ - 1))
            )
        return np.asarray(scores @ self.components_ + self.mean_, dtype=np.float64)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "components_")
        return component_names(self, input_features)


class GaussianRandomProjection(TransformerMixin, BaseEstimator):
    """Dense Gaussian random projection with variance ``1 / n_components``."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> GaussianRandomProjection:
        del y
        Xv = validate_features(self, X, reset=True)
        k = validate_n_components(self.n_components, maximum=Xv.shape[1])
        rng = make_rng(self.random_state)
        self.n_components_ = k
        self.components_ = np.asarray(
            rng.normal(size=(k, Xv.shape[1])) / np.sqrt(float(k)),
            dtype=np.float64,
        )
        self.diagnostics_ = diagnostics(
            "gaussian_projection",
            numerical_rank=int(np.linalg.matrix_rank(self.components_)),
            condition_estimate=float(np.linalg.cond(self.components_)),
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "components_")
        Xv = validate_features(self, X, reset=False, min_samples=1)
        return np.asarray(Xv @ self.components_.T, dtype=np.float64)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "components_")
        return component_names(self, input_features)
