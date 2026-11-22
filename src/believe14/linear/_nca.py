# mypy: disallow-subclassing-any=False
"""Neighborhood components analysis of Goldberger et al. (2004)."""

from __future__ import annotations

from numbers import Integral
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.linalg import (
    apply_centering,
    canonicalize_columns,
    centering_state,
)
from believe14._core.validation import (
    validate_features,
    validate_n_components,
    validate_positive_real,
)

from ._common import component_names, validate_vector


def _nca_loss_gradient(
    flat_map: NDArray[np.float64],
    X: NDArray[np.float64],
    same_class: NDArray[np.bool_],
) -> tuple[float, NDArray[np.float64]]:
    """Negative expected leave-one-out accuracy and its analytic derivative."""
    transform = flat_map.reshape(-1, X.shape[1])
    Z = X @ transform.T
    squared = squareform(pdist(Z, metric="sqeuclidean"))
    if not np.isfinite(squared).all():
        raise FloatingPointError("NCA projected distances overflow float64.")
    np.fill_diagonal(squared, np.inf)
    squared -= squared.min(axis=1, keepdims=True)
    with np.errstate(under="ignore"):
        probabilities = np.exp(-squared)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    success = (probabilities * same_class).sum(axis=1)
    # For an ordered pair (i,j) its coefficient in the gradient is
    # p_ij (1[y_i=y_j] - success_i). Combine both orientations of each pair.
    weights = probabilities * (same_class - success[:, None])
    weights = weights + weights.T
    laplacian_X = weights.sum(axis=1)[:, None] * X - weights @ X
    gradient = (2.0 / len(X)) * Z.T @ laplacian_X
    return -float(success.mean()), np.asarray(gradient.ravel(), dtype=np.float64)


class NeighborhoodComponentsAnalysis(TransformerMixin, BaseEstimator):
    """Learn a linear projection maximizing stochastic nearest-neighbor accuracy.

    The leave-one-out probabilities exclude the observation itself. Optimization
    starts from the principal axes or identity rows in centered, globally scaled
    feature units. The fitted transform uses the original feature units.
    """

    components_: NDArray[np.float64]
    _center_reference_: NDArray[np.float64]
    _center_offset_mean_: NDArray[np.float64]

    def __init__(
        self,
        n_components: int = 2,
        *,
        init: Literal["pca", "identity"] = "pca",
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> None:
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        if y is None:
            raise ValueError("y is required for NeighborhoodComponentsAnalysis.")
        features = validate_features(self, X, reset=True, min_samples=3)
        labels = validate_vector(y, n_samples=len(features))
        classes, encoded, counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        if classes.size < 2 or counts.max() < 2:
            raise ValueError(
                "NCA requires multiple classes and at least one same-class pair."
            )
        k = validate_n_components(
            self.n_components, maximum=min(len(features) - 1, features.shape[1])
        )
        tolerance = validate_positive_real(self.tol, name="tol")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, Integral):
            raise TypeError("max_iter must be an integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")
        reference, offset = centering_state(features)
        centered = apply_centering(features, reference, offset)
        scale = float(np.max(np.abs(centered)))
        if scale == 0:
            raise ValueError("NCA requires nonconstant feature data.")
        normalized = centered / scale
        if self.init == "pca":
            _, _, vt = np.linalg.svd(normalized, full_matrices=False)
            initial = canonicalize_columns(vt[:k].T).T
        elif self.init == "identity":
            initial = np.eye(k, normalized.shape[1])
        else:
            raise ValueError("init must be 'pca' or 'identity'.")
        same = encoded[:, None] == encoded[None, :]
        result = minimize(
            _nca_loss_gradient,
            initial.ravel(),
            args=(normalized, same),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(self.max_iter), "gtol": tolerance, "ftol": 0.0},
        )
        gradient_norm = float(np.linalg.norm(result.jac, ord=np.inf))
        if not np.isfinite(gradient_norm) or not np.isfinite(result.fun):
            raise FloatingPointError("NCA optimization returned a nonfinite result.")
        converged = bool(result.success and gradient_norm <= tolerance)
        with np.errstate(over="raise", invalid="raise"):
            self.components_ = np.asarray(result.x.reshape(k, -1) / scale)
        self.mean_ = reference.copy()
        self._center_reference_ = reference
        self._center_offset_mean_ = offset
        self.classes_ = classes
        self.n_components_ = k
        self.n_iter_ = int(result.nit)
        self.expected_accuracy_ = -float(result.fun)
        self.diagnostics_ = diagnostics(
            "nca_lbfgs",
            converged=converged,
            n_iter=self.n_iter_,
            residual_norm=gradient_norm,
            objective_value=float(result.fun),
            warnings=()
            if converged
            else ("NCA stopped before satisfying the gradient criterion.",),
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "components_")
        features = validate_features(self, X, reset=False, min_samples=1)
        centered = apply_centering(
            features, self._center_reference_, self._center_offset_mean_
        )
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(centered @ self.components_.T)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "components_")
        return component_names(self, input_features)
