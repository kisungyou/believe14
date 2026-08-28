# mypy: disallow-subclassing-any=False
"""Independent and sparse component estimators."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.linalg import (
    apply_centering,
    canonicalize_columns,
    centered_svd,
    centering_state,
    numerical_rank,
    stable_center,
    stable_mean,
)
from believe14._core.validation import (
    make_rng,
    validate_features,
    validate_n_components,
    validate_positive_real,
)

from ._common import component_names, soft_threshold


def _symmetric_decorrelation(W: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project a square matrix onto the row-orthogonal matrices."""

    values, vectors = linalg.eigh(W @ W.T, check_finite=False)
    tolerance = W.shape[0] * np.finfo(np.float64).eps * max(1.0, float(values[-1]))
    if np.any(values <= tolerance):
        raise ValueError("The ICA iteration produced a rank-deficient unmixing matrix.")
    inverse_root = (vectors / np.sqrt(values)) @ vectors.T
    return np.asarray(inverse_root @ W, dtype=np.float64)


class FastICA(TransformerMixin, BaseEstimator):
    """Symmetric FastICA with unit-variance whitening and log-cosh contrast."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        alpha: float = 1.0,
        tol: float = 1e-5,
        max_iter: int = 500,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> FastICA:
        del y
        Xv = validate_features(self, X, reset=True)
        n_samples, n_features = Xv.shape
        k = validate_n_components(
            self.n_components, maximum=min(n_samples - 1, n_features)
        )
        alpha = validate_positive_real(
            self.alpha,
            name="alpha",
        )
        if alpha < 1.0 or alpha > 2.0:
            raise ValueError("alpha must be in [1, 2] for log-cosh FastICA.")
        tol = validate_positive_real(self.tol, name="tol")
        if isinstance(self.max_iter, bool) or not isinstance(
            self.max_iter, (int, np.integer)
        ):
            raise TypeError("max_iter must be an integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")

        centered, _, singular_values, vectors_t = centered_svd(Xv)
        rank = numerical_rank(singular_values, shape=Xv.shape)
        if k > rank:
            raise ValueError(
                f"n_components={k} exceeds the centered numerical rank {rank}."
            )
        component_scales = singular_values[:k] / np.sqrt(float(n_samples))
        if np.any(component_scales == 0.0):
            raise ValueError("The ICA whitening scale is numerically zero.")
        whitening = vectors_t[:k] / component_scales[:, None]
        whitened = centered @ whitening.T
        rng = make_rng(self.random_state)
        W = _symmetric_decorrelation(rng.normal(size=(k, k)))
        converged = False
        residual = np.inf
        for _iteration in range(1, int(self.max_iter) + 1):
            projected = whitened @ W.T
            nonlinear = np.tanh(alpha * projected)
            derivative_mean = np.mean(alpha * (1.0 - nonlinear**2), axis=0)
            update = nonlinear.T @ whitened / float(n_samples)
            update -= derivative_mean[:, None] * W
            W_new = _symmetric_decorrelation(update)
            alignment = np.abs(np.diag(W_new @ W.T))
            residual = float(np.max(np.abs(alignment - 1.0)))
            W = W_new
            if residual <= tol:
                converged = True
                break
        iteration = _iteration

        unmixing = W @ whitening
        unmixing = canonicalize_columns(unmixing.T).T
        mixing = np.asarray(linalg.pinv(unmixing, check_finite=False), dtype=np.float64)
        self.n_components_ = k
        self.mean_ = stable_mean(Xv)
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.whitening_ = whitening
        self.components_ = unmixing
        self.mixing_ = mixing
        self.sources_ = centered @ unmixing.T
        self.n_iter_ = iteration
        self.diagnostics_ = diagnostics(
            "symmetric_fastica",
            converged=converged,
            n_iter=iteration,
            residual_norm=residual,
            numerical_rank=rank,
            condition_estimate=float(np.linalg.cond(whitening)),
            warnings=(
                ()
                if converged
                else ("fixed-point iteration reached max_iter before satisfying tol",)
            ),
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("components_", "mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        centered = apply_centering(
            Xv, self._center_reference_, self._center_offset_mean_
        )
        return np.asarray(centered @ self.components_.T)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("mixing_", "mean_"))
        sources = np.asarray(X, dtype=np.float64)
        if sources.ndim != 2 or sources.shape[1] != self.n_components_:
            raise ValueError(f"X must have exactly {self.n_components_} columns.")
        if not np.all(np.isfinite(sources)):
            raise ValueError("X must contain only finite values.")
        return np.asarray(sources @ self.mixing_.T + self.mean_)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "components_")
        return component_names(self, input_features)


class SparsePCA(TransformerMixin, BaseEstimator):
    """Elastic-net sparse PCA using the Zou--Hastie--Tibshirani criterion.

    Alternation uses exact orthogonal Procrustes updates for the reconstruction
    directions and coordinate descent for each elastic-net loading vector.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        alpha: float = 0.01,
        ridge_alpha: float = 1e-3,
        init: Literal["pca", "random"] = "pca",
        tol: float = 1e-6,
        max_iter: int = 500,
        coordinate_max_iter: int = 1000,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.coordinate_max_iter = coordinate_max_iter
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> SparsePCA:
        del y
        Xv = validate_features(self, X, reset=True)
        n_samples, n_features = Xv.shape
        k = validate_n_components(
            self.n_components, maximum=min(n_samples - 1, n_features)
        )
        alpha = validate_positive_real(
            self.alpha,
            name="alpha",
            strict=False,
        )
        ridge = validate_positive_real(
            self.ridge_alpha,
            name="ridge_alpha",
            strict=False,
        )
        tol = validate_positive_real(self.tol, name="tol")
        if self.init not in {"pca", "random"}:
            raise ValueError("init must be 'pca' or 'random'.")
        for name, value in (
            ("max_iter", self.max_iter),
            ("coordinate_max_iter", self.coordinate_max_iter),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")

        mean, centered = stable_center(Xv)
        gram = centered.T @ centered / float(n_samples)
        if self.init == "pca":
            _, _, singular_values, vectors_t = centered_svd(Xv)
            rank = numerical_rank(singular_values, shape=Xv.shape)
            if k > rank:
                raise ValueError(
                    f"n_components={k} exceeds the centered numerical rank {rank}."
                )
            A = vectors_t[:k].T.copy()
        else:
            rng = make_rng(self.random_state)
            A, _ = np.linalg.qr(rng.normal(size=(n_features, k)))
            rank = int(np.linalg.matrix_rank(centered))
            if k > rank:
                raise ValueError(
                    f"n_components={k} exceeds the centered numerical rank {rank}."
                )

        B = A.copy()
        objective = np.inf
        converged = False
        objective_history: list[float] = []
        kkt_relative = np.inf
        for _iteration in range(1, int(self.max_iter) + 1):
            coordinate_converged = True
            for component in range(k):
                target_covariance = gram @ A[:, component]
                beta = B[:, component].copy()
                cd_done = False
                for _ in range(int(self.coordinate_max_iter)):
                    previous_beta = beta.copy()
                    for feature in range(n_features):
                        partial = target_covariance[feature] - (
                            gram[feature] @ beta
                            - gram[feature, feature] * beta[feature]
                        )
                        denominator = gram[feature, feature] + ridge
                        beta[feature] = (
                            0.0
                            if denominator == 0.0
                            else soft_threshold(partial, alpha) / denominator
                        )
                    if np.max(np.abs(beta - previous_beta)) <= tol:
                        cd_done = True
                        break
                coordinate_converged = coordinate_converged and cd_done
                B[:, component] = beta

            cross = gram @ B
            left, _, right_t = linalg.svd(
                cross, full_matrices=False, check_finite=False
            )
            A = left @ right_t
            reconstruction = centered @ B @ A.T
            new_objective = float(
                0.5 * np.sum((centered - reconstruction) ** 2) / n_samples
                + alpha * np.sum(np.abs(B))
                + 0.5 * ridge * np.sum(B**2)
            )
            if np.isfinite(objective):
                monotonic_tolerance = (
                    1e3 * np.finfo(np.float64).eps * max(1.0, abs(objective))
                )
                if new_objective > objective + monotonic_tolerance:
                    raise FloatingPointError(
                        "SparsePCA objective increased beyond roundoff."
                    )
            objective_history.append(new_objective)
            gradient = gram @ B - gram @ A + ridge * B
            violation = np.where(
                B != 0.0,
                np.abs(gradient + alpha * np.sign(B)),
                np.maximum(np.abs(gradient) - alpha, 0.0),
            )
            kkt_scale = max(1.0, float(np.max(np.abs(gram @ A))), alpha)
            kkt_relative = float(np.max(violation) / kkt_scale)
            if (
                np.isfinite(objective)
                and abs(objective - new_objective) <= tol * (1.0 + abs(objective))
                and kkt_relative <= np.sqrt(tol)
            ):
                converged = coordinate_converged
                objective = new_objective
                break
            objective = new_objective
        iteration = _iteration

        canonical_loadings = canonicalize_columns(B)
        # Apply exactly the canonicalization signs to both alternating factors.
        signs = np.sign(np.sum(B * canonical_loadings, axis=0))
        signs[signs == 0.0] = 1.0
        B = canonical_loadings
        A *= signs
        component_norms = np.linalg.norm(B, axis=0)
        loading_tolerance = (
            np.finfo(np.float64).eps
            * max(1.0, float(np.max(component_norms)))
            * max(B.shape)
        )
        if np.any(component_norms <= loading_tolerance):
            raise ValueError(
                "The elastic-net solution contains a zero component; reduce alpha "
                "or request fewer components."
            )
        normalized_loadings = B / component_norms
        residual = centered - centered @ B @ A.T
        self.n_components_ = k
        self.mean_ = mean
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.components_ = normalized_loadings.T
        self.raw_components_ = B.T
        self.component_norms_ = component_norms
        self.reconstruction_components_ = component_norms[:, None] * A.T
        self.n_iter_ = iteration
        self.objective_ = objective
        self.objective_history_ = np.asarray(objective_history, dtype=np.float64)
        self.kkt_residual_ = kkt_relative
        self.reconstruction_error_ = float(np.sum(residual**2) / n_samples)
        self.diagnostics_ = diagnostics(
            "alternating_elastic_net",
            converged=converged,
            n_iter=iteration,
            objective_value=objective,
            residual_norm=kkt_relative,
            numerical_rank=rank,
            condition_estimate=float(np.linalg.cond(gram + ridge * np.eye(n_features))),
            warnings=(
                ()
                if converged
                else ("alternating optimization did not satisfy all tolerances",)
            ),
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
