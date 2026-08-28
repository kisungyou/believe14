# mypy: disallow-subclassing-any=False
"""Supervised and paired-view linear projection estimators."""

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
    centered_svd,
    centering_state,
    stable_center,
    stable_mean,
)
from believe14._core.validation import (
    validate_features,
    validate_n_components,
    validate_positive_real,
)

from ._common import (
    as_second_view,
    component_names,
    validate_targets,
    validate_vector,
)


def _stable_sample_scale(centered: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return column sample standard deviations without raw squaring."""

    maxima = np.max(np.abs(centered), axis=0)
    scaled = np.divide(
        centered,
        maxima,
        out=np.zeros_like(centered),
        where=maxima > 0.0,
    )
    normalized = np.linalg.norm(scaled, axis=0) / np.sqrt(centered.shape[0] - 1)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = normalized * maxima
    if np.any((maxima > 0.0) & (result == 0.0)):
        raise FloatingPointError("A sample standard deviation underflows float64.")
    return np.asarray(result, dtype=np.float64)


class LinearDiscriminantAnalysis(TransformerMixin, BaseEstimator):
    """Fisher/Rao linear discriminant subspace transformer.

    A zero ``regularization`` solves the finite problem on the numerical range
    of the within-class scatter matrix and rejects infinite Fisher directions.
    Positive values scale their ridge by ``trace(Sw) / n_features``, falling
    back to total-scatter scale only when the within-class trace is zero.
    """

    def __init__(self, n_components: int = 2, *, regularization: float = 0.0) -> None:
        self.n_components = n_components
        self.regularization = regularization

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> LinearDiscriminantAnalysis:
        if y is None:
            raise ValueError("y is required for LinearDiscriminantAnalysis.")
        Xv = validate_features(self, X, reset=True)
        labels = validate_vector(y, n_samples=Xv.shape[0])
        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("y must contain at least two classes.")
        k = validate_n_components(
            self.n_components,
            maximum=min(classes.size - 1, Xv.shape[1]),
        )
        regularization = validate_positive_real(
            self.regularization,
            name="regularization",
            strict=False,
        )
        mean, centered = stable_center(Xv)
        data_scale = float(np.max(np.abs(centered), initial=0.0))
        if data_scale == 0.0:
            raise ValueError("X has zero within- and between-class variation.")
        scaled_centered = centered / data_scale
        within = np.zeros((Xv.shape[1], Xv.shape[1]), dtype=np.float64)
        between = np.zeros_like(within)
        class_means = np.empty((classes.size, Xv.shape[1]), dtype=np.float64)
        counts = np.bincount(encoded)
        for index in range(classes.size):
            centered_group = scaled_centered[encoded == index]
            offset, deviations = stable_center(centered_group)
            with np.errstate(over="raise", invalid="raise"):
                class_means[index] = mean + offset * data_scale
            within += deviations.T @ deviations
            between += counts[index] * np.outer(offset, offset)
        within_symmetric = (within + within.T) * 0.5
        within_values, within_vectors = linalg.eigh(
            within_symmetric, check_finite=False
        )
        within_tolerance = (
            max(Xv.shape)
            * np.finfo(np.float64).eps
            * max(0.0, float(within_values[-1]))
        )
        within_support = within_values > within_tolerance
        if regularization == 0.0 and np.any(~within_support):
            null_basis = within_vectors[:, ~within_support]
            null_energy = float(linalg.norm(null_basis.T @ between @ null_basis, ord=2))
            between_scale = float(linalg.norm(between, ord=2))
            null_tolerance = (
                100.0 * max(Xv.shape) * np.finfo(np.float64).eps * between_scale
            )
            if null_energy > null_tolerance:
                raise ValueError(
                    "The between-class scatter has discriminative energy in "
                    "null(within_scatter), yielding infinite Fisher eigenvalues; "
                    "set positive regularization explicitly."
                )
        ridge_scale = float(np.trace(within)) / Xv.shape[1]
        if ridge_scale <= within_tolerance:
            ridge_scale = float(np.trace(within + between)) / Xv.shape[1]
        ridge = regularization * ridge_scale
        denominator = within + ridge * np.eye(Xv.shape[1])
        values, vectors = linalg.eigh((denominator + denominator.T) * 0.5)
        tolerance = (
            max(Xv.shape) * np.finfo(np.float64).eps * max(0.0, float(values[-1]))
        )
        keep = values > tolerance
        rank = int(np.count_nonzero(keep))
        if rank < k:
            raise ValueError(
                f"The within-class scatter has numerical rank {rank}, smaller "
                f"than n_components={k}; set explicit regularization if desired."
            )
        whitener = vectors[:, keep] / np.sqrt(values[keep])
        reduced = whitener.T @ between @ whitener
        eigenvalues, reduced_vectors = linalg.eigh(
            (reduced + reduced.T) * 0.5, check_finite=False
        )
        order = np.argsort(eigenvalues)[::-1][:k]
        scaled_directions = canonicalize_columns(whitener @ reduced_vectors[:, order])
        selected_values = np.asarray(eigenvalues[order], dtype=np.float64)
        residual = (
            between @ scaled_directions
            - denominator @ scaled_directions * selected_values
        )
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            directions = scaled_directions / data_scale
            scatter_factor = data_scale * data_scale
            public_within = within * scatter_factor
            public_between = between * scatter_factor
        if np.any((within != 0.0) & (public_within == 0.0)) or np.any(
            (between != 0.0) & (public_between == 0.0)
        ):
            raise FloatingPointError("The Fisher scatter matrices underflow float64.")

        self.n_components_ = k
        self.classes_ = classes
        self.class_means_ = class_means
        self.class_counts_ = counts
        self.mean_ = mean
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.components_ = directions.T
        self.eigenvalues_ = selected_values
        self.within_scatter_ = public_within
        self.between_scatter_ = public_between
        self.diagnostics_ = diagnostics(
            "range_space_generalized_eigh",
            residual_norm=float(
                np.linalg.norm(residual)
                / max(
                    1.0,
                    float(np.linalg.norm(between)),
                    float(np.linalg.norm(denominator)),
                )
            ),
            numerical_rank=rank,
            condition_estimate=float(values[keep][-1] / values[keep][0]),
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


class CanonicalCorrelationAnalysis(TransformerMixin, BaseEstimator):
    """Hotelling canonical correlation analysis using covariance support SVDs."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X: ArrayLike, Y: ArrayLike) -> CanonicalCorrelationAnalysis:
        Xv = validate_features(self, X, reset=True)
        Yv = as_second_view(Y, n_samples=Xv.shape[0])
        n_samples = Xv.shape[0]
        Xc, Ux, sx, Vxt = centered_svd(Xv)
        Yc, Uy, sy, Vyt = centered_svd(Yv)
        self.x_mean_ = stable_mean(Xv)
        self.y_mean_ = stable_mean(Yv)
        self._x_center_reference_, self._x_center_offset_mean_ = centering_state(Xv, Xc)
        self._y_center_reference_, self._y_center_offset_mean_ = centering_state(Yv, Yc)
        tx = max(Xc.shape) * np.finfo(np.float64).eps * sx[0]
        ty = max(Yc.shape) * np.finfo(np.float64).eps * sy[0]
        rank_x = int(np.count_nonzero(sx > tx))
        rank_y = int(np.count_nonzero(sy > ty))
        maximum = min(rank_x, rank_y)
        k = validate_n_components(self.n_components, maximum=maximum)
        cross = Ux[:, :rank_x].T @ Uy[:, :rank_y]
        left, correlations, right_t = linalg.svd(
            cross, full_matrices=False, check_finite=False
        )
        scale = np.sqrt(float(n_samples - 1))
        x_weights = (Vxt[:rank_x].T / sx[:rank_x]) @ left[:, :k] * scale
        y_weights = (Vyt[:rank_y].T / sy[:rank_y]) @ right_t.T[:, :k] * scale
        x_weights = canonicalize_columns(x_weights)
        # Align Y scores positively with the sign-canonical X scores.
        x_scores = Xc @ x_weights
        y_scores = Yc @ y_weights
        signs = np.sign(np.sum(x_scores * y_scores, axis=0))
        signs[signs == 0.0] = 1.0
        y_weights *= signs
        y_scores *= signs

        self.n_components_ = k
        self.y_n_features_in_ = Yv.shape[1]
        self.x_weights_ = np.asarray(x_weights, dtype=np.float64)
        self.y_weights_ = np.asarray(y_weights, dtype=np.float64)
        self.x_scores_ = np.asarray(x_scores, dtype=np.float64)
        self.y_scores_ = np.asarray(y_scores, dtype=np.float64)
        self.canonical_correlations_ = np.asarray(correlations[:k], dtype=np.float64)
        covariance_error = self.x_scores_.T @ self.x_scores_ / (n_samples - 1) - np.eye(
            k
        )
        self.diagnostics_ = diagnostics(
            "hotelling_support_svd",
            residual_norm=float(np.linalg.norm(covariance_error)),
            numerical_rank=min(rank_x, rank_y),
            condition_estimate=max(
                float(sx[0] / sx[rank_x - 1]),
                float(sy[0] / sy[rank_y - 1]),
            ),
        )
        return self

    def transform(
        self, X: ArrayLike, Y: ArrayLike | None = None
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        check_is_fitted(self, ("x_weights_", "y_weights_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        Xc = apply_centering(Xv, self._x_center_reference_, self._x_center_offset_mean_)
        x_scores = np.asarray(Xc @ self.x_weights_)
        if Y is None:
            return x_scores
        Yv = as_second_view(Y, n_samples=Xv.shape[0])
        if Yv.shape[1] != self.y_n_features_in_:
            raise ValueError(
                f"Y has {Yv.shape[1]} features; expected {self.y_n_features_in_}."
            )
        Yc = apply_centering(Yv, self._y_center_reference_, self._y_center_offset_mean_)
        return x_scores, np.asarray(Yc @ self.y_weights_)

    def fit_transform(
        self, X: ArrayLike, Y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        fitted = self.fit(X, Y)
        return fitted.x_scores_.copy(), fitted.y_scores_.copy()

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "x_weights_")
        return component_names(self, input_features)


class PLSRegression(TransformerMixin, BaseEstimator):
    """NIPALS PLS2 regression with regression-mode X and Y deflation."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        tol: float = 1e-6,
        max_iter: int = 500,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> PLSRegression:
        if y is None:
            raise ValueError("y is required for PLSRegression.")
        Xv = validate_features(self, X, reset=True)
        Yv, y_was_1d = validate_targets(y, n_samples=Xv.shape[0])
        n_samples, n_features = Xv.shape
        if not isinstance(self.scale, bool):
            raise TypeError("scale must be a boolean.")
        k = validate_n_components(
            self.n_components, maximum=min(n_samples - 1, n_features)
        )
        tol = validate_positive_real(self.tol, name="tol")
        if isinstance(self.max_iter, bool) or not isinstance(
            self.max_iter, (int, np.integer)
        ):
            raise TypeError("max_iter must be an integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")
        self.x_mean_, X_centered = stable_center(Xv)
        self.y_mean_, Y_centered = stable_center(Yv)
        self._x_center_reference_, self._x_center_offset_mean_ = centering_state(
            Xv, X_centered
        )
        self._y_center_reference_, self._y_center_offset_mean_ = centering_state(
            Yv, Y_centered
        )
        if self.scale:
            x_scale = _stable_sample_scale(X_centered)
            y_scale = _stable_sample_scale(Y_centered)
            if np.any(x_scale == 0.0):
                raise ValueError("PLSRegression cannot scale a constant X feature.")
            y_scale[y_scale == 0.0] = 1.0
        else:
            x_scale = np.ones(n_features)
            y_scale = np.ones(Yv.shape[1])
        Xk = X_centered / x_scale
        Yk = Y_centered / y_scale
        initial_y_energy = float(np.max(np.sum(Yk**2, axis=0)))
        y_zero_tolerance = max(Yk.shape) * np.finfo(np.float64).eps * initial_y_energy
        W = np.empty((n_features, k), dtype=np.float64)
        P = np.empty_like(W)
        Q = np.empty((Yv.shape[1], k), dtype=np.float64)
        T = np.empty((n_samples, k), dtype=np.float64)
        iterations = np.empty(k, dtype=np.int64)
        all_converged = True

        for component in range(k):
            column_norms = np.sum(Yk**2, axis=0)
            if float(np.max(column_norms)) <= y_zero_tolerance:
                raise ValueError(
                    "The Y residual became numerically zero before all requested "
                    "PLS components were extracted."
                )
            u = Yk[:, int(np.argmax(column_norms))].copy()
            converged = False
            w = np.zeros(n_features)
            for iteration in range(1, int(self.max_iter) + 1):
                old_w = w.copy()
                w = Xk.T @ u / float(u @ u)
                norm_w = float(np.linalg.norm(w))
                if norm_w == 0.0:
                    raise ValueError("NIPALS produced a zero X weight vector.")
                w /= norm_w
                t = Xk @ w
                q_iteration = Yk.T @ t / float(t @ t)
                q_norm_sq = float(q_iteration @ q_iteration)
                if q_norm_sq == 0.0:
                    raise ValueError("NIPALS produced a zero Y weight vector.")
                u = Yk @ q_iteration / q_norm_sq
                if (
                    iteration > 1
                    and min(np.linalg.norm(w - old_w), np.linalg.norm(w + old_w)) <= tol
                ):
                    converged = True
                    break
            t = Xk @ w
            score_norm = float(t @ t)
            p = Xk.T @ t / score_norm
            q = Yk.T @ t / score_norm
            Xk -= np.outer(t, p)
            Yk -= np.outer(t, q)
            W[:, component] = w
            P[:, component] = p
            Q[:, component] = q
            T[:, component] = t
            iterations[component] = iteration
            all_converged = all_converged and converged

        rotations = W @ linalg.inv(P.T @ W, check_finite=False)
        standardized_coef = rotations @ Q.T
        coef = standardized_coef * y_scale[None, :] / x_scale[:, None]
        intercept = self.y_mean_ - self.x_mean_ @ coef
        fitted_centered = (X_centered / x_scale) @ standardized_coef * y_scale
        residual = Y_centered - fitted_centered
        diagnostic_scale = max(
            float(np.max(np.abs(Y_centered), initial=0.0)),
            float(np.max(np.abs(residual), initial=0.0)),
        )
        if diagnostic_scale == 0.0:
            relative_residual = 0.0
        else:
            relative_residual = float(
                np.linalg.norm(residual / diagnostic_scale)
                / max(
                    float(np.linalg.norm(Y_centered / diagnostic_scale)),
                    np.finfo(np.float64).tiny,
                )
            )
        self.n_components_ = k
        self.x_scale_ = x_scale
        self.y_scale_ = y_scale
        self.x_weights_ = W
        self.x_loadings_ = P
        self.y_loadings_ = Q
        self.x_rotations_ = rotations
        self.y_rotations_ = Q @ linalg.pinv(Q.T @ Q, check_finite=False)
        self.x_scores_ = T
        self.n_targets_ = Yv.shape[1]
        self.coef_ = coef
        self._standardized_coef_ = standardized_coef
        self.intercept_ = intercept
        self.n_iter_ = iterations
        self._y_was_1d = y_was_1d
        self.diagnostics_ = diagnostics(
            "nipals_pls2",
            converged=all_converged,
            n_iter=int(np.max(iterations)),
            objective_value=relative_residual * relative_residual,
            residual_norm=relative_residual,
            numerical_rank=int(np.linalg.matrix_rank(T)),
            condition_estimate=float(np.linalg.cond(P.T @ W)),
            warnings=(
                ()
                if all_converged
                else ("at least one NIPALS component reached max_iter",)
            ),
        )
        return self

    def transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        check_is_fitted(self, ("x_rotations_", "x_mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        x_scores = np.asarray(
            apply_centering(Xv, self._x_center_reference_, self._x_center_offset_mean_)
            / self.x_scale_
            @ self.x_rotations_
        )
        if y is None:
            return x_scores
        Yv, _ = validate_targets(y, n_samples=Xv.shape[0])
        if Yv.shape[1] != self.n_targets_:
            raise ValueError(
                f"y has {Yv.shape[1]} targets; expected {self.n_targets_}."
            )
        y_scores = (
            apply_centering(Yv, self._y_center_reference_, self._y_center_offset_mean_)
            / self.y_scale_
            @ self.y_rotations_
        )
        return x_scores, np.asarray(y_scores)

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        if y is None:
            raise ValueError("y is required for PLSRegression.")
        return self.fit(X, y).transform(X, y)

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("_standardized_coef_", "x_mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        prediction = np.asarray(
            (
                apply_centering(
                    Xv, self._x_center_reference_, self._x_center_offset_mean_
                )
                / self.x_scale_
            )
            @ self._standardized_coef_
            * self.y_scale_
            + self.y_mean_
        )
        return prediction[:, 0] if self._y_was_1d else prediction

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "x_rotations_")
        return component_names(self, input_features)
