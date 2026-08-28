# mypy: disallow-subclassing-any=False
"""Linear Gaussian latent-variable estimators."""

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
    numerical_rank,
    stable_center,
    stable_mean,
)
from believe14._core.validation import (
    validate_features,
    validate_n_components,
    validate_positive_real,
)

from ._common import component_names


def _gaussian_log_likelihood(
    sample_covariance: NDArray[np.float64],
    model_covariance: NDArray[np.float64],
    n_samples: int,
) -> float:
    """Log likelihood of centered observations under a Gaussian covariance."""

    scale = float(np.max(np.abs(model_covariance), initial=0.0))
    if scale == 0.0:
        return -np.inf
    scaled_model = model_covariance / scale
    scaled_sample = sample_covariance / scale
    sign, scaled_logdet = np.linalg.slogdet(scaled_model)
    if sign <= 0.0:
        return -np.inf
    logdet = float(scaled_logdet + model_covariance.shape[0] * np.log(scale))
    solved = linalg.solve(
        scaled_model,
        scaled_sample,
        assume_a="pos",
        check_finite=False,
    )
    p = sample_covariance.shape[0]
    return float(
        -0.5 * n_samples * (p * np.log(2.0 * np.pi) + logdet + float(np.trace(solved)))
    )


class FactorAnalysis(TransformerMixin, BaseEstimator):
    """Gaussian factor analysis fitted by the Rubin--Thayer EM updates.

    The fitted transform is the posterior mean of each latent factor. The
    uniqueness variances are constrained below by ``min_noise_variance``;
    active constraints are reported in ``diagnostics_.warnings``.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        tol: float = 1e-6,
        max_iter: int = 500,
        min_noise_variance: float = 1e-10,
    ) -> None:
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.min_noise_variance = min_noise_variance

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> FactorAnalysis:
        del y
        Xv = validate_features(self, X, reset=True)
        n_samples, n_features = Xv.shape
        k = validate_n_components(
            self.n_components, maximum=min(n_features, n_samples - 1)
        )
        tol = validate_positive_real(self.tol, name="tol")
        floor = validate_positive_real(
            self.min_noise_variance,
            name="min_noise_variance",
        )
        if isinstance(self.max_iter, bool) or not isinstance(
            self.max_iter, (int, np.integer)
        ):
            raise TypeError("max_iter must be an integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")

        mean, centered = stable_center(Xv)
        data_scale = float(np.max(np.abs(centered), initial=0.0))
        if data_scale == 0.0:
            raise ValueError("FactorAnalysis requires nonconstant observations.")
        scaled_centered = centered / data_scale
        covariance = scaled_centered.T @ scaled_centered / float(n_samples)
        log_scaled_floor = np.log(floor) - 2.0 * np.log(data_scale)
        if log_scaled_floor > np.log(np.finfo(np.float64).max):
            raise ValueError(
                "min_noise_variance exceeds the variance scale of every feature."
            )
        scaled_floor = (
            0.0
            if log_scaled_floor < np.log(np.nextafter(0.0, 1.0))
            else float(np.exp(log_scaled_floor))
        )
        marginal = np.diag(covariance)
        if np.any(marginal <= scaled_floor):
            raise ValueError(
                "FactorAnalysis requires every feature to have variance greater "
                "than min_noise_variance."
            )
        eigenvalues, eigenvectors = linalg.eigh(covariance, check_finite=False)
        order = np.argsort(eigenvalues)[::-1][:k]
        initial_noise = np.maximum(0.5 * marginal, scaled_floor)
        loadings = eigenvectors[:, order] * np.sqrt(
            np.maximum(eigenvalues[order] - float(np.mean(initial_noise)), scaled_floor)
        )
        noise = initial_noise
        previous = -np.inf
        converged = False
        clipped = False
        log_likelihood = -np.inf

        for iteration in range(1, int(self.max_iter) + 1):
            precision_loadings = loadings / noise[:, None]
            posterior_precision = np.eye(k) + loadings.T @ precision_loadings
            posterior_covariance = linalg.inv(posterior_precision, check_finite=False)
            beta = posterior_covariance @ precision_loadings.T
            cross_moment = covariance @ beta.T
            factor_moment = posterior_covariance + beta @ covariance @ beta.T
            new_loadings = linalg.solve(
                factor_moment,
                cross_moment.T,
                assume_a="pos",
                check_finite=False,
            ).T
            raw_noise = np.diag(covariance - new_loadings @ cross_moment.T)
            clipped = clipped or bool(np.any(raw_noise < scaled_floor))
            new_noise = np.maximum(raw_noise, scaled_floor)
            model_covariance = new_loadings @ new_loadings.T + np.diag(new_noise)
            log_likelihood = _gaussian_log_likelihood(
                covariance, model_covariance, n_samples
            )
            loadings, noise = new_loadings, new_noise
            if iteration > 1 and abs(log_likelihood - previous) <= tol * (
                1.0 + abs(previous)
            ):
                converged = True
                break
            previous = log_likelihood

        loadings = canonicalize_columns(loadings)
        precision_loadings = loadings / noise[:, None]
        posterior_covariance = linalg.inv(
            np.eye(k) + loadings.T @ precision_loadings, check_finite=False
        )
        scaled_loadings = loadings
        scaled_noise = noise
        scaled_posterior_operator = posterior_covariance @ precision_loadings.T
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            loadings = scaled_loadings * data_scale
            noise = scaled_noise * data_scale * data_scale
            posterior_operator = scaled_posterior_operator / data_scale
        if np.any((scaled_noise > 0.0) & (noise == 0.0)):
            raise FloatingPointError(
                "The FactorAnalysis uniqueness variances underflow float64."
            )
        raw_log_likelihood = float(
            log_likelihood - n_samples * n_features * np.log(data_scale)
        )
        self.n_components_ = k
        self.mean_ = mean
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.components_ = loadings.T
        self.loadings_ = loadings
        self.noise_variance_ = noise
        self.posterior_covariance_ = posterior_covariance
        self.posterior_operator_ = posterior_operator
        self.log_likelihood_ = raw_log_likelihood
        self.n_iter_ = iteration
        warning_messages: list[str] = []
        if clipped:
            warning_messages.append(
                "uniqueness variance reached its explicit lower bound"
            )
        if not converged:
            warning_messages.append("EM reached max_iter before satisfying tol")
        scaled_model_covariance = scaled_loadings @ scaled_loadings.T + np.diag(
            scaled_noise
        )
        self.diagnostics_ = diagnostics(
            "rubin_thayer_em",
            converged=converged,
            n_iter=iteration,
            objective_value=-raw_log_likelihood,
            residual_norm=float(
                np.linalg.norm(covariance - scaled_model_covariance)
                / max(float(np.linalg.norm(covariance)), np.finfo(np.float64).tiny)
            ),
            numerical_rank=int(np.linalg.matrix_rank(loadings)),
            condition_estimate=float(np.linalg.cond(scaled_model_covariance)),
            warnings=tuple(warning_messages),
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("posterior_operator_", "mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        centered = apply_centering(
            Xv, self._center_reference_, self._center_offset_mean_
        )
        return np.asarray(centered @ self.posterior_operator_.T)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("loadings_", "mean_"))
        scores = np.asarray(X, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != self.n_components_:
            raise ValueError(f"X must have exactly {self.n_components_} columns.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("X must contain only finite values.")
        return np.asarray(scores @ self.loadings_.T + self.mean_)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "loadings_")
        return component_names(self, input_features)


class ProbabilisticPCA(TransformerMixin, BaseEstimator):
    """Closed-form maximum-likelihood probabilistic PCA."""

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> ProbabilisticPCA:
        del y
        Xv = validate_features(self, X, reset=True)
        n_samples, n_features = Xv.shape
        if n_features < 2:
            raise ValueError(
                "ProbabilisticPCA requires at least 2 features; got 1 feature(s)."
            )
        k = validate_n_components(
            self.n_components, maximum=min(n_samples - 1, n_features - 1)
        )
        centered, _, singular_values, vectors_t = centered_svd(Xv)
        rank = numerical_rank(singular_values, shape=Xv.shape)
        spectrum = np.zeros(n_features, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                scaled_singular_values = singular_values / np.sqrt(float(n_samples))
                spectrum[: singular_values.size] = scaled_singular_values**2
            except FloatingPointError as error:
                raise FloatingPointError(
                    "The PPCA covariance spectrum is not representable in float64."
                ) from error
        noise = float(np.mean(spectrum[k:]))
        retained = spectrum[:k]
        noise_tolerance = (
            max(n_samples, n_features) * np.finfo(np.float64).eps * float(retained[0])
        )
        if noise <= noise_tolerance:
            raise ValueError(
                "The maximum-likelihood noise variance is on the singular zero "
                "boundary; ProbabilisticPCA requires a nonsingular Gaussian model."
            )
        if np.any(retained <= noise):
            raise ValueError(
                "The requested latent dimension is not separated from the "
                "isotropic noise eigenspace."
            )
        directions = vectors_t[:k].T
        loadings = directions * np.sqrt(retained - noise)
        latent_covariance = loadings.T @ loadings + noise * np.eye(k)
        posterior_operator = linalg.solve(
            latent_covariance,
            loadings.T,
            assume_a="pos",
            check_finite=False,
        )
        model_covariance = loadings @ loadings.T + noise * np.eye(n_features)
        covariance = centered.T @ centered / float(n_samples)
        log_likelihood = _gaussian_log_likelihood(
            covariance, model_covariance, n_samples
        )
        self.n_components_ = k
        self.mean_ = stable_mean(Xv)
        self._center_reference_, self._center_offset_mean_ = centering_state(
            Xv, centered
        )
        self.components_ = loadings.T
        self.loadings_ = loadings
        self.noise_variance_ = noise
        self.explained_variance_ = retained
        self.posterior_operator_ = posterior_operator
        self.log_likelihood_ = log_likelihood
        covariance_scale = max(
            float(np.max(np.abs(covariance), initial=0.0)),
            float(np.max(np.abs(model_covariance), initial=0.0)),
        )
        if covariance_scale == 0.0:
            normalized_residual = 0.0
            condition = float(np.linalg.cond(model_covariance))
        else:
            scaled_covariance = covariance / covariance_scale
            scaled_model = model_covariance / covariance_scale
            denominator = float(np.linalg.norm(scaled_covariance))
            normalized_residual = float(
                np.linalg.norm(scaled_covariance - scaled_model)
                / max(denominator, np.finfo(np.float64).tiny)
            )
            condition = float(np.linalg.cond(scaled_model))
        self.diagnostics_ = diagnostics(
            "closed_form_ml",
            objective_value=-log_likelihood,
            residual_norm=normalized_residual,
            numerical_rank=rank,
            condition_estimate=condition,
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("posterior_operator_", "mean_"))
        Xv = validate_features(self, X, reset=False, min_samples=1)
        centered = apply_centering(
            Xv, self._center_reference_, self._center_offset_mean_
        )
        return np.asarray(centered @ self.posterior_operator_.T)

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, ("loadings_", "mean_"))
        scores = np.asarray(X, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != self.n_components_:
            raise ValueError(f"X must have exactly {self.n_components_} columns.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("X must contain only finite values.")
        return np.asarray(scores @ self.loadings_.T + self.mean_)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "loadings_")
        return component_names(self, input_features)
