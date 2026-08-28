# mypy: disallow-subclassing-any=False
"""Kernel and diffusion spectral embeddings."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.kernels import (
    center_cross_kernel,
    center_kernel,
    rbf_kernel,
)
from believe14._core.linalg import (
    apply_centering,
    canonicalize_columns,
    centering_state,
    stable_center,
)
from believe14._core.validation import (
    as_float_matrix,
    validate_features,
    validate_n_components,
    validate_positive_real,
)

from ._common import output_names, validate_unit_interval


def _validate_kernel_matrix(X: ArrayLike) -> NDArray[np.float64]:
    kernel = as_float_matrix(X)
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("A precomputed training kernel must be square.")
    scale = float(np.max(np.abs(kernel), initial=0.0))
    tolerance = np.finfo(np.float64).eps * kernel.shape[0] * scale * 100.0
    if not np.allclose(kernel, kernel.T, rtol=0.0, atol=tolerance):
        raise ValueError("A precomputed training kernel must be symmetric.")
    return np.asarray(kernel * 0.5 + kernel.T * 0.5, dtype=np.float64)


class KernelPCA(TransformerMixin, BaseEstimator):
    """Kernel PCA with explicit Gram centering and Nyström projection."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        kernel: Literal["linear", "rbf", "poly", "precomputed"] = "rbf",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
    ) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _resolve_kernel_parameters(self, n_features: int) -> None:
        if self.kernel not in {"linear", "rbf", "poly", "precomputed"}:
            raise ValueError(
                "kernel must be 'linear', 'rbf', 'poly', or 'precomputed'."
            )
        gamma = (
            1.0 / n_features
            if self.gamma is None
            else validate_positive_real(self.gamma, name="gamma")
        )
        if isinstance(self.degree, bool) or not isinstance(self.degree, Integral):
            raise TypeError("degree must be an integer.")
        if self.degree < 1:
            raise ValueError("degree must be positive.")
        if isinstance(self.coef0, bool) or not isinstance(self.coef0, Real):
            raise TypeError("coef0 must be a real scalar.")
        coef0 = float(self.coef0)
        if not np.isfinite(coef0):
            raise ValueError("coef0 must be finite.")
        self.gamma_: float = gamma
        self.coef0_: float = coef0

    def _kernel(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        if self.kernel == "linear":
            other = X if Y is None else Y
            return np.asarray(X @ other.T, dtype=np.float64)
        if self.kernel == "rbf":
            return rbf_kernel(X, Y, gamma=self.gamma_)
        if self.kernel == "poly":
            other = X if Y is None else Y
            with np.errstate(over="raise", invalid="raise"):
                base = self.gamma_ * (X @ other.T) + self.coef0_
                result = base ** int(self.degree)
            return np.asarray(result, dtype=np.float64)
        raise RuntimeError("Precomputed kernels are supplied by the caller.")

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> KernelPCA:
        """Fit the centered kernel eigensystem."""

        del y
        if self.kernel == "precomputed":
            kernel = _validate_kernel_matrix(X)
            self.n_features_in_ = kernel.shape[1]
            self._resolve_kernel_parameters(kernel.shape[1])
        else:
            features = validate_features(self, X, reset=True, copy=True)
            self._resolve_kernel_parameters(features.shape[1])
            if self.kernel == "linear":
                feature_mean, kernel_features = stable_center(features)
                self._feature_mean_ = feature_mean[None, :]
                (
                    self._center_reference_,
                    self._center_offset_mean_,
                ) = centering_state(features, kernel_features)
            else:
                kernel_features = features
            kernel = self._kernel(kernel_features)
            self._fit_features_ = kernel_features
        n_components = validate_n_components(
            self.n_components, maximum=kernel.shape[0] - 1
        )
        centered, column_mean, total_mean = center_kernel(kernel)
        values, vectors = linalg.eigh(centered, check_finite=False)
        order = np.argsort(values, kind="stable")[::-1]
        values = np.asarray(values[order], dtype=np.float64)
        vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
        scale = float(linalg.norm(centered, ord=2))
        if scale == 0.0:
            raise ValueError("The centered kernel has zero positive numerical rank.")
        threshold = np.finfo(np.float64).eps * len(kernel) * scale * 100.0
        negative = values < -threshold
        if np.any(negative):
            raise ValueError(
                "The centered kernel is significantly indefinite; KernelPCA "
                "requires a positive-semidefinite kernel."
            )
        if int(np.count_nonzero(values > threshold)) < n_components:
            raise ValueError(
                "n_components exceeds the positive numerical rank of the kernel."
            )
        selected_values = values[:n_components]
        selected_vectors = vectors[:, :n_components]
        dual = selected_vectors / np.sqrt(selected_values)[None, :]
        residual = centered @ selected_vectors - selected_vectors * selected_values
        self.embedding_ = selected_vectors * np.sqrt(selected_values)[None, :]
        self.eigenvalues_ = selected_values
        self.eigenvectors_ = selected_vectors
        self.dual_coef_ = dual
        self.kernel_matrix_ = kernel
        self._kernel_column_mean_ = column_mean
        self._kernel_total_mean_ = total_mean
        self.n_samples_fit_ = kernel.shape[0]
        self.n_components_ = n_components
        self.diagnostics_ = diagnostics(
            "symmetric_eigh",
            residual_norm=float(linalg.norm(residual) / scale),
            numerical_rank=int(np.count_nonzero(values > threshold)),
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project query samples by the centered-kernel Nyström formula."""

        check_is_fitted(self, "dual_coef_")
        if self.kernel == "precomputed":
            cross_kernel = as_float_matrix(X, min_samples=1)
            if cross_kernel.shape[1] != self.n_samples_fit_:
                raise ValueError(
                    "A query precomputed kernel must have one column per fitted "
                    "observation."
                )
        else:
            features = validate_features(self, X, reset=False, min_samples=1)
            if self.kernel == "linear":
                features = apply_centering(
                    features,
                    self._center_reference_,
                    self._center_offset_mean_,
                )
            cross_kernel = self._kernel(features, self._fit_features_)
        centered = center_cross_kernel(
            cross_kernel,
            training_column_mean=self._kernel_column_mean_,
            training_total_mean=self._kernel_total_mean_,
        )
        return np.asarray(centered @ self.dual_coef_, dtype=np.float64)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("kernelpca", self.n_components_)


class DiffusionMap(TransformerMixin, BaseEstimator):
    """Density-normalized diffusion maps with a Nyström extension."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        gamma: float | None = None,
        alpha: float = 1.0,
        diffusion_time: int = 1,
    ) -> None:
        self.n_components = n_components
        self.gamma = gamma
        self.alpha = alpha
        self.diffusion_time = diffusion_time

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> DiffusionMap:
        """Fit the symmetric conjugate of the diffusion operator."""

        del y
        features = validate_features(self, X, reset=True, copy=True)
        n_components = validate_n_components(
            self.n_components, maximum=features.shape[0] - 1
        )
        gamma = (
            1.0 / features.shape[1]
            if self.gamma is None
            else validate_positive_real(self.gamma, name="gamma")
        )
        alpha = validate_unit_interval(self.alpha, name="alpha")
        if isinstance(self.diffusion_time, bool) or not isinstance(
            self.diffusion_time, Integral
        ):
            raise TypeError("diffusion_time must be an integer.")
        if self.diffusion_time < 0:
            raise ValueError("diffusion_time must be nonnegative.")
        kernel = rbf_kernel(features, gamma=gamma)
        support_components = int(
            connected_components(csr_matrix(kernel > 0.0), directed=False)[0]
        )
        if support_components != 1:
            raise ValueError(
                "The diffusion kernel is numerically disconnected; gamma and "
                "the data scale do not define an irreducible diffusion operator."
            )
        density = np.sum(kernel, axis=1, dtype=np.float64)
        if np.any(density <= 0.0):
            raise FloatingPointError("The kernel density contains a nonpositive row.")
        normalized_kernel = kernel / (
            density[:, None] ** alpha * density[None, :] ** alpha
        )
        degree = np.sum(normalized_kernel, axis=1, dtype=np.float64)
        if np.any(degree <= 0.0):
            raise FloatingPointError(
                "The density-normalized diffusion kernel has a nonpositive degree."
            )
        symmetric = normalized_kernel / np.sqrt(degree[:, None] * degree[None, :])
        values, vectors = linalg.eigh(symmetric, check_finite=False)
        order = np.argsort(values, kind="stable")[::-1]
        values = np.asarray(values[order], dtype=np.float64)
        vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
        scale = max(1.0, float(linalg.norm(symmetric, ord=2)))
        threshold = np.finfo(np.float64).eps * len(features) * scale * 100.0
        selected_values = values[1 : n_components + 1]
        if np.any(np.abs(selected_values) <= threshold):
            raise ValueError(
                "A selected diffusion eigenvalue is numerically zero and has no "
                "stable Nyström extension."
            )
        normalization = np.sqrt(float(np.sum(degree, dtype=np.float64)))
        right_vectors = vectors / np.sqrt(degree)[:, None] * normalization
        selected_right = right_vectors[:, 1 : n_components + 1]
        embedding = (
            selected_right * (selected_values ** int(self.diffusion_time))[None, :]
        )
        residual = (
            symmetric @ vectors[:, : n_components + 1]
            - vectors[:, : n_components + 1] * values[: n_components + 1]
        )
        self.embedding_: NDArray[np.float64] = np.asarray(embedding, dtype=np.float64)
        self.eigenvalues_: NDArray[np.float64] = selected_values
        self.eigenvectors_: NDArray[np.float64] = selected_right
        self.diffusion_operator_ = normalized_kernel / degree[:, None]
        self.kernel_density_: NDArray[np.float64] = density
        self.diffusion_degree_ = degree
        self._fit_features_: NDArray[np.float64] = features
        self.gamma_: float = gamma
        self.alpha_: float = alpha
        self.n_components_: int = n_components
        self.diagnostics_ = diagnostics(
            "symmetric_markov_eigh",
            residual_norm=float(linalg.norm(residual) / scale),
            numerical_rank=int(np.count_nonzero(np.abs(values) > threshold)) - 1,
        )
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Extend fitted diffusion eigenfunctions by the Nyström equation."""

        check_is_fitted(self, "eigenvectors_")
        features = validate_features(self, X, reset=False, min_samples=1)
        kernel = rbf_kernel(features, self._fit_features_, gamma=self.gamma_)
        query_density = np.sum(kernel, axis=1, dtype=np.float64)
        if np.any(query_density <= 0.0):
            raise FloatingPointError("A query has zero kernel density.")
        normalized = kernel / (
            query_density[:, None] ** self.alpha_
            * self.kernel_density_[None, :] ** self.alpha_
        )
        query_degree = np.sum(normalized, axis=1, dtype=np.float64)
        if np.any(query_degree <= 0.0):
            raise FloatingPointError("A query has zero normalized diffusion degree.")
        transition = normalized / query_degree[:, None]
        extended = transition @ self.eigenvectors_ / self.eigenvalues_[None, :]
        return np.asarray(
            extended * (self.eigenvalues_ ** int(self.diffusion_time))[None, :],
            dtype=np.float64,
        )

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("diffusionmap", self.n_components_)
