# mypy: disallow-subclassing-any=False
"""Distance-preserving nonlinear embeddings."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.distances import pairwise_distances
from believe14._core.validation import (
    as_float_matrix,
    validate_features,
    validate_n_components,
    validate_positive_real,
)
from believe14.api import FitDiagnostics

from ._common import (
    FloatMatrix,
    classical_embedding,
    initialize_embedding,
    output_names,
    smacof,
    validate_dissimilarities,
)


class ClassicalMDS(BaseEstimator):
    """Classical metric scaling by double centering squared dissimilarities."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        dissimilarity: Literal["euclidean", "precomputed"] = "euclidean",
    ) -> None:
        self.n_components = n_components
        self.dissimilarity = dissimilarity

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> ClassicalMDS:
        """Fit the transductive embedding."""

        del y
        features = None
        if self.dissimilarity == "euclidean":
            features = validate_features(self, X, reset=True)
        distances = validate_dissimilarities(
            X, dissimilarity=self.dissimilarity, feature_matrix=features
        )
        if self.dissimilarity == "precomputed":
            self.n_features_in_ = distances.shape[1]
        embedding, spectrum, gram, residual, rank, warnings = classical_embedding(
            distances, self.n_components
        )
        self.embedding_ = embedding
        self.eigenvalues_ = spectrum
        self.gram_matrix_ = gram
        self.dissimilarity_matrix_ = distances
        self.n_components_ = self.n_components
        self.diagnostics_ = diagnostics(
            "symmetric_eigh",
            residual_norm=residual,
            numerical_rank=rank,
            warnings=warnings,
        )
        return self

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Fit and return the training coordinates."""

        return self.fit(X, y).embedding_.copy()

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("classicalmds", self.n_components_)


class MetricMDS(BaseEstimator):
    """Metric MDS minimizing complete unweighted raw stress with SMACOF."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        dissimilarity: Literal["euclidean", "precomputed"] = "euclidean",
        init: Literal["classical", "random"] = "classical",
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> MetricMDS:
        """Fit the transductive raw-stress embedding."""

        del y
        features = None
        if self.dissimilarity == "euclidean":
            features = validate_features(self, X, reset=True)
        distances = validate_dissimilarities(
            X, dissimilarity=self.dissimilarity, feature_matrix=features
        )
        if self.dissimilarity == "precomputed":
            self.n_features_in_ = distances.shape[1]
        n_components = validate_n_components(
            self.n_components, maximum=distances.shape[0] - 1
        )
        initial = initialize_embedding(
            distances,
            n_components,
            init=self.init,
            random_state=self.random_state,
        )
        embedding, objective, n_iter, converged, residual = smacof(
            distances, initial, max_iter=self.max_iter, tol=self.tol
        )
        self.embedding_ = embedding
        self.dissimilarity_matrix_ = distances
        self.stress_ = objective
        self.n_iter_ = n_iter
        self.n_components_ = n_components
        warnings = () if converged else ("SMACOF reached max_iter.",)
        self.diagnostics_ = diagnostics(
            "smacof",
            converged=converged,
            n_iter=n_iter,
            residual_norm=residual,
            objective_value=objective,
            warnings=warnings,
        )
        return self

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Fit and return the training coordinates."""

        return self.fit(X, y).embedding_.copy()

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("metricmds", self.n_components_)


class SammonMapping(BaseEstimator):
    """Sammon mapping with the paper's normalized inverse-distance stress."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        dissimilarity: Literal["euclidean", "precomputed"] = "euclidean",
        init: Literal["classical", "random"] = "classical",
        max_iter: int = 300,
        tol: float = 1e-7,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _objective_gradient(
        flat_embedding: NDArray[np.float64],
        distances: FloatMatrix,
        n_components: int,
    ) -> tuple[float, NDArray[np.float64]]:
        embedding = flat_embedding.reshape(distances.shape[0], n_components)
        distance_scale = float(np.max(distances, initial=0.0))
        if distance_scale <= 0.0:
            raise ValueError("Sammon stress requires positive dissimilarities.")
        scaled_distances = distances / distance_scale
        scaled_embedding = embedding / distance_scale
        embedded = pairwise_distances(scaled_embedding)
        upper = np.triu_indices_from(distances, k=1)
        normalization = float(np.sum(scaled_distances[upper], dtype=np.float64))
        residual = embedded - scaled_distances
        objective = float(
            np.sum(
                (residual[upper] ** 2) / scaled_distances[upper],
                dtype=np.float64,
            )
            / normalization
        )
        scale = float(np.max(embedded, initial=0.0))
        threshold = np.finfo(np.float64).eps * scale * 100.0
        coincident = (embedded <= threshold) & (scaled_distances > 0.0)
        np.fill_diagonal(coincident, False)
        if np.any(coincident):
            raise FloatingPointError(
                "Sammon stress is nondifferentiable at coincident embedded points."
            )
        coefficients = np.zeros_like(scaled_distances)
        mask = ~np.eye(distances.shape[0], dtype=bool)
        coefficients[mask] = residual[mask] / (scaled_distances[mask] * embedded[mask])
        laplacian = -coefficients
        laplacian[np.diag_indices_from(laplacian)] = np.sum(
            coefficients, axis=1, dtype=np.float64
        )
        gradient = 2.0 * (laplacian @ scaled_embedding) / normalization / distance_scale
        return objective, np.asarray(gradient.ravel(), dtype=np.float64)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> SammonMapping:
        """Fit the transductive Sammon embedding."""

        del y
        features = None
        if self.dissimilarity == "euclidean":
            features = validate_features(self, X, reset=True)
        distances = validate_dissimilarities(
            X, dissimilarity=self.dissimilarity, feature_matrix=features
        )
        if self.dissimilarity == "precomputed":
            self.n_features_in_ = distances.shape[1]
        n_components = validate_n_components(
            self.n_components, maximum=distances.shape[0] - 1
        )
        max_iter = self.max_iter
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
            raise TypeError("max_iter must be an integer.")
        if max_iter < 1:
            raise ValueError("max_iter must be positive.")
        tolerance = validate_positive_real(self.tol, name="tol")
        upper = np.triu_indices_from(distances, k=1)
        if np.any(distances[upper] <= 0.0):
            raise ValueError(
                "Sammon mapping requires strictly positive dissimilarities "
                "between distinct observations."
            )
        initial = initialize_embedding(
            distances,
            n_components,
            init=self.init,
            random_state=self.random_state,
        )
        distance_scale = float(np.max(distances, initial=0.0))
        scaled_distances = distances / distance_scale
        scaled_initial = initial / distance_scale
        result = optimize.minimize(
            self._objective_gradient,
            scaled_initial.ravel(),
            args=(scaled_distances, n_components),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": int(max_iter), "ftol": tolerance, "gtol": tolerance},
        )
        scaled_embedding = np.asarray(result.x, dtype=np.float64).reshape(
            distances.shape[0], n_components
        )
        scaled_embedding -= np.mean(scaled_embedding, axis=0, keepdims=True)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            embedding = scaled_embedding * distance_scale
        gradient_norm = float(linalg.norm(np.asarray(result.jac, dtype=np.float64)))
        self.embedding_: NDArray[np.float64] = embedding
        self.dissimilarity_matrix_ = distances
        self.stress_ = float(result.fun)
        self.n_iter_ = int(result.nit)
        self.n_components_: int = n_components
        converged = bool(result.success)
        warnings = () if converged else (str(result.message),)
        self.diagnostics_ = diagnostics(
            "lbfgsb",
            converged=converged,
            n_iter=self.n_iter_,
            residual_norm=gradient_norm,
            objective_value=self.stress_,
            warnings=warnings,
        )
        return self

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Fit and return the training coordinates."""

        return self.fit(X, y).embedding_.copy()

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("sammonmapping", self.n_components_)


class FastMap(TransformerMixin, BaseEstimator):
    """FastMap using deterministic farthest-pivot selection."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        dissimilarity: Literal["euclidean", "precomputed"] = "euclidean",
        pivot_iterations: int = 5,
    ) -> None:
        self.n_components = n_components
        self.dissimilarity = dissimilarity
        self.pivot_iterations = pivot_iterations

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> FastMap:
        """Fit pivots and training coordinates."""

        del y
        features = None
        if self.dissimilarity == "euclidean":
            features = validate_features(self, X, reset=True, copy=True)
        distances = validate_dissimilarities(
            X, dissimilarity=self.dissimilarity, feature_matrix=features
        )
        if self.dissimilarity == "precomputed":
            self.n_features_in_ = distances.shape[1]
        n_components = validate_n_components(
            self.n_components, maximum=distances.shape[0] - 1
        )
        if isinstance(self.pivot_iterations, bool) or not isinstance(
            self.pivot_iterations, Integral
        ):
            raise TypeError("pivot_iterations must be an integer.")
        if self.pivot_iterations < 1:
            raise ValueError("pivot_iterations must be positive.")
        distance_scale = float(np.max(distances, initial=0.0))
        if distance_scale <= 0.0:
            raise ValueError("FastMap requires at least one positive dissimilarity.")
        scaled_distances = distances / distance_scale
        residual_squared = scaled_distances * scaled_distances
        scaled_embedding = np.zeros(
            (distances.shape[0], n_components), dtype=np.float64
        )
        pivots = np.zeros((n_components, 2), dtype=np.int64)
        scaled_pivot_distances = np.zeros(n_components, dtype=np.float64)
        active = 0
        residual_tolerance = distances.shape[0] * np.finfo(np.float64).eps
        for component in range(n_components):
            first = 0
            for _ in range(int(self.pivot_iterations)):
                second = int(np.argmax(residual_squared[first]))
                updated_first = int(np.argmax(residual_squared[second]))
                if updated_first == first:
                    break
                first = updated_first
            second = int(np.argmax(residual_squared[first]))
            pivot_distance = float(np.sqrt(max(residual_squared[first, second], 0.0)))
            pivots[component] = (first, second)
            scaled_pivot_distances[component] = pivot_distance
            if residual_squared[first, second] <= residual_tolerance:
                pivots[component:] = first
                scaled_pivot_distances[component:] = 0.0
                break
            numerator = (
                residual_squared[:, first]
                + pivot_distance * pivot_distance
                - residual_squared[:, second]
            )
            coordinate = numerator / (2.0 * pivot_distance)
            scaled_embedding[:, component] = coordinate
            differences = coordinate[:, None] - coordinate[None, :]
            updated = residual_squared - differences * differences
            numerical_tolerance = residual_tolerance
            if np.any(updated < -numerical_tolerance):
                raise FloatingPointError(
                    "FastMap residual squared distances became significantly negative."
                )
            residual_squared = np.maximum(updated, 0.0)
            np.fill_diagonal(residual_squared, 0.0)
            active += 1

        with np.errstate(over="raise", invalid="raise", under="ignore"):
            embedding = scaled_embedding * distance_scale
            pivot_distances = scaled_pivot_distances * distance_scale
        if active > 0 and not np.any(embedding[:, :active]):
            raise FloatingPointError("The FastMap embedding underflows float64.")
        self.embedding_: NDArray[np.float64] = embedding
        self.pivot_indices_: NDArray[np.int64] = pivots
        self.pivot_distances_: NDArray[np.float64] = pivot_distances
        self._scaled_pivot_distances_: NDArray[np.float64] = scaled_pivot_distances
        self._scaled_embedding_: NDArray[np.float64] = scaled_embedding
        self._distance_scale_: float = distance_scale
        self.n_active_components_: int = active
        self.n_components_: int = n_components
        self.n_samples_fit_: int = distances.shape[0]
        self.dissimilarity_matrix_ = distances
        if features is not None:
            self._fit_features_: NDArray[np.float64] = features
        warning = ()
        if active < n_components:
            warning = ("FastMap exhausted the positive residual metric rank.",)
        self.diagnostics_: FitDiagnostics = diagnostics(
            "fastmap_pivots",
            numerical_rank=active,
            warnings=warning,
        )
        return self

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: object
    ) -> NDArray[np.float64]:
        """Fit and return exact stored training coordinates."""

        if fit_params:
            raise TypeError("FastMap.fit_transform does not accept fit parameters.")
        return self.fit(X, y).embedding_.copy()

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Embed queries using feature or query-to-training pivot distances."""

        check_is_fitted(self, "pivot_indices_")
        query: NDArray[np.float64] | None = None
        if self.dissimilarity == "precomputed":
            cross_distances = as_float_matrix(X, min_samples=1)
            if cross_distances.shape[1] != self.n_samples_fit_:
                raise ValueError(
                    "A query dissimilarity matrix must have one column per fitted "
                    "observation."
                )
            if np.any(cross_distances < 0.0):
                raise ValueError("Query dissimilarities must be nonnegative.")
        else:
            query = validate_features(self, X, reset=False, min_samples=1)
            cross_distances = np.empty((query.shape[0], 0), dtype=np.float64)
        scaled_result = np.zeros(
            (cross_distances.shape[0], self.n_components_), dtype=np.float64
        )
        for component in range(self.n_active_components_):
            first, second = self.pivot_indices_[component]
            pivot_distance = self._scaled_pivot_distances_[component]
            if query is None:
                with np.errstate(over="raise", invalid="raise"):
                    first_squared = (
                        cross_distances[:, first] / self._distance_scale_
                    ) ** 2
                    second_squared = (
                        cross_distances[:, second] / self._distance_scale_
                    ) ** 2
            else:
                first_distances = pairwise_distances(
                    query, self._fit_features_[[first]]
                )[:, 0]
                second_distances = pairwise_distances(
                    query, self._fit_features_[[second]]
                )[:, 0]
                with np.errstate(over="raise", invalid="raise"):
                    first_squared = (first_distances / self._distance_scale_) ** 2
                    second_squared = (second_distances / self._distance_scale_) ** 2
            if component:
                first_squared -= np.sum(
                    (
                        scaled_result[:, :component]
                        - self._scaled_embedding_[first, :component]
                    )
                    ** 2,
                    axis=1,
                )
                second_squared -= np.sum(
                    (
                        scaled_result[:, :component]
                        - self._scaled_embedding_[second, :component]
                    )
                    ** 2,
                    axis=1,
                )
            scale = max(
                float(np.max(np.abs(first_squared), initial=0.0)),
                float(np.max(np.abs(second_squared), initial=0.0)),
            )
            tolerance = np.finfo(np.float64).eps * scale * 1e3
            if np.any(first_squared < -tolerance) or np.any(
                second_squared < -tolerance
            ):
                raise FloatingPointError(
                    "FastMap query residual squared distances became negative."
                )
            first_squared = np.maximum(first_squared, 0.0)
            second_squared = np.maximum(second_squared, 0.0)
            scaled_result[:, component] = (
                first_squared + pivot_distance * pivot_distance - second_squared
            ) / (2.0 * pivot_distance)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = scaled_result * self._distance_scale_
        return np.asarray(result, dtype=np.float64)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("fastmap", self.n_components_)
