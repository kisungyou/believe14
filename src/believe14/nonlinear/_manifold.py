# mypy: disallow-subclassing-any=False
"""Graph and local-geometry manifold embeddings."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.distances import exact_neighbors
from believe14._core.graphs import (
    graph_shortest_paths,
    neighbor_graph,
    require_connected,
)
from believe14._core.linalg import canonicalize_columns, centered_svd
from believe14._core.validation import (
    validate_features,
    validate_n_components,
    validate_n_neighbors,
    validate_positive_real,
)

from ._common import classical_embedding, output_names


class Isomap(BaseEstimator):
    """Isomap with an exact connected k-neighbor graph and classical scaling."""

    def __init__(self, n_components: int = 2, *, n_neighbors: int = 5) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Isomap:
        """Fit the transductive geodesic-distance embedding."""

        del y
        features = validate_features(self, X, reset=True)
        n_components = validate_n_components(
            self.n_components, maximum=features.shape[0] - 1
        )
        n_neighbors = validate_n_neighbors(
            self.n_neighbors, n_samples=features.shape[0]
        )
        neighbor_distances, _ = exact_neighbors(features, n_neighbors)
        if np.any(neighbor_distances <= 0.0):
            raise ValueError(
                "Isomap does not admit duplicate observations or zero-length "
                "neighborhood edges."
            )
        graph = neighbor_graph(
            features,
            n_neighbors=n_neighbors,
            weighting="binary",
            symmetrize="union",
        )
        geodesic = graph_shortest_paths(graph)
        embedding, spectrum, _, residual, rank, warnings = classical_embedding(
            geodesic, n_components
        )
        self.embedding_ = embedding
        self.geodesic_distances_ = geodesic
        self.eigenvalues_ = spectrum
        self.n_neighbors_ = n_neighbors
        self.n_components_ = n_components
        self.diagnostics_ = diagnostics(
            "exact_shortest_path+classical_mds",
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
        return output_names("isomap", self.n_components_)


class LocallyLinearEmbedding(BaseEstimator):
    """Standard LLE with regularized local barycentric weights."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 5,
        regularization: float = 1e-3,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.regularization = regularization

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> LocallyLinearEmbedding:
        """Fit the transductive reconstruction-weight embedding."""

        del y
        features = validate_features(self, X, reset=True)
        n_components = validate_n_components(
            self.n_components, maximum=features.shape[0] - 1
        )
        n_neighbors = validate_n_neighbors(
            self.n_neighbors, n_samples=features.shape[0], minimum=n_components + 1
        )
        regularization = validate_positive_real(
            self.regularization, name="regularization"
        )
        graph = neighbor_graph(
            features,
            n_neighbors=n_neighbors,
            weighting="binary",
            symmetrize="union",
        )
        require_connected(graph)
        _, indices = exact_neighbors(features, n_neighbors)
        weights = np.zeros((features.shape[0], features.shape[0]), dtype=np.float64)
        reconstruction = np.zeros_like(features)
        for row, neighbors in enumerate(indices):
            offsets = features[neighbors] - features[row]
            offset_scale = float(np.max(np.abs(offsets), initial=0.0))
            if offset_scale == 0.0:
                raise ValueError(
                    "LLE encountered a degenerate neighborhood with zero scatter."
                )
            scaled_offsets = offsets / offset_scale
            covariance = scaled_offsets @ scaled_offsets.T
            trace = float(np.trace(covariance))
            scale = float(linalg.norm(covariance, ord=2))
            threshold = np.finfo(np.float64).eps * scale * 100.0
            if trace <= threshold:
                raise ValueError(
                    "LLE encountered a degenerate neighborhood with zero scatter."
                )
            covariance.flat[:: n_neighbors + 1] += regularization * trace
            local = linalg.solve(
                covariance,
                np.ones(n_neighbors, dtype=np.float64),
                assume_a="pos",
                check_finite=False,
            )
            denominator = float(np.sum(local, dtype=np.float64))
            denominator_tolerance = (
                np.finfo(np.float64).eps
                * n_neighbors
                * float(np.max(np.abs(local), initial=0.0))
                * 100.0
            )
            if abs(denominator) <= denominator_tolerance:
                raise FloatingPointError("An LLE weight normalization is singular.")
            normalized_weights = local / denominator
            weights[row, neighbors] = normalized_weights
            reconstruction[row] = -(normalized_weights @ offsets)
        identity_minus = np.eye(features.shape[0], dtype=np.float64) - weights
        alignment = identity_minus.T @ identity_minus
        values, vectors = linalg.eigh(alignment, check_finite=False)
        order = np.argsort(values, kind="stable")
        values = np.asarray(values[order], dtype=np.float64)
        vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
        selected = vectors[:, 1 : n_components + 1]
        selected_values = values[1 : n_components + 1]
        residual = alignment @ selected - selected * selected_values
        scale = float(linalg.norm(alignment, ord=2))
        reconstruction_scale = float(np.max(np.abs(reconstruction), initial=0.0))
        if reconstruction_scale == 0.0:
            reconstruction_error = 0.0
        else:
            normalized_error = float(
                np.sum((reconstruction / reconstruction_scale) ** 2)
            )
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                reconstruction_error = (
                    normalized_error * reconstruction_scale * reconstruction_scale
                )
            if reconstruction_error == 0.0:
                raise FloatingPointError(
                    "The LLE reconstruction error underflows float64."
                )
        self.embedding_ = selected
        self.reconstruction_weights_ = weights
        self.reconstruction_error_ = float(reconstruction_error)
        self.eigenvalues_ = selected_values
        self.n_neighbors_ = n_neighbors
        self.n_components_ = n_components
        self.diagnostics_ = diagnostics(
            "barycentric_weights+symmetric_eigh",
            residual_norm=float(linalg.norm(residual) / scale),
            objective_value=float(np.sum(selected_values, dtype=np.float64)),
            numerical_rank=int(np.linalg.matrix_rank(alignment)),
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
        return output_names("locallylinearembedding", self.n_components_)


class LaplacianEigenmaps(BaseEstimator):
    """Laplacian Eigenmaps using a connected, symmetric k-neighbor graph."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 5,
        weighting: Literal["binary", "heat"] = "heat",
        gamma: float = 1.0,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.gamma = gamma

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> LaplacianEigenmaps:
        """Fit the transductive graph-Laplacian embedding."""

        del y
        features = validate_features(self, X, reset=True)
        n_components = validate_n_components(
            self.n_components, maximum=features.shape[0] - 1
        )
        n_neighbors = validate_n_neighbors(
            self.n_neighbors, n_samples=features.shape[0]
        )
        if self.weighting not in {"binary", "heat"}:
            raise ValueError("weighting must be 'binary' or 'heat'.")
        gamma = validate_positive_real(self.gamma, name="gamma")
        graph = neighbor_graph(
            features,
            n_neighbors=n_neighbors,
            weighting=self.weighting,
            gamma=gamma if self.weighting == "heat" else None,
            symmetrize="union",
        )
        require_connected(graph)
        affinity = graph.weights
        degree_values = np.sum(affinity, axis=1, dtype=np.float64)
        if np.any(degree_values <= 0.0):
            raise FloatingPointError("The graph contains a zero-degree vertex.")
        degree = np.diag(degree_values)
        laplacian = degree - affinity
        values, vectors = linalg.eigh(laplacian, degree, check_finite=False)
        order = np.argsort(values, kind="stable")
        values = np.asarray(values[order], dtype=np.float64)
        vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
        selected = vectors[:, 1 : n_components + 1]
        selected_values = values[1 : n_components + 1]
        residual = laplacian @ selected - (degree @ selected) * selected_values
        scale = max(
            1.0,
            float(linalg.norm(laplacian, ord=2))
            + float(np.max(selected_values)) * float(linalg.norm(degree, ord=2)),
        )
        self.embedding_ = selected
        self.eigenvalues_ = selected_values
        self.affinity_matrix_ = affinity
        self.degree_ = degree_values
        self.n_neighbors_ = n_neighbors
        self.gamma_ = gamma
        self.n_components_ = n_components
        self.diagnostics_ = diagnostics(
            "generalized_symmetric_eigh",
            residual_norm=float(linalg.norm(residual) / scale),
            objective_value=float(np.sum(selected_values, dtype=np.float64)),
            numerical_rank=int(np.linalg.matrix_rank(laplacian)),
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
        return output_names("laplacianeigenmaps", self.n_components_)


class LocalTangentSpaceAlignment(BaseEstimator):
    """Local Tangent Space Alignment with exact local SVDs."""

    def __init__(self, n_components: int = 2, *, n_neighbors: int = 8) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> LocalTangentSpaceAlignment:
        """Fit the transductive local-tangent alignment embedding."""

        del y
        features = validate_features(self, X, reset=True)
        n_components = validate_n_components(
            self.n_components,
            maximum=min(features.shape[0] - 1, features.shape[1]),
        )
        n_neighbors = validate_n_neighbors(
            self.n_neighbors, n_samples=features.shape[0], minimum=n_components + 1
        )
        graph = neighbor_graph(
            features,
            n_neighbors=n_neighbors,
            weighting="binary",
            symmetrize="union",
        )
        require_connected(graph)
        _, neighbor_indices = exact_neighbors(features, n_neighbors)
        alignment = np.zeros((features.shape[0], features.shape[0]), dtype=np.float64)
        for row, neighbors in enumerate(neighbor_indices):
            local_indices = np.concatenate(
                (np.asarray([row], dtype=np.int64), neighbors)
            )
            local = features[local_indices]
            _, left, singular_values, _ = centered_svd(local)
            threshold = (
                max(local.shape) * np.finfo(np.float64).eps * singular_values[0] * 100.0
            )
            if int(np.count_nonzero(singular_values > threshold)) < n_components:
                raise ValueError(
                    "LTSA encountered a neighborhood with insufficient tangent rank."
                )
            constant = np.full((n_neighbors + 1, 1), 1.0 / np.sqrt(n_neighbors + 1))
            basis = np.column_stack((constant, left[:, :n_components]))
            local_alignment = np.eye(n_neighbors + 1) - basis @ basis.T
            alignment[np.ix_(local_indices, local_indices)] += local_alignment
        alignment = (alignment + alignment.T) * 0.5
        values, vectors = linalg.eigh(alignment, check_finite=False)
        order = np.argsort(values, kind="stable")
        values = np.asarray(values[order], dtype=np.float64)
        vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
        selected = vectors[:, 1 : n_components + 1]
        selected_values = values[1 : n_components + 1]
        residual = alignment @ selected - selected * selected_values
        scale = max(1.0, float(linalg.norm(alignment, ord=2)))
        self.embedding_ = selected
        self.eigenvalues_ = selected_values
        self.alignment_matrix_ = alignment
        self.n_neighbors_ = n_neighbors
        self.n_components_ = n_components
        self.diagnostics_ = diagnostics(
            "local_svd+symmetric_eigh",
            residual_norm=float(linalg.norm(residual) / scale),
            objective_value=float(np.sum(selected_values, dtype=np.float64)),
            numerical_rank=int(np.linalg.matrix_rank(alignment)),
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
        return output_names("localtangentspacealignment", self.n_components_)
