# mypy: disallow-subclassing-any=False
"""Landmark classical scaling and its graph-geodesic counterpart."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.distances import pairwise_distances
from believe14._core.linalg import _float_units, apply_centering, centering_state
from believe14._core.validation import (
    as_float_matrix,
    validate_features,
    validate_n_components,
    validate_n_neighbors,
    validate_precomputed,
)
from believe14.api import FitDiagnostics
from believe14.neighbors import ExactNeighborGraph, build_neighbor_graph

from ._common import FloatMatrix, classical_embedding, output_names


def _select_landmarks(
    size: int, k: int, count: int | None, indices: ArrayLike | None
) -> NDArray[np.intp]:
    if indices is None:
        m = (
            min(size, max(k + 1, ceil(sqrt(size))))
            if count is None
            else validate_n_components(count, maximum=size, name="n_landmarks")
        )
        if m <= k:
            raise ValueError("At least n_components + 1 landmarks are required.")
        return np.linspace(0, size - 1, num=m, dtype=np.intp)
    if count is not None:
        raise ValueError("Specify only one of n_landmarks and landmark_indices.")
    chosen = np.asarray(indices)
    if chosen.ndim != 1 or chosen.dtype.kind not in "iu":
        raise ValueError("landmark_indices must be a one-dimensional integer array.")
    if (
        len(chosen) <= k
        or np.any(chosen < 0)
        or np.any(chosen >= size)
        or len(np.unique(chosen)) != len(chosen)
    ):
        raise ValueError(
            "Provide at least n_components + 1 distinct, in-range landmark indices."
        )
    return np.array(chosen, dtype=np.intp, copy=True)


@dataclass(frozen=True)
class _LandmarkFrame:
    scale: float
    mean_square: FloatMatrix
    dual: FloatMatrix
    spectrum: FloatMatrix
    fit_diagnostics: FitDiagnostics

    @classmethod
    def fit(cls, distances: FloatMatrix, k: int, solver: str) -> _LandmarkFrame:
        scale = float(distances.max())
        if scale == 0:
            raise ValueError("The landmarks have zero separation.")
        normalized = distances / scale
        coordinates, values, _, residual, rank, warnings = classical_embedding(
            normalized, k
        )
        dual = coordinates / values[:k]
        with np.errstate(over="raise", invalid="raise"):
            spectrum = (values * scale) * scale
        if np.any(spectrum[:k] == 0):
            raise FloatingPointError("Landmark eigenvalues underflow float64.")
        return cls(
            scale,
            (normalized**2).mean(axis=0),
            dual,
            spectrum,
            diagnostics(
                solver, residual_norm=residual, numerical_rank=rank, warnings=warnings
            ),
        )

    def project(
        self, distances: FloatMatrix, baseline: FloatMatrix | None = None
    ) -> FloatMatrix:
        """Project squared-distance contrasts by the landmark Nyström formula."""
        with np.errstate(over="raise", invalid="raise"):
            # Orthogonality to the constant vector permits subtracting the
            # first squared distance. Factor that difference before squaring.
            first = distances[:, :1]
            delta = (distances - first) / self.scale
            # Take the small difference before scaling. For metric distances
            # |delta| <= 1, even arbitrarily far outside the landmark set.
            cross = delta * (0.5 * distances) + delta * (0.5 * first)
            if baseline is not None:
                cross += delta * baseline
            anchor = (0.5 * (self.mean_square - self.mean_square[0])) * self.scale
            return np.asarray((anchor - cross) @ self.dual)


class LandmarkMDS(TransformerMixin, BaseEstimator):
    """Classical scaling on landmarks with distance-based extension.

    Use explicit landmark indices or select n_landmarks evenly spaced row
    indices. The default count is ceil(sqrt(n)), at least n_components + 1.
    In precomputed mode supply a square training distance matrix; transform
    takes distances to all training rows, in their original order.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_landmarks: int | None = None,
        landmark_indices: ArrayLike | None = None,
        dissimilarity: Literal["euclidean", "precomputed"] = "euclidean",
    ) -> None:
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.landmark_indices = landmark_indices
        self.dissimilarity = dissimilarity

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.dissimilarity == "precomputed"
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> LandmarkMDS:
        del y
        if self.dissimilarity == "precomputed":
            data = validate_precomputed(X)
            self.n_features_in_ = data.shape[1]
        elif self.dissimilarity == "euclidean":
            data = validate_features(self, X, reset=True)
        else:
            raise ValueError("dissimilarity must be 'euclidean' or 'precomputed'.")
        k = validate_n_components(self.n_components, maximum=len(data) - 1)
        indices = _select_landmarks(
            len(data), k, self.n_landmarks, self.landmark_indices
        )
        if self.dissimilarity == "precomputed":
            block = data[np.ix_(indices, indices)]
        else:
            landmarks = data[indices]
            block = pairwise_distances(landmarks)
        frame = _LandmarkFrame.fit(block, k, "landmark_classical_mds")
        if self.dissimilarity == "precomputed":
            embedding = frame.project(data[:, indices])
        else:
            # For Euclidean features the same Nyström formula is a linear
            # projection. This avoids subtracting rounded large query radii.
            location = centering_state(landmarks)
            centered = apply_centering(landmarks, *location)
            basis = (centered / frame.scale).T @ frame.dual
            embedding = apply_centering(data, *location) @ basis
            self._location_ = location
            self._basis_ = basis
        if not np.isfinite(embedding).all():
            raise FloatingPointError("Landmark coordinates exceed float64.")
        self._frame_ = frame
        self.landmark_indices_ = indices
        self.n_landmarks_ = len(indices)
        self.n_samples_fit_ = len(data)
        self.n_components_ = k
        self.embedding_ = embedding
        self.eigenvalues_ = frame.spectrum
        self.diagnostics_ = frame.fit_diagnostics
        return self

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> FloatMatrix:
        return self.fit(X, y).embedding_.copy()

    def transform(self, X: ArrayLike) -> FloatMatrix:
        check_is_fitted(self, "_frame_")
        if self.dissimilarity == "precomputed":
            distances = as_float_matrix(X, min_samples=1)
            if distances.shape[1] != self.n_samples_fit_ or np.any(distances < 0):
                raise ValueError(
                    "Provide one nonnegative query distance per training row."
                )
            return self._frame_.project(distances[:, self.landmark_indices_])
        data = validate_features(self, X, reset=False, min_samples=1)
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(apply_centering(data, *self._location_) @ self._basis_)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "_frame_")
        return output_names("landmarkmds", self.n_components_)


def _query_edges(
    point: FloatMatrix, training: FloatMatrix, count: int
) -> tuple[float, NDArray[np.intp], FloatMatrix]:
    """Find query edges and preserve their length offsets from a common base."""
    distances = pairwise_distances(point[None, :], training)[0]
    order = np.argsort(distances, kind="stable")
    nearest = order[:count]
    boundary = distances[nearest[-1]]
    # Refine candidates within the rounding error of the cutoff. Exact squared
    # lengths also resolve neighbors indistinguishable at a very distant query.
    band = (8 * training.shape[1] * np.finfo(float).eps) * boundary
    close = np.flatnonzero(np.abs(distances - boundary) <= band)
    point_units = [_float_units(float(value)) for value in point]

    def squared_units(index: int) -> int:
        return sum(
            (a - _float_units(float(b))) ** 2
            for a, b in zip(point_units, training[index], strict=True)
        )

    if len(close) > 1 and np.any(distances[order[count:]] - boundary <= band):
        before = nearest[boundary - distances[nearest] > band]
        refined = sorted(close.tolist(), key=squared_units)[: count - len(before)]
        nearest = np.concatenate((before, np.asarray(refined, dtype=np.intp)))
    base = float(distances[nearest[0]])
    if base == 0.0 or count == 1:
        return base, nearest, distances[nearest] - base
    # (r_j - r_0) = (r_j² - r_0²)/(r_j + r_0). Computing the numerator in
    # squared subnormal units preserves the difference when r_j rounds to r_0.
    reference = squared_units(int(nearest[0]))
    base_units = _float_units(base)
    offsets = np.array(
        [
            (squared_units(int(j)) - reference)
            / ((base_units + _float_units(float(distances[j]))) << 1074)
            for j in nearest
        ]
    )
    return base, nearest, offsets


class LandmarkIsomap(TransformerMixin, BaseEstimator):
    """Landmark scaling of shortest paths on an exact union k-neighbor graph.

    Fit computes shortest paths only from landmarks. Each new query connects
    to its k closest training points, preserving the fitted graph.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 5,
        n_landmarks: int | None = None,
        landmark_indices: ArrayLike | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_landmarks = n_landmarks
        self.landmark_indices = landmark_indices

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        neighbors: ExactNeighborGraph | None = None,
    ) -> LandmarkIsomap:
        del y
        data = validate_features(self, X, reset=True, copy=True)
        k = validate_n_components(self.n_components, maximum=len(data) - 1)
        n_neighbors = validate_n_neighbors(self.n_neighbors, n_samples=len(data))
        indices = _select_landmarks(
            len(data), k, self.n_landmarks, self.landmark_indices
        )
        graph = (
            build_neighbor_graph(data, n_neighbors=n_neighbors)
            if neighbors is None
            else neighbors
        )
        distances, destinations = graph.query(data, n_neighbors=n_neighbors)
        if np.any(distances <= 0):
            raise ValueError(
                "LandmarkIsomap requires positive edges and distinct observations."
            )
        rows = np.repeat(np.arange(len(data)), n_neighbors)
        directed = csr_matrix(
            (distances.ravel(), (rows, destinations.ravel())),
            shape=(len(data), len(data)),
        )
        undirected = directed.maximum(directed.T)
        paths = np.asarray(dijkstra(undirected, directed=False, indices=indices))
        if not np.isfinite(paths).all():
            raise ValueError(
                "The graph is disconnected or its path distances exceed float64."
            )
        # The same undirected path can accumulate in different orders.
        block = paths[:, indices]
        block = np.minimum(block, block.T)
        frame = _LandmarkFrame.fit(block, k, "landmark_shortest_paths+classical_mds")
        embedding = frame.project(paths.T)
        self.landmark_indices_ = indices
        self.n_landmarks_ = len(indices)
        self.n_neighbors_ = n_neighbors
        self.n_components_ = k
        self.landmark_distances_ = paths.T
        self._training_data_ = data
        self._frame_ = frame
        self.embedding_ = embedding
        self.eigenvalues_ = frame.spectrum
        self.diagnostics_ = frame.fit_diagnostics
        return self

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        neighbors: ExactNeighborGraph | None = None,
    ) -> FloatMatrix:
        return self.fit(X, y, neighbors=neighbors).embedding_.copy()

    def transform(self, X: ArrayLike) -> FloatMatrix:
        check_is_fitted(self, "_frame_")
        data = validate_features(self, X, reset=False, min_samples=1)
        query_paths = np.empty((len(data), self.n_landmarks_))
        baselines = np.empty((len(data), 1))
        for row, observation in enumerate(data):
            base, nearest, offsets = _query_edges(
                observation, self._training_data_, self.n_neighbors_
            )
            baselines[row, 0] = base
            with np.errstate(over="ignore"):
                query_paths[row] = np.min(
                    offsets[:, None] + self.landmark_distances_[nearest], axis=0
                )
        if not np.isfinite(query_paths).all():
            raise FloatingPointError("Query path distances exceed float64.")
        return self._frame_.project(query_paths, baselines)

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        check_is_fitted(self, "_frame_")
        return output_names("landmarkisomap", self.n_components_)
