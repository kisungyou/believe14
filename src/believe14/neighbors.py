"""Reusable exact Euclidean neighborhoods with deterministic tie handling."""

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core.distances import pairwise_distances
from ._core.validation import as_float_matrix, validate_n_neighbors


def _observation_digest(values: NDArray[np.float64]) -> bytes:
    # Canonicalize signed zero so equal float64 observations hash identically.
    return hashlib.sha256(values + 0.0).digest()


@dataclass(frozen=True, init=False, eq=False, slots=True)
class ExactNeighborGraph:
    """Immutable, directed nearest-neighbor lists for a particular ordered dataset.

    Each row excludes itself. Distinct observations at equal distance are
    ordered by their original row index. A graph built for k neighbors can
    supply every smaller neighborhood without repeating the distance search.
    """

    _shape: tuple[int, int]
    _digest: bytes
    _distances: bytes
    _indices: bytes
    n_neighbors: int

    def __init__(self, X: ArrayLike, n_neighbors: int = 5) -> None:
        values = as_float_matrix(X)
        n = len(values)
        k = validate_n_neighbors(n_neighbors, n_samples=n)
        distances = np.empty((n, k))
        indices = np.empty((n, k), dtype=np.int64)
        for row in range(n):
            d = pairwise_distances(values[row : row + 1], values)[0]
            d[row] = np.inf
            idx = np.argsort(d, kind="stable")[:k]
            indices[row], distances[row] = idx, d[idx]
        object.__setattr__(self, "_shape", values.shape)
        object.__setattr__(self, "_digest", _observation_digest(values))
        object.__setattr__(self, "_distances", distances.tobytes())
        object.__setattr__(self, "_indices", indices.tobytes())
        object.__setattr__(self, "n_neighbors", k)

    @property
    def distances(self) -> NDArray[np.float64]:
        """Euclidean distances ordered from nearest to farthest."""
        return np.frombuffer(self._distances, dtype=np.float64).reshape(
            -1, self.n_neighbors
        )

    @property
    def indices(self) -> NDArray[np.int64]:
        """Training row indices of the selected neighbors."""
        return np.frombuffer(self._indices, dtype=np.int64).reshape(
            -1, self.n_neighbors
        )

    def query(
        self, X: ArrayLike, n_neighbors: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Return a prefix after checking that X matches the original observations."""
        values = as_float_matrix(X)
        k = validate_n_neighbors(n_neighbors, n_samples=len(values))
        if k > self.n_neighbors:
            raise ValueError("Requested more neighbors than this graph contains.")
        if values.shape != self._shape or _observation_digest(values) != self._digest:
            raise ValueError(
                "The neighbor graph was built for different observations or row order."
            )
        return self.distances[:, :k], self.indices[:, :k]


def build_neighbor_graph(X: ArrayLike, n_neighbors: int = 5) -> ExactNeighborGraph:
    """Compute exact sorted neighbors using O(nk + np) working memory."""
    return ExactNeighborGraph(X, n_neighbors)
