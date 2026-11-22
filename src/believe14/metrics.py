"""Local neighborhood fidelity measures for dimensionality reduction."""

import numpy as np
from numpy.typing import ArrayLike

from ._core.distances import pairwise_distances
from ._core.validation import as_float_matrix, validate_n_neighbors


def trustworthiness(
    X: ArrayLike, embedding: ArrayLike, *, n_neighbors: int = 5
) -> float:
    """Score neighborhood intrusions in the embedding on a scale of zero to one.

    Each embedded neighbor is penalized by how far its original rank exceeds
    k (Venna and Kaski, 2001). Self is excluded; ties follow input row order.
    The standard normalization requires k < n/2.
    """
    original, reduced = as_float_matrix(X), as_float_matrix(embedding)
    n = len(original)
    if len(reduced) != n:
        raise ValueError(
            "X and embedding must contain the same observations in row order."
        )
    k = validate_n_neighbors(n_neighbors, n_samples=n)
    if 2 * k >= n:
        raise ValueError(
            "n_neighbors must be strictly less than half the sample count."
        )
    rank = np.empty(n, dtype=np.int64)
    ordinal = np.arange(1, n + 1)
    total = 0
    for row in range(n):
        d = pairwise_distances(original[row : row + 1], original)[0]
        z = pairwise_distances(reduced[row : row + 1], reduced)[0]
        d[row] = z[row] = np.inf
        rank[np.argsort(d, kind="stable")] = ordinal
        selected = np.argsort(z, kind="stable")[:k]
        total += int(np.maximum(rank[selected] - k, 0).sum())
    return 1.0 - (2 * total) / (n * k * (2 * n - 3 * k - 1))


def continuity(X: ArrayLike, embedding: ArrayLike, *, n_neighbors: int = 5) -> float:
    """Score missing original neighbors using their ranks in the embedding."""
    return trustworthiness(embedding, X, n_neighbors=n_neighbors)
