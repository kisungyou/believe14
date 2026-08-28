"""Exact dense distance and neighborhood primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pairwise_squared_distances(
    X: NDArray[np.float64], Y: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Compute stable squared Euclidean distances with explicit overflow checks."""

    distances = pairwise_distances(X, Y)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        result = distances * distances
    if np.any(np.isinf(result)):
        raise FloatingPointError("Squared Euclidean distances overflow float64.")
    if np.any((distances > 0.0) & (result == 0.0)):
        raise FloatingPointError("Squared Euclidean distances underflow float64.")
    result = np.maximum(result, 0.0)
    if Y is None:
        result = result * 0.5 + result.T * 0.5
        np.fill_diagonal(result, 0.0)
    return np.asarray(result, dtype=np.float64)


def pairwise_scaled_squared_distances(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    multiplier: float,
) -> NDArray[np.float64]:
    """Compute ``multiplier * ||x-y||**2`` without materializing raw squares.

    Infinite results are retained intentionally: callers such as an RBF kernel
    map them to their mathematically correct limiting value of zero.
    """

    other = X if Y is None else Y
    result = np.zeros((X.shape[0], other.shape[0]), dtype=np.float64)
    log_multiplier = np.log(multiplier)
    maximum_log = np.log(np.finfo(np.float64).max)
    minimum_log = np.log(np.nextafter(0.0, 1.0))
    for row, observation in enumerate(X):
        with np.errstate(over="ignore", invalid="ignore"):
            differences = other - observation
        overflowed = np.any(~np.isfinite(differences), axis=1)
        norms = np.hypot.reduce(np.abs(differences), axis=1)
        log_values = np.full(other.shape[0], -np.inf, dtype=np.float64)
        ordinary = (~overflowed) & (norms > 0.0)
        log_values[ordinary] = log_multiplier + 2.0 * np.log(norms[ordinary])
        for index in np.flatnonzero(overflowed):
            pair_scale = max(
                float(np.max(np.abs(observation), initial=0.0)),
                float(np.max(np.abs(other[index]), initial=0.0)),
            )
            scaled_difference = other[index] / pair_scale - observation / pair_scale
            scaled_norm = float(np.hypot.reduce(np.abs(scaled_difference)))
            if scaled_norm > 0.0:
                log_values[index] = (
                    log_multiplier
                    + 2.0 * np.log(pair_scale)
                    + 2.0 * np.log(scaled_norm)
                )
        high = log_values > maximum_log
        finite = (log_values >= minimum_log) & ~high
        result[row, high] = np.inf
        result[row, finite] = np.exp(log_values[finite])
    if Y is None:
        result = result * 0.5 + result.T * 0.5
        np.fill_diagonal(result, 0.0)
    return np.asarray(result, dtype=np.float64)


def scaled_squares(
    values: NDArray[np.float64], *, multiplier: float
) -> NDArray[np.float64]:
    """Compute a positive multiplier times nonnegative squares stably."""

    result = np.zeros_like(values, dtype=np.float64)
    positive = values > 0.0
    logs = np.log(multiplier) + 2.0 * np.log(values[positive])
    maximum_log = np.log(np.finfo(np.float64).max)
    minimum_log = np.log(np.nextafter(0.0, 1.0))
    finite = (logs <= maximum_log) & (logs >= minimum_log)
    selected = np.empty_like(logs)
    selected[logs > maximum_log] = np.inf
    selected[logs < minimum_log] = 0.0
    selected[finite] = np.exp(logs[finite])
    result[positive] = selected
    return result


def pairwise_distances(
    X: NDArray[np.float64], Y: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Compute scaled Euclidean distances with explicit overflow checks."""

    other = X if Y is None else Y
    result = np.empty((X.shape[0], other.shape[0]), dtype=np.float64)
    for row, observation in enumerate(X):
        with np.errstate(over="ignore", invalid="ignore"):
            differences = other - observation
        if np.any(~np.isfinite(differences)):
            raise FloatingPointError("Euclidean distances overflow float64.")
        result[row] = np.hypot.reduce(np.abs(differences), axis=1)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("Euclidean distances overflow float64.")
    if Y is None:
        result = result * 0.5 + result.T * 0.5
        np.fill_diagonal(result, 0.0)
    return np.asarray(result, dtype=np.float64)


def exact_neighbors(
    X: NDArray[np.float64], n_neighbors: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return deterministic exact neighbors, excluding each point itself."""

    distances = pairwise_distances(X)
    np.fill_diagonal(distances, np.inf)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbors]
    rows = np.arange(X.shape[0])[:, None]
    selected = distances[rows, indices]
    return (
        np.asarray(selected, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
    )
