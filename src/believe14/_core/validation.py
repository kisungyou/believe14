"""Input and parameter validation shared by all estimators."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array, validate_data


def validate_features(
    estimator: Any,
    X: ArrayLike,
    *,
    reset: bool,
    min_samples: int = 2,
    min_features: int = 1,
    copy: bool = False,
) -> NDArray[np.float64]:
    """Validate a finite dense feature matrix and track fitted feature count."""

    result = validate_data(
        estimator,
        X=X,
        reset=reset,
        accept_sparse=False,
        dtype=np.float64,
        order="C",
        copy=copy,
        ensure_all_finite=True,
        ensure_2d=True,
        ensure_min_samples=min_samples,
        ensure_min_features=min_features,
    )
    return np.asarray(result, dtype=np.float64)


def as_float_matrix(
    X: ArrayLike,
    *,
    min_samples: int = 2,
    min_features: int = 1,
    copy: bool = False,
) -> NDArray[np.float64]:
    """Return a validated finite dense float64 matrix."""

    result = check_array(
        X,
        accept_sparse=False,
        dtype=np.float64,
        order="C",
        copy=copy,
        ensure_all_finite=True,
        ensure_2d=True,
        ensure_min_samples=min_samples,
        ensure_min_features=min_features,
    )
    return np.asarray(result, dtype=np.float64)


def validate_precomputed(
    D: ArrayLike,
    *,
    squared: bool = False,
    atol: float | None = None,
) -> NDArray[np.float64]:
    """Validate a finite, symmetric, hollow, nonnegative dissimilarity matrix."""

    matrix = as_float_matrix(D)
    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError("A precomputed dissimilarity matrix must be square.")
    scale = float(np.max(np.abs(matrix), initial=0.0))
    tolerance = 100.0 * np.finfo(np.float64).eps * n_rows * scale
    if atol is not None:
        tolerance = float(atol)
    if np.any(matrix < -tolerance):
        raise ValueError("Dissimilarities must be nonnegative.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError("A precomputed dissimilarity matrix must be symmetric.")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=tolerance):
        raise ValueError(
            "A precomputed dissimilarity matrix must have a zero diagonal."
        )
    matrix = np.maximum(0.0, matrix * 0.5 + matrix.T * 0.5)
    np.fill_diagonal(matrix, 0.0)
    if squared and not np.all(np.isfinite(matrix)):
        raise ValueError("Squared dissimilarities must be finite.")
    return matrix


def validate_n_components(
    n_components: int,
    *,
    maximum: int,
    name: str = "n_components",
) -> int:
    """Validate a strictly positive integer component count."""

    if isinstance(n_components, bool) or not isinstance(n_components, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(n_components)
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}], got {value}.")
    return value


def validate_n_neighbors(
    n_neighbors: int,
    *,
    n_samples: int,
    minimum: int = 1,
) -> int:
    """Validate a neighbor count that excludes the observation itself."""

    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, Integral):
        raise TypeError("n_neighbors must be an integer.")
    value = int(n_neighbors)
    if value < minimum or value >= n_samples:
        raise ValueError(
            f"n_neighbors must be in [{minimum}, {n_samples - 1}], got {value}."
        )
    return value


def validate_positive_real(value: float, *, name: str, strict: bool = True) -> float:
    """Validate a finite real scalar."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if (strict and result <= 0.0) or (not strict and result < 0.0):
        relation = "positive" if strict else "nonnegative"
        raise ValueError(f"{name} must be {relation}.")
    return result


def make_rng(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Construct or forward a local generator without touching global RNG state."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise TypeError("random_state must be None, an integer, or a Generator.")
    return np.random.default_rng(int(random_state))
