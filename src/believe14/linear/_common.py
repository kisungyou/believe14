"""Private helpers shared by the linear estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array

from believe14._core.validation import as_float_matrix


def component_names(
    estimator: Any, input_features: ArrayLike | None = None
) -> NDArray[np.object_]:
    """Return stable generic names for latent coordinates."""

    if input_features is not None:
        supplied = np.asarray(input_features, dtype=object)
        if supplied.ndim != 1 or supplied.shape[0] != estimator.n_features_in_:
            raise ValueError(
                "input_features must contain exactly one name per fitted feature."
            )
        if hasattr(estimator, "feature_names_in_") and not np.array_equal(
            supplied, estimator.feature_names_in_
        ):
            raise ValueError("input_features must match feature_names_in_.")
    return np.asarray(
        [
            f"{estimator.__class__.__name__.lower()}{i}"
            for i in range(estimator.n_components_)
        ],
        dtype=object,
    )


def validate_vector(y: ArrayLike, *, n_samples: int, name: str = "y") -> NDArray[Any]:
    """Validate a finite one-dimensional response or label vector."""

    result = check_array(
        y,
        ensure_2d=False,
        dtype=None,
        ensure_all_finite=True,
        ensure_min_samples=n_samples,
    )
    result = np.asarray(result)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if result.shape[0] != n_samples:
        raise ValueError(f"{name} has {result.shape[0]} samples; expected {n_samples}.")
    return result


def validate_targets(
    y: ArrayLike, *, n_samples: int
) -> tuple[NDArray[np.float64], bool]:
    """Validate a one- or two-dimensional floating-point target matrix."""

    result = check_array(
        y,
        ensure_2d=False,
        dtype=np.float64,
        ensure_all_finite=True,
        ensure_min_samples=n_samples,
    )
    array = np.asarray(result, dtype=np.float64)
    was_1d = array.ndim == 1
    if was_1d:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n_samples:
        raise ValueError("y must contain one row per observation in X.")
    return array, was_1d


def covariance_support(
    X_centered: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, float]:
    """Return a covariance whitening map on its numerical support."""

    n_samples, n_features = X_centered.shape
    scale = float(np.max(np.abs(X_centered), initial=0.0))
    if scale == 0.0:
        raise ValueError("X has zero numerical variance.")
    scaled = X_centered / scale
    covariance = (scaled.T @ scaled) / float(n_samples)
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    tolerance = (
        max(n_samples, n_features)
        * np.finfo(np.float64).eps
        * max(0.0, float(values[0]))
    )
    keep = values > tolerance
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise ValueError("X has zero numerical variance.")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        whitening = vectors[:, keep] / np.sqrt(values[keep]) / scale
    condition = float(values[0] / values[keep][-1])
    return whitening, values[keep], rank, condition


def response_slices(
    y: ArrayLike, *, n_samples: int, n_slices: int
) -> NDArray[np.int64]:
    """Assign a numeric response to deterministic, tie-preserving slices."""

    raw = validate_vector(y, n_samples=n_samples)
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("y must be numeric for inverse-regression slicing.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite values.")
    if isinstance(n_slices, bool) or not isinstance(n_slices, (int, np.integer)):
        raise TypeError("n_slices must be an integer.")
    if n_slices < 2:
        raise ValueError("n_slices must be at least 2.")
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError("y must contain at least two distinct values.")
    if unique.size <= n_slices:
        return np.searchsorted(unique, values).astype(np.int64)
    probabilities = np.linspace(0.0, 1.0, n_slices + 1)[1:-1]
    edges = np.unique(np.quantile(values, probabilities, method="inverted_cdf"))
    return np.searchsorted(edges, values, side="left").astype(np.int64)


def as_second_view(Y: ArrayLike, *, n_samples: int) -> NDArray[np.float64]:
    """Validate a paired dense floating-point feature view."""

    result = as_float_matrix(Y, min_samples=n_samples)
    if result.shape[0] != n_samples:
        raise ValueError("X and Y must contain the same number of observations.")
    return result


def soft_threshold(value: float, threshold: float) -> float:
    """Scalar soft-thresholding operator."""

    return float(np.sign(value) * max(abs(value) - threshold, 0.0))
