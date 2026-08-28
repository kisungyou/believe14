"""Private numerical helpers shared by nonlinear estimators."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from believe14._core.linalg import canonicalize_columns, stable_center
from believe14._core.validation import (
    as_float_matrix,
    make_rng,
    validate_n_components,
    validate_positive_real,
    validate_precomputed,
)

FloatMatrix = NDArray[np.float64]


def validate_dissimilarities(
    X: ArrayLike,
    *,
    dissimilarity: Literal["euclidean", "precomputed"],
    feature_matrix: FloatMatrix | None = None,
) -> FloatMatrix:
    """Return the exact dissimilarities requested by an estimator."""

    if dissimilarity == "precomputed":
        return validate_precomputed(X)
    if dissimilarity != "euclidean":
        raise ValueError("dissimilarity must be 'euclidean' or 'precomputed'.")
    if feature_matrix is None:
        feature_matrix = as_float_matrix(X)
    from believe14._core.distances import pairwise_distances

    return pairwise_distances(feature_matrix)


def classical_embedding(
    distances: FloatMatrix,
    n_components: int,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix, float, int, tuple[str, ...]]:
    """Compute Torgerson's classical-scaling coordinates and full spectrum."""

    n_samples = distances.shape[0]
    validate_n_components(n_components, maximum=n_samples - 1)
    distance_scale = float(np.max(distances, initial=0.0))
    scaled_distances = (
        distances if distance_scale == 0.0 else distances / distance_scale
    )
    squared = scaled_distances * scaled_distances
    row_mean = np.mean(squared, axis=1, keepdims=True, dtype=np.float64)
    total_mean = float(np.mean(squared, dtype=np.float64))
    scaled_gram = -0.5 * (squared - row_mean - row_mean.T + total_mean)
    scaled_gram = (scaled_gram + scaled_gram.T) * 0.5
    values, vectors = linalg.eigh(scaled_gram, check_finite=False)
    order = np.argsort(values, kind="stable")[::-1]
    values = np.asarray(values[order], dtype=np.float64)
    vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
    scale = float(linalg.norm(scaled_gram, ord=2))
    if scale == 0.0:
        raise ValueError("The double-centered Gram matrix is numerically zero.")
    tolerance = np.finfo(np.float64).eps * n_samples * scale * 100.0
    positive = values > tolerance
    positive_rank = int(np.count_nonzero(positive))
    if positive_rank < n_components:
        raise ValueError(
            "The requested embedding dimension exceeds the positive numerical "
            "rank of the double-centered Gram matrix."
        )
    selected_values = values[:n_components]
    if np.any(selected_values <= tolerance):
        raise ValueError("The requested classical-MDS axes are not positive.")
    selected_vectors = vectors[:, :n_components]
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        embedding = (
            selected_vectors * np.sqrt(selected_values)[None, :] * distance_scale
        )
    residual = scaled_gram @ selected_vectors - selected_vectors * selected_values
    residual_norm = float(linalg.norm(residual) / scale)
    warnings: tuple[str, ...] = ()
    if np.any(values < -tolerance):
        warnings = (
            "The dissimilarities are non-Euclidean; the full Gram spectrum "
            "contains significant negative eigenvalues.",
        )
    largest_squared_state = max(
        float(np.max(np.abs(values), initial=0.0)),
        float(np.max(np.abs(scaled_gram), initial=0.0)),
    )
    if distance_scale > 0.0 and largest_squared_state > 0.0:
        state_log_scale = 2.0 * np.log(distance_scale) + np.log(largest_squared_state)
        if state_log_scale > np.log(np.finfo(np.float64).max):
            raise FloatingPointError("The classical-MDS Gram matrix overflows float64.")
        if state_log_scale < np.log(np.finfo(np.float64).smallest_subnormal):
            raise FloatingPointError(
                "The classical-MDS Gram matrix underflows float64."
            )
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        squared_scale = distance_scale * distance_scale
        spectrum = values * squared_scale
        gram = scaled_gram * squared_scale
    return embedding, spectrum, gram, residual_norm, positive_rank, warnings


def stress(distances: FloatMatrix, embedding: FloatMatrix) -> float:
    """Return unweighted raw stress, with every unordered pair counted once."""

    from believe14._core.distances import pairwise_distances

    residual = pairwise_distances(embedding) - distances
    return float(0.5 * np.sum(residual * residual, dtype=np.float64))


def smacof(
    distances: FloatMatrix,
    initial: FloatMatrix,
    *,
    max_iter: int,
    tol: float,
) -> tuple[FloatMatrix, float, int, bool, float]:
    """Minimize complete, unweighted raw stress by SMACOF majorization."""

    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    tolerance = validate_positive_real(tol, name="tol")
    distance_scale = float(np.max(distances, initial=0.0))
    if distance_scale <= 0.0:
        raise ValueError("SMACOF requires at least one positive dissimilarity.")
    scaled_distances = distances / distance_scale
    _, embedding = stable_center(np.asarray(initial, dtype=np.float64))
    embedding /= distance_scale
    previous = stress(scaled_distances, embedding)
    zero_tolerance = np.finfo(np.float64).eps * 100.0
    converged = False
    relative_change = np.inf

    from believe14._core.distances import pairwise_distances

    for _iteration in range(1, int(max_iter) + 1):
        embedded_distances = pairwise_distances(embedding)
        invalid = (embedded_distances <= zero_tolerance) & (
            scaled_distances > zero_tolerance
        )
        np.fill_diagonal(invalid, False)
        if np.any(invalid):
            raise FloatingPointError(
                "SMACOF encountered coincident embedded points with a positive "
                "target dissimilarity; choose a nondegenerate initialization."
            )
        ratios = np.zeros_like(scaled_distances)
        mask = embedded_distances > zero_tolerance
        ratios[mask] = scaled_distances[mask] / embedded_distances[mask]
        np.fill_diagonal(ratios, 0.0)
        majorizer = -ratios
        majorizer[np.diag_indices_from(majorizer)] = -np.sum(
            majorizer, axis=1, dtype=np.float64
        )
        updated = majorizer @ embedding / distances.shape[0]
        _, updated = stable_center(updated)
        current = stress(scaled_distances, updated)
        monotonic_tolerance = 1e3 * np.finfo(np.float64).eps * max(1.0, abs(previous))
        if current > previous + monotonic_tolerance:
            raise FloatingPointError("SMACOF stress increased beyond roundoff.")
        relative_change = abs(previous - current) / max(1.0, abs(previous))
        embedding = updated
        previous = current
        if relative_change <= tolerance:
            converged = True
            break

    with np.errstate(over="raise", invalid="raise", under="ignore"):
        final_embedding = embedding * distance_scale
        objective = previous * distance_scale * distance_scale
    if previous > 0.0 and objective == 0.0:
        raise FloatingPointError("The raw SMACOF stress underflows float64.")
    return (
        np.asarray(final_embedding, dtype=np.float64),
        float(objective),
        _iteration,
        converged,
        float(relative_change),
    )


def initialize_embedding(
    distances: FloatMatrix,
    n_components: int,
    *,
    init: Literal["classical", "random"],
    random_state: int | np.random.Generator | None,
) -> FloatMatrix:
    """Create an explicitly selected deterministic or local-random start."""

    if init == "classical":
        return classical_embedding(distances, n_components)[0]
    if init != "random":
        raise ValueError("init must be 'classical' or 'random'.")
    rng = make_rng(random_state)
    positive = distances[distances > 0.0]
    distance_scale = float(np.max(positive, initial=0.0))
    if distance_scale <= 0.0:
        raise ValueError("Random initialization requires positive dissimilarities.")
    scale = float(np.mean(positive / distance_scale, dtype=np.float64))
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = (
            rng.normal(scale=scale * 1e-2, size=(len(distances), n_components))
            * distance_scale
        )
    if not np.any(result):
        raise FloatingPointError("The random MDS initialization underflows float64.")
    _, result = stable_center(np.asarray(result, dtype=np.float64))
    return np.asarray(result, dtype=np.float64)


def validate_unit_interval(value: float, *, name: str) -> float:
    """Validate a real value in the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return result


def output_names(prefix: str, n_components: int) -> NDArray[np.object_]:
    """Create deterministic feature names for an embedding."""

    return np.asarray(
        [f"{prefix}{index}" for index in range(n_components)], dtype=object
    )
