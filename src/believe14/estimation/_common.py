"""Private numerical helpers for intrinsic-dimension estimators."""

from __future__ import annotations

from math import comb, fsum
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import digamma, i0e, i1e

from believe14._core.distances import exact_neighbors, pairwise_distances


def positive_integer(value: int, *, name: str, minimum: int = 1) -> int:
    """Validate an integer lower bound without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def unit_interval(value: float, *, name: str, open_left: bool = False) -> float:
    """Validate a finite scalar in ``[0, 1)`` or ``(0, 1)``."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    lower_ok = result > 0.0 if open_left else result >= 0.0
    if not np.isfinite(result) or not lower_ok or result >= 1.0:
        left = "(" if open_left else "["
        raise ValueError(f"{name} must be in {left}0, 1).")
    return result


def distances_without_duplicates(
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute distances and reject exact or numerically collapsed duplicates."""

    distances = pairwise_distances(X)
    upper = distances[np.triu_indices(X.shape[0], k=1)]
    if np.any(upper == 0.0):
        raise ValueError(
            "X contains duplicate observations, or distinct observations whose "
            "distance underflows to zero; this estimator requires positive radii."
        )
    return distances


def neighbor_distances(
    X: NDArray[np.float64], n_neighbors: int
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return exact neighbors and reject zero neighbor radii."""

    distances, indices = exact_neighbors(X, n_neighbors)
    if np.any(distances <= 0.0):
        raise ValueError(
            "X contains duplicate observations, or distinct observations whose "
            "distance underflows to zero; neighbor radii must be positive."
        )
    return distances, indices


def normalized_minimum_distances(
    X: NDArray[np.float64], n_neighbors: int
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Compute Equation (11) of Ceruti et al. using ``k + 1`` neighbors."""

    distances, indices = neighbor_distances(X, n_neighbors + 1)
    ratios = distances[:, 0] / distances[:, -1]
    if np.any(ratios >= 1.0):
        raise ValueError(
            "The first and (k + 1)-st neighbor radii tie. The normalized "
            "minimum-distance likelihood requires every ratio to lie in (0, 1)."
        )
    return (
        np.asarray(ratios, dtype=np.float64),
        np.asarray(indices[:, :n_neighbors], dtype=np.int64),
        np.asarray(distances[:, :n_neighbors], dtype=np.float64),
    )


def mind_log_likelihood(
    dimension: float, ratios: NDArray[np.float64], n_neighbors: int
) -> float:
    """Log likelihood for ``g(r; k, d)`` from Lombardi et al."""

    powers = np.power(ratios, dimension)
    return float(
        ratios.size * np.log(float(n_neighbors) * dimension)
        + (dimension - 1.0) * np.sum(np.log(ratios))
        + (n_neighbors - 1.0) * np.sum(np.log1p(-powers))
    )


def mind_score(
    dimension: float, ratios: NDArray[np.float64], n_neighbors: int
) -> float:
    """Derivative of the MiND log likelihood with respect to dimension."""

    logs = np.log(ratios)
    powers = np.power(ratios, dimension)
    return float(
        ratios.size / dimension
        + np.sum(logs)
        - (n_neighbors - 1.0) * np.sum(powers * logs / (1.0 - powers))
    )


def maximize_mind_likelihood(
    ratios: NDArray[np.float64],
    n_neighbors: int,
    max_dimension: int,
) -> tuple[float, float, float, bool, int]:
    """Maximize the bounded MiND likelihood, including both endpoints."""

    if max_dimension == 1:
        value = mind_log_likelihood(1.0, ratios, n_neighbors)
        return 1.0, value, abs(mind_score(1.0, ratios, n_neighbors)), True, 1
    result = minimize_scalar(
        lambda value: -mind_log_likelihood(value, ratios, n_neighbors),
        bounds=(1.0, float(max_dimension)),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    candidates = (
        (1.0, -mind_log_likelihood(1.0, ratios, n_neighbors)),
        (float(result.x), float(result.fun)),
        (
            float(max_dimension),
            -mind_log_likelihood(float(max_dimension), ratios, n_neighbors),
        ),
    )
    estimate, objective = min(candidates, key=lambda item: item[1])
    score = abs(mind_score(estimate, ratios, n_neighbors))
    return estimate, -objective, score, bool(result.success), int(result.nfev)


def inverse_bessel_ratio(mean_resultant_length: float) -> float:
    """Approximation in Equation (8) of Ceruti et al."""

    eta = float(mean_resultant_length)
    if eta < 0.0 or eta >= 1.0 or not np.isfinite(eta):
        raise ValueError(
            "mean_resultant_length must be finite and in [0, 1); a value of "
            "one is an angular degeneracy with infinite concentration."
        )
    if eta < 0.53:
        return float(2.0 * eta + eta**3 + 5.0 * eta**5 / 6.0)
    if eta < 0.85:
        return float(-0.4 + 1.39 * eta + 0.43 / (1.0 - eta))
    # Factored form avoids cancellation as eta approaches its boundary at one.
    return float(1.0 / (eta * (1.0 - eta) * (3.0 - eta)))


def angular_parameters(
    X: NDArray[np.float64],
    neighbor_indices: NDArray[np.int64],
    neighbor_radii: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Estimate and average local von Mises parameters in Equations (6)--(8)."""

    n_samples, n_neighbors = neighbor_indices.shape
    rows, cols = np.triu_indices(n_neighbors, k=1)
    means = np.empty(n_samples, dtype=np.float64)
    concentrations = np.empty(n_samples, dtype=np.float64)
    for sample in range(n_samples):
        directions = X[neighbor_indices[sample]] - X[sample]
        directions = directions / neighbor_radii[sample, :, None]
        cosines = np.clip(directions @ directions.T, -1.0, 1.0)[rows, cols]
        angles = np.arccos(cosines)
        cosine_mean = float(np.mean(np.cos(angles)))
        sine_mean = float(np.mean(np.sin(angles)))
        means[sample] = np.arctan2(sine_mean, cosine_mean)
        resultant = float(np.hypot(cosine_mean, sine_mean))
        concentrations[sample] = inverse_bessel_ratio(resultant)
    return (
        float(np.mean(means)),
        float(np.mean(concentrations)),
        means,
        concentrations,
    )


def sample_unit_ball(
    rng: np.random.Generator, n_samples: int, dimension: int
) -> NDArray[np.float64]:
    """Draw uniformly from the unit ``dimension``-ball."""

    directions = rng.normal(size=(n_samples, dimension))
    norms = np.linalg.norm(directions, axis=1)
    while np.any(norms == 0.0):  # probability zero, retained as an explicit policy
        mask = norms == 0.0
        directions[mask] = rng.normal(size=(int(np.count_nonzero(mask)), dimension))
        norms[mask] = np.linalg.norm(directions[mask], axis=1)
    radii = np.power(rng.random(n_samples), 1.0 / float(dimension))
    return np.asarray(directions / norms[:, None] * radii[:, None], dtype=np.float64)


def norm_kl(
    data_dimension: float, reference_dimension: float, n_neighbors: int
) -> float:
    """Closed-form minimum-distance KL divergence, paper Equation (3)."""

    ratio = reference_dimension / data_dimension
    harmonic_k = fsum(1.0 / index for index in range(1, n_neighbors + 1))
    harmonic_previous = fsum(1.0 / index for index in range(1, n_neighbors))
    alternating = fsum(
        (-1.0) ** index * comb(n_neighbors, index) * float(digamma(1.0 + index / ratio))
        for index in range(n_neighbors + 1)
    )
    value = (
        harmonic_k * ratio
        - 1.0
        - harmonic_previous
        - np.log(ratio)
        - (n_neighbors - 1.0) * alternating
    )
    if value < 0.0 and value > -1e-10:
        return 0.0
    return float(value)


def _log_i0(value: float) -> float:
    return float(np.log(i0e(value)) + abs(value))


def von_mises_kl(
    mean_1: float, concentration_1: float, mean_2: float, concentration_2: float
) -> float:
    """Closed-form von Mises KL divergence, paper Equation (9)."""

    if concentration_1 == 0.0:
        bessel_ratio = 0.0
    else:
        bessel_ratio = float(i1e(concentration_1) / i0e(concentration_1))
    value = (
        _log_i0(concentration_2)
        - _log_i0(concentration_1)
        + bessel_ratio * (concentration_1 - concentration_2 * np.cos(mean_2 - mean_1))
    )
    if value < 0.0 and value > -1e-10:
        return 0.0
    return float(value)
