"""Private numerical helpers for intrinsic-dimension estimators."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import i0e, i1e

from believe14._core.distances import exact_neighbors, pairwise_distances
from believe14.neighbors import ExactNeighborGraph


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
    X: NDArray[np.float64],
    n_neighbors: int,
    *,
    neighbors: ExactNeighborGraph | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return exact neighbors and reject zero neighbor radii."""

    if neighbors is None:
        distances, indices = exact_neighbors(X, n_neighbors)
    else:
        distances, indices = neighbors.query(X, n_neighbors=n_neighbors)
    if np.any(distances <= 0.0):
        raise ValueError(
            "X contains duplicate observations, or distinct observations whose "
            "distance underflows to zero; neighbor radii must be positive."
        )
    return distances, indices


def normalized_minimum_distances(
    X: NDArray[np.float64],
    n_neighbors: int,
    *,
    neighbors: ExactNeighborGraph | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Compute Equation (11) of Ceruti et al. using ``k + 1`` neighbors."""

    distances, indices = neighbor_distances(X, n_neighbors + 1, neighbors=neighbors)
    ratios = distances[:, 0] / distances[:, -1]
    if np.any(ratios == 0.0):
        raise FloatingPointError("Normalized neighbor radii underflow float64.")
    if n_neighbors > 1 and np.any(ratios >= 1.0):
        raise ValueError(
            "The first and (k + 1)-st neighbor radii tie. The normalized "
            "minimum-distance likelihood requires every ratio to lie in (0, 1)."
        )
    return (
        np.asarray(ratios, dtype=np.float64),
        np.asarray(indices[:, :n_neighbors], dtype=np.int64),
        np.asarray(distances[:, :n_neighbors], dtype=np.float64),
    )


def log_distance_ratios(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Log ratios without forming potentially unrepresentable quotients.

    Close radii use log1p to retain differences lost by subtracting logarithms.
    """

    first, second = np.broadcast_arrays(numerator, denominator)
    result = np.log(first) - np.log(second)
    close = np.abs(result) < np.log(2.0)
    result[close] = np.log1p((first[close] - second[close]) / second[close])
    return np.asarray(result, dtype=np.float64)


def mind_log_likelihood(
    dimension: float, ratios: NDArray[np.float64], n_neighbors: int
) -> float:
    """Log likelihood for ``g(r; k, d)`` from Lombardi et al."""

    logs = np.log(ratios)
    if n_neighbors == 1:
        return float(ratios.size * np.log(dimension) + (dimension - 1.0) * np.sum(logs))
    return float(
        ratios.size * (np.log(float(n_neighbors)) + np.log(dimension))
        + (dimension - 1.0) * np.sum(logs)
        + (n_neighbors - 1.0) * np.sum(np.log(-np.expm1(dimension * logs)))
    )


def mind_score(
    dimension: float, ratios: NDArray[np.float64], n_neighbors: int
) -> float:
    """Derivative of the MiND log likelihood with respect to dimension."""

    logs = np.log(ratios)
    if n_neighbors == 1:
        return float(ratios.size / dimension + np.sum(logs))
    powers = np.power(ratios, dimension)
    return float(
        ratios.size / dimension
        + np.sum(logs)
        - (n_neighbors - 1.0) * np.sum(powers * logs / -np.expm1(dimension * logs))
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
    if not np.all(np.isfinite([estimate, objective, score])):
        raise FloatingPointError("The MiND likelihood or score is not finite.")
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
    n_angles = n_neighbors * (n_neighbors - 1) // 2
    means = np.empty(n_samples, dtype=np.float64)
    concentrations = np.empty(n_samples, dtype=np.float64)
    for sample in range(n_samples):
        directions = X[neighbor_indices[sample]] - X[sample]
        directions = directions / neighbor_radii[sample, :, None]
        # The half-angle formula retains angular differences near alignment.
        # Work one direction at a time to bound scratch storage by O(k*p+k^2).
        directions /= np.hypot.reduce(directions, axis=1)[:, None]
        angles = np.empty(n_angles)
        offset = 0
        for index in range(n_neighbors - 1):
            rest = directions[index + 1 :]
            chord = np.hypot.reduce(rest - directions[index], axis=1)
            antichord = np.hypot.reduce(rest + directions[index], axis=1)
            angles[offset : offset + len(rest)] = 2 * np.arctan2(chord, antichord)
            offset += len(rest)
        cosine_mean = float(np.mean(np.cos(angles)))
        sine_mean = float(np.mean(np.sin(angles)))
        means[sample] = np.arctan2(sine_mean, cosine_mean)
        resultant = float(np.hypot(cosine_mean, sine_mean))
        if resultant < 0.85:
            concentrations[sample] = inverse_bessel_ratio(resultant)
        else:
            # 1-eta^2 is the total variance of unit vectors on the circle.
            # Rotate by the first angle, then shift their common real part.
            delta = angles - angles[0]
            deficit_sq = np.var(-2 * np.sin(delta / 2) ** 2) + np.var(np.sin(delta))
            if deficit_sq == 0:
                raise ValueError("Identical local angles have infinite concentration.")
            with np.errstate(over="raise", divide="raise"):
                concentrations[sample] = (1 + resultant) / (
                    resultant * (3 - resultant) * deficit_sq
                )
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
    """Minimum-distance KL using an exponentially weighted density integral.

    With u = r**data_dimension and t = -k*log(1-u), t has unit
    exponential density. This avoids the exponentially ill-conditioned
    alternating binomial sum in Equation (3), including for large k.
    """

    if (
        not np.all(np.isfinite([data_dimension, reference_dimension]))
        or min(data_dimension, reference_dimension) <= 0.0
    ):
        raise ValueError("KL dimensions must be finite and positive.")
    k = positive_integer(n_neighbors, name="n_neighbors")
    ratio = reference_dimension / data_dimension
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise FloatingPointError("The dimension ratio is not representable.")
    if ratio == 1.0:
        return 0.0
    delta = ratio - 1.0
    log_ratio = np.log(ratio)

    def integrand(t: float) -> float:
        if t == 0.0 or t > 740.0:
            return 0.0
        z = t / k
        log_u = np.log(-np.expm1(-z)) if z < np.log(2.0) else np.log1p(-np.exp(-z))
        if k == 1:
            complement = 0.0
        elif abs(delta) < 0.5:
            # Compute the ratio of complements directly near equal dimensions.
            complement = np.log1p(-np.exp(log_u + z) * np.expm1(delta * log_u))
        else:
            powered_log = ratio * log_u
            log_complement = (
                np.log(-np.expm1(powered_log))
                if powered_log > -np.log(2.0)
                else np.log1p(-np.exp(powered_log))
            )
            complement = log_complement + z
        return float(np.exp(-t) * (-log_ratio - delta * log_u - (k - 1) * complement))

    result = quad(
        integrand, 0.0, np.inf, epsabs=2e-12, epsrel=2e-10, limit=200, full_output=1
    )
    value, error = float(result[0]), float(result[1])
    if (
        len(result) != 3
        or not np.all(np.isfinite([value, error]))
        or error > max(1e-10, abs(value) * 1e-8)
    ):
        raise FloatingPointError("Minimum-distance KL integration did not converge.")
    if value < -max(error * 4.0, 1e-12):
        raise FloatingPointError("Minimum-distance KL is significantly negative.")
    return max(0.0, value)


def von_mises_kl(
    mean_1: float, concentration_1: float, mean_2: float, concentration_2: float
) -> float:
    """Closed-form von Mises KL divergence, paper Equation (9)."""

    ratio = float(i1e(concentration_1) / i0e(concentration_1))
    if concentration_1 < 1e5:
        deficit = 1.0 - ratio
    else:
        # Large-k expansion of 1-I1(k)/I0(k) through cubic order.
        inv = 1.0 / concentration_1
        deficit = inv * (0.5 + inv * (0.125 + 0.125 * inv))
    value = (
        np.log(i0e(concentration_2))
        - np.log(i0e(concentration_1))
        + deficit * (concentration_2 - concentration_1)
        + (ratio * concentration_2) * (2.0 * np.sin((mean_2 - mean_1) / 2.0) ** 2)
    )
    if value < 0.0 and value > -1e-10:
        return 0.0
    if not np.isfinite(value) or value < 0.0:
        raise FloatingPointError(
            "The von Mises KL is non-finite or significantly negative."
        )
    return float(value)
