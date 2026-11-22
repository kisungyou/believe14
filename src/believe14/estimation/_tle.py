# mypy: disallow-subclassing-any=False
"""Local intrinsic dimension from distances measured about moving centers."""

from __future__ import annotations

from fractions import Fraction
from math import fsum, log
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

from believe14._core.diagnostics import diagnostics
from believe14._core.linalg import stable_mean
from believe14._core.validation import validate_features, validate_n_neighbors
from believe14.neighbors import ExactNeighborGraph

from ._common import log_distance_ratios, neighbor_distances


def _subnormal_neighborhood(
    X: NDArray[np.float64], sample: int, k: int, outer: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover a tiny neighborhood before float64 distance rounding breaks ties."""
    offsets = X - X[sample]
    # Each initially selected offset has all coordinates bounded by outer.
    # Thus a ball of radius sqrt(p)*outer contains at least k observations.
    # Twice this width also covers the rounding in the candidate bound.
    bound = (2 * np.sqrt(X.shape[1])) * outer
    candidates = np.flatnonzero(np.all(np.abs(offsets) <= bound, axis=1))
    candidates = candidates[candidates != sample]
    # Every positive binary64 subnormal becomes normal after scaling by 2^64.
    scaled = np.ldexp(offsets[candidates], 64)
    lengths = np.hypot.reduce(scaled, axis=1)
    order = np.argsort(lengths, kind="stable")[:k]
    return scaled[order], lengths[order]


def _log_radial_deficits(
    query: NDArray[np.float64],
    points: NDArray[np.float64],
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log(1-r_i²/r²), resolving nearly tied outer radii."""
    r = distances[-1]
    with np.errstate(divide="ignore"):
        result = np.log((r - distances) / r) + np.log1p(distances / r)
    # When radii agree to rounding precision, identify the true outer
    # surface among selected neighbors using their exact float coordinates.
    near = np.flatnonzero(r - distances <= 4 * np.finfo(float).eps * r)
    if len(near) > 1:
        origin = [Fraction(x) for x in query]
        squared = [
            sum(
                (
                    (Fraction(x) - q) ** 2
                    for x, q in zip(points[index], origin, strict=True)
                ),
                Fraction(0),
            )
            for index in near
        ]
        outer = max(squared)
        for index, value in zip(near, squared, strict=True):
            deficit = (outer - value) / outer
            result[index] = (
                log(deficit.numerator) - log(deficit.denominator)
                if deficit
                else -np.inf
            )
    return np.asarray(result, dtype=np.float64)


def _adjusted_log_radii(
    query: NDArray[np.float64],
    points: NDArray[np.float64],
    distances: NDArray[np.float64],
    log_deficit: NDArray[np.float64],
    *,
    reflected: bool,
) -> NDArray[np.float64]:
    """Log of first-contact radii in a unit ball, excluding self measurements."""
    radius = distances[-1]
    i, j = np.where(~np.eye(len(points), dtype=bool))
    offsets = points - query
    starts = offsets / radius
    # Measure separations before normalization. Distinct points very close
    # to one another can round to the same vector relative to a distant query.
    if reflected:
        starts = -starts
        with np.errstate(over="ignore"):
            displacement = offsets[j] + offsets[i]
            separation = np.hypot.reduce(displacement, axis=1)
        halved = ~np.isfinite(separation)
        if np.any(halved):
            displacement[halved] = offsets[j[halved]] / 2 + offsets[i[halved]] / 2
        # Opposing offsets can lose the query's small displacement in both
        # subtractions. Recover the original four-term sum at those entries.
        cancellation = (
            (np.abs(displacement) < np.sqrt(np.finfo(float).eps) * np.abs(offsets[i]))
            & (query != 0)
            & ~halved[:, None]
        )
        for pair, feature in zip(*np.nonzero(cancellation), strict=True):
            q = query[feature]
            displacement[pair, feature] = fsum(
                (points[j[pair], feature], -q, points[i[pair], feature], -q)
            )
    else:
        displacement = points[j] - points[i]
        halved = np.zeros(len(i), dtype=bool)
    separation = np.hypot.reduce(displacement, axis=1)
    if np.any(separation == 0):
        raise ValueError(
            "A neighbor coincides with a moving center; TLE has a zero adjusted radius."
        )
    log_separation = log_distance_ratios(separation, distances[-1:])
    log_separation[halved] += np.log(2)

    # Squaring ||(1-t)*x - v||=t gives a*u² + 2*b*u - 1 = 0
    # for u=t/||v-x||. Use separate forms of the positive root to avoid
    # cancellation for negative b, including a=0 at the ball's surface.
    log_a = log_deficit[i]
    unit = displacement / separation[:, None]
    b = -np.einsum("ij,ij->i", starts[i], unit)
    h = np.hypot(b, np.exp(log_a / 2))
    boundary = np.isneginf(log_deficit[j])
    positive = ~boundary & (b >= 0)
    negative = ~boundary & (b < 0)
    if np.any(negative & np.isneginf(log_a)) or np.any(positive & (h + b == 0)):
        raise FloatingPointError("Cannot resolve the TLE contact at the ball boundary.")
    log_t = np.zeros_like(b)
    log_t[positive] = log_separation[positive] - np.log((h + b)[positive])
    log_t[negative] = (
        log_separation[negative] + np.log((h - b)[negative]) - log_a[negative]
    )
    # When a contact nearly touches the outer surface, subtracting logarithms
    # can round a small negative log to zero. Solve instead for 1-t using
    # the radial deficit of the target and 1-x·v = (a_x+a_v+||v-x||²)/2.
    near_surface = ~boundary & (log_t > -0.25)
    if np.any(near_surface):
        log_d = log_separation[near_surface]
        target = log_deficit[j[near_surface]]
        log_half_sum = np.logaddexp(
            np.logaddexp(log_a[near_surface], target), 2 * log_d
        ) - np.log(2)
        log_denom = np.logaddexp(log_half_sum, log_d + np.log(h[near_surface]))
        complement = np.exp(target - log_denom)
        log_t[near_surface] = np.log1p(-complement)
    if not np.isfinite(log_t).all() or np.any(log_t > 128 * np.finfo(float).eps):
        raise FloatingPointError(
            "Adjusted TLE radii must be positive and inside the neighborhood."
        )
    return np.asarray(np.minimum(log_t, 0.0), dtype=np.float64)


class TightLocalEstimator(BaseEstimator):
    """Estimate one intrinsic dimension per observation using reflected moving centers.

    Uses the TLE variant of Amsaleg et al. (2022), with ordered distinct
    neighbor pairs and their reflected centers. The arithmetic mean of local
    estimates is provided as a descriptive global summary. Neighborhoods are
    exact Euclidean neighborhoods excluding the observation itself.
    """

    def __init__(self, *, n_neighbors: int = 20) -> None:
        self.n_neighbors = n_neighbors

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        neighbors: ExactNeighborGraph | None = None,
    ) -> Self:
        """Fit the pointwise likelihoods at the selected neighborhood size."""
        del y
        features = validate_features(self, X, reset=True, min_samples=4)
        k = validate_n_neighbors(self.n_neighbors, n_samples=len(features), minimum=3)
        distances, indices = neighbor_distances(features, k, neighbors=neighbors)
        local = np.empty(len(features))
        for i, (row, order) in enumerate(zip(distances, indices, strict=True)):
            points = features[order]
            query = features[i]
            if row[-1] < np.finfo(float).tiny:
                points, row = _subnormal_neighborhood(features, i, k, row[-1])
                query = np.zeros_like(query)
            deficit = _log_radial_deficits(query, points, row)
            direct = _adjusted_log_radii(query, points, row, deficit, reflected=False)
            reflected = _adjusted_log_radii(query, points, row, deficit, reflected=True)
            denominator = -float(direct.sum() + reflected.sum())
            if denominator <= 0:
                raise ValueError(
                    "TLE has zero likelihood denominator at a neighborhood."
                )
            local[i] = 2 * k * (k - 1) / denominator
        if not np.all(np.isfinite(local)):
            raise FloatingPointError("Local TLE dimensions must be finite.")
        self.local_dimensions_ = local
        self.local_dimensions_by_k_ = local[:, None].copy()
        self.k_values_ = np.array([k], dtype=np.int64)
        self.dimension_ = float(stable_mean(local[:, None])[0])
        self.dimensions_by_k_ = np.array([self.dimension_])
        self.diagnostics_ = diagnostics("tight_local_reflected_mle")
        return self
