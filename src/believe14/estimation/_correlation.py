# mypy: disallow-subclassing-any=False
"""Grassberger--Procaccia correlation dimension."""

from __future__ import annotations

from numbers import Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

from believe14._core.diagnostics import diagnostics
from believe14._core.validation import validate_features

from ._common import distances_without_duplicates, positive_integer


class CorrelationDimension(BaseEstimator):
    """Estimate correlation dimension from a declared radius interval.

    The fitted slope is that of ``log(C_n(r))`` against ``log(r)``, where
    ``C_n(r)`` is the fraction of unordered pairs with distance strictly below
    ``r``.  If ``radii`` is omitted, logarithmically spaced radii between two
    empirical positive-distance quantiles define the advertised scale range.

    Parameters
    ----------
    radii : array-like of shape (n_radii,), default=None
        Absolute, positive, strictly increasing radii.  Supplying radii makes
        the scale choice entirely user controlled.
    n_radii : int, default=20
        Number of logarithmically spaced radii used when ``radii`` is omitted.
    quantile_range : tuple of float, default=(0.05, 0.2)
        Pair-distance quantiles defining the automatic scale interval.

    References
    ----------
    Grassberger, P. and Procaccia, I. (1983), Physica D 9, 189--208.
    """

    def __init__(
        self,
        *,
        radii: ArrayLike | None = None,
        n_radii: int = 20,
        quantile_range: tuple[float, float] = (0.05, 0.2),
    ) -> None:
        self.radii = radii
        self.n_radii = n_radii
        self.quantile_range = quantile_range

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Estimate the slope of the empirical correlation integral."""

        del y
        Xv = validate_features(self, X, reset=True, min_samples=3)
        distances = distances_without_duplicates(Xv)
        pair_distances = distances[np.triu_indices(Xv.shape[0], k=1)]
        radii = self._validated_radii(pair_distances)
        correlation = np.asarray(
            [np.mean(pair_distances < radius) for radius in radii],
            dtype=np.float64,
        )
        valid = (correlation > 0.0) & (correlation < 1.0)
        if np.count_nonzero(valid) < 2:
            raise ValueError(
                "The chosen scale interval produces fewer than two finite, "
                "non-saturated correlation-integral values."
            )
        radii = radii[valid]
        correlation = correlation[valid]
        log_radii = np.log(radii)
        log_correlation = np.log(correlation)
        centered = log_radii - np.mean(log_radii)
        denominator = float(centered @ centered)
        if denominator == 0.0:
            raise ValueError("The usable radii are not distinct on the log scale.")
        slope = float(centered @ (log_correlation - np.mean(log_correlation)))
        slope /= denominator
        intercept = float(np.mean(log_correlation) - slope * np.mean(log_radii))
        residual = log_correlation - (intercept + slope * log_radii)
        residual_norm = float(
            np.linalg.norm(residual) / max(1.0, np.linalg.norm(log_correlation))
        )
        if slope <= 0.0:
            raise ValueError(
                "The fitted correlation integral has nonpositive slope; choose a "
                "scale range in which C_n(r) grows with r."
            )
        self.dimension_ = slope
        self.radii_ = radii
        self.correlation_integral_ = correlation
        self.intercept_ = intercept
        self.diagnostics_ = diagnostics(
            "correlation_integral_log_ols",
            residual_norm=residual_norm,
            objective_value=float(residual @ residual),
            numerical_rank=2,
        )
        return self

    def _validated_radii(
        self, pair_distances: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self.radii is not None:
            radii = np.asarray(self.radii, dtype=np.float64)
            if radii.ndim != 1 or radii.size < 2:
                raise ValueError(
                    "radii must be a one-dimensional array of length >= 2."
                )
            if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise ValueError("radii must contain only finite positive values.")
            if np.any(np.diff(radii) <= 0.0):
                raise ValueError("radii must be strictly increasing.")
            return radii
        count = positive_integer(self.n_radii, name="n_radii", minimum=2)
        if len(self.quantile_range) != 2:
            raise ValueError("quantile_range must contain exactly two values.")
        lower, upper = self.quantile_range
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, Real)
            or not isinstance(upper, Real)
        ):
            raise TypeError("quantile_range values must be real scalars.")
        lower_float, upper_float = float(lower), float(upper)
        if not 0.0 < lower_float < upper_float < 1.0:
            raise ValueError("quantile_range must satisfy 0 < lower < upper < 1.")
        endpoints = np.quantile(pair_distances, [lower_float, upper_float])
        if endpoints[0] <= 0.0 or endpoints[0] >= endpoints[1]:
            raise ValueError(
                "The requested distance quantiles do not define a positive interval; "
                "supply explicit radii for discrete or heavily tied data."
            )
        return np.geomspace(endpoints[0], endpoints[1], count)
