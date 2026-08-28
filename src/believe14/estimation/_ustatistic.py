# mypy: disallow-subclassing-any=False
"""Hein--Audibert U-statistic intrinsic dimension estimator."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator

from believe14._core.diagnostics import diagnostics
from believe14._core.validation import make_rng, validate_features

from ._common import distances_without_duplicates, positive_integer


def _kernel_mean(
    distances: NDArray[np.float64], base_bandwidth: float, bandwidth_factor: float
) -> float:
    with np.errstate(over="ignore", invalid="raise"):
        ratios = np.minimum(
            distances / base_bandwidth / bandwidth_factor,
            1.0,
        )
    values = 1.0 - ratios * ratios
    return float(np.mean(values))


def _partition_kernel_mean(
    distances: NDArray[np.float64],
    order: NDArray[np.int64],
    divisor: int,
    base_bandwidth: float,
    bandwidth_factor: float,
) -> float:
    """Average the compact-kernel means for one paper subsample scale."""

    size = order.size // divisor
    used = order[: size * divisor]
    groups = [used[offset::divisor] for offset in range(divisor)]
    estimates: list[float] = []
    for left in range(divisor):
        for right in range(left, divisor):
            block = distances[np.ix_(groups[left], groups[right])]
            if left == right:
                values = block[np.triu_indices(size, k=1)]
            else:
                values = block.ravel()
            estimates.append(_kernel_mean(values, base_bandwidth, bandwidth_factor))
    return float(np.mean(estimates))


class UStatisticDimension(BaseEstimator):
    """Estimate dimension with the Hein--Audibert five-scale U-statistic.

    The implementation follows Sections 3.1--3.2: the compact kernel is
    ``k(x)=(1-x)_+``, subsample divisors are 1 through 5, weights are ``1/r``,
    and the selected integer candidate has the smallest absolute log-log slope.
    A local random permutation makes the interleaved paper partitions independent
    of an externally sorted input while preserving their i.i.d. interpretation.

    References
    ----------
    Hein, M. and Audibert, J.-Y. (2005), ICML, 289--296.
    """

    def __init__(
        self,
        *,
        max_dimension: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.max_dimension = max_dimension
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Evaluate all paper candidate slopes and choose the flattest."""

        del y
        Xv = validate_features(self, X, reset=True, min_samples=10)
        max_allowed = min(Xv.shape[1], 15)
        if self.max_dimension is None:
            max_dimension = max_allowed
        else:
            max_dimension = positive_integer(self.max_dimension, name="max_dimension")
            if max_dimension > max_allowed:
                raise ValueError(
                    "max_dimension cannot exceed min(n_features, 15), the "
                    "advertised range in Hein and Audibert."
                )
        distances = distances_without_duplicates(Xv)
        working = distances.copy()
        np.fill_diagonal(working, np.inf)
        nearest = np.min(working, axis=1)
        nearest_scale = float(np.max(nearest, initial=0.0))
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            base_bandwidth = float(
                np.mean(nearest / nearest_scale, dtype=np.float64) * nearest_scale
            )
        if not np.isfinite(base_bandwidth) or base_bandwidth <= 0.0:
            raise ValueError("The mean nearest-neighbor bandwidth is not positive.")
        rng = make_rng(self.random_state)
        order = np.asarray(rng.permutation(Xv.shape[0]), dtype=np.int64)
        divisors = np.arange(1, 6, dtype=np.int64)
        dimensions = np.arange(1, max_dimension + 1, dtype=np.int64)
        bandwidth_factors = np.empty((max_dimension, 5), dtype=np.float64)
        log_bandwidths = np.empty_like(bandwidth_factors)
        kernel_means = np.empty_like(bandwidth_factors)
        log_statistics = np.empty_like(bandwidth_factors)
        slopes = np.empty(max_dimension, dtype=np.float64)
        regression_residuals = np.empty(max_dimension, dtype=np.float64)
        log_total = np.log(float(Xv.shape[0]))
        for dim_index, candidate in enumerate(dimensions):
            for scale_index, divisor in enumerate(divisors):
                size = Xv.shape[0] // int(divisor)
                bandwidth_factor = (
                    Xv.shape[0] / float(size) * np.log(float(size)) / log_total
                ) ** (1.0 / float(candidate))
                bandwidth_factors[dim_index, scale_index] = bandwidth_factor
                log_bandwidths[dim_index, scale_index] = np.log(
                    base_bandwidth
                ) + np.log(bandwidth_factor)
                kernel_means[dim_index, scale_index] = _partition_kernel_mean(
                    distances,
                    order,
                    int(divisor),
                    base_bandwidth,
                    bandwidth_factor,
                )
                if kernel_means[dim_index, scale_index] > 0.0:
                    log_statistics[dim_index, scale_index] = (
                        np.log(kernel_means[dim_index, scale_index])
                        - int(candidate) * log_bandwidths[dim_index, scale_index]
                    )
            if np.any(kernel_means[dim_index] <= 0.0):
                raise ValueError(
                    "A compact-kernel U-statistic is zero at an advertised scale; "
                    "the five-scale log regression is undefined for this dataset."
                )
            x_values = log_bandwidths[dim_index]
            y_values = log_statistics[dim_index]
            weights = 1.0 / divisors.astype(np.float64)
            x_mean = float(np.sum(weights * x_values) / np.sum(weights))
            y_mean = float(np.sum(weights * y_values) / np.sum(weights))
            centered_x = x_values - x_mean
            denominator = float(np.sum(weights * centered_x**2))
            if denominator == 0.0:
                raise ValueError("The five paper bandwidths collapse on the log scale.")
            slope = float(
                np.sum(weights * centered_x * (y_values - y_mean)) / denominator
            )
            residual = y_values - (y_mean + slope * centered_x)
            slopes[dim_index] = slope
            regression_residuals[dim_index] = float(np.sum(weights * residual**2))
        winner = int(np.argmin(np.abs(slopes)))
        warning_messages: tuple[str, ...] = ()
        if winner in {0, max_dimension - 1}:
            warning_messages = (
                "The flattest U-statistic slope occurs on the candidate boundary.",
            )
        self.dimension_ = float(dimensions[winner])
        self.candidate_dimensions_ = dimensions
        self.bandwidth_factors_ = bandwidth_factors
        self.log_bandwidths_ = log_bandwidths
        self.kernel_means_ = kernel_means
        self.log_u_statistics_ = log_statistics
        self.slopes_ = slopes
        self.sample_order_ = order
        self.base_bandwidth_ = base_bandwidth
        self.diagnostics_ = diagnostics(
            "hein_audibert_weighted_slope_search",
            n_iter=max_dimension,
            residual_norm=abs(float(slopes[winner])),
            objective_value=float(regression_residuals[winner]),
            warnings=warning_messages,
        )
        return self
