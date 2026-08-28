"""Kernel construction and centering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .distances import pairwise_scaled_squared_distances


def rbf_kernel(
    X: NDArray[np.float64],
    Y: NDArray[np.float64] | None = None,
    *,
    gamma: float,
) -> NDArray[np.float64]:
    """Compute an RBF kernel as exp(-gamma * squared_distance)."""

    exponent = pairwise_scaled_squared_distances(X, Y, multiplier=gamma)
    return np.exp(-exponent)


def center_kernel(
    kernel: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Center a square Gram matrix using a common reference and scaling.

    The second return value is the stable column effect ``column_mean -
    total_mean`` needed to center query kernels.  The third value remains for
    the private cross-centering interface and is identically zero.
    """

    reference = float(kernel[0, 0])
    try:
        with np.errstate(over="raise", invalid="raise"):
            shifted = kernel - reference
        scale = float(np.max(np.abs(shifted), initial=0.0))
        if scale == 0.0:
            return np.zeros_like(kernel), np.zeros(kernel.shape[1]), 0.0
        scaled = shifted / scale
    except FloatingPointError:
        scale = float(np.max(np.abs(kernel), initial=0.0))
        if scale == 0.0:
            return np.zeros_like(kernel), np.zeros(kernel.shape[1]), 0.0
        scaled_kernel = kernel / scale
        scaled = scaled_kernel - scaled_kernel[0, 0]
    column_mean = np.mean(scaled, axis=0, dtype=np.float64)
    total_mean = float(np.mean(column_mean, dtype=np.float64))
    centered_scaled = scaled - column_mean[None, :] - column_mean[:, None] + total_mean
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        centered = centered_scaled * scale
        column_effect = (column_mean - total_mean) * scale
    return (
        np.asarray(centered * 0.5 + centered.T * 0.5, dtype=np.float64),
        np.asarray(column_effect, dtype=np.float64),
        0.0,
    )


def center_cross_kernel(
    kernel: NDArray[np.float64],
    *,
    training_column_mean: NDArray[np.float64],
    training_total_mean: float,
) -> NDArray[np.float64]:
    """Center query-to-training kernel values using training statistics."""

    del training_total_mean
    reference = kernel[:, :1]
    try:
        with np.errstate(over="raise", invalid="raise"):
            shifted = kernel - reference
        scale = float(np.max(np.abs(shifted), initial=0.0))
        if scale == 0.0:
            row_centered = np.zeros_like(kernel)
        else:
            scaled = shifted / scale
            row_centered = (
                scaled - np.mean(scaled, axis=1, keepdims=True, dtype=np.float64)
            ) * scale
    except FloatingPointError:
        scale = float(np.max(np.abs(kernel), initial=0.0))
        if scale == 0.0:
            row_centered = np.zeros_like(kernel)
        else:
            scaled_kernel = kernel / scale
            shifted = scaled_kernel - scaled_kernel[:, :1]
            row_centered = (
                shifted - np.mean(shifted, axis=1, keepdims=True, dtype=np.float64)
            ) * scale
    return np.asarray(row_centered - training_column_mean[None, :], dtype=np.float64)
