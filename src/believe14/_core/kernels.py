"""Kernel construction and centering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .distances import pairwise_scaled_squared_distances
from .linalg import stable_center, stable_mean


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
    """Compute J K J, plus the training column effect for Nyström extension.

    The two-way contrast removes large row and column offsets before centering.
    This is equivalent to replacing each feature-space vector by its difference
    from the first training observation. J annihilates that translation.
    """

    try:
        with np.errstate(over="raise", invalid="raise"):
            differences = kernel - kernel[0]
            contrast = differences - differences[:, :1]
        _, left_centered = stable_center(contrast)
        _, transposed = stable_center(left_centered.T)
        centered = transposed.T
    except FloatingPointError:
        # The anchored contrast can exceed float64 even when J K J fits.
        # In that case apply each centering operator directly.
        _, left_centered = stable_center(kernel)
        _, transposed = stable_center(left_centered.T)
        centered = transposed.T
    _, row_centered_t = stable_center(kernel.T)
    column_effect = stable_mean(row_centered_t.T)
    # Only entries that actually differ require symmetrization. Equal
    # subnormal entries would change under two separate halvings.
    different = centered != centered.T
    result = centered.copy()
    result[different] = centered[different] * 0.5 + centered.T[different] * 0.5
    return result, column_effect, 0.0


def center_cross_kernel(
    kernel: NDArray[np.float64],
    *,
    training_column_mean: NDArray[np.float64],
    training_total_mean: float,
) -> NDArray[np.float64]:
    """Apply independent row centering and the fitted training column effect."""

    del training_total_mean
    # Each query must be centered independently of every other query.
    _, centered_t = stable_center(kernel.T)
    return np.asarray(centered_t.T - training_column_mean[None, :], dtype=np.float64)
