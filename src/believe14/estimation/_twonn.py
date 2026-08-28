# mypy: disallow-subclassing-any=False
"""Two-nearest-neighbor intrinsic dimension."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from believe14._core.diagnostics import diagnostics
from believe14._core.validation import validate_features

from ._common import neighbor_distances, unit_interval


class TwoNN(BaseEstimator):
    """Estimate dimension by the Facco et al. two-neighbor ratio regression.

    The largest 10 percent of ratios are discarded by default exactly as in the
    paper.  Positive distance ties are retained; exact neighbor ties are broken
    stably by input row index and only the radii enter the statistic.

    Parameters
    ----------
    discard_fraction : float, default=0.1
        Fraction of the largest ``r_2 / r_1`` ratios omitted before fitting the
        line through the origin.

    References
    ----------
    Facco, E. et al. (2017), Scientific Reports 7, 12140.
    """

    def __init__(self, *, discard_fraction: float = 0.1) -> None:
        self.discard_fraction = discard_fraction

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the origin-constrained empirical-CDF regression."""

        del y
        Xv = validate_features(self, X, reset=True, min_samples=3)
        discard = unit_interval(
            self.discard_fraction, name="discard_fraction", open_left=True
        )
        distances, _ = neighbor_distances(Xv, 2)
        ratios = np.sort(distances[:, 1] / distances[:, 0], kind="stable")
        n_samples = Xv.shape[0]
        n_keep = int(np.floor((1.0 - discard) * n_samples))
        if n_keep < 2 or n_keep >= n_samples:
            raise ValueError(
                "discard_fraction must retain at least two observations and omit "
                "at least one observation."
            )
        retained = ratios[:n_keep]
        x_values = np.log(retained)
        ranks = np.arange(1, n_keep + 1, dtype=np.float64)
        y_values = -np.log1p(-ranks / float(n_samples))
        denominator = float(x_values @ x_values)
        if denominator == 0.0:
            raise ValueError(
                "All retained neighbor ratios equal one, so the TwoNN slope is "
                "undefined."
            )
        slope = float(x_values @ y_values / denominator)
        residual = y_values - slope * x_values
        self.dimension_ = slope
        self.ratios_ = ratios
        self.fit_x_ = x_values
        self.fit_y_ = y_values
        self.diagnostics_ = diagnostics(
            "twonn_origin_ols",
            residual_norm=float(
                np.linalg.norm(residual) / max(1.0, np.linalg.norm(y_values))
            ),
            objective_value=float(residual @ residual),
            numerical_rank=1,
        )
        return self
