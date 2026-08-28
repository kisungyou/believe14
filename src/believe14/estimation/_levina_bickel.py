# mypy: disallow-subclassing-any=False
"""Levina--Bickel Poisson-process maximum likelihood estimator."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from believe14._core.diagnostics import diagnostics
from believe14._core.validation import validate_features

from ._common import neighbor_distances, positive_integer


class LevinaBickelMLE(BaseEstimator):
    """Estimate dimension from local Poisson-process likelihoods.

    For each ``k`` in ``[k_min, k_max]``, local estimates use Equation (8) of
    Levina and Bickel.  ``bias_correction=True`` replaces its ``k - 1`` factor
    by the paper's asymptotically unbiased ``k - 2`` factor.  The global result
    is the arithmetic mean over observations and then over ``k`` (Equation 9).

    References
    ----------
    Levina, E. and Bickel, P. J. (2004), Advances in NIPS 17, 777--784.
    """

    def __init__(
        self,
        *,
        k_min: int = 10,
        k_max: int = 20,
        bias_correction: bool = True,
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.bias_correction = bias_correction

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the averaged local likelihood estimator."""

        del y
        Xv = validate_features(self, X, reset=True, min_samples=5)
        k_min = positive_integer(self.k_min, name="k_min", minimum=3)
        k_max = positive_integer(self.k_max, name="k_max", minimum=3)
        if k_min > k_max:
            raise ValueError("k_min must not exceed k_max.")
        if k_max >= Xv.shape[0]:
            raise ValueError("k_max must be smaller than the number of samples.")
        if not isinstance(self.bias_correction, (bool, np.bool_)):
            raise TypeError("bias_correction must be boolean.")
        distances, _ = neighbor_distances(Xv, k_max)
        estimates = np.empty((k_max - k_min + 1, Xv.shape[0]), dtype=np.float64)
        for offset, k_value in enumerate(range(k_min, k_max + 1)):
            log_ratios = np.log(
                distances[:, [k_value - 1]] / distances[:, : k_value - 1]
            )
            denominator = np.sum(log_ratios, axis=1)
            if np.any(denominator <= 0.0):
                raise ValueError(
                    "Neighbor-distance ties make at least one local likelihood "
                    "singular; increase k or remove the tied configuration."
                )
            numerator = k_value - 2 if self.bias_correction else k_value - 1
            estimates[offset] = float(numerator) / denominator
        by_k = np.mean(estimates, axis=1)
        self.dimension_ = float(np.mean(by_k))
        self.local_dimensions_ = np.mean(estimates, axis=0)
        self.dimensions_by_k_ = by_k
        self.k_values_ = np.arange(k_min, k_max + 1, dtype=np.int64)
        self.diagnostics_ = diagnostics(
            "levina_bickel_local_poisson_mle",
            residual_norm=float(np.std(by_k) / max(1.0, abs(self.dimension_))),
            objective_value=float(np.var(by_k)),
        )
        return self
