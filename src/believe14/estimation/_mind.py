# mypy: disallow-subclassing-any=False
"""Minimum-neighbor-distance maximum likelihood dimension estimator."""

from __future__ import annotations

from typing import Self

from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from believe14._core.diagnostics import diagnostics
from believe14._core.validation import validate_features

from ._common import (
    maximize_mind_likelihood,
    normalized_minimum_distances,
    positive_integer,
)


class MiNDML(BaseEstimator):
    """Maximum likelihood for normalized minimum-neighbor distances.

    The statistic is ``rho_i = T_1(i) / T_{k+1}(i)`` and its density is
    ``g(r; k, d) = k d r**(d-1) (1-r**d)**(k-1)``.  The likelihood is optimized
    continuously on ``[1, max_dimension]``; this class does not silently round
    the estimate to an integer.

    References
    ----------
    Lombardi, G. et al. (2011), ECML PKDD, LNCS 6912, 374--389.
    """

    def __init__(
        self, *, n_neighbors: int = 10, max_dimension: int | None = None
    ) -> None:
        self.n_neighbors = n_neighbors
        self.max_dimension = max_dimension

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Maximize the normalized minimum-distance likelihood."""

        del y
        Xv = validate_features(self, X, reset=True, min_samples=4)
        n_neighbors = positive_integer(self.n_neighbors, name="n_neighbors", minimum=1)
        if n_neighbors + 1 >= Xv.shape[0]:
            raise ValueError("n_neighbors + 1 must be smaller than n_samples.")
        if self.max_dimension is None:
            max_dimension = Xv.shape[1]
        else:
            max_dimension = positive_integer(self.max_dimension, name="max_dimension")
            if max_dimension > Xv.shape[1]:
                raise ValueError("max_dimension cannot exceed n_features.")
        ratios, _, _ = normalized_minimum_distances(Xv, n_neighbors)
        estimate, log_likelihood, score, converged, evaluations = (
            maximize_mind_likelihood(ratios, n_neighbors, max_dimension)
        )
        warning_messages: tuple[str, ...] = ()
        if estimate in {1.0, float(max_dimension)}:
            warning_messages = (
                "The likelihood maximum is on the advertised dimension boundary.",
            )
        self.dimension_ = estimate
        self.normalized_distances_ = ratios
        self.log_likelihood_ = log_likelihood
        self.diagnostics_ = diagnostics(
            "bounded_scalar_mind_ml",
            converged=converged,
            n_iter=evaluations,
            residual_norm=score / max(1.0, float(Xv.shape[0])),
            objective_value=-log_likelihood,
            warnings=warning_messages,
        )
        return self
