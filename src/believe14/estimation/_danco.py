# mypy: disallow-subclassing-any=False
"""DANCo intrinsic-dimension estimator."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from believe14._core.diagnostics import diagnostics
from believe14._core.linalg import centered_svd, numerical_rank
from believe14._core.validation import make_rng, validate_features

from ._common import (
    angular_parameters,
    maximize_mind_likelihood,
    norm_kl,
    normalized_minimum_distances,
    positive_integer,
    sample_unit_ball,
    von_mises_kl,
)


class DANCo(BaseEstimator):
    """Dimension from Angle and Norm Concentration (DANCo).

    For every integer candidate from two through ``max_dimension``, one equally
    sized uniform unit-ball calibration sample is drawn using a local random
    generator.  Equation (13)'s sum of the norm and von Mises KL divergences
    chooses the dimension.  Equal-distance neighbors are ordered stably by row
    index; a tie between the first and ``(k+1)``-st radii is rejected because the
    norm likelihood is then singular.  The 0.1.0 angular model is defined only
    for intrinsic dimensions of at least two; collinear data are rejected.

    References
    ----------
    Ceruti, C. et al. (2014), Pattern Recognition 47, 2569--2581.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 10,
        max_dimension: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.max_dimension = max_dimension
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Calibrate and minimize the joint paper divergence."""

        del y
        Xv = validate_features(self, X, reset=True, min_samples=5)
        n_neighbors = positive_integer(self.n_neighbors, name="n_neighbors", minimum=2)
        if n_neighbors + 1 >= Xv.shape[0]:
            raise ValueError("n_neighbors + 1 must be smaller than n_samples.")
        if self.max_dimension is None:
            max_dimension = Xv.shape[1]
        else:
            max_dimension = positive_integer(
                self.max_dimension, name="max_dimension", minimum=2
            )
            if max_dimension > Xv.shape[1]:
                raise ValueError("max_dimension cannot exceed n_features.")
        if max_dimension < 2:
            raise ValueError(
                "max_dimension must be at least 2 because DANCo's angular "
                "concentration model is not defined for one-dimensional data."
            )
        _, _, singular_values, _ = centered_svd(Xv)
        rank = numerical_rank(singular_values, shape=Xv.shape)
        if rank < 2:
            raise ValueError(
                "DANCo's angular statistic is degenerate for collinear data; "
                "believe14 0.1.0 supports intrinsic dimensions of at least 2."
            )
        ratios, indices, radii = normalized_minimum_distances(Xv, n_neighbors)
        data_ml, _, _, data_converged, data_evaluations = maximize_mind_likelihood(
            ratios, n_neighbors, max_dimension
        )
        data_mean, data_concentration, local_means, local_concentrations = (
            angular_parameters(Xv, indices, radii)
        )
        rng = make_rng(self.random_state)
        candidates = np.arange(2, max_dimension + 1, dtype=np.int64)
        n_candidates = candidates.size
        scores = np.empty(n_candidates, dtype=np.float64)
        calibration_ml = np.empty(n_candidates, dtype=np.float64)
        calibration_means = np.empty(n_candidates, dtype=np.float64)
        calibration_concentrations = np.empty(n_candidates, dtype=np.float64)
        evaluations = data_evaluations
        all_converged = data_converged
        for index, candidate in enumerate(candidates):
            calibration = sample_unit_ball(rng, Xv.shape[0], int(candidate))
            cal_ratios, cal_indices, cal_radii = normalized_minimum_distances(
                calibration, n_neighbors
            )
            cal_ml, _, _, converged, count = maximize_mind_likelihood(
                cal_ratios, n_neighbors, max_dimension
            )
            cal_mean, cal_concentration, _, _ = angular_parameters(
                calibration, cal_indices, cal_radii
            )
            calibration_ml[index] = cal_ml
            calibration_means[index] = cal_mean
            calibration_concentrations[index] = cal_concentration
            scores[index] = norm_kl(data_ml, cal_ml, n_neighbors) + von_mises_kl(
                data_mean,
                data_concentration,
                cal_mean,
                cal_concentration,
            )
            evaluations += count
            all_converged = all_converged and converged
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError(
                "DANCo calibration produced a non-finite KL divergence."
            )
        winner = int(np.argmin(scores))
        warning_messages: tuple[str, ...] = ()
        if winner in {0, n_candidates - 1}:
            warning_messages = (
                "The minimum DANCo divergence occurs on the candidate boundary.",
            )
        self.dimension_ = float(candidates[winner])
        self.normalized_distances_ = ratios
        self.local_angle_means_ = local_means
        self.local_angle_concentrations_ = local_concentrations
        self.angle_mean_ = data_mean
        self.angle_concentration_ = data_concentration
        self.norm_dimension_ = data_ml
        self.candidate_dimensions_ = candidates
        self.calibration_norm_dimensions_ = calibration_ml
        self.calibration_angle_means_ = calibration_means
        self.calibration_angle_concentrations_ = calibration_concentrations
        self.divergences_ = scores
        self.diagnostics_ = diagnostics(
            "danco_equation_13_calibration",
            converged=all_converged,
            n_iter=evaluations,
            objective_value=float(scores[winner]),
            numerical_rank=rank,
            warnings=warning_messages,
        )
        return self
