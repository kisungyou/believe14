from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from believe14.api import FitDiagnostics
from believe14.estimation import (
    CorrelationDimension,
    DANCo,
    LevinaBickelMLE,
    MiNDML,
    TwoNN,
    UStatisticDimension,
)


def _line_fixture() -> np.ndarray:
    return np.array([[0.0], [1.0], [3.0], [7.0]])


@pytest.mark.parametrize(
    "radii, message",
    [
        ([1.0], "one-dimensional array of length"),
        ([[1.0, 2.0]], "one-dimensional array of length"),
        ([1.0, np.inf], "finite positive"),
        ([-1.0, 1.0], "finite positive"),
        ([2.0, 1.0], "strictly increasing"),
    ],
)
def test_correlation_dimension_rejects_invalid_custom_radii(
    radii: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CorrelationDimension(radii=radii).fit(_line_fixture())


def test_correlation_dimension_rejects_saturated_and_flat_scale_ranges() -> None:
    X = _line_fixture()
    with pytest.raises(ValueError, match="fewer than two"):
        CorrelationDimension(radii=[0.1, 0.2]).fit(X)
    with pytest.raises(ValueError, match="nonpositive slope"):
        CorrelationDimension(radii=[1.1, 1.2]).fit(X)


@pytest.mark.parametrize(
    "estimator, message",
    [
        (CorrelationDimension(n_radii=True), "n_radii must be an integer"),
        (CorrelationDimension(n_radii=1), "n_radii must be at least 2"),
        (
            CorrelationDimension(quantile_range=(0.1,)),
            "quantile_range must contain exactly two",
        ),
        (
            CorrelationDimension(quantile_range=(True, 0.2)),
            "quantile_range values must be real",
        ),
        (TwoNN(discard_fraction=True), "discard_fraction must be a real"),
        (TwoNN(discard_fraction=0.0), r"discard_fraction must be in \(0, 1\)"),
        (LevinaBickelMLE(k_min=2, k_max=3), "k_min must be at least 3"),
        (LevinaBickelMLE(k_min=3.5, k_max=4), "k_min must be an integer"),
        (MiNDML(n_neighbors=False), "n_neighbors must be an integer"),
        (MiNDML(n_neighbors=0), "n_neighbors must be at least 1"),
        (UStatisticDimension(max_dimension=False), "max_dimension must be an integer"),
        (DANCo(n_neighbors=1), "n_neighbors must be at least 2"),
        (DANCo(max_dimension=1), "max_dimension must be at least 2"),
    ],
)
def test_integer_and_scalar_parameter_failures_are_explicit(
    estimator: BaseEstimator, message: str
) -> None:
    X = np.random.default_rng(91).normal(size=(30, 2))
    with pytest.raises((TypeError, ValueError), match=message):
        estimator.fit(X)


def test_correlation_dimension_rejects_invalid_automatic_scale_specification() -> None:
    X = _line_fixture()
    with pytest.raises(ValueError, match="0 < lower < upper < 1"):
        CorrelationDimension(quantile_range=(0.3, 0.2)).fit(X)

    # Every pairwise distance between distinct standard basis vectors is sqrt(2).
    with pytest.raises(ValueError, match="do not define a positive interval"):
        CorrelationDimension().fit(np.eye(3))


def test_twonn_rejects_insufficient_trim_and_unit_neighbor_ratios() -> None:
    with pytest.raises(ValueError, match="retain at least two"):
        TwoNN(discard_fraction=0.75).fit(np.arange(5.0)[:, None])

    # A regular simplex has identical first- and second-neighbor distances.
    with pytest.raises(ValueError, match="ratios equal one"):
        TwoNN().fit(np.eye(4))


def test_levina_bickel_rejects_reversed_k_range_and_nonboolean_correction() -> None:
    X = np.random.default_rng(92).normal(size=(20, 2))
    with pytest.raises(ValueError, match="k_min must not exceed k_max"):
        LevinaBickelMLE(k_min=5, k_max=4).fit(X)
    with pytest.raises(TypeError, match="bias_correction must be boolean"):
        LevinaBickelMLE(k_min=3, k_max=4, bias_correction=1).fit(X)


def test_levina_bickel_rejects_singular_neighbor_ties() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match="Neighbor-distance ties"):
        LevinaBickelMLE(k_min=3, k_max=3).fit(X)


def test_mind_exact_one_dimensional_optimizer_boundary() -> None:
    X = np.random.default_rng(93).uniform(size=(50, 1))
    model = MiNDML(n_neighbors=3).fit(X)

    assert model.dimension_ == 1.0
    assert model.diagnostics_.converged
    assert model.diagnostics_.n_iter == 1
    assert model.diagnostics_.warnings == (
        "The likelihood maximum is on the advertised dimension boundary.",
    )


@pytest.mark.parametrize(
    "estimator, message",
    [
        (MiNDML(n_neighbors=3, max_dimension=3), "max_dimension cannot exceed"),
        (
            UStatisticDimension(max_dimension=3, random_state=0),
            "max_dimension cannot exceed",
        ),
        (DANCo(n_neighbors=3, max_dimension=3), "max_dimension cannot exceed"),
    ],
)
def test_dimension_search_bounds_cannot_exceed_advertised_ambient_range(
    estimator: BaseEstimator, message: str
) -> None:
    X = np.random.default_rng(94).normal(size=(30, 2))
    with pytest.raises(ValueError, match=message):
        estimator.fit(X)


def test_neighbor_count_boundary_is_explicit_for_mind_and_danco() -> None:
    X = np.random.default_rng(95).normal(size=(6, 2))
    for estimator in (
        MiNDML(n_neighbors=5, max_dimension=2),
        DANCo(n_neighbors=5, max_dimension=2, random_state=0),
    ):
        with pytest.raises(ValueError, match=r"n_neighbors \+ 1"):
            estimator.fit(X)


def test_normalized_minimum_distance_ties_fail_for_mind_and_danco() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    for estimator in (
        MiNDML(n_neighbors=2, max_dimension=2),
        DANCo(n_neighbors=2, max_dimension=2, random_state=0),
    ):
        with pytest.raises(ValueError, match=r"first and \(k \+ 1\)-st neighbor"):
            estimator.fit(X)


def test_ustatistic_default_bound_and_interior_solution_diagnostics() -> None:
    default_X = np.random.default_rng(96).uniform(size=(80, 2))
    default_model = UStatisticDimension(random_state=4).fit(default_X)
    assert default_model.candidate_dimensions_.tolist() == [1, 2]

    intrinsic = np.random.default_rng(97).uniform(size=(200, 2))
    embedded = np.column_stack((intrinsic, np.zeros(intrinsic.shape[0])))
    interior = UStatisticDimension(max_dimension=3, random_state=4).fit(embedded)
    assert interior.dimension_ == 2.0
    assert interior.diagnostics_.warnings == ()


def test_ustatistic_rejects_zero_compact_kernel_statistic() -> None:
    # The constant second coordinate pins distance scaling at one. Consecutive
    # first coordinates are exactly 1/8 apart, so every divisor-one kernel value
    # is exactly zero at the mean nearest-neighbor bandwidth.
    line = (np.arange(10.0) - 4.5) / 8.0
    X = np.column_stack((line, np.ones(line.size)))
    with pytest.raises(ValueError, match="compact-kernel U-statistic is zero"):
        UStatisticDimension(max_dimension=1, random_state=0).fit(X)


def test_ustatistic_nearest_neighbor_mean_does_not_overflow() -> None:
    diagonal = np.linspace(0.7, 1.0, 10) * 1.0e308
    X = np.zeros((10, 15), dtype=np.float64)
    X[np.arange(10), np.arange(10)] = diagonal
    model = UStatisticDimension(max_dimension=3, random_state=0).fit(X)
    assert np.isfinite(model.base_bandwidth_)
    assert np.all(np.isfinite(model.log_bandwidths_))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UStatisticDimension(max_dimension=2, random_state="invalid"),
        lambda: DANCo(
            n_neighbors=3,
            max_dimension=2,
            random_state="invalid",
        ),
    ],
)
def test_invalid_random_state_is_rejected(
    factory: Callable[[], BaseEstimator],
) -> None:
    X = np.random.default_rng(98).normal(size=(30, 2))
    with pytest.raises(TypeError, match="random_state must be"):
        factory().fit(X)


def test_successful_fits_expose_finite_truthful_diagnostics() -> None:
    X = np.random.default_rng(99).normal(size=(100, 3))
    estimators = (
        CorrelationDimension(),
        TwoNN(),
        LevinaBickelMLE(k_min=5, k_max=8),
        UStatisticDimension(max_dimension=3, random_state=7),
        MiNDML(n_neighbors=5, max_dimension=3),
        DANCo(n_neighbors=5, max_dimension=3, random_state=7),
    )
    for estimator in estimators:
        fitted = estimator.fit(X)
        fit_diagnostics = fitted.diagnostics_  # type: ignore[attr-defined]
        assert isinstance(fit_diagnostics, FitDiagnostics)
        assert fit_diagnostics.solver
        assert fit_diagnostics.converged
        assert isinstance(fit_diagnostics.warnings, tuple)
        if fit_diagnostics.n_iter is not None:
            assert fit_diagnostics.n_iter >= 1
        if fit_diagnostics.objective_value is not None:
            assert np.isfinite(fit_diagnostics.objective_value)
        if fit_diagnostics.residual_norm is not None:
            assert np.isfinite(fit_diagnostics.residual_norm)
            assert fit_diagnostics.residual_norm >= 0.0
