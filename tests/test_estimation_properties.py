from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from believe14.estimation import (
    CorrelationDimension,
    DANCo,
    LevinaBickelMLE,
    MiNDML,
    TwoNN,
    UStatisticDimension,
)


def _factories() -> tuple[Callable[[], BaseEstimator], ...]:
    return (
        CorrelationDimension,
        TwoNN,
        lambda: LevinaBickelMLE(k_min=5, k_max=8),
        lambda: UStatisticDimension(max_dimension=3, random_state=19),
        lambda: MiNDML(n_neighbors=5, max_dimension=3),
        lambda: DANCo(n_neighbors=5, max_dimension=3, random_state=19),
    )


def test_euclidean_invariances() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(100, 3))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = 7.5 * X @ orthogonal + np.array([10.0, -4.0, 2.0])
    for factory in _factories():
        original = factory().fit(X)
        changed = factory().fit(transformed)
        assert changed.dimension_ == pytest.approx(
            original.dimension_, rel=1e-9, abs=1e-9
        )


@pytest.mark.parametrize(
    "factory",
    [
        CorrelationDimension,
        TwoNN,
        lambda: LevinaBickelMLE(k_min=5, k_max=8),
        lambda: MiNDML(n_neighbors=5, max_dimension=3),
        lambda: DANCo(n_neighbors=5, max_dimension=3, random_state=31),
    ],
)
def test_row_permutation_invariance(factory: Callable[[], BaseEstimator]) -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(90, 3))
    permutation = rng.permutation(X.shape[0])
    expected = factory().fit(X).dimension_  # type: ignore[attr-defined]
    observed = factory().fit(X[permutation]).dimension_  # type: ignore[attr-defined]
    assert observed == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(
    "factory",
    [
        CorrelationDimension,
        TwoNN,
        lambda: LevinaBickelMLE(k_min=4, k_max=6),
        lambda: UStatisticDimension(max_dimension=2, random_state=0),
        lambda: MiNDML(n_neighbors=4, max_dimension=2),
        lambda: DANCo(n_neighbors=4, max_dimension=2, random_state=0),
    ],
)
def test_duplicate_observations_fail_explicitly(
    factory: Callable[[], BaseEstimator],
) -> None:
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 2))
    X[-1] = X[0]
    with pytest.raises(ValueError, match="duplicate"):
        factory().fit(X)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UStatisticDimension(max_dimension=2, random_state=52),
        lambda: DANCo(n_neighbors=5, max_dimension=2, random_state=52),
    ],
)
def test_stochastic_estimators_are_reproducible_and_leave_global_rng_untouched(
    factory: Callable[[], BaseEstimator],
) -> None:
    X = np.random.default_rng(15).normal(size=(100, 2))
    np.random.seed(1234)
    state_before = np.random.get_state()
    first = factory().fit(X)
    state_after = np.random.get_state()
    second = factory().fit(X)

    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    assert second.dimension_ == first.dimension_  # type: ignore[attr-defined]


def test_generator_instance_is_advanced() -> None:
    X = np.random.default_rng(16).normal(size=(80, 2))
    generator = np.random.default_rng(77)
    before = repr(generator.bit_generator.state)
    UStatisticDimension(max_dimension=2, random_state=generator).fit(X)
    after = repr(generator.bit_generator.state)
    assert after != before


def test_synthetic_two_dimensional_recovery() -> None:
    X = np.random.default_rng(0).uniform(size=(300, 2))
    continuous = (
        CorrelationDimension().fit(X).dimension_,
        TwoNN().fit(X).dimension_,
        LevinaBickelMLE(k_min=5, k_max=10).fit(X).dimension_,
        MiNDML(n_neighbors=5, max_dimension=2).fit(X).dimension_,
    )
    for estimate in continuous:
        assert 1.5 < estimate <= 2.5
    assert (
        UStatisticDimension(max_dimension=2, random_state=11).fit(X).dimension_ == 2.0
    )
    assert (
        DANCo(n_neighbors=5, max_dimension=2, random_state=11).fit(X).dimension_ == 2.0
    )


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("calibration_seed", [3, 11])
def test_danco_recovers_multiple_unit_balls_below_a_loose_upper_bound(
    dimension: int, calibration_seed: int
) -> None:
    rng = np.random.default_rng(dimension)
    directions = rng.normal(size=(100, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.random(100) ** (1.0 / dimension)
    intrinsic = directions * radii[:, None]
    X = np.pad(intrinsic, ((0, 0), (0, 5 - dimension)))

    model = DANCo(
        n_neighbors=8,
        max_dimension=4,
        random_state=calibration_seed,
    ).fit(X)

    assert model.candidate_dimensions_.tolist() == [2, 3, 4]
    assert model.dimension_ == float(dimension)


def test_danco_rejects_one_dimensional_and_embedded_collinear_flats() -> None:
    line = np.linspace(-2.0, 2.0, 40)[:, None]
    with pytest.raises(ValueError, match="max_dimension must be at least 2"):
        DANCo(n_neighbors=5).fit(line)

    embedded = np.column_stack((line[:, 0], 2.0 * line[:, 0], -line[:, 0]))
    with pytest.raises(ValueError, match="degenerate for collinear data"):
        DANCo(n_neighbors=5, max_dimension=3).fit(embedded)


def test_neighbor_boundary_and_parameter_failures_are_explicit() -> None:
    X = np.arange(18.0).reshape(9, 2)
    with pytest.raises(ValueError, match="k_max"):
        LevinaBickelMLE(k_min=4, k_max=9).fit(X)
    with pytest.raises(ValueError, match=r"n_neighbors \+ 1"):
        MiNDML(n_neighbors=8).fit(X)
    with pytest.raises(ValueError, match="max_dimension"):
        DANCo(n_neighbors=3, max_dimension=3).fit(X)
    with pytest.raises(ValueError, match="quantile_range"):
        CorrelationDimension(quantile_range=(0.2, 0.1)).fit(X)


def test_neighbor_tie_singularity_is_not_silently_perturbed() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    with pytest.raises(ValueError, match="tie"):
        MiNDML(n_neighbors=1, max_dimension=2).fit(circle)
