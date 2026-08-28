from __future__ import annotations

import numpy as np
import pytest

from believe14._core.distances import (
    exact_neighbors,
    pairwise_distances,
    pairwise_squared_distances,
)
from believe14._core.graphs import (
    graph_shortest_paths,
    neighbor_graph,
    require_connected,
)
from believe14._core.linalg import (
    canonicalize_columns,
    centered_svd,
    condition_estimate,
    generalized_eigh,
    numerical_rank,
    symmetric_eigh,
)
from believe14._core.validation import (
    as_float_matrix,
    make_rng,
    validate_n_components,
    validate_n_neighbors,
    validate_positive_real,
    validate_precomputed,
)


def test_pairwise_distances_are_exact_symmetric_and_hollow() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0], [-3.0, 4.0]])
    observed = pairwise_distances(X)
    expected = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 6.0], [5.0, 6.0, 0.0]])
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_array_equal(np.diag(observed), 0.0)


def test_pairwise_distances_rescale_extreme_finite_values() -> None:
    X = np.array([[-1.0e300], [1.0e300]])
    assert pairwise_distances(X)[0, 1] == pytest.approx(2.0e300)
    with pytest.raises(FloatingPointError, match="overflow"):
        pairwise_squared_distances(X)
    with pytest.raises(FloatingPointError, match="overflow"):
        pairwise_distances(np.array([[-1.0e308], [1.0e308]]))


def test_cross_distances_and_zero_scale_are_explicit() -> None:
    zeros = np.zeros((2, 3))
    np.testing.assert_array_equal(pairwise_squared_distances(zeros), np.zeros((2, 2)))
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    Y = np.array([[0.0, 2.0]])
    np.testing.assert_allclose(pairwise_squared_distances(X, Y)[:, 0], [4.0, 5.0])


def test_exact_neighbors_break_ties_by_row_index() -> None:
    X = np.array([[0.0], [-1.0], [1.0]])
    distances, indices = exact_neighbors(X, 1)
    np.testing.assert_array_equal(indices[:, 0], [1, 0, 0])
    np.testing.assert_allclose(distances[:, 0], 1.0)


def test_symmetric_eigh_reports_residual_and_canonical_sign() -> None:
    matrix = np.diag([3.0, 2.0, 1.0])
    result = symmetric_eigh(matrix, n_components=2)
    np.testing.assert_allclose(result.values, [3.0, 2.0])
    assert result.residual_norm < 1e-14
    np.testing.assert_array_equal(result.vectors, canonicalize_columns(result.vectors))


def test_linear_algebra_services_cover_both_eigen_orders_and_rank_edges() -> None:
    empty = np.empty((3, 0))
    assert canonicalize_columns(empty).shape == (3, 0)
    assert numerical_rank(np.array([]), shape=(3, 0)) == 0
    assert numerical_rank(np.array([3.0, 1e-20]), shape=(2, 2)) == 1

    matrix = np.diag([1.0, 3.0, 2.0])
    smallest = symmetric_eigh(matrix, n_components=2, largest=False)
    np.testing.assert_allclose(smallest.values, [1.0, 2.0])
    complete = symmetric_eigh(matrix)
    np.testing.assert_allclose(complete.values, [3.0, 2.0, 1.0])

    metric = np.diag([2.0, 4.0, 5.0])
    generalized = generalized_eigh(matrix, metric, n_components=2, largest=False)
    expected = np.sort(np.diag(matrix) / np.diag(metric))[:2]
    np.testing.assert_allclose(generalized.values, expected)
    assert generalized.residual_norm < 1e-14
    largest = generalized_eigh(matrix, metric, n_components=1)
    assert largest.values[0] == pytest.approx(0.75)
    assert np.isinf(condition_estimate(np.diag([1.0, 0.0])))


def test_centered_svd_reconstructs_centered_data() -> None:
    X = np.array([[0.0, 1.0], [2.0, 4.0], [4.0, -1.0]])
    centered, left, singular_values, vectors_t = centered_svd(X)
    np.testing.assert_allclose(
        left @ np.diag(singular_values) @ vectors_t,
        centered,
        atol=1e-14,
    )
    for factor in (1e-150, 1e150):
        changed, changed_left, changed_values, changed_vectors_t = centered_svd(
            X * factor
        )
        np.testing.assert_allclose(changed / factor, centered, atol=1e-14)
        np.testing.assert_allclose(
            changed_left @ changed_left.T, left @ left.T, atol=1e-14
        )
        np.testing.assert_allclose(changed_values / factor, singular_values)
        np.testing.assert_allclose(
            changed_vectors_t.T @ changed_vectors_t,
            vectors_t.T @ vectors_t,
            atol=1e-14,
        )


def test_graph_shortest_paths_ignore_absent_dense_edges() -> None:
    X = np.array([[0.0], [1.0], [3.0]])
    graph = neighbor_graph(X, n_neighbors=1)
    observed = graph_shortest_paths(graph)
    np.testing.assert_allclose(observed[0, 2], 3.0)


def test_neighbor_graph_modes_and_failure_contracts() -> None:
    X = np.array([[0.0], [1.0], [2.5], [4.0]])
    heat = neighbor_graph(
        X, n_neighbors=2, weighting="heat", gamma=0.5, symmetrize="mean"
    )
    assert np.allclose(heat.weights, heat.weights.T)
    mutual = neighbor_graph(X, n_neighbors=2, symmetrize="mutual")
    assert np.allclose(mutual.weights, mutual.weights.T)
    with pytest.raises(ValueError, match="gamma"):
        neighbor_graph(X, n_neighbors=1, weighting="heat")
    with pytest.raises(ValueError, match="symmetrize"):
        neighbor_graph(X, n_neighbors=1, symmetrize="invalid")  # type: ignore[arg-type]

    disconnected = neighbor_graph(
        np.array([[0.0], [0.1], [10.0], [10.1]]), n_neighbors=1
    )
    with pytest.raises(ValueError, match="disconnected"):
        require_connected(disconnected)
    with pytest.raises(ValueError, match="disconnected"):
        graph_shortest_paths(disconnected)


def test_local_rng_does_not_touch_global_state() -> None:
    np.random.seed(123)
    before = np.random.get_state()
    make_rng(42).normal(size=10)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_rng_and_scalar_validation_boundaries() -> None:
    supplied = np.random.default_rng(3)
    assert make_rng(supplied) is supplied
    assert isinstance(make_rng(None), np.random.Generator)
    with pytest.raises(TypeError, match="random_state"):
        make_rng(True)
    with pytest.raises(TypeError, match="real scalar"):
        validate_positive_real(True, name="value")
    with pytest.raises(ValueError, match="finite"):
        validate_positive_real(np.inf, name="value")
    with pytest.raises(ValueError, match="positive"):
        validate_positive_real(0.0, name="value")
    assert validate_positive_real(0.0, name="value", strict=False) == 0.0
    with pytest.raises(ValueError, match="nonnegative"):
        validate_positive_real(-1.0, name="value", strict=False)


def test_neighbor_and_precomputed_validation_boundaries() -> None:
    with pytest.raises(TypeError, match="integer"):
        validate_n_neighbors(True, n_samples=4)
    with pytest.raises(ValueError, match=r"\[2, 3\]"):
        validate_n_neighbors(1, n_samples=4, minimum=2)

    valid = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(validate_precomputed(valid, atol=0.0), valid)
    with pytest.raises(ValueError, match="square"):
        validate_precomputed(np.ones((2, 3)))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_precomputed(np.array([[0.0, -1.0], [-1.0, 0.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        validate_precomputed(np.array([[0.0, 1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="zero diagonal"):
        validate_precomputed(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        as_float_matrix([[0.0, np.inf], [1.0, 2.0]])


@pytest.mark.parametrize("value", [True, 0, -1, 4])
def test_invalid_component_counts_fail(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_n_components(value, maximum=3)  # type: ignore[arg-type]
