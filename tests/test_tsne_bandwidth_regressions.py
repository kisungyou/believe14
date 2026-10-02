"""Independent checks of t-SNE bandwidths across wide distance scales."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import entr

from believe14.nonlinear import TSNE
from believe14.nonlinear._stochastic import _joint_probabilities


def _independent_line_probabilities(X: np.ndarray, perplexity: float) -> np.ndarray:
    """Solve ordinary precision roots after long-double row rescaling."""

    result = np.zeros((len(X), len(X)))
    for index, coordinate in enumerate(X[:, 0]):
        distances = np.abs(
            X[np.arange(len(X)) != index, 0].astype(np.longdouble)
            - np.longdouble(coordinate)
        )
        minimum = distances.min()
        gaps = (distances - minimum) * (distances + minimum)
        normalized = np.asarray(gaps / np.median(gaps[gaps > 0]), dtype=float)

        def probabilities(
            precision: float, row_gaps: np.ndarray = normalized
        ) -> np.ndarray:
            with np.errstate(under="ignore"):
                weights = np.exp(-precision * row_gaps)
            return weights / weights.sum()

        def difference(precision: float) -> float:
            return float(entr(probabilities(precision)).sum() - np.log(perplexity))

        upper = 1.0
        while difference(upper) > 0:
            upper *= 2
        root = brentq(difference, 0, upper, xtol=1e-14)
        result[index, np.arange(len(X)) != index] = probabilities(root)
    return result


def _assert_probability_contract(
    joint: np.ndarray, conditional: np.ndarray, residuals: np.ndarray, perplexity: float
) -> None:
    assert np.all(np.isfinite(joint))
    assert np.all(np.isfinite(conditional))
    assert np.all(joint >= 0)
    assert np.all(conditional >= 0)
    np.testing.assert_array_equal(joint, joint.T)
    np.testing.assert_array_equal(np.diag(joint), 0)
    np.testing.assert_array_equal(np.diag(conditional), 0)
    np.testing.assert_allclose(joint.sum(), 1, atol=2e-15, rtol=0)
    np.testing.assert_allclose(conditional.sum(axis=1), 1, atol=2e-15, rtol=0)
    independently_computed = entr(conditional).sum(axis=1) - np.log(perplexity)
    np.testing.assert_allclose(residuals, independently_computed, atol=2e-15, rtol=0)
    assert np.max(np.abs(independently_computed)) <= 1e-8


def test_attainable_wide_scale_perplexity_matches_independent_precision_roots() -> None:
    X = np.r_[np.arange(20.0), 1e14][:, None]
    expected = _independent_line_probabilities(X, perplexity=5)
    model = TSNE(
        1,
        perplexity=5,
        init="random",
        early_exaggeration_iter=0,
        max_iter=1,
        random_state=19,
    ).fit(X)
    np.testing.assert_allclose(
        model.conditional_probabilities_, expected, atol=2e-8, rtol=0
    )
    _assert_probability_contract(
        model.joint_probabilities_,
        model.conditional_probabilities_,
        model.perplexity_entropy_residuals_,
        5,
    )


@pytest.mark.parametrize("exponent", [-950, -500, 500, 950])
def test_wide_scale_probabilities_are_uniform_scale_invariant(exponent: int) -> None:
    X = np.r_[np.arange(20.0), 1e14][:, None]
    expected = _independent_line_probabilities(X, perplexity=5)
    changed = np.ldexp(X[:, 0], exponent)
    distances = np.abs(changed[:, None] - changed[None, :])
    joint, conditional, residuals = _joint_probabilities(distances, 5, squared=False)
    # Powers of two preserve the represented input geometry exactly.
    np.testing.assert_allclose(conditional, expected, atol=2e-8, rtol=0)
    _assert_probability_contract(joint, conditional, residuals, 5)


def test_log_precision_handles_squares_outside_the_float64_range() -> None:
    tiny = np.finfo(float).smallest_subnormal
    large = np.finfo(float).max
    distances = np.array(
        [
            [0, tiny, 2 * tiny, large],
            [tiny, 0, 3 * tiny, large / 2],
            [2 * tiny, 3 * tiny, 0, large / 4],
            [large, large / 2, large / 4, 0],
        ]
    )
    joint, conditional, residuals = _joint_probabilities(distances, 1.5, squared=False)
    # The remote neighbor has zero representable Gaussian weight in the first
    # three rows. Their remaining probabilities solve the binary entropy law,
    # independently of the bandwidth or the very different distance gaps.
    binary_probability = brentq(
        lambda p: float(entr(p) + entr(1 - p) - np.log(1.5)), 0.5, 1.0
    )
    for row, nearest, second in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        np.testing.assert_allclose(
            conditional[row, [nearest, second]],
            [binary_probability, 1 - binary_probability],
            atol=1e-8,
            rtol=0,
        )
        assert conditional[row, 3] == 0
    _assert_probability_contract(joint, conditional, residuals, 1.5)


def test_squared_distance_contract_retains_zero_and_subnormal_gaps() -> None:
    tiny = np.finfo(float).smallest_subnormal
    large = np.finfo(float).max
    squared = np.array(
        [
            [0, 0, tiny, large],
            [0, 0, 2 * tiny, large / 2],
            [tiny, 2 * tiny, 0, large / 4],
            [large, large / 2, large / 4, 0],
        ]
    )
    joint, conditional, residuals = _joint_probabilities(
        squared_distances=squared, perplexity=1.5
    )
    _assert_probability_contract(joint, conditional, residuals, 1.5)
    assert conditional[0, 1] > conditional[0, 2] > 0


def test_tied_nearest_endpoint_and_uniform_endpoint_remain_exact() -> None:
    square = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    distances = np.linalg.norm(square[:, None] - square[None, :], axis=2)
    for perplexity in (2, 3):
        joint, conditional, residuals = _joint_probabilities(
            distances, perplexity, squared=False
        )
        expected = (distances == 1).astype(float) / 2
        if perplexity == 3:
            expected = (distances > 0).astype(float) / 3
        np.testing.assert_array_equal(conditional, expected)
        _assert_probability_contract(joint, conditional, residuals, perplexity)
    with pytest.raises(ValueError, match="nearest distances are tied"):
        _joint_probabilities(distances, 1.5, squared=False)
