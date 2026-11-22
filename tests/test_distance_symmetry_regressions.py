"""Small representable distances must survive all symmetric distance paths."""

import numpy as np

from believe14._core.distances import (
    pairwise_distances,
    pairwise_scaled_squared_distances,
    pairwise_squared_distances,
)
from believe14._core.validation import validate_precomputed
from believe14.estimation import TwoNN
from believe14.nonlinear import TSNE


def test_duplicate_free_subnormal_distance_matrix_matches_rectangular_path():
    unit = np.finfo(float).smallest_subnormal
    integers = np.array([0, 1, 4, 10, 21, 35])
    X = integers[:, None] * unit
    expected = np.abs(integers[:, None] - integers) * unit
    np.testing.assert_array_equal(pairwise_distances(X), expected)
    np.testing.assert_array_equal(pairwise_distances(X, X), expected)
    np.testing.assert_array_equal(validate_precomputed(expected), expected)
    np.testing.assert_allclose(
        TwoNN().fit(X).dimension_, TwoNN().fit(integers[:, None]).dimension_
    )


def test_representable_squared_distances_survive_symmetrization():
    X = np.array([[0.0], [2.0**-537]])
    expected = np.finfo(float).smallest_subnormal
    assert pairwise_squared_distances(X)[0, 1] == expected
    assert pairwise_scaled_squared_distances(X, multiplier=1.0)[0, 1] == expected


def test_tiny_distance_units_preserve_tsne_perplexity():
    X = np.array([[0.0], [1.0], [4.0], [8.0]])
    params = dict(
        perplexity=1.2,
        init="random",
        early_exaggeration_iter=0,
        max_iter=1,
        random_state=17,
    )
    ordinary = TSNE(1, **params).fit(X)
    tiny = TSNE(1, **params).fit(X * np.finfo(float).smallest_subnormal)
    np.testing.assert_allclose(
        tiny.conditional_probabilities_, ordinary.conditional_probabilities_, atol=2e-8
    )
