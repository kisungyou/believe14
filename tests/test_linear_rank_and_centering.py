"""Regression tests for centering, exhausted PLS rank, and Fisher null directions."""

from itertools import permutations
from math import fsum

import numpy as np
import pytest

from believe14.linear import PCA, LinearDiscriminantAnalysis, PLSRegression


def test_pca_and_identity_regression_preserve_queries_across_wide_training_range():
    X = np.array([[-1e16], [1e16]])
    query = np.array([[1.0], [2.0], [3.0], [-0.125]])
    pca = PCA(1).fit(X)
    np.testing.assert_array_equal(pca.inverse_transform(pca.transform(query)), query)
    pls = PLSRegression(1).fit(X, X[:, 0])
    np.testing.assert_allclose(pls.predict(query), query[:, 0], rtol=1e-15)


def test_fitted_mean_is_permutation_invariant_under_cancellation():
    values = [1e16, 1.0, -1e16]
    expected = fsum(values) / len(values)
    for order in permutations(values):
        pca = PCA(1).fit(np.array(order)[:, None])
        assert pca.mean_[0] == expected
        np.testing.assert_allclose(pca.transform([[1.0]])[0, 0], 1 - expected)


def test_standardized_pls_preserves_predictions_when_features_change_units():
    X = np.array(
        [[1.0, 0.0], [4.0, 1.0], [2.0, 3.0], [5.0, 6.0], [6.0, 4.0], [8.0, 5.0]]
    )
    scaled = np.ldexp(X, [570, -570])
    y = X[:, 1]
    original = PLSRegression(1).fit(X, y)
    changed = PLSRegression(1).fit(scaled, y)
    assert np.ldexp(changed.x_mean_[1], 570) == 19 / 6
    np.testing.assert_allclose(changed.predict(scaled), original.predict(X))


@pytest.mark.parametrize("scale", [False, True])
def test_pls_rejects_rank_exhaustion_with_nonzero_target_residual(scale):
    rng = np.random.default_rng(4)
    x = rng.normal(size=12)
    X = np.column_stack((x, 2 * x))
    y = x + rng.normal(0, 0.1, len(x))
    design = np.column_stack((np.ones(len(x)), x))
    expected = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(
        PLSRegression(1, scale=scale).fit(X, y).predict(X), expected
    )
    with pytest.raises(ValueError, match="numerical rank"):
        PLSRegression(2, scale=scale).fit(X, y)


@pytest.mark.parametrize("exponent", [-560, 530])
def test_unscaled_pls_extreme_identity(exponent):
    x = np.ldexp(np.array([-1.0, 0.0, 1.0]), exponent)
    model = PLSRegression(1, scale=False).fit(x[:, None], x)
    np.testing.assert_allclose(model.coef_, [[1.0]], rtol=3e-15)
    np.testing.assert_allclose(model.predict(x[:, None]), x, atol=0.0, rtol=3e-15)


def test_lda_detects_small_exact_class_separator_and_valid_null_control():
    X = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1e-7], [2.0, 1e-7]])
    labels = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="infinite Fisher"):
        LinearDiscriminantAnalysis(1).fit(X, labels)
    X[:, 1] = 0
    assert LinearDiscriminantAnalysis(1).fit(X, labels).eigenvalues_[
        0
    ] == pytest.approx(0.25)


def test_lda_exact_per_class_means_around_rounded_global_offset():
    X = (1e16 + np.array([0.0, 0.0, 2.0, 2.0]))[:, None]
    model = LinearDiscriminantAnalysis(1, regularization=1).fit(X, [0, 0, 1, 1])
    np.testing.assert_array_equal(model.class_means_[:, 0], X[[0, 2], 0])


def test_pls_retains_weak_independent_target_score_and_rotation():
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    Y = X * [1.0, 1e-9]
    model = PLSRegression(2, scale=False).fit(X, Y)
    np.testing.assert_allclose(model.predict(X), Y, rtol=1e-14, atol=0.0)
    xs, ys = model.transform(X, Y)
    np.testing.assert_allclose(np.abs(ys), np.abs(xs), rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(model.y_loadings_.T @ model.y_rotations_, np.eye(2))
