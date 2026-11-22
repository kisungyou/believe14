"""Regression tests for Gram centering, input tags and nonlinear fit diagnostics."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import get_tags

from believe14.nonlinear import (
    PHATE,
    ClassicalMDS,
    FastMap,
    KernelPCA,
    MetricMDS,
    SammonMapping,
)
from believe14.nonlinear._common import smacof


@pytest.mark.parametrize("kernel", ["precomputed", "poly"])
def test_shifted_rank_one_kernel_retains_positive_spectrum(kernel):
    X = (10_000 + np.arange(4.0))[:, None]
    K = X @ X.T
    model = KernelPCA(1, kernel=kernel, gamma=1, degree=1, coef0=0)
    data = K if kernel == "precomputed" else X
    embedded = model.fit_transform(data)
    expected = X - X.mean()
    np.testing.assert_allclose(model.eigenvalues_, [5], atol=1e-13)
    np.testing.assert_allclose(embedded @ embedded.T, expected @ expected.T, atol=1e-12)
    np.testing.assert_allclose(embedded, model.embedding_, atol=1e-12)


def test_kernel_query_mixed_magnitudes_are_batch_invariant():
    model = KernelPCA(1, kernel="precomputed").fit([[1.0, -1.0], [-1.0, 1.0]])
    amplitudes = np.array([1e-40, 1e290, -1e-40])
    data = np.column_stack([amplitudes, -amplitudes])
    result = model.transform(data)
    for i, row in enumerate(data):
        np.testing.assert_allclose(
            result[i], model.transform(row[None, :])[0], rtol=1e-14, atol=0
        )
    np.testing.assert_allclose(
        np.abs(result[:, 0]), np.abs(amplitudes), atol=0, rtol=1e-14
    )


def test_precomputed_kernel_pipeline_uses_train_test_kernel_blocks():
    X = np.random.default_rng(15).normal(size=(12, 3))
    K, y = X @ X.T, X[:, 0]
    pipeline = make_pipeline(KernelPCA(2, kernel="precomputed"), Ridge())
    actual = cross_val_score(pipeline, K, y, cv=3, error_score="raise")
    expected = [
        pipeline.fit(K[np.ix_(train, train)], y[train]).score(
            K[np.ix_(test, train)], y[test]
        )
        for train, test in KFold(3).split(X)
    ]
    np.testing.assert_allclose(actual, expected)
    assert not get_tags(KernelPCA(kernel="linear")).input_tags.pairwise


@pytest.mark.parametrize("cls", [ClassicalMDS, MetricMDS, SammonMapping, FastMap])
def test_precomputed_distance_tags_follow_estimator_mode(cls):
    assert get_tags(cls(dissimilarity="precomputed")).input_tags.pairwise
    assert not get_tags(cls()).input_tags.pairwise


def test_relative_smacof_convergence_at_low_stress():
    X = np.random.default_rng(15).normal(size=(9, 3))
    X[:, 2] *= 0.03
    init = ClassicalMDS(2).fit_transform(X)
    original = pdist(X)
    initial_stress = np.sum((pdist(init) - original) ** 2)
    one = MetricMDS(2, max_iter=1).fit(X)
    after_one = np.sum((pdist(one.embedding_) - original) ** 2)
    assert not one.diagnostics_.converged
    assert one.diagnostics_.residual_norm == pytest.approx(
        (initial_stress - after_one) / initial_stress
    )
    final = MetricMDS(2).fit(X)
    assert final.diagnostics_.converged
    assert final.n_iter_ > 1
    assert final.stress_ < 0.9 * after_one
    assert final.diagnostics_.residual_norm <= final.tol


def test_smacof_starts_at_zero_stress():
    X = np.array([[-1.0], [1.0]])
    embedding, objective, _, converged, residual = smacof(
        squareform(pdist(X)), X, tol=1e-8, max_iter=2
    )
    assert objective == residual == 0
    assert converged
    np.testing.assert_array_equal(pdist(embedding), pdist(X))


def test_phate_positive_bandwidth_with_remote_cluster():
    X = np.array([0.0, 1.0, 2.0, 1e14])[:, None]
    distances = squareform(pdist(X))
    bandwidths = np.sort(distances, axis=1)[:, 1]
    directed = np.exp(-((distances / bandwidths[:, None]) ** 2))
    expected = (directed + directed.T) / 2
    model = PHATE(1, n_neighbors=1, decay=2.0, diffusion_time=1).fit(X)
    np.testing.assert_allclose(model.affinity_matrix_, expected, atol=1e-15)
    assert np.all(np.isfinite(model.potential_distances_))


def test_phate_resolves_one_ulp_bandwidth_differences_at_large_scale():
    band = 2.0**500
    just_beyond = np.nextafter(band, np.inf)
    X = np.array([0.0, band, just_beyond, -band])[:, None]
    model = PHATE(1, n_neighbors=2, diffusion_time=1, decay=1e16).fit(X)
    exponent = np.exp(model.decay * np.log1p(np.finfo(float).eps))
    expected = (np.exp(-exponent) + np.exp(-1.0)) / 2
    assert model.affinity_matrix_[0, 2] == pytest.approx(expected, rel=1e-14)
