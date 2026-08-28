"""Cross-family regressions for float64 scale and translation stability."""

from __future__ import annotations

import numpy as np
import pytest

from believe14._core.distances import pairwise_distances
from believe14._core.kernels import center_kernel, rbf_kernel
from believe14._core.linalg import centered_svd
from believe14._core.validation import validate_precomputed
from believe14.estimation import UStatisticDimension
from believe14.linear import (
    PCA,
    CanonicalCorrelationAnalysis,
    FactorAnalysis,
    FisherScore,
    LinearDiscriminantAnalysis,
    PLSRegression,
)
from believe14.nonlinear import (
    PHATE,
    TSNE,
    DiffusionMap,
    FastMap,
    LocallyLinearEmbedding,
    MetricMDS,
)


def _projector(rows: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(rows.T)
    return basis @ basis.T


def test_distances_and_centering_preserve_large_common_offsets() -> None:
    X = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]])
    np.testing.assert_array_equal(pairwise_distances(X + 1.0e12), pairwise_distances(X))

    integers = np.arange(48.0).reshape(12, 4)
    _, _, _, baseline_vectors = centered_svd(integers)
    _, _, _, shifted_vectors = centered_svd(integers + 1.0e15)
    np.testing.assert_allclose(
        _projector(shifted_vectors[:2]),
        _projector(baseline_vectors[:2]),
        atol=1e-14,
    )


def test_tiny_precomputed_validation_does_not_repair_invalid_input() -> None:
    invalid = np.array([[1.0e-150, -1.0e-150], [2.0e-150, 3.0e-150]])
    with pytest.raises(ValueError, match="nonnegative"):
        validate_precomputed(invalid)


def test_kernel_arithmetic_handles_large_constants_and_balanced_gamma() -> None:
    constant = np.full((20, 20), 1.0e300)
    centered, column_effect, total_effect = center_kernel(constant)
    np.testing.assert_array_equal(centered, 0.0)
    np.testing.assert_array_equal(column_effect, 0.0)
    assert total_effect == 0.0

    X = np.random.default_rng(1).normal(size=(20, 3)) * 1.0e155
    kernel = rbf_kernel(X, gamma=1.0e-310)
    assert np.all(np.isfinite(kernel))
    np.testing.assert_array_equal(np.diag(kernel), 1.0)


def test_dimensionless_linear_results_survive_tiny_scale_and_translation() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 5))
    labels = np.repeat(np.arange(3), 20)
    baseline = FisherScore(2).fit(X, labels).scores_
    tiny = FisherScore(2).fit(X * 1.0e-170, labels).scores_
    np.testing.assert_allclose(tiny, baseline, rtol=1e-13, atol=1e-15)

    first_view = rng.integers(-20, 21, size=(80, 5)).astype(np.float64)
    second_view = rng.integers(-20, 21, size=(80, 4)).astype(np.float64)
    baseline_cca = CanonicalCorrelationAnalysis(2).fit(first_view, second_view)
    shifted_cca = CanonicalCorrelationAnalysis(2).fit(
        first_view + 1.0e15, second_view - 1.0e15
    )
    np.testing.assert_allclose(
        _projector(shifted_cca.x_weights_.T),
        _projector(baseline_cca.x_weights_.T),
        atol=1e-13,
    )
    baseline_x, baseline_y = baseline_cca.transform(first_view, second_view)
    shifted_x, shifted_y = shifted_cca.transform(
        first_view + 1.0e15, second_view - 1.0e15
    )
    np.testing.assert_allclose(shifted_x, baseline_x, atol=1e-13)
    np.testing.assert_allclose(shifted_y, baseline_y, atol=1e-13)
    np.testing.assert_allclose(shifted_x, shifted_cca.x_scores_, atol=1e-13)

    lda_labels = np.repeat(np.arange(4), 20)
    baseline_lda = LinearDiscriminantAnalysis(2, regularization=0.1).fit(
        first_view, lda_labels
    )
    shifted_lda = LinearDiscriminantAnalysis(2, regularization=0.1).fit(
        first_view + 1.0e15, lda_labels
    )
    np.testing.assert_allclose(
        shifted_lda.transform(first_view + 1.0e15),
        baseline_lda.transform(first_view),
        atol=1e-13,
    )


def test_pls_prediction_uses_centered_coordinates() -> None:
    X = np.arange(300.0).reshape(60, 5)
    target = X[:, 0] - 2.0 * X[:, 1]
    baseline = PLSRegression(1).fit(X, target).predict(X)
    translated = PLSRegression(1).fit(X + 1.0e15, target).predict(X + 1.0e15)
    np.testing.assert_allclose(translated, baseline, rtol=1e-13, atol=1e-13)


def test_variance_models_fail_honestly_or_retain_representable_state() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(FloatingPointError, match="underflow"):
        PCA(2).fit(rng.normal(size=(20, 3)) * 1.0e-170)

    model = FactorAnalysis(2, max_iter=20).fit(rng.normal(size=(60, 5)) * 1.0e150)
    assert np.all(np.isfinite(model.noise_variance_))
    assert np.all(model.noise_variance_ > 0.0)


@pytest.mark.parametrize("factor", [1.0e-150, 1.0e150])
def test_distance_objectives_and_lle_are_scale_aware(factor: float) -> None:
    X = np.random.default_rng(4).normal(size=(24, 3)) * factor
    fastmap = FastMap(2).fit(X)
    assert np.any(fastmap.embedding_)
    np.testing.assert_allclose(
        pairwise_distances(fastmap.embedding_) / factor,
        pairwise_distances(FastMap(2).fit_transform(X / factor)),
        rtol=1e-12,
        atol=1e-12,
    )
    metric = MetricMDS(2, max_iter=5).fit(X)
    assert np.any(metric.embedding_)
    lle = LocallyLinearEmbedding(2, n_neighbors=5).fit(X)
    assert np.all(np.isfinite(lle.embedding_))


def test_distributional_methods_handle_scaled_inputs_truthfully() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="numerically disconnected"):
        DiffusionMap(2).fit(rng.normal(size=(20, 4)) * 1.0e150)

    for factor in (1.0e-150, 1.0e150):
        tsne = TSNE(
            perplexity=4,
            init="random",
            random_state=0,
            early_exaggeration_iter=1,
            max_iter=3,
        ).fit(rng.normal(size=(18, 3)) * factor)
        assert np.max(np.abs(tsne.perplexity_entropy_residuals_)) <= 1.0e-8

        estimate = UStatisticDimension(max_dimension=3, random_state=0).fit(
            rng.normal(size=(80, 3)) * factor
        )
        assert estimate.dimension_ in {1.0, 2.0, 3.0}

    phate = PHATE(2, n_neighbors=3, mds_max_iter=5).fit(
        np.r_[np.arange(10.0), 1.0e10][:, None]
    )
    assert np.all(np.isfinite(phate.embedding_))
