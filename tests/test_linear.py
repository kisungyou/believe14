"""Independent formula, geometry, and contract tests for linear estimators."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from believe14.api import FitDiagnostics
from believe14.linear import (
    PCA,
    CanonicalCorrelationAnalysis,
    FactorAnalysis,
    FastICA,
    FisherScore,
    GaussianRandomProjection,
    LinearDiscriminantAnalysis,
    PLSRegression,
    ProbabilisticPCA,
    SlicedAverageVarianceEstimation,
    SlicedInverseRegression,
    SparsePCA,
)


def _rng_data(seed: int = 17) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(80, 6))


def _projector(rows: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(rows.T)
    return basis @ basis.T


def test_pca_matches_literal_centered_svd_and_whitening() -> None:
    X = _rng_data()
    model = PCA(3, whiten=True).fit(X)
    centered = X - X.mean(axis=0)
    _, singular_values, vectors_t = np.linalg.svd(centered, full_matrices=False)
    assert_allclose(
        _projector(model.components_),
        _projector(vectors_t[:3]),
        atol=1e-12,
    )
    assert_allclose(model.singular_values_, singular_values[:3], atol=1e-12)
    scores = model.transform(X)
    assert_allclose(np.cov(scores, rowvar=False), np.eye(3), atol=1e-12)
    assert_allclose(
        model.inverse_transform(scores),
        centered @ model.components_.T @ model.components_ + model.mean_,
        atol=1e-12,
    )


def test_pca_and_ppca_are_stable_at_extreme_representable_scales() -> None:
    X = _rng_data()[:, :4]
    pca = PCA(2).fit(X)
    ppca = ProbabilisticPCA(2).fit(X)
    for factor in (1e-150, 1e150):
        changed_pca = PCA(2).fit(X * factor)
        changed_ppca = ProbabilisticPCA(2).fit(X * factor)
        assert_allclose(
            _projector(changed_pca.components_),
            _projector(pca.components_),
            atol=1e-12,
        )
        assert_allclose(
            changed_pca.explained_variance_ / factor / factor,
            pca.explained_variance_,
            rtol=1e-12,
        )
        assert_allclose(
            _projector(changed_ppca.loadings_.T),
            _projector(ppca.loadings_.T),
            atol=1e-12,
        )
        assert changed_ppca.noise_variance_ / factor / factor == pytest.approx(
            ppca.noise_variance_
        )
    with pytest.raises(FloatingPointError, match="not representable"):
        PCA(2).fit(X * 1e300)
    with pytest.raises(FloatingPointError, match="not representable"):
        ProbabilisticPCA(2).fit(X * 1e300)


def test_pca_rejects_nonboolean_whitening_flag() -> None:
    with pytest.raises(TypeError, match="whiten must be a boolean"):
        PCA(2, whiten="false").fit(_rng_data())  # type: ignore[arg-type]


def test_gaussian_projection_scaling_replay_and_global_rng() -> None:
    X = _rng_data()
    np.random.seed(812)
    state_before = np.random.get_state()
    first = GaussianRandomProjection(3, random_state=9).fit(X)
    second = GaussianRandomProjection(3, random_state=9).fit(X)
    state_after = np.random.get_state()
    assert_allclose(first.components_, second.components_)
    assert_allclose(first.transform(X), X @ first.components_.T)
    assert state_before[0] == state_after[0]
    assert_allclose(state_before[1], state_after[1])
    assert np.isclose(np.var(first.components_) * 3.0, 1.0, rtol=0.8)


def test_factor_analysis_em_recovers_covariance_structure() -> None:
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(600, 2))
    loadings = np.array([[1.2, 0.0], [0.8, 0.1], [0.0, 1.1], [0.2, 0.9], [0.4, -0.3]])
    noise_sd = np.array([0.2, 0.3, 0.25, 0.35, 0.4])
    X = latent @ loadings.T + rng.normal(size=(600, 5)) * noise_sd
    model = FactorAnalysis(2, tol=1e-7, max_iter=1000).fit(X)
    observed = np.cov(X, rowvar=False, bias=True)
    fitted = model.loadings_ @ model.loadings_.T + np.diag(model.noise_variance_)
    assert model.diagnostics_.converged
    assert np.linalg.norm(observed - fitted) / np.linalg.norm(observed) < 0.08
    expected_posterior = (X - model.mean_) @ np.linalg.solve(fitted, model.loadings_)
    assert_allclose(model.transform(X), expected_posterior, atol=1e-9)


def test_ppca_uses_ml_spectrum_and_posterior_mean() -> None:
    X = _rng_data()
    model = ProbabilisticPCA(2).fit(X)
    centered = X - X.mean(axis=0)
    covariance = centered.T @ centered / X.shape[0]
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    assert_allclose(model.noise_variance_, np.mean(eigenvalues[2:]), atol=1e-12)
    fitted_covariance = (
        model.loadings_ @ model.loadings_.T + model.noise_variance_ * np.eye(X.shape[1])
    )
    expected = centered @ np.linalg.solve(fitted_covariance, model.loadings_)
    assert_allclose(model.transform(X), expected, atol=1e-10)


def test_ppca_rejects_singular_zero_noise_boundary() -> None:
    rng = np.random.default_rng(101)
    latent = rng.normal(size=(100, 2))
    X = latent @ rng.normal(size=(2, 5))
    with pytest.raises(ValueError, match="singular zero boundary"):
        ProbabilisticPCA(2).fit(X)


def test_fastica_whitening_independence_and_inverse() -> None:
    rng = np.random.default_rng(44)
    source = np.column_stack(
        (rng.laplace(size=1500), rng.uniform(-np.sqrt(3), np.sqrt(3), size=1500))
    )
    mixing = np.array([[1.0, 0.5], [0.25, 1.3]])
    X = source @ mixing.T
    model = FastICA(2, random_state=3, tol=1e-7, max_iter=1000).fit(X)
    scores = model.transform(X)
    assert model.diagnostics_.converged
    assert_allclose(scores.T @ scores / X.shape[0], np.eye(2), atol=1e-10)
    assert_allclose(model.inverse_transform(scores), X, atol=1e-10)
    fourth_cross = np.mean((scores[:, 0] ** 2 - 1) * (scores[:, 1] ** 2 - 1))
    assert abs(fourth_cross) < 0.2


def test_sparse_pca_satisfies_declared_reconstruction_and_sparsifies() -> None:
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(200, 2))
    X = np.column_stack(
        (
            latent[:, 0] + 0.05 * rng.normal(size=200),
            latent[:, 0] + 0.05 * rng.normal(size=200),
            latent[:, 1] + 0.05 * rng.normal(size=200),
            latent[:, 1] + 0.05 * rng.normal(size=200),
            0.05 * rng.normal(size=200),
            0.05 * rng.normal(size=200),
        )
    )
    model = SparsePCA(2, alpha=0.03, ridge_alpha=0.01, max_iter=1000).fit(X)
    centered = X - model.mean_
    reconstruction = model.transform(X) @ model.reconstruction_components_ + model.mean_
    assert_allclose(
        reconstruction,
        centered @ model.components_.T @ model.reconstruction_components_ + model.mean_,
    )
    assert np.count_nonzero(np.abs(model.components_) < 1e-12) >= 2
    assert_allclose(np.linalg.norm(model.components_, axis=1), 1.0, atol=1e-12)
    assert np.isfinite(model.objective_)
    assert np.all(np.diff(model.objective_history_) <= 1e-12)
    assert model.kkt_residual_ < 2e-3


def test_sparse_pca_handles_a_constant_coordinate_without_ridge() -> None:
    X = _rng_data()[:, :4]
    X[:, -1] = 7.0
    model = SparsePCA(1, alpha=0.01, ridge_alpha=0.0, max_iter=1000).fit(X)
    assert model.raw_components_[0, -1] == 0.0
    assert np.all(np.isfinite(model.components_))


def test_lda_fisher_direction_and_label_recode_invariance() -> None:
    rng = np.random.default_rng(8)
    X0 = rng.normal(loc=[-2.0, 0.0, 0.0], scale=0.7, size=(100, 3))
    X1 = rng.normal(loc=[2.0, 0.0, 0.0], scale=0.7, size=(100, 3))
    X = np.vstack((X0, X1))
    labels = np.repeat(["left", "right"], 100)
    first = LinearDiscriminantAnalysis(1).fit(X, labels)
    second = LinearDiscriminantAnalysis(1).fit(X, np.where(labels == "left", 91, -4))
    assert_allclose(_projector(first.components_), _projector(second.components_))
    assert abs(first.components_[0, 0]) > 5 * abs(first.components_[0, 1])


def test_lda_does_not_discard_infinite_null_scatter_direction() -> None:
    X = np.array([[-2.0, -1.0], [-2.0, 1.0], [2.0, -1.0], [2.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="infinite Fisher eigenvalues"):
        LinearDiscriminantAnalysis(1).fit(X, labels)
    regularized = LinearDiscriminantAnalysis(1, regularization=1e-3).fit(X, labels)
    assert abs(regularized.components_[0, 0]) > 100 * abs(regularized.components_[0, 1])
    zero_within = np.array([[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    fallback = LinearDiscriminantAnalysis(1, regularization=0.1).fit(
        zero_within, labels
    )
    assert fallback.diagnostics_.condition_estimate is not None
    assert np.isfinite(fallback.diagnostics_.condition_estimate)


def test_cca_has_unit_variance_scores_and_singular_values() -> None:
    rng = np.random.default_rng(11)
    shared = rng.normal(size=(200, 2))
    X = shared @ np.array([[1.0, 0.2, -0.4], [0.1, 1.2, 0.3]])
    X += 0.05 * rng.normal(size=X.shape)
    Y = shared @ np.array([[0.8, -0.3], [0.2, 1.1]])
    Y += 0.05 * rng.normal(size=Y.shape)
    model = CanonicalCorrelationAnalysis(2)
    x_scores, y_scores = model.fit_transform(X, Y)
    assert_allclose(np.cov(x_scores, rowvar=False), np.eye(2), atol=1e-10)
    assert_allclose(np.cov(y_scores, rowvar=False), np.eye(2), atol=1e-10)
    correlations = np.diag(x_scores.T @ y_scores / float(X.shape[0] - 1))
    assert_allclose(correlations, model.canonical_correlations_, atol=1e-10)


def test_pls2_prediction_matches_exact_low_rank_linear_response() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(160, 4))
    coefficients = np.array([[1.5, -0.2], [-0.5, 0.7], [0.0, 0.0], [0.0, 0.0]])
    Y = X @ coefficients + np.array([2.0, -1.0])
    model = PLSRegression(4, tol=1e-10, max_iter=1000).fit(X, Y)
    assert_allclose(model.predict(X), Y, atol=1e-9)
    assert model.transform(X).shape == (160, 4)


def test_unscaled_pls_is_uniform_scale_equivariant() -> None:
    rng = np.random.default_rng(121)
    X = rng.normal(size=(200, 4))
    Y = X[:, :2] @ np.array([[1.0, -0.4], [0.3, 0.8]])
    baseline = PLSRegression(2, scale=False, tol=1e-10).fit(X, Y)
    for factor in (1e-10, 1e10):
        scaled = PLSRegression(2, scale=False, tol=1e-10).fit(factor * X, factor * Y)
        assert_allclose(
            _projector(baseline.x_rotations_.T),
            _projector(scaled.x_rotations_.T),
            atol=1e-10,
        )
        assert_allclose(
            scaled.predict(factor * X) / factor,
            baseline.predict(X),
            rtol=1e-10,
            atol=1e-10,
        )


def test_sir_recovers_linear_central_subspace_and_is_monotone_invariant() -> None:
    rng = np.random.default_rng(14)
    X = rng.normal(size=(1000, 4))
    y = 2.0 * X[:, 0] + 0.1 * rng.normal(size=X.shape[0])
    first = SlicedInverseRegression(1, n_slices=10).fit(X, y)
    second = SlicedInverseRegression(1, n_slices=10).fit(X, np.exp(y / 5.0))
    axis = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert np.linalg.norm(_projector(first.components_) - _projector(axis)) < 0.18
    assert_allclose(
        _projector(first.components_), _projector(second.components_), atol=1e-12
    )


def test_save_recovers_symmetric_quadratic_signal() -> None:
    rng = np.random.default_rng(18)
    X = rng.normal(size=(2000, 3))
    y = X[:, 0] ** 2 + 0.05 * rng.normal(size=X.shape[0])
    model = SlicedAverageVarianceEstimation(1, n_slices=8).fit(X, y)
    axis = np.array([[1.0, 0.0, 0.0]])
    assert np.linalg.norm(_projector(model.components_) - _projector(axis)) < 0.2


def test_supervised_subspaces_are_uniform_scale_equivariant() -> None:
    rng = np.random.default_rng(31)
    X = rng.normal(size=(300, 4))
    continuous = X[:, 0] + 0.2 * X[:, 1]
    labels = continuous > 0.0
    factories = [
        lambda data: LinearDiscriminantAnalysis(1).fit(data, labels),
        lambda data: SlicedInverseRegression(1, n_slices=6).fit(data, continuous),
        lambda data: SlicedAverageVarianceEstimation(1, n_slices=6).fit(
            data, continuous
        ),
    ]
    for factory in factories:
        ordinary = factory(X)
        tiny = factory(X * 1e-10)
        assert_allclose(
            _projector(ordinary.components_),
            _projector(tiny.components_),
            atol=1e-10,
        )


def test_fisher_score_matches_literal_weighted_formula() -> None:
    X = np.array([[0.0, 4.0, 1.0], [2.0, 5.0, 1.0], [6.0, 4.0, 1.0], [8.0, 5.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = FisherScore(1).fit(X, y)
    overall = X.mean(axis=0)
    numerator = sum(
        np.sum(y == group) * (X[y == group].mean(axis=0) - overall) ** 2
        for group in (0, 1)
    )
    denominator = sum(
        np.sum((X[y == group] - X[y == group].mean(axis=0)) ** 2, axis=0)
        for group in (0, 1)
    )
    expected = np.divide(
        numerator,
        denominator,
        out=np.zeros(3),
        where=denominator > 0,
    )
    assert_allclose(model.scores_, expected)
    assert_allclose(model.transform(X), X[:, [0]])


@pytest.mark.parametrize(
    "estimator",
    [
        PCA(),
        GaussianRandomProjection(random_state=1),
        FactorAnalysis(),
        ProbabilisticPCA(),
        FastICA(random_state=1),
        SparsePCA(),
        LinearDiscriminantAnalysis(),
        CanonicalCorrelationAnalysis(),
        PLSRegression(),
        SlicedInverseRegression(),
        SlicedAverageVarianceEstimation(),
        FisherScore(2),
    ],
)
def test_all_estimators_clone_with_constructor_parameters(estimator: object) -> None:
    cloned = clone(estimator)
    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)


def test_iterative_nonconvergence_is_truthful() -> None:
    X = _rng_data()
    models = [
        FactorAnalysis(2, max_iter=1, tol=1e-15),
        FastICA(2, max_iter=1, tol=1e-15, random_state=0),
        SparsePCA(2, max_iter=1, tol=1e-15),
    ]
    for model in models:
        model.fit(X)
        assert isinstance(model.diagnostics_, FitDiagnostics)
        assert not model.diagnostics_.converged
        assert model.diagnostics_.warnings

    targets = np.column_stack((X[:, 0] + X[:, 1], X[:, 2] - X[:, 3]))
    pls = PLSRegression(1, max_iter=1, tol=1e-15).fit(X, targets)
    assert not pls.diagnostics_.converged
    assert pls.diagnostics_.warnings


def test_pls_rejects_nonboolean_scale() -> None:
    X = _rng_data()
    with pytest.raises(TypeError, match="scale must be a boolean"):
        PLSRegression(1, scale=1).fit(X, X[:, 0])


def test_feature_names_pipeline_pickle_and_refit_contracts() -> None:
    X = _rng_data()
    model = PCA(2).fit(X)
    assert model.get_feature_names_out([f"f{i}" for i in range(6)]).tolist() == [
        "pca0",
        "pca1",
    ]
    with pytest.raises(ValueError, match="one name per fitted feature"):
        model.get_feature_names_out(["too", "short"])
    pipeline = Pipeline([("reduce", PCA(2)), ("project", PCA(1))]).fit(X)
    assert pipeline.transform(X).shape == (80, 1)
    restored = pickle.loads(pickle.dumps(model))
    assert_allclose(restored.transform(X), model.transform(X))
    model.fit(X[:, :4])
    assert model.n_features_in_ == 4
    assert model.transform(X[:, :4]).shape == (80, 2)


def test_fit_rejects_nonfinite_dense_data() -> None:
    X = _rng_data()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        PCA().fit(X)
