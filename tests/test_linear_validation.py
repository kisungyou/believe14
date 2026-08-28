"""Failure-mode and fitted-state coverage for linear estimators."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

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
from believe14.linear._common import as_second_view, response_slices, validate_vector
from believe14.linear._components import _symmetric_decorrelation
from believe14.linear._probabilistic import _gaussian_log_likelihood


@pytest.fixture
def data() -> np.ndarray:
    return np.random.default_rng(51).normal(size=(30, 4))


def test_pca_degenerate_whitening_inverse_and_unfitted_contracts(
    data: np.ndarray,
) -> None:
    constant_rank = np.column_stack((data[:, 0], data[:, 0], np.ones(len(data))))
    with pytest.raises(ValueError, match="retained components"):
        PCA(3, whiten=True).fit(constant_rank)
    model = PCA(2).fit(data)
    with pytest.raises(ValueError, match="exactly 2 columns"):
        model.inverse_transform(np.ones((3, 1)))
    invalid = np.ones((3, 2))
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        model.inverse_transform(invalid)
    with pytest.raises(NotFittedError):
        GaussianRandomProjection().get_feature_names_out()
    assert GaussianRandomProjection(2, random_state=2).fit(
        data
    ).get_feature_names_out().tolist() == [
        "gaussianrandomprojection0",
        "gaussianrandomprojection1",
    ]


def test_common_response_and_paired_view_validation(data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_vector(np.ones((30, 2)), n_samples=30)
    with pytest.raises(ValueError, match="expected 30"):
        validate_vector(np.arange(31), n_samples=30)
    with pytest.raises(TypeError, match="numeric"):
        response_slices(np.array(["a", "b"] * 15), n_samples=30, n_slices=2)
    with pytest.raises(TypeError, match="integer"):
        response_slices(np.arange(30), n_samples=30, n_slices=True)
    with pytest.raises(ValueError, match="at least 2"):
        response_slices(np.arange(30), n_samples=30, n_slices=1)
    with pytest.raises(ValueError, match="distinct"):
        response_slices(np.ones(30), n_samples=30, n_slices=2)
    labels = response_slices(np.repeat([0.0, 1.0, 2.0], 10), n_samples=30, n_slices=5)
    assert np.unique(labels).size == 3
    with pytest.raises(ValueError, match="same number"):
        as_second_view(np.ones((31, 2)), n_samples=30)


def test_probabilistic_model_failure_and_inverse_contracts(data: np.ndarray) -> None:
    with pytest.raises(TypeError, match="max_iter"):
        FactorAnalysis(1, max_iter=True).fit(data)
    with pytest.raises(ValueError, match="max_iter"):
        FactorAnalysis(1, max_iter=0).fit(data)
    constant = data.copy()
    constant[:, -1] = 1.0
    with pytest.raises(ValueError, match="every feature"):
        FactorAnalysis(1).fit(constant)
    factor = FactorAnalysis(1, max_iter=3).fit(data)
    with pytest.raises(ValueError, match="exactly 1 columns"):
        factor.inverse_transform(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        factor.inverse_transform(np.array([[np.nan]]))
    assert factor.get_feature_names_out().tolist() == ["factoranalysis0"]

    with pytest.raises(ValueError, match=r"1 feature\(s\)"):
        ProbabilisticPCA(1).fit(data[:, :1])
    isotropic = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match="not separated"):
        ProbabilisticPCA(1).fit(isotropic)
    ppca = ProbabilisticPCA(1).fit(data)
    with pytest.raises(ValueError, match="exactly 1 columns"):
        ppca.inverse_transform(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        ppca.inverse_transform(np.array([[np.inf]]))
    assert ppca.get_feature_names_out().tolist() == ["probabilisticpca0"]
    assert _gaussian_log_likelihood(np.eye(2), np.zeros((2, 2)), 3) == -np.inf


def test_fastica_and_sparse_parameter_and_degeneracy_policies(
    data: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        FastICA(1, alpha=0.5).fit(data)
    with pytest.raises(TypeError, match="max_iter"):
        FastICA(1, max_iter=True).fit(data)
    with pytest.raises(ValueError, match="max_iter"):
        FastICA(1, max_iter=0).fit(data)
    with pytest.raises(ValueError, match="numerical rank"):
        FastICA(2).fit(np.column_stack((data[:, 0], data[:, 0])))
    with pytest.raises(ValueError, match="rank-deficient"):
        _symmetric_decorrelation(np.zeros((2, 2)))
    ica = FastICA(2, random_state=1).fit(data)
    with pytest.raises(ValueError, match="exactly 2 columns"):
        ica.inverse_transform(np.ones((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        ica.inverse_transform(np.array([[np.inf, 0.0]]))
    assert len(ica.get_feature_names_out()) == 2

    with pytest.raises(ValueError, match="init"):
        SparsePCA(1, init="bad").fit(data)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="coordinate_max_iter"):
        SparsePCA(1, coordinate_max_iter=True).fit(data)
    with pytest.raises(ValueError, match="max_iter"):
        SparsePCA(1, max_iter=0).fit(data)
    with pytest.raises(ValueError, match="zero component"):
        SparsePCA(1, alpha=100.0).fit(data)
    random_model = SparsePCA(1, init="random", random_state=5, max_iter=5).fit(data)
    assert random_model.transform(data).shape == (30, 1)
    assert random_model.get_feature_names_out().tolist() == ["sparsepca0"]


def test_supervised_required_targets_and_degenerate_classes(data: np.ndarray) -> None:
    supervised = (
        LinearDiscriminantAnalysis(1),
        PLSRegression(1),
        SlicedInverseRegression(1),
        SlicedAverageVarianceEstimation(1),
        FisherScore(1),
    )
    for estimator in supervised:
        with pytest.raises(ValueError, match="y is required"):
            estimator.fit(data)
    for estimator in (LinearDiscriminantAnalysis(1), FisherScore(1)):
        with pytest.raises(ValueError, match="at least two classes"):
            estimator.fit(data, np.zeros(len(data)))
    rank_one = np.column_stack((data[:, 0], data[:, 0]))
    labels = data[:, 0] > 0.0
    with pytest.raises(ValueError, match="numerical rank"):
        LinearDiscriminantAnalysis(2).fit(rank_one, np.arange(len(data)) % 3)
    regularized = LinearDiscriminantAnalysis(1, regularization=0.1).fit(
        rank_one, labels
    )
    assert regularized.transform(rank_one).shape == (30, 1)
    assert len(regularized.get_feature_names_out()) == 1


def test_cca_transform_variants_and_view_validation(data: np.ndarray) -> None:
    Y = np.column_stack((data[:, 0] + data[:, 1], data[:, 2]))
    model = CanonicalCorrelationAnalysis(1).fit(data, Y)
    assert model.transform(data).shape == (30, 1)
    x_scores, y_scores = model.transform(data, Y)
    assert x_scores.shape == y_scores.shape == (30, 1)
    with pytest.raises(ValueError, match="same number"):
        model.transform(data, np.ones((31, 2)))
    with pytest.raises(ValueError, match="expected 2"):
        model.transform(data, np.ones((30, 3)))
    assert model.get_feature_names_out().tolist() == ["canonicalcorrelationanalysis0"]


def test_pls_target_transform_and_failure_paths(data: np.ndarray) -> None:
    target = np.column_stack((data[:, 0], data[:, 1]))
    model = PLSRegression(1).fit(data, target)
    x_scores, y_scores = model.transform(data, target)
    assert x_scores.shape == y_scores.shape == (30, 1)
    with pytest.raises(ValueError, match="expected 2"):
        model.transform(data, data[:, 0])
    with pytest.raises(ValueError, match="y is required"):
        PLSRegression(1).fit_transform(data)
    with pytest.raises(TypeError, match="max_iter"):
        PLSRegression(1, max_iter=True).fit(data, target)
    with pytest.raises(ValueError, match="max_iter"):
        PLSRegression(1, max_iter=0).fit(data, target)
    constant_feature = data.copy()
    constant_feature[:, -1] = 0.0
    with pytest.raises(ValueError, match="constant X feature"):
        PLSRegression(1).fit(constant_feature, target)
    with pytest.raises(ValueError, match="Y residual"):
        PLSRegression(1).fit(data, np.ones(len(data)))
    one_target = PLSRegression(1).fit(data, data[:, 0])
    assert one_target.predict(data).ndim == 1
    assert one_target.get_feature_names_out().tolist() == ["plsregression0"]


@pytest.mark.parametrize("factor", [1.0e-170, 1.0e154])
def test_pls_sample_scaling_is_numerically_neutral(factor: float) -> None:
    rng = np.random.default_rng(181)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=(50, 2))
    baseline = PLSRegression(2).fit(X, y)
    changed = PLSRegression(2).fit(X * factor, y * factor)
    np.testing.assert_allclose(
        changed.x_scores_, baseline.x_scores_, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        changed.predict(X * factor) / factor,
        baseline.predict(X),
        rtol=1e-5,
        atol=1e-5,
    )


def test_lda_public_scatters_retain_original_units() -> None:
    rng = np.random.default_rng(182)
    X = rng.normal(size=(60, 4))
    labels = np.repeat(np.arange(3), 20)
    baseline = LinearDiscriminantAnalysis(2, regularization=0.1).fit(X, labels)
    changed = LinearDiscriminantAnalysis(2, regularization=0.1).fit(10.0 * X, labels)
    np.testing.assert_allclose(
        changed.within_scatter_, 100.0 * baseline.within_scatter_
    )
    np.testing.assert_allclose(
        changed.between_scatter_, 100.0 * baseline.between_scatter_
    )


def test_inverse_regression_slice_and_state_failures(data: np.ndarray) -> None:
    y = np.arange(len(data), dtype=np.float64)
    with pytest.raises(TypeError, match="n_slices"):
        SlicedInverseRegression(1, n_slices=True).fit(data, y)
    with pytest.raises(ValueError, match="zero numerical variance"):
        SlicedInverseRegression(1, n_slices=2).fit(np.ones_like(data), y)
    with pytest.raises(ValueError, match="at least two observations"):
        SlicedAverageVarianceEstimation(1, n_slices=30).fit(data, y)
    sir = SlicedInverseRegression(1, n_slices=3).fit(data, y)
    save = SlicedAverageVarianceEstimation(1, n_slices=3).fit(data, y)
    assert sir.transform(data).shape == save.transform(data).shape == (30, 1)
    assert sir.get_feature_names_out().tolist() == ["slicedinverseregression0"]
    assert save.get_feature_names_out().tolist() == ["slicedaveragevarianceestimation0"]


def test_feature_name_mismatch_and_fisher_degeneracy(data: np.ndarray) -> None:
    model = PCA(1).fit(data)
    model.feature_names_in_ = np.asarray(["a", "b", "c", "d"], dtype=object)
    with pytest.raises(ValueError, match="match feature_names_in"):
        model.get_feature_names_out(["a", "b", "c", "wrong"])
    with pytest.raises(NotFittedError):
        FisherScore(1).get_support()
    separated = np.array([[0.0, 1.0], [0.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    selector = FisherScore(1).fit(separated, [0, 0, 1, 1])
    assert np.isinf(selector.scores_[0])
    assert selector.diagnostics_.warnings
    assert_allclose(selector.transform(separated), separated[:, [0]])
