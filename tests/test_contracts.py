"""Package-wide estimator, registry, serialization, and output contracts."""

from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

from believe14.api import Capability, FitDiagnostics
from believe14.estimation import (
    CorrelationDimension,
    DANCo,
    LevinaBickelMLE,
    MiNDML,
    TwoNN,
    UStatisticDimension,
)
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
from believe14.nonlinear import (
    PHATE,
    TSNE,
    ClassicalMDS,
    DiffusionMap,
    FastMap,
    Isomap,
    KernelPCA,
    LaplacianEigenmaps,
    LocallyLinearEmbedding,
    LocalTangentSpaceAlignment,
    MetricMDS,
    SammonMapping,
)
from believe14.registry import list_estimators


def _data() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(814)
    latent = rng.normal(size=(48, 3))
    mixing = rng.normal(size=(3, 5))
    X = latent @ mixing + 0.05 * rng.normal(size=(48, 5))
    Y = latent @ np.array([[1.0, -0.2], [0.3, 0.9], [-0.4, 0.5]])
    Y += 0.03 * rng.normal(size=Y.shape)
    labels = np.repeat(np.arange(3), 16)
    response = X[:, 0] + 0.3 * X[:, 1]
    return X, Y, labels, response


FitCall = Callable[[BaseEstimator], BaseEstimator]


def _scenarios() -> dict[str, tuple[BaseEstimator, FitCall]]:
    X, Y, labels, response = _data()

    def x_fit(estimator: BaseEstimator) -> BaseEstimator:
        return estimator.fit(X)

    def label_fit(estimator: BaseEstimator) -> BaseEstimator:
        return estimator.fit(X, labels)

    def response_fit(estimator: BaseEstimator) -> BaseEstimator:
        return estimator.fit(X, response)

    def paired_fit(estimator: BaseEstimator) -> BaseEstimator:
        return estimator.fit(X, Y)

    return {
        "PCA": (PCA(2), x_fit),
        "GaussianRandomProjection": (
            GaussianRandomProjection(2, random_state=3),
            x_fit,
        ),
        "FactorAnalysis": (FactorAnalysis(2, max_iter=1000), x_fit),
        "ProbabilisticPCA": (ProbabilisticPCA(2), x_fit),
        "FastICA": (FastICA(2, random_state=3, max_iter=1000), x_fit),
        "SparsePCA": (SparsePCA(2, max_iter=200), x_fit),
        "LinearDiscriminantAnalysis": (
            LinearDiscriminantAnalysis(2, regularization=1e-8),
            label_fit,
        ),
        "CanonicalCorrelationAnalysis": (
            CanonicalCorrelationAnalysis(2),
            paired_fit,
        ),
        "PLSRegression": (PLSRegression(2), paired_fit),
        "SlicedInverseRegression": (
            SlicedInverseRegression(2, n_slices=6),
            response_fit,
        ),
        "SlicedAverageVarianceEstimation": (
            SlicedAverageVarianceEstimation(2, n_slices=6),
            response_fit,
        ),
        "FisherScore": (FisherScore(3), label_fit),
        "ClassicalMDS": (ClassicalMDS(2), x_fit),
        "MetricMDS": (MetricMDS(2, max_iter=100), x_fit),
        "SammonMapping": (SammonMapping(2, max_iter=100), x_fit),
        "FastMap": (FastMap(2), x_fit),
        "KernelPCA": (KernelPCA(2, kernel="linear"), x_fit),
        "Isomap": (Isomap(2, n_neighbors=10), x_fit),
        "LocallyLinearEmbedding": (
            LocallyLinearEmbedding(2, n_neighbors=10),
            x_fit,
        ),
        "LaplacianEigenmaps": (
            LaplacianEigenmaps(2, n_neighbors=10),
            x_fit,
        ),
        "DiffusionMap": (DiffusionMap(2), x_fit),
        "LocalTangentSpaceAlignment": (
            LocalTangentSpaceAlignment(2, n_neighbors=10),
            x_fit,
        ),
        "TSNE": (
            TSNE(
                2,
                perplexity=8,
                early_exaggeration_iter=10,
                max_iter=40,
                random_state=3,
            ),
            x_fit,
        ),
        "PHATE": (
            PHATE(2, n_neighbors=8, decay=5, diffusion_time=4),
            x_fit,
        ),
        "CorrelationDimension": (CorrelationDimension(), x_fit),
        "TwoNN": (TwoNN(), x_fit),
        "LevinaBickelMLE": (
            LevinaBickelMLE(k_min=5, k_max=9),
            x_fit,
        ),
        "UStatisticDimension": (
            UStatisticDimension(max_dimension=3, random_state=3),
            x_fit,
        ),
        "MiNDML": (MiNDML(n_neighbors=5, max_dimension=3), x_fit),
        "DANCo": (
            DANCo(n_neighbors=5, max_dimension=3, random_state=3),
            x_fit,
        ),
    }


@pytest.fixture(scope="module")
def fitted_estimators() -> dict[str, BaseEstimator]:
    fitted: dict[str, BaseEstimator] = {}
    for name, (estimator, fit_call) in _scenarios().items():
        assert fit_call(estimator) is estimator
        fitted[name] = estimator
    return fitted


def test_every_registry_record_has_a_successful_fit_and_diagnostics(
    fitted_estimators: dict[str, BaseEstimator],
) -> None:
    assert set(fitted_estimators) == {info.name for info in list_estimators()}
    for estimator in fitted_estimators.values():
        record = estimator.diagnostics_  # type: ignore[attr-defined]
        assert isinstance(record, FitDiagnostics)
        assert record.solver
        for value in (
            record.objective_value,
            record.residual_norm,
            record.condition_estimate,
        ):
            assert value is None or np.isfinite(value)
        assert record.n_iter is None or record.n_iter >= 0


def test_registry_capabilities_match_the_concrete_api() -> None:
    method_capabilities = {
        Capability.FIT_TRANSFORM: "fit_transform",
        Capability.TRANSFORM: "transform",
        Capability.INVERSE_TRANSFORM: "inverse_transform",
        Capability.PREDICT: "predict",
    }
    for info in list_estimators():
        estimator = info.estimator()
        for capability, method_name in method_capabilities.items():
            assert hasattr(estimator, method_name) is (capability in info.capabilities)
        if info.family == "estimation":
            assert not hasattr(estimator, "score")


def test_fitted_estimators_are_cloneable_and_pickleable(
    fitted_estimators: dict[str, BaseEstimator],
) -> None:
    for estimator in fitted_estimators.values():
        copied = clone(estimator)
        assert copied.get_params(deep=False) == estimator.get_params(deep=False)
        restored = pickle.loads(pickle.dumps(estimator))
        assert restored.get_params(deep=False) == estimator.get_params(deep=False)
        assert restored.diagnostics_ == estimator.diagnostics_  # type: ignore[attr-defined]


def test_fitted_reducers_report_output_names(
    fitted_estimators: dict[str, BaseEstimator],
) -> None:
    for info in list_estimators():
        estimator = fitted_estimators[info.name]
        if info.family == "estimation":
            continue
        names = estimator.get_feature_names_out()  # type: ignore[attr-defined]
        if info.name == "FisherScore":
            assert names.shape == (3,)
        else:
            assert names.shape == (2,)
        assert len(set(names.tolist())) == len(names)


def test_only_the_cited_inductive_estimators_expose_transform() -> None:
    inductive = {info.name for info in list_estimators(capability=Capability.TRANSFORM)}
    assert inductive == {
        "PCA",
        "GaussianRandomProjection",
        "FactorAnalysis",
        "ProbabilisticPCA",
        "FastICA",
        "SparsePCA",
        "LinearDiscriminantAnalysis",
        "CanonicalCorrelationAnalysis",
        "PLSRegression",
        "SlicedInverseRegression",
        "SlicedAverageVarianceEstimation",
        "FisherScore",
        "FastMap",
        "KernelPCA",
        "DiffusionMap",
    }
