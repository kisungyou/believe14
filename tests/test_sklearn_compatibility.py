"""Applicable official scikit-learn estimator conformance checks."""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

from believe14.linear import (
    PCA,
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
    ClassicalMDS,
    DiffusionMap,
    FastMap,
    KernelPCA,
    MetricMDS,
)


@pytest.mark.parametrize(
    "estimator",
    [
        PCA(1),
        GaussianRandomProjection(1, random_state=0),
        FactorAnalysis(1, max_iter=50),
        ProbabilisticPCA(1),
        FastICA(1, random_state=0, max_iter=50),
        SparsePCA(1, alpha=0.01, max_iter=50),
        LinearDiscriminantAnalysis(1),
        PLSRegression(1),
        SlicedInverseRegression(1),
        SlicedAverageVarianceEstimation(1),
        FisherScore(1),
        ClassicalMDS(1),
        MetricMDS(1, max_iter=20),
        FastMap(1),
        KernelPCA(1, kernel="linear"),
        DiffusionMap(1),
    ],
    ids=lambda estimator: type(estimator).__name__,
)
def test_applicable_official_estimator_checks(estimator: BaseEstimator) -> None:
    # Paired-view and deliberately transductive/degeneracy-rejecting estimators
    # have bespoke contract tests; this set is the subset to which the generic
    # sklearn checks apply without weakening either library's scientific domain.
    check_estimator(estimator, on_skip=None)
