"""Paper-aligned linear dimensionality reduction estimators."""

from ._components import FastICA, SparsePCA
from ._inverse_regression import (
    SlicedAverageVarianceEstimation,
    SlicedInverseRegression,
)
from ._pca import PCA, GaussianRandomProjection
from ._probabilistic import FactorAnalysis, ProbabilisticPCA
from ._selection import FisherScore
from ._supervised import (
    CanonicalCorrelationAnalysis,
    LinearDiscriminantAnalysis,
    PLSRegression,
)

__all__ = [
    "PCA",
    "CanonicalCorrelationAnalysis",
    "FactorAnalysis",
    "FastICA",
    "FisherScore",
    "GaussianRandomProjection",
    "LinearDiscriminantAnalysis",
    "PLSRegression",
    "ProbabilisticPCA",
    "SlicedAverageVarianceEstimation",
    "SlicedInverseRegression",
    "SparsePCA",
]
