"""Nonlinear dimensionality-reduction estimators."""

from ._kernel import DiffusionMap, KernelPCA
from ._landmarks import LandmarkIsomap, LandmarkMDS
from ._manifold import (
    Isomap,
    LaplacianEigenmaps,
    LocallyLinearEmbedding,
    LocalTangentSpaceAlignment,
)
from ._mds import ClassicalMDS, FastMap, MetricMDS, SammonMapping
from ._stochastic import PHATE, TSNE

__all__ = [
    "PHATE",
    "TSNE",
    "ClassicalMDS",
    "DiffusionMap",
    "FastMap",
    "Isomap",
    "KernelPCA",
    "LandmarkIsomap",
    "LandmarkMDS",
    "LaplacianEigenmaps",
    "LocalTangentSpaceAlignment",
    "LocallyLinearEmbedding",
    "MetricMDS",
    "SammonMapping",
]
