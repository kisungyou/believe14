"""Public typing and diagnostic records for believe14 estimators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, Self, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator


class Capability(StrEnum):
    """Capabilities advertised by a public estimator."""

    FIT_TRANSFORM = "fit_transform"
    TRANSFORM = "transform"
    INVERSE_TRANSFORM = "inverse_transform"
    PREDICT = "predict"
    FEATURE_SELECTION = "feature_selection"
    SUPERVISED = "supervised"
    PAIRED_INPUT = "paired_input"
    PRECOMPUTED = "precomputed"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True, slots=True, kw_only=True)
class FitDiagnostics:
    """Numerical status recorded by every successful fit."""

    solver: str
    converged: bool
    n_iter: int | None = None
    residual_norm: float | None = None
    objective_value: float | None = None
    numerical_rank: int | None = None
    condition_estimate: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class EstimatorInfo:
    """Immutable public registry entry."""

    name: str
    qualified_name: str
    family: Literal["linear", "nonlinear", "estimation"]
    approaches: frozenset[str]
    supervision: Literal["unsupervised", "supervised", "paired"]
    input_kinds: frozenset[str]
    capabilities: frozenset[Capability]
    out_of_sample: Literal["linear", "nystrom", "pivot", "native"] | None
    complexity: str
    references: tuple[str, ...]
    estimator: type[BaseEstimator]


class EstimatorProtocol(Protocol):
    """Static structural contract shared by public estimators."""

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> Self: ...


class DimensionEstimatorProtocol(EstimatorProtocol, Protocol):
    """Contract for global intrinsic-dimension estimators."""

    dimension_: float
    diagnostics_: FitDiagnostics

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...


class TransductiveEmbeddingProtocol(EstimatorProtocol, Protocol):
    """Contract for embeddings defined only on their training observations."""

    embedding_: NDArray[np.float64]
    diagnostics_: FitDiagnostics

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]: ...


class InductiveEmbeddingProtocol(TransductiveEmbeddingProtocol, Protocol):
    """Contract for embeddings with a justified out-of-sample map."""

    def transform(self, X: ArrayLike) -> NDArray[np.float64]: ...


class InvertibleEmbeddingProtocol(InductiveEmbeddingProtocol, Protocol):
    """Contract for embeddings with a defined reconstruction map."""

    def inverse_transform(self, X: ArrayLike) -> NDArray[np.float64]: ...


class FeatureSelectorProtocol(EstimatorProtocol, Protocol):
    """Contract for supervised or unsupervised feature selectors."""

    diagnostics_: FitDiagnostics

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...

    def transform(self, X: ArrayLike) -> NDArray[np.float64]: ...

    def get_support(
        self, indices: bool = False
    ) -> NDArray[np.bool_] | NDArray[np.intp]: ...


class PairedViewEmbeddingProtocol(EstimatorProtocol, Protocol):
    """Contract for estimators trained on two aligned data views."""

    diagnostics_: FitDiagnostics

    def fit(self, X: ArrayLike, Y: ArrayLike) -> Self: ...

    @overload
    def transform(self, X: ArrayLike, Y: None = None) -> NDArray[np.float64]: ...

    @overload
    def transform(
        self, X: ArrayLike, Y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
