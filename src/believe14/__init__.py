"""Paper-first dimensionality reduction and dimension estimation."""

from importlib.metadata import PackageNotFoundError, version

from .api import (
    Capability,
    DimensionEstimatorProtocol,
    EstimatorInfo,
    FeatureSelectorProtocol,
    FitDiagnostics,
    InductiveEmbeddingProtocol,
    InvertibleEmbeddingProtocol,
    PairedViewEmbeddingProtocol,
    TransductiveEmbeddingProtocol,
)
from .registry import get_estimator, list_estimators

try:
    __version__ = version("believe14")
except PackageNotFoundError:  # pragma: no cover - editable source without metadata
    __version__ = "0.1.0"

__all__ = [
    "Capability",
    "DimensionEstimatorProtocol",
    "EstimatorInfo",
    "FeatureSelectorProtocol",
    "FitDiagnostics",
    "InductiveEmbeddingProtocol",
    "InvertibleEmbeddingProtocol",
    "PairedViewEmbeddingProtocol",
    "TransductiveEmbeddingProtocol",
    "__version__",
    "get_estimator",
    "list_estimators",
]
