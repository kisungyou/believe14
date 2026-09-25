"""Explicit registry of the 30 public believe14 estimators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from typing import Literal

from .api import Capability, EstimatorInfo

Family = Literal["linear", "nonlinear", "estimation"]
Supervision = Literal["unsupervised", "supervised", "paired"]


@dataclass(frozen=True, slots=True)
class _Spec:
    name: str
    family: Family
    approaches: tuple[str, ...]
    supervision: Supervision
    input_kinds: tuple[str, ...]
    capabilities: tuple[Capability, ...]
    out_of_sample: Literal["linear", "nystrom", "pivot", "native"] | None
    complexity: str
    references: tuple[str, ...]
    validation_status: Literal["validated", "experimental"] = "validated"

    @property
    def module(self) -> str:
        return f"believe14.{self.family}"


_FT = Capability.FIT_TRANSFORM
_T = Capability.TRANSFORM
_I = Capability.INVERSE_TRANSFORM
_S = Capability.SUPERVISED
_P = Capability.PAIRED_INPUT
_R = Capability.PRECOMPUTED
_Z = Capability.STOCHASTIC


_SPECS: tuple[_Spec, ...] = (
    _Spec(
        "PCA",
        "linear",
        ("spectral",),
        "unsupervised",
        ("features",),
        (_FT, _T, _I),
        "linear",
        "O(min(np^2,n^2p)) time; O(np) memory",
        ("Pearson (1901)",),
    ),
    _Spec(
        "GaussianRandomProjection",
        "linear",
        ("random_projection",),
        "unsupervised",
        ("features",),
        (_FT, _T, _Z),
        "linear",
        "O(np + pk + min(pk^2,p^2k)) time; O(np + pk) memory",
        ("Bingham and Mannila (2001)",),
    ),
    _Spec(
        "FactorAnalysis",
        "linear",
        ("probabilistic", "latent_variable"),
        "unsupervised",
        ("features",),
        (_FT, _T, _I),
        "linear",
        "O(np^2 + T p^3) time; O(np + p^2) memory",
        ("Rubin and Thayer (1982)",),
    ),
    _Spec(
        "ProbabilisticPCA",
        "linear",
        ("probabilistic", "spectral"),
        "unsupervised",
        ("features",),
        (_FT, _T, _I),
        "linear",
        "O(np^2 + p^3) time; O(np + p^2) memory",
        ("Tipping and Bishop (1999)",),
    ),
    _Spec(
        "FastICA",
        "linear",
        ("independent_components", "fixed_point"),
        "unsupervised",
        ("features",),
        (_FT, _T, _I, _Z),
        "linear",
        "O(min(np^2,n^2p) + T(nk^2 + k^3) + pk^2) time; O(np + pk) memory",
        ("Hyvarinen (1999)",),
    ),
    _Spec(
        "SparsePCA",
        "linear",
        ("sparse", "optimization"),
        "unsupervised",
        ("features",),
        (_FT, _T, _Z),
        "linear",
        "O(np^2 + p^3 + T(Ckp^2 + npk + pk^2)) time; O(np + p^2 + T) memory",
        ("Zou, Hastie, and Tibshirani (2006)",),
    ),
    _Spec(
        "LinearDiscriminantAnalysis",
        "linear",
        ("supervised", "spectral"),
        "supervised",
        ("features", "labels"),
        (_FT, _T, _S),
        "linear",
        "O(np^2 + p^3 + n log n + Ln) time; O(np + p^2) memory",
        ("Fisher (1936); Rao (1948)",),
    ),
    _Spec(
        "CanonicalCorrelationAnalysis",
        "linear",
        ("multiview", "spectral"),
        "paired",
        ("paired_features",),
        (_FT, _T, _P),
        "linear",
        "O(n(p+q)^2 + (p+q)^3) time; O(n(p+q) + (p+q)^2) memory",
        ("Hotelling (1936)",),
    ),
    _Spec(
        "PLSRegression",
        "linear",
        ("supervised", "latent_variable", "regression"),
        "supervised",
        ("features", "targets"),
        (_FT, _T, _S, Capability.PREDICT),
        "linear",
        "O(Tkn(p+q) + (p+q)k^2 + k^3 + pqk) time; O(n(p+q) + pq) memory",
        ("Wold et al. (1984)",),
    ),
    _Spec(
        "SlicedInverseRegression",
        "linear",
        ("supervised", "inverse_regression"),
        "supervised",
        ("features", "targets"),
        (_FT, _T, _S),
        "linear",
        "O(np^2 + p^3 + n log n + Sn) time; O(np + p^2) memory",
        ("Li (1991)",),
    ),
    _Spec(
        "SlicedAverageVarianceEstimation",
        "linear",
        ("supervised", "inverse_regression"),
        "supervised",
        ("features", "targets"),
        (_FT, _T, _S),
        "linear",
        "O(np^2 + (S+1)p^3 + n log n + Sn) time; O(np + p^2) memory",
        ("Cook (2000)",),
    ),
    _Spec(
        "FisherScore",
        "linear",
        ("supervised", "feature_selection"),
        "supervised",
        ("features", "labels"),
        (_FT, _T, _I, _S, Capability.FEATURE_SELECTION),
        "linear",
        "O(np + n log n + p log p + Ln) time; O(np) memory",
        ("Duda, Hart, and Stork (2001)",),
    ),
    _Spec(
        "ClassicalMDS",
        "nonlinear",
        ("distance", "spectral"),
        "unsupervised",
        ("features", "precomputed_distances"),
        (_FT, _R),
        None,
        "O(n^2 p + n^3) time; O(np + n^2) memory",
        ("Torgerson (1952)",),
    ),
    _Spec(
        "MetricMDS",
        "nonlinear",
        ("distance", "stress", "optimization"),
        "unsupervised",
        ("features", "precomputed_distances"),
        (_FT, _R, _Z),
        None,
        "O(n^2 p + n^3 + T n^2 k) time; O(np + n^2) memory",
        ("de Leeuw (1977)",),
    ),
    _Spec(
        "SammonMapping",
        "nonlinear",
        ("distance", "stress", "optimization"),
        "unsupervised",
        ("features", "precomputed_distances"),
        (_FT, _R, _Z),
        None,
        "O(n^2 p + n^3 + E n^2 k) time; O(np + n^2) memory",
        ("Sammon (1969)",),
    ),
    _Spec(
        "FastMap",
        "nonlinear",
        ("distance", "pivot"),
        "unsupervised",
        ("features", "precomputed_distances"),
        (_FT, _T, _R),
        "pivot",
        "O(n^2 p + k n^2 + Rkn) time; O(np + n^2) memory",
        ("Faloutsos and Lin (1995)",),
    ),
    _Spec(
        "KernelPCA",
        "nonlinear",
        ("kernel", "spectral"),
        "unsupervised",
        ("features", "precomputed_kernel"),
        (_FT, _T, _R),
        "nystrom",
        "O(n^2 p + n^3) time; O(np + n^2) memory",
        ("Scholkopf, Smola, and Muller (1998)",),
    ),
    _Spec(
        "Isomap",
        "nonlinear",
        ("graph", "geodesic", "spectral"),
        "unsupervised",
        ("features",),
        (_FT,),
        None,
        "O(n^2 p + n^3) time; O(np + n^2) memory",
        ("Tenenbaum, de Silva, and Langford (2000)",),
    ),
    _Spec(
        "LocallyLinearEmbedding",
        "nonlinear",
        ("graph", "local_reconstruction", "spectral"),
        "unsupervised",
        ("features",),
        (_FT,),
        None,
        "O(n^2 p + n h^2 p + n h^3 + n^3) time; O(np + n^2) memory",
        ("Roweis and Saul (2000)",),
    ),
    _Spec(
        "LaplacianEigenmaps",
        "nonlinear",
        ("graph", "spectral"),
        "unsupervised",
        ("features",),
        (_FT,),
        None,
        "O(n^2 p + n^3) time; O(np + n^2) memory",
        ("Belkin and Niyogi (2003)",),
    ),
    _Spec(
        "DiffusionMap",
        "nonlinear",
        ("graph", "diffusion", "spectral"),
        "unsupervised",
        ("features",),
        (_FT, _T),
        "nystrom",
        "O(n^2 p + n^3) time; O(np + n^2) memory",
        ("Coifman and Lafon (2006)",),
    ),
    _Spec(
        "LocalTangentSpaceAlignment",
        "nonlinear",
        ("graph", "local_tangent", "spectral"),
        "unsupervised",
        ("features",),
        (_FT,),
        None,
        "O(n^2 p + n min(hp^2,h^2p) + n h^2 k + n^3) time; O(np + n^2) memory",
        ("Zhang and Zha (2004)",),
    ),
    _Spec(
        "TSNE",
        "nonlinear",
        ("probabilistic", "stochastic", "optimization"),
        "unsupervised",
        ("features",),
        (_FT, _Z),
        None,
        "O(n^2 p + min(np^2,n^2p) + E n^2 k) time; O(np + n^2) memory",
        ("van der Maaten and Hinton (2008)",),
    ),
    _Spec(
        "PHATE",
        "nonlinear",
        ("graph", "diffusion", "stress"),
        "unsupervised",
        ("features",),
        (_FT,),
        None,
        "O(n^2 p + n^3 log(t+1) + T n^2 k) time; O(np + n^2) memory",
        ("Moon et al. (2019)",),
    ),
    _Spec(
        "CorrelationDimension",
        "estimation",
        ("correlation_integral", "scaling"),
        "unsupervised",
        ("features",),
        (),
        None,
        "O(n^2(p + log n) + B n^2) time; O(np + n^2) memory",
        ("Grassberger and Procaccia (1983)",),
    ),
    _Spec(
        "TwoNN",
        "estimation",
        ("nearest_neighbors", "distance_ratio"),
        "unsupervised",
        ("features",),
        (),
        None,
        "O(n^2(p + log n)) time; O(np + n^2) memory",
        ("Facco et al. (2017)",),
    ),
    _Spec(
        "LevinaBickelMLE",
        "estimation",
        ("nearest_neighbors", "likelihood"),
        "unsupervised",
        ("features",),
        (),
        None,
        "O(n^2(p + log n) + n h^2) time; O(np + n^2) memory",
        ("Levina and Bickel (2004)",),
    ),
    _Spec(
        "UStatisticDimension",
        "estimation",
        ("u_statistic", "scaling"),
        "unsupervised",
        ("features",),
        (_Z,),
        None,
        "O(n^2(p + D)) time; O(np + n^2) memory",
        ("Hein and Audibert (2005)",),
        validation_status="experimental",
    ),
    _Spec(
        "MiNDML",
        "estimation",
        ("nearest_neighbors", "likelihood"),
        "unsupervised",
        ("features",),
        (),
        None,
        "O(n^2(p + log n) + Tn) time; O(np + n^2) memory",
        ("Lombardi et al. (2011)",),
    ),
    _Spec(
        "DANCo",
        "estimation",
        ("nearest_neighbors", "angles", "concentration"),
        "unsupervised",
        ("features",),
        (_Z,),
        None,
        "O(n^2(p + D^2 + D log n) + n h^2(p + D^2) + DTn + DQ "
        "+ min(np^2,n^2p)) time; O(n^2 + np + nD + h^2) memory",
        ("Ceruti et al. (2014)",),
    ),
)


@cache
def _materialize(spec: _Spec) -> EstimatorInfo:
    module = import_module(spec.module)
    estimator = getattr(module, spec.name)
    return EstimatorInfo(
        name=spec.name,
        qualified_name=f"{spec.module}.{spec.name}",
        family=spec.family,
        approaches=frozenset(spec.approaches),
        supervision=spec.supervision,
        input_kinds=frozenset(spec.input_kinds),
        capabilities=frozenset(spec.capabilities),
        out_of_sample=spec.out_of_sample,
        complexity=spec.complexity,
        references=spec.references,
        estimator=estimator,
        validation_status=spec.validation_status,
    )


def list_estimators(
    *,
    family: Family | None = None,
    capability: Capability | str | None = None,
    supervision: Supervision | None = None,
    approach: str | None = None,
    validation_status: Literal["validated", "experimental"] | None = None,
) -> tuple[EstimatorInfo, ...]:
    """List explicitly allowlisted public estimators matching all filters.

    Validation status refers to the evidence recorded in each method's ledger;
    it does not guarantee accuracy outside the tested scenarios.
    """

    if validation_status not in {None, "validated", "experimental"}:
        raise ValueError("validation_status must be 'validated' or 'experimental'.")
    requested_capability = Capability(capability) if capability is not None else None
    infos: list[EstimatorInfo] = []
    for spec in _SPECS:
        if family is not None and spec.family != family:
            continue
        if supervision is not None and spec.supervision != supervision:
            continue
        if approach is not None and approach not in spec.approaches:
            continue
        if (
            validation_status is not None
            and spec.validation_status != validation_status
        ):
            continue
        if (
            requested_capability is not None
            and requested_capability not in spec.capabilities
        ):
            continue
        infos.append(_materialize(spec))
    return tuple(infos)


def get_estimator(name: str) -> EstimatorInfo:
    """Return a public estimator record by class or qualified name."""

    for spec in _SPECS:
        if name in {spec.name, f"{spec.module}.{spec.name}"}:
            return _materialize(spec)
    raise KeyError(f"Unknown public estimator: {name!r}")


def public_estimator_names() -> tuple[str, ...]:
    """Return the frozen public class-name inventory without importing methods."""

    return tuple(spec.name for spec in _SPECS)
