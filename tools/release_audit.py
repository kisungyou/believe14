"""Run deterministic installed-artifact audits and archive raw JSON evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_info

import believe14
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
from believe14.registry import public_estimator_names

JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
DIMENSION_ESTIMATOR_NAMES = {
    "CorrelationDimension",
    "TwoNN",
    "LevinaBickelMLE",
    "UStatisticDimension",
    "MiNDML",
    "DANCo",
}
CATALOG_MAX_NORMALIZED_RESIDUAL = 0.5


def _git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_metadata() -> dict[str, JSONValue]:
    status = _git_value("status", "--porcelain")
    return {
        "commit": _git_value("rev-parse", "HEAD") or "uncommitted",
        "ref": _git_value("symbolic-ref", "--short", "-q", "HEAD")
        or _git_value("describe", "--tags", "--exact-match")
        or "detached",
        "dirty": bool(status) if status is not None else None,
    }


def _artifact_metadata(paths: tuple[Path, ...]) -> dict[str, JSONValue]:
    records: dict[str, JSONValue] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Audit artifact does not exist: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return records


def _array_metadata(array: np.ndarray) -> dict[str, JSONValue]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _environment_packages() -> dict[str, JSONValue]:
    packages: dict[str, JSONValue] = {}
    for installed in importlib.metadata.distributions():
        name = installed.metadata.get("Name")
        if name:
            packages[name] = installed.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _jsonable(value: Any) -> JSONValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return str(value)


def _numeric_outputs(estimator: BaseEstimator) -> tuple[dict[str, JSONValue], bool]:
    outputs: dict[str, JSONValue] = {}
    all_finite = True
    for name, value in vars(estimator).items():
        if not name.endswith("_") or name.startswith("_"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            finite = bool(np.all(np.isfinite(value)))
            outputs[name] = {"shape": list(value.shape), "finite": finite}
        elif isinstance(value, np.number | int | float) and not isinstance(value, bool):
            finite = bool(np.isfinite(value))
            outputs[name] = {"shape": [], "finite": finite}
        else:
            continue
        all_finite = all_finite and finite
    return outputs, all_finite


def _diagnostic_record(
    estimator: BaseEstimator, *, wall_seconds: float, python_peak_bytes: int
) -> dict[str, JSONValue]:
    record = estimator.diagnostics_  # type: ignore[attr-defined]
    outputs, output_finite = _numeric_outputs(estimator)
    result: dict[str, JSONValue] = {
        "solver": record.solver,
        "converged": record.converged,
        "n_iter": record.n_iter,
        "objective": record.objective_value,
        "normalized_residual": record.residual_norm,
        "numerical_rank": record.numerical_rank,
        "condition_estimate": record.condition_estimate,
        "warnings": list(record.warnings),
        "parameters": _jsonable(estimator.get_params(deep=False)),
        "checked_numeric_outputs": outputs,
        "output_finite": output_finite and bool(outputs),
        "wall_seconds": wall_seconds,
        "python_peak_bytes": python_peak_bytes,
    }
    if hasattr(estimator, "embedding_"):
        embedding = np.asarray(estimator.embedding_, dtype=np.float64)  # type: ignore[attr-defined]
        result["output_shape"] = list(embedding.shape)
        result["embedding_shape"] = list(embedding.shape)
    if hasattr(estimator, "dimension_"):
        result["dimension"] = float(estimator.dimension_)  # type: ignore[attr-defined]
    return result


def _profile_fit(
    estimator: BaseEstimator, X: np.ndarray, y: np.ndarray | None = None
) -> tuple[BaseEstimator, float, int]:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        fitted = estimator.fit(X) if y is None else estimator.fit(X, y)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return fitted, elapsed, peak


def _catalog_audit(seed: int) -> dict[str, JSONValue]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(80, 3))
    X = latent @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(80, 6))
    Y = latent @ rng.normal(size=(3, 2)) + 0.03 * rng.normal(size=(80, 2))
    labels = np.repeat(np.arange(4), 20)
    response = X[:, 0] - 0.4 * X[:, 1]

    x_estimators: tuple[BaseEstimator, ...] = (
        PCA(2),
        GaussianRandomProjection(2, random_state=seed),
        FactorAnalysis(2, max_iter=1000),
        ProbabilisticPCA(2),
        FastICA(2, random_state=seed, max_iter=1000),
        SparsePCA(2, max_iter=500),
        ClassicalMDS(2),
        MetricMDS(2, max_iter=300),
        SammonMapping(2, max_iter=300),
        FastMap(2),
        KernelPCA(2),
        Isomap(2, n_neighbors=12),
        LocallyLinearEmbedding(2, n_neighbors=12),
        LaplacianEigenmaps(2, n_neighbors=12),
        DiffusionMap(2),
        LocalTangentSpaceAlignment(2, n_neighbors=12),
        TSNE(
            2,
            perplexity=10,
            early_exaggeration_iter=50,
            max_iter=300,
            random_state=seed,
        ),
        PHATE(2, n_neighbors=10, decay=5, diffusion_time=5),
        CorrelationDimension(),
        TwoNN(),
        LevinaBickelMLE(k_min=5, k_max=12),
        UStatisticDimension(max_dimension=4, random_state=seed),
        MiNDML(n_neighbors=6, max_dimension=4),
        DANCo(n_neighbors=6, max_dimension=4, random_state=seed),
    )
    label_estimators: tuple[BaseEstimator, ...] = (
        LinearDiscriminantAnalysis(2, regularization=1e-8),
        FisherScore(3),
    )
    response_estimators: tuple[BaseEstimator, ...] = (
        SlicedInverseRegression(2, n_slices=8),
        SlicedAverageVarianceEstimation(2, n_slices=8),
    )
    paired_estimators: tuple[BaseEstimator, ...] = (
        CanonicalCorrelationAnalysis(2),
        PLSRegression(2),
    )
    records: dict[str, JSONValue] = {}
    for estimator in x_estimators:
        fitted, elapsed, peak = _profile_fit(estimator, X)
        record = _diagnostic_record(
            fitted, wall_seconds=elapsed, python_peak_bytes=peak
        )
        record["audit_seed"] = seed
        record["inputs"] = {"X": _array_metadata(X)}
        records[type(estimator).__name__] = record
    for estimator in label_estimators:
        fitted, elapsed, peak = _profile_fit(estimator, X, labels)
        record = _diagnostic_record(
            fitted, wall_seconds=elapsed, python_peak_bytes=peak
        )
        record["audit_seed"] = seed
        record["inputs"] = {
            "X": _array_metadata(X),
            "y": _array_metadata(labels),
        }
        records[type(estimator).__name__] = record
    for estimator in response_estimators:
        fitted, elapsed, peak = _profile_fit(estimator, X, response)
        record = _diagnostic_record(
            fitted, wall_seconds=elapsed, python_peak_bytes=peak
        )
        record["audit_seed"] = seed
        record["inputs"] = {
            "X": _array_metadata(X),
            "y": _array_metadata(response),
        }
        records[type(estimator).__name__] = record
    for estimator in paired_estimators:
        fitted, elapsed, peak = _profile_fit(estimator, X, Y)
        record = _diagnostic_record(
            fitted, wall_seconds=elapsed, python_peak_bytes=peak
        )
        record["audit_seed"] = seed
        record["inputs"] = {
            "X": _array_metadata(X),
            "Y": _array_metadata(Y),
        }
        records[type(estimator).__name__] = record
    expected = set(public_estimator_names())
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise RuntimeError(f"Catalog audit mismatch; missing={missing}, extra={extra}")
    return records


def _flat_sample(
    rng: np.random.Generator,
    *,
    n_samples: int,
    dimension: int,
    ambient_dimension: int = 6,
    noise: float = 0.0,
) -> np.ndarray:
    latent = rng.uniform(-1.0, 1.0, size=(n_samples, dimension))
    basis, _ = np.linalg.qr(rng.normal(size=(ambient_dimension, dimension)))
    X = latent @ basis.T
    if noise:
        X += noise * rng.normal(size=X.shape)
    return np.asarray(X, dtype=np.float64)


def _sphere_sample(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    X = rng.normal(size=(n_samples, 3))
    return np.asarray(X / np.linalg.norm(X, axis=1, keepdims=True), dtype=np.float64)


def _swiss_roll_sample(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n_samples)
    height = rng.uniform(-1.0, 1.0, size=n_samples)
    X = np.column_stack((angle * np.cos(angle), height, angle * np.sin(angle)))
    return np.asarray(X / np.std(X, axis=0, keepdims=True), dtype=np.float64)


@dataclass(frozen=True)
class DimensionScenario:
    """Fully specified synthetic dimension-estimation scenario."""

    name: str
    geometry: str
    truth: float
    n_samples: int
    dimension: int
    ambient_dimension: int
    noise: float = 0.0

    def parameters(self) -> dict[str, JSONValue]:
        return {
            "geometry": self.geometry,
            "truth": self.truth,
            "n_samples": self.n_samples,
            "dimension": self.dimension,
            "ambient_dimension": self.ambient_dimension,
            "noise": self.noise,
        }

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if self.geometry == "flat":
            return _flat_sample(
                rng,
                n_samples=self.n_samples,
                dimension=self.dimension,
                ambient_dimension=self.ambient_dimension,
                noise=self.noise,
            )
        if self.geometry == "sphere":
            return _sphere_sample(rng, self.n_samples)
        if self.geometry == "swiss_roll":
            return _swiss_roll_sample(rng, self.n_samples)
        raise RuntimeError(f"Unknown audit geometry: {self.geometry}")


def _dimension_scenarios() -> tuple[DimensionScenario, ...]:
    return (
        DimensionScenario("flat_d1_n240", "flat", 1.0, 240, 1, 6),
        DimensionScenario("flat_d1_n360", "flat", 1.0, 360, 1, 6),
        DimensionScenario("flat_d2_n240", "flat", 2.0, 240, 2, 6),
        DimensionScenario("flat_d2_n360", "flat", 2.0, 360, 2, 6),
        DimensionScenario("flat_d3_n240", "flat", 3.0, 240, 3, 6),
        DimensionScenario("flat_d3_n360", "flat", 3.0, 360, 3, 6),
        DimensionScenario(
            "flat_d2_n300_noise_1e-3", "flat", 2.0, 300, 2, 6, noise=1e-3
        ),
        DimensionScenario("sphere_d2_n300", "sphere", 2.0, 300, 2, 3),
        DimensionScenario("swiss_roll_d2_n300", "swiss_roll", 2.0, 300, 2, 3),
    )


def _dimension_factories(
    seed: int, ambient: int
) -> dict[str, Callable[[], BaseEstimator]]:
    upper = min(5, ambient)
    return {
        "CorrelationDimension": CorrelationDimension,
        "TwoNN": TwoNN,
        "LevinaBickelMLE": lambda: LevinaBickelMLE(k_min=5, k_max=12),
        "UStatisticDimension": lambda: UStatisticDimension(
            max_dimension=upper, random_state=seed
        ),
        "MiNDML": lambda: MiNDML(n_neighbors=6, max_dimension=upper),
        "DANCo": lambda: DANCo(n_neighbors=6, max_dimension=upper, random_state=seed),
    }


def _dimension_audit(seeds: tuple[int, ...]) -> dict[str, JSONValue]:
    raw: dict[str, JSONValue] = {}
    for scenario in _dimension_scenarios():
        scenario_values: dict[str, list[JSONValue]] = {}
        samples: list[JSONValue] = []
        for seed in seeds:
            X = scenario.sample(np.random.default_rng(seed))
            sample = _array_metadata(X)
            sample["seed"] = seed
            samples.append(sample)
            for name, factory in _dimension_factories(seed, X.shape[1]).items():
                # DANCo's angular model is singular for one-dimensional flats.
                if name == "DANCo" and scenario.truth < 2.0:
                    continue
                values = scenario_values.setdefault(name, [])
                estimator = factory()
                parameters = _jsonable(estimator.get_params(deep=False))
                try:
                    fitted, elapsed, peak = _profile_fit(estimator, X)
                    values.append(
                        {
                            "seed": seed,
                            "estimate": float(fitted.dimension_),  # type: ignore[attr-defined]
                            "parameters": parameters,
                            "diagnostics": _jsonable(  # type: ignore[attr-defined]
                                asdict(fitted.diagnostics_)
                            ),
                            "wall_seconds": elapsed,
                            "python_peak_bytes": peak,
                        }
                    )
                except (FloatingPointError, ValueError) as error:
                    values.append(
                        {
                            "seed": seed,
                            "parameters": parameters,
                            "failure": type(error).__name__,
                            "message": str(error),
                        }
                    )
        summaries: dict[str, JSONValue] = {}
        for name, values in scenario_values.items():
            successful = [
                value
                for value in values
                if isinstance(value, dict)
                and isinstance(value.get("estimate"), int | float)
            ]
            estimates = np.asarray(
                [value["estimate"] for value in successful], dtype=np.float64
            )
            failures = len(values) - len(estimates)
            if estimates.size:
                errors = estimates - scenario.truth
                summary: dict[str, JSONValue] = {
                    "truth": scenario.truth,
                    "raw": values,
                    "mean": float(np.mean(estimates)),
                    "bias": float(np.mean(errors)),
                    "rmse": float(np.sqrt(np.mean(errors * errors))),
                    "standard_deviation": float(np.std(estimates)),
                    "failure_rate": failures / len(values),
                }
            else:
                summary = {
                    "truth": scenario.truth,
                    "raw": values,
                    "failure_rate": 1.0,
                }
            summaries[name] = summary
        raw[scenario.name] = {
            "truth": scenario.truth,
            "parameters": scenario.parameters(),
            "seeds": list(seeds),
            "samples": samples,
            "methods": summaries,
            "not_applicable": (
                {
                    "DANCo": (
                        "The 0.1.0 angular-concentration contract excludes "
                        "one-dimensional manifolds."
                    )
                }
                if scenario.truth < 2.0
                else {}
            ),
        }
    return raw


def _release_decision(
    catalog: dict[str, JSONValue], dimension_scenarios: dict[str, JSONValue]
) -> dict[str, JSONValue]:
    catalog_failures: list[JSONValue] = []
    expected_catalog = set(public_estimator_names())
    for name in sorted(expected_catalog - set(catalog)):
        catalog_failures.append(f"missing:{name}")
    for name in sorted(set(catalog) - expected_catalog):
        catalog_failures.append(f"unexpected:{name}")
    for name, raw_record in catalog.items():
        record = raw_record if isinstance(raw_record, dict) else {}
        reasons: list[str] = []
        if record.get("converged") is not True:
            reasons.append("not-converged")
        if record.get("output_finite") is not True:
            reasons.append("non-finite-or-unchecked-output")
        checked_outputs = record.get("checked_numeric_outputs")
        if not isinstance(checked_outputs, dict) or not checked_outputs:
            reasons.append("no-checked-numeric-output")
        for field in ("objective", "normalized_residual", "condition_estimate"):
            value = record.get(field)
            if value is not None and (
                not isinstance(value, int | float) or not np.isfinite(value)
            ):
                reasons.append(f"non-finite-{field}")
        residual = record.get("normalized_residual")
        if (
            isinstance(residual, int | float)
            and residual > CATALOG_MAX_NORMALIZED_RESIDUAL
        ):
            reasons.append("residual-above-threshold")
        numerical_rank = record.get("numerical_rank")
        if numerical_rank is not None and (
            not isinstance(numerical_rank, int) or numerical_rank <= 0
        ):
            reasons.append("invalid-numerical-rank")
        catalog_failures.extend(f"{name}:{reason}" for reason in reasons)

    dimension_failures: list[JSONValue] = []
    expected_scenarios = {
        scenario.name: scenario for scenario in _dimension_scenarios()
    }
    for name in sorted(set(expected_scenarios) - set(dimension_scenarios)):
        dimension_failures.append(f"missing-scenario:{name}")
    for name in sorted(set(dimension_scenarios) - set(expected_scenarios)):
        dimension_failures.append(f"unexpected-scenario:{name}")
    for scenario, raw_scenario in dimension_scenarios.items():
        scenario_record = raw_scenario if isinstance(raw_scenario, dict) else {}
        raw_methods = scenario_record.get("methods", {})
        methods = raw_methods if isinstance(raw_methods, dict) else {}
        not_applicable = scenario_record.get("not_applicable", {})
        excluded = set(not_applicable) if isinstance(not_applicable, dict) else set()
        expected_methods = DIMENSION_ESTIMATOR_NAMES - excluded
        for name in sorted(expected_methods - set(methods)):
            dimension_failures.append(f"{scenario}:missing-method:{name}")
        for name in sorted(set(methods) - expected_methods):
            dimension_failures.append(f"{scenario}:unexpected-method:{name}")
        raw_seeds = scenario_record.get("seeds", [])
        seeds = raw_seeds if isinstance(raw_seeds, list) else []
        for name, raw_summary in methods.items():
            summary = raw_summary if isinstance(raw_summary, dict) else {}
            max_rmse = 0.5 if name in {"UStatisticDimension", "DANCo"} else 0.75
            rmse = summary.get("rmse")
            bias = summary.get("bias")
            standard_deviation = summary.get("standard_deviation")
            failure_rate = summary.get("failure_rate")
            raw_values = summary.get("raw", [])
            values = raw_values if isinstance(raw_values, list) else []
            observed_seeds = {
                value.get("seed") for value in values if isinstance(value, dict)
            }
            complete = len(values) == len(seeds) and observed_seeds == set(seeds)
            passed = (
                isinstance(rmse, int | float)
                and np.isfinite(rmse)
                and rmse <= max_rmse
                and isinstance(bias, int | float)
                and np.isfinite(bias)
                and abs(bias) <= max_rmse
                and isinstance(standard_deviation, int | float)
                and np.isfinite(standard_deviation)
                and standard_deviation <= 0.75
                and failure_rate == 0.0
                and complete
            )
            summary["release_thresholds"] = {
                "max_rmse": max_rmse,
                "max_absolute_bias": max_rmse,
                "max_standard_deviation": 0.75,
                "max_failure_rate": 0.0,
                "required_replicates": len(seeds),
            }
            summary["passed"] = passed
            if not passed:
                dimension_failures.append(f"{scenario}:{name}")
    passed = not catalog_failures and not dimension_failures
    return {
        "passed": passed,
        "catalog_max_normalized_residual": CATALOG_MAX_NORMALIZED_RESIDUAL,
        "catalog_failures": catalog_failures,
        "dimension_failures": dimension_failures,
    }


def run_audit(*, artifact_paths: tuple[Path, ...] = ()) -> dict[str, JSONValue]:
    """Return all raw evidence and reproducibility metadata."""

    seeds = (1701, 1702, 1703, 1704, 1705)
    catalog = _catalog_audit(seeds[0])
    dimension_scenarios = _dimension_audit(seeds)
    decision = _release_decision(catalog, dimension_scenarios)
    git = _git_metadata()
    return {
        "schema_version": 2,
        "release": believe14.__version__,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "commit": git["commit"],
        "git": git,
        "platform": {
            "description": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": sys.version,
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "environment_packages": _environment_packages(),
        "numpy_build": _jsonable(np.show_config(mode="dicts")),
        "threadpools": _jsonable(threadpool_info()),
        "artifacts": _artifact_metadata(artifact_paths),
        "random_generator": "numpy.random.default_rng (PCG64)",
        "seeds": list(seeds),
        "catalog": catalog,
        "dimension_scenarios": dimension_scenarios,
        "release_gate": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[], type=Path)
    arguments = parser.parse_args()
    results = run_audit(artifact_paths=tuple(arguments.artifact))
    arguments.output.write_text(
        json.dumps(results, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision = results["release_gate"]
    if not isinstance(decision, dict) or not decision.get("passed", False):
        raise SystemExit("believe14 release audit failed; inspect the JSON artifact.")


if __name__ == "__main__":
    main()
