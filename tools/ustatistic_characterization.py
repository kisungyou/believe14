"""Freeze and run a focused, untuned UStatisticDimension validation study.

Run ``python -m tools.ustatistic_characterization freeze`` before ``run``.
The frozen design uses 50 previously unused independent data and partition seeds
per scenario. It never alters the estimator, the release gate, or its thresholds.
The JSON evidence is deliberately stored under ignored ``build/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

import believe14
from believe14.estimation import UStatisticDimension

DESIGN_VERSION = 1
REPLICATES = 50
BOOTSTRAP_DRAWS = 4096
DEFAULT_DIRECTORY = Path("build/ustatistic-characterization")


@dataclass(frozen=True)
class Scenario:
    """A fully specified data-generating distribution, independent of outcomes."""

    name: str
    geometry: str
    dimension: int
    n_samples: int
    role: str = "primary"
    ambient_dimension: int = 8
    noise: float = 0.0

    def parameters(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "target": "latent_manifold_dimension",
            "support_dimension": (
                self.ambient_dimension if self.noise else self.dimension
            ),
            "search_max_dimension": self.ambient_dimension,
        }


def scenarios() -> tuple[Scenario, ...]:
    """Return the prespecified main grid and distribution-sensitivity checks."""

    primary = tuple(
        Scenario(f"flat_d{dimension}_n{size}", "flat", dimension, size)
        for dimension in (2, 3, 5)
        for size in (120, 240, 480, 960)
    )
    return (
        *primary,
        Scenario("gaussian_d3_n480", "gaussian", 3, 480, "sensitivity"),
        Scenario("sphere_d2_n480", "sphere", 2, 480, "sensitivity"),
        Scenario("swiss_roll_d2_n480", "swiss_roll", 2, 480, "sensitivity"),
        Scenario(
            "noisy_flat_d3_n480_sigma0.01",
            "flat",
            3,
            480,
            "sensitivity",
            noise=0.01,
        ),
    )


def sample(scenario: Scenario, seed: int) -> np.ndarray:
    """Draw one sample, recording the distinction between noise and dimension."""

    rng = np.random.default_rng(seed)
    size, dimension = scenario.n_samples, scenario.dimension
    if scenario.geometry == "flat":
        latent = rng.uniform(-1.0, 1.0, size=(size, dimension))
    elif scenario.geometry == "gaussian":
        latent = rng.normal(size=(size, dimension))
    elif scenario.geometry == "sphere":
        latent = rng.normal(size=(size, dimension + 1))
        latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    elif scenario.geometry == "swiss_roll":
        angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=size)
        height = rng.uniform(-1.0, 1.0, size=size)
        latent = np.column_stack((angle * np.cos(angle), height, angle * np.sin(angle)))
        # Match the existing release-audit geometry, including its rescaling.
        latent /= np.std(latent, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown geometry: {scenario.geometry}")
    basis, _ = np.linalg.qr(
        rng.normal(size=(scenario.ambient_dimension, latent.shape[1]))
    )
    X = latent @ basis.T
    if scenario.noise:
        X += scenario.noise * rng.normal(size=X.shape)
    return np.asarray(X, dtype=np.float64)


def array_metadata(X: np.ndarray) -> dict[str, Any]:
    """Hash canonical little-endian float64 bytes, with an explicit shape."""

    canonical = np.asarray(X, dtype="<f8", order="C")
    return {
        "shape": list(canonical.shape),
        "dtype": "<f8",
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def source_fingerprints() -> dict[str, str]:
    """Bind a design to the actual loaded estimator and its numerical helpers."""

    package = Path(inspect.getfile(UStatisticDimension)).resolve().parents[1]
    files = (
        "estimation/_ustatistic.py",
        "estimation/_common.py",
        "_core/distances.py",
        "_core/validation.py",
        "_core/diagnostics.py",
        "api.py",
    )
    return {
        **{
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in files
        },
        "characterization_tool": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def make_design() -> dict[str, Any]:
    """Specify streams and analysis choices before drawing any outcomes."""

    entries = []
    for index, scenario in enumerate(scenarios()):
        base = 925_300_000 + 10_000 * index
        entries.append(
            {
                "parameters": scenario.parameters(),
                "replicates": [
                    {"data_seed": base + 2 * i, "estimator_seed": base + 2 * i + 1}
                    for i in range(REPLICATES)
                ],
                "bootstrap_seed": 935_300_000 + index,
            }
        )
    return {
        "design_version": DESIGN_VERSION,
        "purpose": "fresh validation of the unchanged published estimator",
        "candidate_tuning": "none; no candidate evaluated or selected",
        "reserved_candidate_seed_range": [925_200_000, 925_299_999],
        "source_sha256": source_fingerprints(),
        "scenarios": entries,
        "analysis": {
            "replicates_per_scenario": REPLICATES,
            "rmse_reference": 0.5,
            "reference_role": "descriptive comparison; does not change release gate",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_interval": "pointwise percentile 95%, conditional on success",
            "failure_interval": "Wilson score 95% over all attempted replicates",
            "scope": "pointwise evidence only; no simultaneous coverage guarantee",
            "noise_interpretation": (
                "Noisy samples have ambient-dimensional support; error against the "
                "latent dimension describes finite-scale robustness only."
            ),
        },
    }


def _wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4 * count**2))
    half /= denominator
    return [max(0.0, float(center - half)), min(1.0, float(center + half))]


def summarize(
    records: list[dict[str, Any]], truth: int, bootstrap_seed: int
) -> dict[str, Any]:
    """Keep failed/nonfinite attempts in the denominator and expose uncertainty."""

    if not records:
        raise ValueError("At least one attempted replicate is required.")
    estimates = np.array(
        [
            record["estimate"]
            for record in records
            if record.get("success") is True
            and isinstance(record.get("estimate"), int | float)
            and not isinstance(record["estimate"], bool)
            and np.isfinite(record["estimate"])
            and record["estimate"] > 0
        ],
        dtype=np.float64,
    )
    count, successful = len(records), estimates.size
    failures = count - successful
    correct = int(np.count_nonzero(estimates == truth))
    result: dict[str, Any] = {
        "attempted_replicates": count,
        "successful_replicates": successful,
        "failures": failures,
        "failure_rate": failures / count,
        "failure_rate_ci95": _wilson(failures, count),
        "exact_accuracy_all_attempts": correct / count,
        "exact_accuracy_ci95": _wilson(correct, count),
        "estimate_counts": dict(sorted(Counter(str(v) for v in estimates).items())),
        "error_metrics_conditioned_on_success": True,
        "bias": None,
        "rmse": None,
        "rmse_bootstrap_ci95": None,
    }
    if successful:
        errors = estimates - truth
        result["bias"] = float(np.mean(errors))
        result["rmse"] = float(np.sqrt(np.mean(errors**2)))
    if successful > 1:
        errors = estimates - truth
        draws = np.random.default_rng(bootstrap_seed).choice(
            errors, size=(BOOTSTRAP_DRAWS, successful)
        )
        interval = np.quantile(np.sqrt(np.mean(draws**2, axis=1)), [0.025, 0.975])
        result["rmse_bootstrap_ci95"] = interval.tolist()
    return result


def fit_record(scenario: Scenario, seeds: dict[str, int]) -> dict[str, Any]:
    """Run one fit without silently discarding exceptions or invalid output."""

    X = sample(scenario, seeds["data_seed"])
    record: dict[str, Any] = {
        **seeds,
        "sample": array_metadata(X),
        "estimator_parameters": {
            "max_dimension": scenario.ambient_dimension,
            "random_state": seeds["estimator_seed"],
        },
        "success": False,
    }
    started = time.perf_counter()
    try:
        estimator = UStatisticDimension(**record["estimator_parameters"]).fit(X)
        arrays = {
            name: getattr(estimator, name)
            for name in (
                "candidate_dimensions_",
                "bandwidth_factors_",
                "log_bandwidths_",
                "kernel_means_",
                "log_u_statistics_",
                "slopes_",
            )
        }
        diagnostic = asdict(estimator.diagnostics_)
        if (
            not all(np.all(np.isfinite(value)) for value in arrays.values())
            or not np.isfinite(estimator.dimension_)
            or not np.isfinite(estimator.base_bandwidth_)
            or not diagnostic["converged"]
            or not 1 <= estimator.dimension_ <= scenario.ambient_dimension
            or any(
                isinstance(value, float) and not np.isfinite(value)
                for value in diagnostic.values()
            )
        ):
            raise FloatingPointError("Nonfinite or invalid fit/diagnostic output.")
        record.update(
            success=True,
            estimate=float(estimator.dimension_),
            base_bandwidth=float(estimator.base_bandwidth_),
            candidate_dimensions=estimator.candidate_dimensions_.tolist(),
            slopes=estimator.slopes_.tolist(),
            diagnostics=diagnostic,
        )
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    record["wall_seconds"] = time.perf_counter() - started
    return record


def freeze_design(path: Path) -> None:
    """Exclusively create a design; never replace previously frozen evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"frozen_at": datetime.now(UTC).isoformat(), "design": make_design()}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_frozen_design(design_path: Path, output: Path) -> None:
    """Run sequentially with one BLAS thread and checkpoint after each scenario."""

    frozen_bytes = design_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    if frozen.get("design") != make_design():
        raise ValueError("Frozen design or numerical source differs from this run.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "frozen_design_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "frozen_design": frozen,
        "started_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "believe14": believe14.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "blas_threads": 1,
        },
        "complete": False,
        "scenarios": {},
    }
    with threadpool_limits(limits=1):
        for scenario, entry in zip(
            scenarios(), frozen["design"]["scenarios"], strict=True
        ):
            records = [fit_record(scenario, seeds) for seeds in entry["replicates"]]
            summary = summarize(records, scenario.dimension, entry["bootstrap_seed"])
            evidence["scenarios"][scenario.name] = {
                "parameters": scenario.parameters(),
                "summary": summary,
                "replicates": records,
            }
            # An interrupted run remains visibly incomplete and is never mistaken
            # for a study that completed its predetermined number of replicates.
            temporary = output.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            temporary.replace(output)
            print(
                f"{scenario.name}: RMSE={summary['rmse']}, "
                f"95% interval={summary['rmse_bootstrap_ci95']}, "
                f"failures={summary['failures']}",
                flush=True,
            )
    evidence["complete"] = True
    evidence["completed_at"] = datetime.now(UTC).isoformat()
    output.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    arguments = parser.parse_args()
    design_path = arguments.directory / "design.json"
    if arguments.action == "freeze":
        freeze_design(design_path)
        print(f"Frozen design: {design_path}")
    else:
        run_frozen_design(design_path, arguments.directory / "evidence.json")


if __name__ == "__main__":
    main()
