"""Prospective simultaneous accuracy certification of the reference estimator.

Freeze the fixed design before running any fits. This tool does not change the
release gate or the estimator. Certification concerns only the nine original
benchmark distributions and their exact estimator configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import beta
from threadpoolctl import threadpool_limits

import believe14
from believe14.estimation import UStatisticDimension

if __package__:
    from tools import release_audit
else:
    # Installed-artifact audits also invoke tools by absolute path with python -I.
    # Load only the adjacent trusted helper, without changing the import path.
    _spec = importlib.util.spec_from_file_location(
        "_believe14_certification_release_audit",
        Path(__file__).with_name("release_audit.py"),
    )
    if _spec is None or _spec.loader is None:
        raise ImportError("Cannot load the adjacent release-audit generators.")
    release_audit = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = release_audit
    _spec.loader.exec_module(release_audit)

DESIGN_VERSION = 1
REPLICATES = 500
FAMILY_ALPHA = 0.05
TOTAL_TAILS = 25
TAIL_ALPHA = FAMILY_ALPHA / TOTAL_TAILS
MAX_RMSE = 0.5
SEED_BASE = 926_400_000
DEFAULT_DIRECTORY = Path("build/ustatistic-certification")
ARRAY_NAMES = (
    "candidate_dimensions_",
    "bandwidth_factors_",
    "log_bandwidths_",
    "kernel_means_",
    "log_u_statistics_",
    "slopes_",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _same_json(left: object, right: object) -> bool:
    try:
        return _json_bytes(left) == _json_bytes(right)
    except (TypeError, ValueError, OverflowError):
        return False


def scenarios() -> tuple[release_audit.DimensionScenario, ...]:
    """Reuse the original distributions, sample sizes and search bounds exactly."""

    return release_audit._dimension_scenarios()


def _factory(scenario: release_audit.DimensionScenario, seed: int) -> Any:
    return release_audit._dimension_factories(
        seed,
        scenario.ambient_dimension,
        max_dimension=max(5, scenario.dimension + 3),
    )["UStatisticDimension"]()


def _maximum_error(scenario: release_audit.DimensionScenario) -> int:
    upper = int(_factory(scenario, 0).get_params()["max_dimension"])
    return max(scenario.dimension - 1, upper - scenario.dimension)


def array_metadata(X: np.ndarray) -> dict[str, Any]:
    canonical = np.asarray(X, dtype="<f8", order="C")
    return {
        "shape": list(canonical.shape),
        "dtype": "<f8",
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def source_fingerprints() -> dict[str, str]:
    """Fingerprint numerical code and generators without binding release policy."""

    package = Path(inspect.getfile(UStatisticDimension)).resolve().parents[1]
    files = (
        "estimation/_ustatistic.py",
        "estimation/_common.py",
        "_core/distances.py",
        "_core/validation.py",
        "_core/diagnostics.py",
        "api.py",
    )
    result = {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in files
    }
    for name in (
        "DimensionScenario",
        "_dimension_scenarios",
        "_dimension_factories",
        "_flat_sample",
        "_sphere_sample",
        "_swiss_roll_sample",
    ):
        source = inspect.getsource(getattr(release_audit, name)).encode()
        result[f"release_audit.{name}"] = hashlib.sha256(source).hexdigest()
    result["certification_tool"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return result


def make_design() -> dict[str, Any]:
    """Return trusted constants; never derive the protocol from supplied evidence."""

    entries = []
    for index, scenario in enumerate(scenarios()):
        base = SEED_BASE + 10_000 * index
        entries.append(
            {
                "name": scenario.name,
                "parameters": scenario.parameters(),
                "maximum_absolute_error": _maximum_error(scenario),
                "replicates": [
                    {
                        "data_seed": base + 2 * i,
                        "estimator_seed": base + 2 * i + 1,
                        "estimator_parameters": _factory(
                            scenario, base + 2 * i + 1
                        ).get_params(deep=False),
                    }
                    for i in range(REPLICATES)
                ],
            }
        )
    if len(entries) != 9 or sum(e["maximum_absolute_error"] for e in entries) != 25:
        raise ValueError("The original nine-scenario certification scope has changed.")
    return {
        "design_version": DESIGN_VERSION,
        "purpose": "prospective certification of the unchanged reference estimator",
        "source_sha256": source_fingerprints(),
        "scenarios": entries,
        "analysis": {
            "replicates_per_scenario": REPLICATES,
            "maximum_rmse": MAX_RMSE,
            "family_alpha": FAMILY_ALPHA,
            "total_binomial_tails": TOTAL_TAILS,
            "alpha_per_tail": TAIL_ALPHA,
            "bound": "one-sided Clopper-Pearson tails, weighted by 2*j-1",
            "decision": "all nine simultaneous RMSE upper bounds <= 0.5",
            "failure_policy": (
                "any failed, invalid, or missing fit prevents certification"
            ),
            "assumptions": [
                "independent identically distributed trials within each fixed scenario",
                "independent data and estimator pseudorandom streams",
                "fixed estimator, fixed sample count, no selection or repeated testing",
                "integer estimates in each prespecified candidate range",
            ],
            "scope": "only the nine original distributions and exact configurations",
            "noise_interpretation": (
                "The noisy scenario targets latent dimension as finite-scale "
                "robustness; "
                "its support dimension is the ambient dimension."
            ),
            "historical_evidence": "previous failed panels remain failed and retained",
        },
    }


def binomial_upper(count: int, trials: int, alpha: float = TAIL_ALPHA) -> float:
    """Exact one-sided upper bound, including nonzero uncertainty at zero count."""

    if (
        type(count) is not int
        or type(trials) is not int
        or trials < 1
        or not 0 <= count <= trials
        or isinstance(alpha, bool)
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("Invalid binomial count, sample size or alpha.")
    if count == trials:
        return 1.0
    if count == 0:
        return float(-np.expm1(np.log(alpha) / trials))
    return float(beta.isf(alpha, count + 1, trials - count))


def _finite_real(value: object) -> bool:
    try:
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and bool(np.isfinite(float(value)))
        )
    except (ValueError, OverflowError):
        return False


def _valid_estimate(value: object, upper: int) -> bool:
    return _finite_real(value) and 1 <= value <= upper and float(value).is_integer()


def _numeric_tree(value: object) -> bool:
    if isinstance(value, list):
        return all(_numeric_tree(item) for item in value)
    return _finite_real(value)


def summarize(records: list[dict[str, Any]], scenario_name: str) -> dict[str, Any]:
    """Recompute conservative tail bounds; failures never disappear from counts."""

    scenario = next((s for s in scenarios() if s.name == scenario_name), None)
    if scenario is None:
        raise ValueError("Unknown trusted certification scenario.")
    upper = int(_factory(scenario, 0).get_params()["max_dimension"])
    maximum_error = _maximum_error(scenario)
    estimates = [
        float(record["estimate"])
        for record in records
        if record.get("success") is True
        and "failure" not in record
        and _valid_estimate(record.get("estimate"), upper)
    ]
    failures = len(records) - len(estimates)
    errors = np.abs(np.asarray(estimates) - scenario.truth).astype(int).tolist()
    # A failed fit cannot be certified. Worst-case imputation additionally keeps
    # every attempt in each binomial denominator instead of conditioning on success.
    errors.extend([maximum_error] * failures)
    tails = []
    for threshold in range(1, maximum_error + 1):
        count = sum(error >= threshold for error in errors)
        tails.append(
            {
                "absolute_error_at_least": threshold,
                "count": count,
                "trials": len(records),
                "alpha": TAIL_ALPHA,
                "probability_upper": (
                    binomial_upper(count, len(records)) if records else 1.0
                ),
            }
        )
    mse_upper = float(
        sum(
            (2 * t["absolute_error_at_least"] - 1) * t["probability_upper"]
            for t in tails
        )
    )
    complete = len(records) == REPLICATES
    return {
        "attempted_replicates": len(records),
        "successful_replicates": len(estimates),
        "failures": failures,
        "failure_rate": failures / len(records) if records else None,
        "estimate_counts": dict(sorted(Counter(str(v) for v in estimates).items())),
        "observed_rmse": (
            float(np.sqrt(np.mean((np.asarray(estimates) - scenario.truth) ** 2)))
            if estimates and not failures
            else None
        ),
        "tails": tails,
        "mse_upper": mse_upper,
        "rmse_upper": float(np.sqrt(mse_upper)),
        "complete": complete,
        "passed": complete and failures == 0 and mse_upper <= MAX_RMSE**2,
    }


def _valid_fitted_output(record: dict[str, Any], upper: int) -> bool:
    """Check saved numerical state, not a caller's assertion of success."""

    if (
        "failure" in record
        or record.get("success") is not True
        or not _valid_estimate(record.get("estimate"), upper)
    ):
        return False
    try:
        if not isinstance(record.get("arrays"), dict) or set(record["arrays"]) != set(
            ARRAY_NAMES
        ):
            return False
        if not all(_numeric_tree(value) for value in record["arrays"].values()):
            return False
        arrays = {
            name: np.asarray(record["arrays"][name], dtype=np.float64)
            for name in ARRAY_NAMES
        }
        for name, value in arrays.items():
            shape = (
                (upper,) if name in {"candidate_dimensions_", "slopes_"} else (upper, 5)
            )
            if value.shape != shape or not np.all(np.isfinite(value)):
                return False
        if not np.array_equal(arrays["candidate_dimensions_"], np.arange(1, upper + 1)):
            return False
        if (
            np.any(arrays["kernel_means_"] <= 0)
            or np.any(arrays["kernel_means_"] > 1)
            or np.any(arrays["bandwidth_factors_"] <= 0)
            or record["estimate"] != np.argmin(np.abs(arrays["slopes_"])) + 1
            or not _finite_real(record["base_bandwidth"])
            or record["base_bandwidth"] <= 0
        ):
            return False
        diagnostic = record["diagnostics"]
        fields = {
            "solver",
            "converged",
            "n_iter",
            "residual_norm",
            "objective_value",
            "numerical_rank",
            "condition_estimate",
            "warnings",
        }
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != fields
            or diagnostic["solver"] != "hein_audibert_weighted_slope_search"
            or diagnostic["converged"] is not True
            or type(diagnostic["n_iter"]) is not int
            or diagnostic["n_iter"] != upper
            or diagnostic["numerical_rank"] is not None
            or diagnostic["condition_estimate"] is not None
            or not isinstance(diagnostic["warnings"], list | tuple)
            or not all(isinstance(item, str) for item in diagnostic["warnings"])
            or any(
                not _finite_real(diagnostic[name]) or diagnostic[name] < 0
                for name in ("residual_norm", "objective_value")
            )
        ):
            return False
        with np.errstate(all="raise"):
            count = record["sample"]["shape"][0]
            sizes = count // np.arange(1, 6)
            dimensions = np.arange(1, upper + 1)[:, None]
            factors = (count / sizes * np.log(sizes) / np.log(count)) ** (
                1 / dimensions
            )
            x = np.log(record["base_bandwidth"]) + np.log(factors)
            y = np.log(arrays["kernel_means_"]) - dimensions * x
            weights = 1 / np.arange(1, 6)
            x_mean = np.average(x, weights=weights, axis=1, keepdims=True)
            y_mean = np.average(y, weights=weights, axis=1, keepdims=True)
            centered = x - x_mean
            slopes = np.sum(weights * centered * (y - y_mean), axis=1)
            slopes /= np.sum(weights * centered**2, axis=1)
            residuals = y - (y_mean + slopes[:, None] * centered)
            winner = int(np.argmin(np.abs(arrays["slopes_"])))
            objective = np.sum(weights * residuals[winner] ** 2)
        return all(
            np.allclose(actual, expected, rtol=1e-11, atol=1e-12)
            for actual, expected in (
                (arrays["bandwidth_factors_"], factors),
                (arrays["log_bandwidths_"], x),
                (arrays["log_u_statistics_"], y),
                (arrays["slopes_"], slopes),
                (diagnostic["residual_norm"], abs(arrays["slopes_"][winner])),
                (diagnostic["objective_value"], objective),
            )
        )
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError):
        return False


def fit_record(
    scenario: release_audit.DimensionScenario, replicate: dict[str, Any]
) -> dict[str, Any]:
    """Record a single fit, retaining exceptions and every finite-state diagnostic."""

    X = scenario.sample(np.random.default_rng(replicate["data_seed"]))
    record = {**replicate, "sample": array_metadata(X), "success": False}
    started = time.perf_counter()
    try:
        estimator = _factory(scenario, replicate["estimator_seed"]).fit(X)
        candidate = {
            **record,
            "success": True,
            "estimate": float(estimator.dimension_),
            "base_bandwidth": float(estimator.base_bandwidth_),
            "arrays": {name: getattr(estimator, name).tolist() for name in ARRAY_NAMES},
            "diagnostics": asdict(estimator.diagnostics_),
        }
        upper = int(estimator.get_params()["max_dimension"])
        if not _valid_fitted_output(candidate, upper):
            raise FloatingPointError("Invalid, noninteger or nonfinite fitted state.")
        record = candidate
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    record["wall_seconds"] = time.perf_counter() - started
    return record


def certification_decision(evidence: object) -> dict[str, Any]:
    """Fail closed, regenerate every dataset hash, and recompute all decisions.

    Stored summaries and caller-supplied analysis choices are never authoritative.
    Source fingerprints bind this evidence to the currently loaded numerical code.
    Consistency checks do not prove that a fit was executed: use evidence from a
    trusted producer, or independently replay fits before trusting external JSON.
    """

    failures: list[str] = []
    summaries: dict[str, Any] = {}
    try:
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be an object")
        frozen = evidence["frozen_design"]
        design = make_design()
        if not isinstance(frozen, dict) or not _same_json(frozen.get("design"), design):
            raise ValueError("frozen design or numerical source differs")
        timestamps = [
            datetime.fromisoformat(frozen["frozen_at"]),
            datetime.fromisoformat(evidence["started_at"]),
        ]
        if evidence.get("complete") is True:
            timestamps.append(datetime.fromisoformat(evidence["completed_at"]))
        if any(
            stamp.utcoffset() is None for stamp in timestamps
        ) or timestamps != sorted(timestamps):
            raise ValueError("missing or inconsistent study timestamps")
        if (
            evidence.get("frozen_design_sha256")
            != hashlib.sha256(_json_bytes(frozen)).hexdigest()
        ):
            raise ValueError("frozen design fingerprint differs")
        if evidence.get("complete") is not True:
            failures.append("incomplete-study")
        supplied = evidence.get("scenarios")
        if not isinstance(supplied, dict) or set(supplied) != {
            s.name for s in scenarios()
        }:
            raise ValueError("missing or unexpected scenarios")
        for scenario, entry in zip(scenarios(), design["scenarios"], strict=True):
            block = supplied[scenario.name]
            if not isinstance(block, dict) or not _same_json(
                block.get("parameters"), scenario.parameters()
            ):
                raise ValueError(f"{scenario.name}: inconsistent parameters")
            records = block.get("replicates")
            if not isinstance(records, list) or len(records) != REPLICATES:
                raise ValueError(f"{scenario.name}: missing or extra replicates")
            checked = []
            for index, (record, expected) in enumerate(
                zip(records, entry["replicates"], strict=True)
            ):
                if not isinstance(record, dict) or any(
                    not _same_json(record.get(key), value)
                    for key, value in expected.items()
                ):
                    raise ValueError(
                        f"{scenario.name}:{index}: inconsistent seeds/parameters"
                    )
                X = scenario.sample(np.random.default_rng(expected["data_seed"]))
                if not _same_json(record.get("sample"), array_metadata(X)):
                    raise ValueError(
                        f"{scenario.name}:{index}: sample fingerprint differs"
                    )
                upper = expected["estimator_parameters"]["max_dimension"]
                valid = _valid_fitted_output(record, upper)
                checked.append({**record, "success": valid})
            summary = summarize(checked, scenario.name)
            summaries[scenario.name] = summary
            if not summary["passed"]:
                failures.append(f"{scenario.name}:uncertified")
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        failures.append(f"invalid-evidence:{exc}")
    return {
        "passed": not failures,
        "failures": failures,
        "family_confidence": 1 - FAMILY_ALPHA,
        "maximum_rmse": MAX_RMSE,
        "scenarios": summaries,
    }


def freeze_design(path: Path) -> None:
    """Exclusively create a plan before outcomes exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = {"frozen_at": datetime.now(UTC).isoformat(), "design": make_design()}
    with path.open("xb") as stream:
        stream.write(_json_bytes(frozen))


def _checkpoint(output: Path, evidence: dict[str, Any]) -> None:
    temporary = output.with_suffix(".tmp")
    temporary.write_bytes(_json_bytes(evidence))
    temporary.replace(output)


def run_frozen_design(design_path: Path, output: Path) -> dict[str, Any]:
    """Run once, sequentially, checkpointing partial evidence as incomplete."""

    frozen = json.loads(design_path.read_bytes())
    if not isinstance(frozen, dict) or not _same_json(
        frozen.get("design"), make_design()
    ):
        raise ValueError("Frozen design or numerical source differs from this run.")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "frozen_design_sha256": hashlib.sha256(_json_bytes(frozen)).hexdigest(),
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
    with output.open("xb") as stream:
        stream.write(_json_bytes(evidence))
    with threadpool_limits(limits=1):
        for scenario, entry in zip(
            scenarios(), frozen["design"]["scenarios"], strict=True
        ):
            block: dict[str, Any] = {
                "parameters": scenario.parameters(),
                "replicates": [],
            }
            evidence["scenarios"][scenario.name] = block
            for index, replicate in enumerate(entry["replicates"]):
                block["replicates"].append(fit_record(scenario, replicate))
                if (index + 1) % 50 == 0:
                    _checkpoint(output, evidence)
            block["summary"] = summarize(block["replicates"], scenario.name)
            _checkpoint(output, evidence)
            print(
                f"{scenario.name}: observed RMSE={block['summary']['observed_rmse']}, "
                f"simultaneous upper={block['summary']['rmse_upper']}, "
                f"failures={block['summary']['failures']}",
                flush=True,
            )
        evidence["complete"] = True
        evidence["completed_at"] = datetime.now(UTC).isoformat()
        evidence["decision"] = certification_decision(evidence)
        _checkpoint(output, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "verify"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    arguments = parser.parse_args()
    design_path = arguments.directory / "design.json"
    output = arguments.directory / "evidence.json"
    if arguments.action == "freeze":
        freeze_design(design_path)
        print(f"Frozen design: {design_path}")
    elif arguments.action == "run":
        evidence = run_frozen_design(design_path, output)
        print(json.dumps(evidence["decision"], indent=2))
        if not evidence["decision"]["passed"]:
            raise SystemExit(1)
    else:
        decision = certification_decision(json.loads(output.read_bytes()))
        print(json.dumps(decision, indent=2))
        if not decision["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
