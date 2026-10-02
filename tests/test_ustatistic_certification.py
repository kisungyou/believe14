from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tools import ustatistic_certification as study


def test_design_keeps_original_nine_configurations_and_fresh_disjoint_streams() -> None:
    design = study.make_design()
    assert [e["maximum_absolute_error"] for e in design["scenarios"]] == [
        4,
        4,
        3,
        3,
        3,
        3,
        3,
        1,
        1,
    ]
    seeds = []
    for scenario, entry in zip(study.scenarios(), design["scenarios"], strict=True):
        assert entry["parameters"] == scenario.parameters()
        assert len(entry["replicates"]) == 500
        for replicate in entry["replicates"]:
            seeds.extend((replicate["data_seed"], replicate["estimator_seed"]))
            assert replicate["estimator_parameters"] == study._factory(
                scenario, replicate["estimator_seed"]
            ).get_params(deep=False)
    assert len(seeds) == len(set(seeds)) == 9000
    assert min(seeds) == 926_400_000
    assert max(seeds) < 926_500_000
    assert design["analysis"]["total_binomial_tails"] == 25
    assert 25 * design["analysis"]["alpha_per_tail"] == pytest.approx(0.05)
    assert "certification_tool" in design["source_sha256"]
    assert "release_audit._flat_sample" in design["source_sha256"]


@pytest.mark.parametrize("trials,count", [(7, 1), (12, 4), (30, 29), (500, 3)])
def test_binomial_upper_inverts_independent_binomial_cdf(
    trials: int, count: int
) -> None:
    upper = study.binomial_upper(count, trials)
    cdf = math.fsum(
        math.comb(trials, k) * upper**k * (1 - upper) ** (trials - k)
        for k in range(count + 1)
    )
    assert cdf == pytest.approx(study.TAIL_ALPHA, rel=2e-12, abs=1e-15)


def test_binomial_endpoints_and_zero_count_uncertainty() -> None:
    upper = study.binomial_upper(0, 500)
    assert (1 - upper) ** 500 == pytest.approx(0.002)
    assert upper > 0
    assert study.binomial_upper(500, 500) == 1
    assert 4 * math.sqrt(upper) == pytest.approx(0.44456347163257665)


@pytest.mark.parametrize(
    "count,trials,alpha",
    [
        (-1, 5, 0.05),
        (6, 5, 0.05),
        (0, 0, 0.05),
        (True, 5, 0.05),
        (0, 5, 0),
        (0, 5, 1),
        (0, 5, float("nan")),
        (0, 5, True),
    ],
)
def test_invalid_binomial_inputs_are_rejected(
    count: int, trials: int, alpha: float
) -> None:
    with pytest.raises(ValueError):
        study.binomial_upper(count, trials, alpha)


def test_tail_identity_recovers_squared_error_and_uses_every_attempt() -> None:
    estimates = [1, 2, 3, 4, 5, 1, 3]
    records = [{"success": True, "estimate": value} for value in estimates]
    summary = study.summarize(records, "flat_d1_n240")
    tail_mse = sum(
        (2 * tail["absolute_error_at_least"] - 1) * tail["count"] / len(records)
        for tail in summary["tails"]
    )
    assert tail_mse == pytest.approx(np.mean((np.array(estimates) - 1) ** 2))
    assert summary["observed_rmse"] == pytest.approx(math.sqrt(tail_mse))
    assert summary["complete"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize("invalid", [True, 0, -1, 7, 2.5, float("nan"), float("inf")])
def test_noninteger_and_out_of_range_estimates_are_failures(invalid: object) -> None:
    records = [{"success": True, "estimate": 3.0}] * 499 + [
        {"success": True, "estimate": invalid}
    ]
    result = study.summarize(records, "flat_d3_n240")
    assert result["attempted_replicates"] == 500
    assert result["successful_replicates"] == 499
    assert result["failures"] == 1
    assert result["failure_rate"] == 1 / 500
    assert result["observed_rmse"] is None
    assert all(t["trials"] == 500 and t["count"] == 1 for t in result["tails"])
    assert result["passed"] is False


def test_no_missing_or_failed_replicate_can_certify() -> None:
    correct = {"success": True, "estimate": 3}
    assert study.summarize([correct] * 500, "flat_d3_n240")["passed"] is True
    assert study.summarize([correct] * 499, "flat_d3_n240")["passed"] is False
    assert study.summarize([correct] * 501, "flat_d3_n240")["passed"] is False
    failed = study.summarize([{"success": False}] * 500, "flat_d3_n240")
    assert failed["failures"] == 500
    assert failed["rmse_upper"] == 3
    assert failed["observed_rmse"] is None
    empty = study.summarize([], "flat_d3_n240")
    assert empty["passed"] is False
    assert empty["rmse_upper"] == 3
    contradictory = study.summarize(
        [{**correct, "failure": "failed"}] * 500, "flat_d3_n240"
    )
    assert contradictory["failures"] == 500
    assert contradictory["passed"] is False
    with pytest.raises(ValueError, match="Unknown trusted"):
        study.summarize([correct] * 500, "invented")


@pytest.mark.parametrize(
    "mutation", ["alpha", "seed", "threshold", "source", "scenario"]
)
def test_frozen_design_refuses_tampering_before_any_fit(
    tmp_path: Path, mutation: str
) -> None:
    path, output = tmp_path / "design.json", tmp_path / "evidence.json"
    study.freeze_design(path)
    with pytest.raises(FileExistsError):
        study.freeze_design(path)
    frozen = json.loads(path.read_bytes())
    if mutation == "alpha":
        frozen["design"]["analysis"]["alpha_per_tail"] = 0.5
    elif mutation == "seed":
        frozen["design"]["scenarios"][0]["replicates"][0]["data_seed"] += 1
    elif mutation == "threshold":
        frozen["design"]["analysis"]["maximum_rmse"] = 100
    elif mutation == "source":
        frozen["design"]["source_sha256"]["estimation/_ustatistic.py"] = "forged"
    else:
        frozen["design"]["scenarios"].pop()
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="Frozen design"):
        study.run_frozen_design(path, output)
    assert not output.exists()


def _ideal_record(scenario: Any, replicate: dict[str, Any]) -> dict[str, Any]:
    """Synthetic arithmetic state for testing the parser, without fitting data."""

    upper = replicate["estimator_parameters"]["max_dimension"]
    dimensions = np.arange(1, upper + 1)
    count = scenario.n_samples
    sizes = count // np.arange(1, 6)
    t = np.log(count / sizes * np.log(sizes) / np.log(count))
    x = t[None, :] / dimensions[:, None]
    log_means = -10 + scenario.truth * x
    return {
        **replicate,
        "sample": study.array_metadata(
            scenario.sample(np.random.default_rng(replicate["data_seed"]))
        ),
        "success": True,
        "estimate": scenario.truth,
        "base_bandwidth": 1.0,
        "arrays": {
            "candidate_dimensions_": dimensions.tolist(),
            "bandwidth_factors_": np.exp(x).tolist(),
            "log_bandwidths_": x.tolist(),
            "kernel_means_": np.exp(log_means).tolist(),
            "log_u_statistics_": (log_means - dimensions[:, None] * x).tolist(),
            "slopes_": (scenario.truth - dimensions).tolist(),
        },
        "diagnostics": {
            "solver": "hein_audibert_weighted_slope_search",
            "converged": True,
            "n_iter": upper,
            "residual_norm": 0.0,
            "objective_value": 0.0,
            "numerical_rank": None,
            "condition_estimate": None,
            "warnings": [],
        },
    }


@pytest.fixture(scope="module")
def synthetic_evidence() -> dict[str, Any]:
    frozen = {"frozen_at": "2026-09-25T12:00:00+00:00", "design": study.make_design()}
    evidence: dict[str, Any] = {
        "frozen_design": frozen,
        "frozen_design_sha256": hashlib.sha256(study._json_bytes(frozen)).hexdigest(),
        "complete": True,
        "started_at": "2026-09-25T12:01:00+00:00",
        "completed_at": "2026-09-25T12:02:00+00:00",
        "scenarios": {},
        "decision": {"passed": True},
    }
    for scenario, entry in zip(
        study.scenarios(), frozen["design"]["scenarios"], strict=True
    ):
        evidence["scenarios"][scenario.name] = {
            "parameters": scenario.parameters(),
            "summary": {"passed": True, "rmse_upper": 0},
            "replicates": [
                _ideal_record(scenario, item) for item in entry["replicates"]
            ],
        }
    return evidence


def test_decision_recomputes_forged_summary_from_raw_records(
    synthetic_evidence: dict[str, Any],
) -> None:
    decision = study.certification_decision(synthetic_evidence)
    assert decision["passed"] is True
    assert decision["scenarios"]["flat_d1_n240"]["rmse_upper"] > 0.44
    altered = copy.deepcopy(synthetic_evidence)
    altered["scenarios"]["flat_d1_n240"]["replicates"][0]["success"] = False
    decision = study.certification_decision(altered)
    assert decision["passed"] is False
    assert decision["scenarios"]["flat_d1_n240"]["failures"] == 1


@pytest.mark.parametrize("mutation", ["hash", "seed", "nan", "partial", "extra"])
def test_decision_rejects_tampered_evidence(
    synthetic_evidence: dict[str, Any], mutation: str
) -> None:
    altered = copy.deepcopy(synthetic_evidence)
    first = altered["scenarios"]["flat_d1_n240"]["replicates"][0]
    if mutation == "hash":
        first["sample"]["sha256"] = "forged"
    elif mutation == "seed":
        first["estimator_seed"] += 1
    elif mutation == "nan":
        first["arrays"]["slopes_"][0] = float("nan")
    elif mutation == "partial":
        altered["complete"] = False
    else:
        altered["scenarios"]["untrusted"] = {}
    assert study.certification_decision(altered)["passed"] is False


@pytest.mark.parametrize("malformed", [None, [], "json", {}, {"frozen_design": []}])
def test_malformed_evidence_fails_closed(malformed: object) -> None:
    decision = study.certification_decision(malformed)
    assert decision["passed"] is False
    assert decision["failures"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "string",
        "boolean",
        "slope",
        "objective",
        "kernel",
        "warnings",
        "contradictory_failure",
    ],
)
def test_fitted_output_requires_complete_finite_consistent_diagnostics(
    synthetic_evidence: dict[str, Any], mutation: str
) -> None:
    record = copy.deepcopy(
        synthetic_evidence["scenarios"]["flat_d1_n240"]["replicates"][0]
    )
    if mutation == "missing":
        del record["diagnostics"]["objective_value"]
    elif mutation == "string":
        record["diagnostics"]["residual_norm"] = "0"
    elif mutation == "boolean":
        record["base_bandwidth"] = True
    elif mutation == "slope":
        record["arrays"]["slopes_"][-1] -= 0.1
    elif mutation == "objective":
        record["diagnostics"]["objective_value"] = 1.0
    elif mutation == "kernel":
        record["arrays"]["kernel_means_"][0][0] *= 2
    elif mutation == "warnings":
        record["diagnostics"]["warnings"] = "not an array"
    else:
        record["failure"] = {"type": "ValueError", "message": "failed"}
    assert study._valid_fitted_output(record, 5) is False


def test_existing_reference_fit_is_accepted_by_numerical_state_verifier() -> None:
    # This is an already-used historical seed, outside the untouched study range.
    scenario = study.scenarios()[0]
    replicate = {
        "data_seed": 1701,
        "estimator_seed": 1701,
        "estimator_parameters": study._factory(scenario, 1701).get_params(deep=False),
    }
    record = study.fit_record(scenario, replicate)
    assert record["success"] is True
    assert study._valid_fitted_output(record, 5) is True
    json.dumps(record, allow_nan=False)


def test_interruption_leaves_incomplete_evidence_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    design, output = tmp_path / "design.json", tmp_path / "evidence.json"
    study.freeze_design(design)

    def interrupted(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(study, "fit_record", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study.run_frozen_design(design, output)
    evidence = json.loads(output.read_bytes())
    assert evidence["complete"] is False
    assert study.certification_decision(evidence)["passed"] is False
    with pytest.raises(FileExistsError):
        study.run_frozen_design(design, output)


def test_isolated_script_can_load_the_adjacent_trusted_helpers() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(Path(study.__file__).resolve()), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "freeze" in result.stdout
    assert "verify" in result.stdout
