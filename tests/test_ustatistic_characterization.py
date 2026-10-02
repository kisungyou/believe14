from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tools import ustatistic_characterization as study


def test_frozen_plan_covers_prespecified_grid_with_disjoint_fresh_streams() -> None:
    design = study.make_design()
    primary = [
        entry
        for entry in design["scenarios"]
        if entry["parameters"]["role"] == "primary"
    ]
    assert {
        (entry["parameters"]["dimension"], entry["parameters"]["n_samples"])
        for entry in primary
    } == {(d, n) for d in (2, 3, 5) for n in (120, 240, 480, 960)}
    seeds = [
        seed
        for entry in design["scenarios"]
        for replicate in entry["replicates"]
        for seed in replicate.values()
    ]
    assert len(seeds) == len(set(seeds))
    assert all(seed >= 925_300_000 for seed in seeds)
    assert all(len(entry["replicates"]) >= 30 for entry in design["scenarios"])
    assert design["candidate_tuning"].startswith("none")


def test_samples_reproduce_and_noise_changes_support_dimension() -> None:
    flat = study.Scenario("flat", "flat", 3, 80)
    noisy = study.Scenario("noisy", "flat", 3, 80, noise=0.01)
    X = study.sample(flat, 91831)
    assert np.array_equal(X, study.sample(flat, 91831))
    assert study.array_metadata(X) == study.array_metadata(X[:, ::-1][:, ::-1])
    assert study.array_metadata(X) != study.array_metadata(study.sample(flat, 91832))
    assert np.linalg.matrix_rank(X - X.mean(axis=0)) == 3
    noisy_X = study.sample(noisy, 91831)
    assert np.linalg.matrix_rank(noisy_X - noisy_X.mean(axis=0)) == 8
    assert noisy.parameters()["support_dimension"] == 8
    assert noisy.parameters()["dimension"] == 3


def test_sphere_sample_has_unit_radius_after_isometric_embedding() -> None:
    sphere = study.Scenario("sphere", "sphere", 2, 40)
    X = study.sample(sphere, 91833)
    assert X.shape == (40, 8)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)
    assert np.linalg.matrix_rank(X) == 3


def test_summary_preserves_failure_denominators_and_known_rmse() -> None:
    records = [
        {"success": True, "estimate": 2.0},
        {"success": True, "estimate": 3.0},
        {"success": True, "estimate": 4.0},
        {"success": False, "failure": "domain error"},
        {"success": True, "estimate": float("nan")},
    ]
    summary = study.summarize(records, 3, 1942)
    assert summary["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert summary["bias"] == 0
    assert summary["failure_rate"] == 2 / 5
    assert summary["exact_accuracy_all_attempts"] == 1 / 5
    low, high = summary["failure_rate_ci95"]
    assert low < 2 / 5 < high
    assert study.summarize(records, 3, 1942) == summary
    json.dumps(summary, allow_nan=False)


def test_zero_observed_failures_still_has_nonzero_uncertainty() -> None:
    summary = study.summarize([{"success": True, "estimate": 3.0}] * 50, 3, 1942)
    assert summary["failure_rate"] == 0
    assert summary["failure_rate_ci95"][1] > 0.07
    assert summary["exact_accuracy_ci95"][0] < 1
    assert summary["rmse_bootstrap_ci95"] == [0.0, 0.0]


def test_all_failed_study_has_no_error_estimate() -> None:
    summary = study.summarize([{"success": False}] * 3, 3, 1942)
    assert summary["failures"] == 3
    assert summary["rmse"] is None
    assert summary["rmse_bootstrap_ci95"] is None
    with pytest.raises(ValueError, match="attempted replicate"):
        study.summarize([], 3, 1942)


def test_estimator_exception_is_recorded_with_sample_and_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_fit(self: object, X: np.ndarray) -> None:
        raise ValueError("expected simulated domain failure")

    monkeypatch.setattr(study.UStatisticDimension, "fit", broken_fit)
    scenario = study.Scenario("flat", "flat", 3, 20)
    result = study.fit_record(scenario, {"data_seed": 92001, "estimator_seed": 92002})
    assert result["success"] is False
    assert result["failure"]["type"] == "ValueError"
    assert len(result["sample"]["sha256"]) == 64
    assert result["estimator_parameters"]["random_state"] == 92002
    json.dumps(result, allow_nan=False)


def test_nonfinite_fitted_state_is_counted_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = study.UStatisticDimension.fit

    def nonfinite_fit(
        self: study.UStatisticDimension, X: np.ndarray
    ) -> study.UStatisticDimension:
        original_fit(self, X)
        self.slopes_[0] = np.inf
        return self

    monkeypatch.setattr(study.UStatisticDimension, "fit", nonfinite_fit)
    scenario = study.Scenario("flat", "flat", 3, 120)
    result = study.fit_record(scenario, {"data_seed": 92003, "estimator_seed": 92004})
    assert result["success"] is False
    assert result["failure"]["type"] == "FloatingPointError"
    assert "estimate" not in result
    json.dumps(result, allow_nan=False)


def test_frozen_design_cannot_be_overwritten_or_silently_modified(
    tmp_path: Path,
) -> None:
    design_path, output = tmp_path / "design.json", tmp_path / "evidence.json"
    study.freeze_design(design_path)
    with pytest.raises(FileExistsError):
        study.freeze_design(design_path)
    altered = json.loads(design_path.read_text())
    altered["design"]["scenarios"][0]["replicates"][0]["data_seed"] += 1
    design_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="Frozen design"):
        study.run_frozen_design(design_path, output)
    assert not output.exists()
