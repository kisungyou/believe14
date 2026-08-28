"""Opt-in empirical gates exercised by nightly and release workflows."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from tools.release_audit import (
    _catalog_audit,
    _dimension_audit,
    _release_decision,
)

from believe14.linear import PCA, ProbabilisticPCA


@pytest.mark.research
def test_prespecified_dimension_scenarios_pass_distributional_gates() -> None:
    scenarios = _dimension_audit((1701, 1702, 1703))
    catalog = _catalog_audit(1701)
    decision = _release_decision(catalog, scenarios)
    assert decision["passed"] is True
    assert set(scenarios) == {
        "flat_d1_n240",
        "flat_d1_n360",
        "flat_d2_n240",
        "flat_d2_n360",
        "flat_d3_n240",
        "flat_d3_n360",
        "flat_d2_n300_noise_1e-3",
        "sphere_d2_n300",
        "swiss_roll_d2_n300",
    }
    for scenario in scenarios.values():
        assert scenario["parameters"]  # type: ignore[index]
        assert scenario["seeds"] == [1701, 1702, 1703]  # type: ignore[index]
        assert len(scenario["samples"]) == 3  # type: ignore[arg-type,index]
        for sample in scenario["samples"]:  # type: ignore[union-attr]
            assert len(sample["sha256"]) == 64  # type: ignore[index]
        for summary in scenario["methods"].values():  # type: ignore[union-attr]
            assert all(value["parameters"] for value in summary["raw"])  # type: ignore[index]

    incomplete_catalog = dict(catalog)
    incomplete_catalog.pop("PCA")
    assert _release_decision(incomplete_catalog, scenarios)["passed"] is False
    incomplete_scenarios = copy.deepcopy(scenarios)
    incomplete_scenarios.pop("sphere_d2_n300")
    assert _release_decision(catalog, incomplete_scenarios)["passed"] is False


@pytest.mark.research
def test_complete_catalog_audit_runs_public_calls() -> None:
    records = _catalog_audit(1701)
    assert len(records) == 30
    for record in records.values():
        assert record["converged"] is True  # type: ignore[index]
        assert record.get("output_finite", True) is True  # type: ignore[union-attr]


@pytest.mark.research
def test_repeated_and_ill_conditioned_spectra_are_audited_by_subspace() -> None:
    rng = np.random.default_rng(1714)
    left, _ = np.linalg.qr(rng.normal(size=(120, 4)))
    right, _ = np.linalg.qr(rng.normal(size=(6, 4)))
    singular_values = np.array([10.0, 10.0, 1e-5, 1e-10])
    X = left @ np.diag(singular_values) @ right.T
    first = PCA(2).fit(X)
    second = PCA(2).fit(X @ np.diag([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]))
    first_projector = first.components_.T @ first.components_
    reflected = np.diag([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
    second_projector = reflected @ second.components_.T @ second.components_ @ reflected
    np.testing.assert_allclose(first_projector, second_projector, atol=1e-10)
    assert ProbabilisticPCA(2).fit(X).diagnostics_.condition_estimate is not None
