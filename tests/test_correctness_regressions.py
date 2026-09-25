"""Regression coverage for the September 2026 independent correctness audit."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.distance import pdist
from tools.release_audit import (
    _catalog_audit,
    _dimension_audit,
    _release_decision,
)

from believe14.estimation import DANCo, LevinaBickelMLE, MiNDML, TwoNN
from believe14.estimation._common import norm_kl
from believe14.linear import ProbabilisticPCA
from believe14.nonlinear import TSNE, ClassicalMDS, FastMap, LocalTangentSpaceAlignment


def heterogeneous_scale_data() -> np.ndarray:
    """Distinct finite observations with finite distances and log distance ratios."""
    return np.vstack(
        [
            np.zeros((1, 2)),
            [[1e-200, 0.0]],
            np.random.default_rng(3).normal(size=(8, 2)) * 1e200,
        ]
    )


def test_danco_kl_identity_at_supported_neighbor_count() -> None:
    assert norm_kl(3.0, 3.0, 60) == pytest.approx(0.0, abs=1e-10)


def test_danco_kl_matches_density_integral_at_larger_k() -> None:
    k = 50
    ratio = 4.0 / 3.0

    def integrand(u: float) -> float:
        log_density_ratio = (
            -np.log(ratio)
            + (1 - ratio) * np.log(u)
            + (k - 1) * (np.log1p(-u) - np.log1p(-(u**ratio)))
        )
        return float(k * (1 - u) ** (k - 1) * log_density_ratio)

    expected, error = quad(integrand, 0, 1, epsabs=1e-11, epsrel=1e-11, limit=300)
    assert error < 1e-9
    assert norm_kl(3.0, 4.0, k) == pytest.approx(expected, abs=1e-9)


def test_danco_public_divergences_are_nonnegative() -> None:
    data = np.random.default_rng(120).normal(size=(120, 5))
    fitted = DANCo(n_neighbors=80, max_dimension=5, random_state=3).fit(data)
    assert np.all(fitted.divergences_ >= -1e-10)


def test_ltsa_excludes_constant_from_flat_nullspace() -> None:
    data = np.random.default_rng(0).normal(size=(40, 2))
    embedded = LocalTangentSpaceAlignment(2, n_neighbors=6).fit_transform(data)
    centered = embedded - embedded.mean(axis=0)
    np.testing.assert_allclose(centered.T @ centered, np.eye(2), atol=1e-10)


def test_ltsa_flat_geometry_is_invariant_to_row_permutation() -> None:
    data = np.random.default_rng(0).normal(size=(40, 2))
    permutation = np.random.default_rng(5).permutation(len(data))
    first = LocalTangentSpaceAlignment(2, n_neighbors=6).fit_transform(data)
    second = LocalTangentSpaceAlignment(2, n_neighbors=6).fit_transform(
        data[permutation]
    )[np.argsort(permutation)]
    np.testing.assert_allclose(pdist(first), pdist(second), rtol=1e-8, atol=1e-10)


def test_fastmap_feature_queries_preserve_axis_coordinate_far_off_axis() -> None:
    fitted = FastMap(1).fit([[0.0, 0.0], [1.0, 0.0], [0.3, 0.2]])
    np.testing.assert_array_equal(fitted.pivot_indices_, [[0, 1]])
    query = np.array([[0.5, 1e9], [1.0, 1e9], [2.0, 1e9]])
    np.testing.assert_allclose(fitted.transform(query)[:, 0], query[:, 0], atol=1e-12)


def test_ppca_retains_finite_likelihood_for_representable_covariance() -> None:
    ordinary = np.random.default_rng(5).normal(size=(1000, 3))
    baseline = ProbabilisticPCA(1).fit(ordinary)
    # All covariance entries are ~1e306; only the unnormalized cross-product
    # overflows. Suppress warnings here to inspect the returned successful fit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fitted = ProbabilisticPCA(1).fit(ordinary * 1e153)
    expected = baseline.log_likelihood_ - ordinary.size * np.log(1e153)
    assert np.isfinite(fitted.log_likelihood_)
    assert fitted.log_likelihood_ == pytest.approx(expected, rel=1e-12)
    assert np.isfinite(fitted.diagnostics_.residual_norm)


def test_classical_mds_does_not_overflow_representable_gram() -> None:
    fitted = ClassicalMDS(1).fit([[-8e153], [8e153]])
    # Largest eigenvalue is 1.28e308 and Gram entries have magnitude 6.4e307.
    np.testing.assert_allclose(
        fitted.gram_matrix_ / 1e307, [[6.4, -6.4], [-6.4, 6.4]], rtol=1e-14
    )


def test_twonn_does_not_return_nan_dimension_from_finite_log_ratios() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            fitted = TwoNN().fit(heterogeneous_scale_data())
        except FloatingPointError:
            return  # An explicit range failure is an acceptable minimum policy.
    assert np.isfinite(fitted.dimension_)
    assert fitted.dimension_ > 0


def test_mind_does_not_report_success_with_nan_likelihood() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            fitted = MiNDML(n_neighbors=3).fit(heterogeneous_scale_data())
        except FloatingPointError:
            return
    assert np.isfinite(fitted.log_likelihood_)
    assert np.isfinite(fitted.diagnostics_.residual_norm)


def test_levina_bickel_local_dimensions_match_log_difference_oracle() -> None:
    data = heterogeneous_scale_data()
    # Direct hypot avoids overflow in cdist's squared-distance accumulation.
    distances = np.hypot.reduce(data[:, None, :] - data[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.sort(distances, axis=1)
    estimates = []
    for k in (3, 4):
        log_ratios = np.log(neighbors[:, [k - 1]]) - np.log(neighbors[:, : k - 1])
        estimates.append((k - 2) / log_ratios.sum(axis=1))
    expected = np.mean(estimates, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            fitted = LevinaBickelMLE(k_min=3, k_max=4).fit(data)
        except FloatingPointError:
            return
    np.testing.assert_allclose(fitted.local_dimensions_, expected, rtol=1e-12)


@pytest.fixture(scope="module")
def audit_records() -> tuple[dict, dict]:
    return _catalog_audit(1701), _dimension_audit((1701, 1702, 1703))


def test_release_gate_rejects_unapproved_method_exemptions(audit_records) -> None:
    catalog, original = audit_records
    scenarios = copy.deepcopy(original)
    scenario = scenarios["flat_d2_n240"]
    scenario["not_applicable"] = dict.fromkeys(scenario["methods"], "missing")
    scenario["methods"] = {}
    assert not _release_decision(catalog, scenarios)["passed"]


def test_release_gate_rejects_missing_all_replicates(audit_records) -> None:
    catalog, original = audit_records
    scenarios = copy.deepcopy(original)
    scenario = scenarios["flat_d2_n240"]
    scenario["seeds"] = []
    scenario["samples"] = []
    for summary in scenario["methods"].values():
        summary["raw"] = []
    assert not _release_decision(catalog, scenarios)["passed"]


def test_release_gate_rejects_nonconverged_dimension_fits(audit_records) -> None:
    catalog, scenarios = copy.deepcopy(audit_records)
    for summary in scenarios["flat_d2_n240"]["methods"].values():
        for result in summary["raw"]:
            result["diagnostics"]["converged"] = False
    assert not _release_decision(catalog, scenarios)["passed"]


@pytest.mark.parametrize("k", [1, 3, 10, 50, 100, 1000, 10000])
@pytest.mark.parametrize("ratio", [0.25, 0.75, 1.00001, 4.0, 16.0])
def test_norm_kl_against_independent_beta_density(k: int, ratio: float) -> None:
    # v=k*u; integrating the beta density differs from the production
    # exponential substitution. Beyond v=50 the omitted mass is negligible.
    def density_integrand(v: float) -> float:
        u = v / k
        log_ratio = -np.log(ratio) + (1 - ratio) * np.log(u)
        if k > 1:
            log_ratio += (k - 1) * (np.log1p(-u) - np.log1p(-(u**ratio)))
        return float((1 - u) ** (k - 1) * log_ratio)

    expected, error = quad(
        density_integrand,
        0.0,
        min(float(k), 50.0),
        epsabs=1e-9,
        epsrel=1e-10,
        limit=300,
    )
    assert error < max(1e-8, abs(expected) * 1e-8)
    actual = norm_kl(3.0, ratio * 3.0, k)
    assert actual >= 0.0
    assert actual == pytest.approx(expected, abs=1e-8, rel=1e-9)


@pytest.mark.parametrize("k", [2, 3, 1000, 1000000])
def test_norm_kl_identity_is_exact(k: int) -> None:
    assert norm_kl(7.3, 7.3, k) == 0.0


def test_danco_rejects_single_angle_neighborhood() -> None:
    with pytest.raises(ValueError, match="n_neighbors must be at least 3"):
        DANCo(n_neighbors=2).fit(np.random.default_rng(903).normal(size=(30, 3)))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_ltsa_flat_subspace_matches_affine_coordinates(dimension: int) -> None:
    rng = np.random.default_rng(219 + dimension)
    latent = rng.normal(size=(80, dimension))
    rotation, _ = np.linalg.qr(rng.normal(size=(dimension + 2, dimension)))
    features = latent @ rotation.T + 7.0
    embedded = LocalTangentSpaceAlignment(dimension, n_neighbors=12).fit_transform(
        features
    )
    expected, _ = np.linalg.qr(latent - latent.mean(axis=0))
    np.testing.assert_allclose(embedded.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(embedded @ embedded.T, expected @ expected.T, atol=1e-10)


def test_fastmap_precomputed_rejects_irrecoverable_query_projection() -> None:
    from scipy.spatial.distance import cdist, squareform

    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.2]])
    fitted = FastMap(1, dissimilarity="precomputed").fit(squareform(pdist(features)))
    query = cdist([[0.5, 1e9]], features)
    with pytest.raises(FloatingPointError, match="ill-conditioned"):
        fitted.transform(query)


def test_fastmap_multiaxis_features_replay_and_are_orthonormal() -> None:
    data = np.random.default_rng(210).normal(size=(70, 4))
    fitted = FastMap(4).fit(data)
    np.testing.assert_allclose(fitted.transform(data), fitted.embedding_, atol=1e-13)
    np.testing.assert_allclose(pdist(fitted.embedding_), pdist(data), atol=1e-12)
    query = np.eye(4)
    operator = fitted.transform(query) - fitted.transform(np.zeros((1, 4)))
    np.testing.assert_allclose(operator.T @ operator, np.eye(4), atol=1e-13)


def test_ppca_noise_average_does_not_overflow() -> None:
    data = np.random.default_rng(220).normal(size=(120, 30))
    baseline = ProbabilisticPCA(1).fit(data)
    scaled = ProbabilisticPCA(1).fit(data * 3e153)
    assert scaled.noise_variance_ / 9e306 == pytest.approx(
        baseline.noise_variance_, rel=1e-12
    )
    assert scaled.log_likelihood_ == pytest.approx(
        baseline.log_likelihood_ - data.size * np.log(3e153), rel=1e-12
    )


def test_classical_mds_representable_subnormal_gram() -> None:
    magnitude = 2e-161
    fitted = ClassicalMDS(1).fit([[-magnitude], [magnitude]])
    expected = magnitude * magnitude
    np.testing.assert_allclose(
        fitted.gram_matrix_ / expected, [[1.0, -1.0], [-1.0, 1.0]], atol=0.013, rtol=0
    )
    assert np.all(np.isfinite(fitted.embedding_))


@pytest.mark.parametrize("seed", [2, 3, 7])
def test_tsne_default_quality_survives_a_tighter_restart(seed: int) -> None:
    from scipy.optimize import minimize

    from believe14.nonlinear._stochastic import _kl_divergence, _tsne_objective_gradient

    features = np.random.default_rng(seed).normal(size=(100, 5))
    fitted = TSNE(random_state=seed).fit(features)
    result = minimize(
        _tsne_objective_gradient,
        fitted.embedding_.ravel(),
        args=(fitted.joint_probabilities_, 2),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 300},
    )
    restarted = _kl_divergence(result.x.reshape(100, 2), fitted.joint_probabilities_)
    # A stationarity/quality check independent of the implementation's stop flag.
    assert fitted.kl_divergence_ - restarted < 1e-4
    assert fitted.diagnostics_.residual_norm < 1e-4
    assert fitted.stopping_reason_
    if not fitted.gradient_converged_:
        assert fitted.diagnostics_.warnings


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_seed",
        "wrong_sample",
        "wrong_parameters",
        "altered_mean",
        "altered_estimate",
        "nonfinite_estimate",
        "nonfinite_diagnostic",
        "missing_diagnostics",
        "missing_numeric_outputs",
        "wrong_truth",
        "summary_only",
    ],
)
def test_release_gate_rejects_inconsistent_evidence(
    audit_records, mutation: str
) -> None:
    catalog, scenarios = copy.deepcopy(audit_records)
    scenario = scenarios["flat_d2_n240"]
    summary = scenario["methods"]["TwoNN"]
    value = summary["raw"][0]
    if mutation == "duplicate_seed":
        summary["raw"][1]["seed"] = value["seed"]
    elif mutation == "wrong_sample":
        scenario["samples"][0]["sha256"] = "a" * 64
    elif mutation == "wrong_parameters":
        value["parameters"]["discard_fraction"] = 0.5
    elif mutation == "altered_mean":
        summary["mean"] += 0.1
    elif mutation == "altered_estimate":
        value["estimate"] += 0.1
    elif mutation == "nonfinite_estimate":
        value["estimate"] = float("nan")
    elif mutation == "nonfinite_diagnostic":
        value["diagnostics"]["residual_norm"] = float("inf")
    elif mutation == "missing_diagnostics":
        del value["diagnostics"]
    elif mutation == "missing_numeric_outputs":
        value["checked_numeric_outputs"] = {}
    elif mutation == "wrong_truth":
        scenario["truth"] = 3.0
    elif mutation == "summary_only":
        summary["raw"] = []
    assert _release_decision(catalog, scenarios)["passed"] is False


def test_release_gate_requires_the_trusted_seed_count(audit_records) -> None:
    catalog, scenarios = audit_records
    assert (
        _release_decision(
            catalog, scenarios, expected_seeds=(1701, 1702, 1703, 1704, 1705)
        )["passed"]
        is False
    )


def test_dimension_intervals_and_nonconverged_failure_accounting(audit_records) -> None:
    from tools.release_audit import _dimension_summary

    _, scenarios = audit_records
    summary = copy.deepcopy(scenarios["flat_d2_n240"]["methods"]["TwoNN"])
    assert summary["failure_rate_ci95"][1] > 0.5
    assert summary["mean_ci95"][0] <= summary["mean"] <= summary["mean_ci95"][1]
    summary["raw"][0]["diagnostics"]["converged"] = False
    corrected = _dimension_summary(summary["raw"], 2.0)
    assert corrected["failure_rate"] == pytest.approx(1 / 3)
    assert corrected["successful_replicates"] == 2


def test_complete_release_records_pass(audit_records) -> None:
    catalog, scenarios = audit_records
    assert _release_decision(catalog, scenarios)["passed"] is True


def test_experimental_accuracy_is_visible_and_still_blocks_release(
    audit_records,
) -> None:
    from tools.release_audit import _dimension_summary

    catalog, scenarios = copy.deepcopy(audit_records)
    scenario = scenarios["flat_d2_n240"]
    values = scenario["methods"]["UStatisticDimension"]["raw"]
    for value in values:
        value["estimate"] = 1.0
    scenario["methods"]["UStatisticDimension"] = _dimension_summary(values, 2.0)
    decision = _release_decision(catalog, scenarios)
    assert decision["passed"] is False
    assert "flat_d2_n240:UStatisticDimension" in decision["dimension_failures"]
    assert decision["supported_scope"]["passed"] is True
    assert decision["supported_scope"]["nonblocking_accuracy_failures"] == [
        "flat_d2_n240:UStatisticDimension"
    ]
    summary = scenario["methods"]["UStatisticDimension"]
    assert summary["passed"] is False
    assert summary["release_thresholds"]["max_rmse"] == 0.5


@pytest.mark.parametrize("mutation", ["missing", "nonfinite", "nonconverged", "forged"])
def test_experimental_status_never_exempts_invalid_evidence(
    audit_records, mutation: str
) -> None:
    catalog, scenarios = copy.deepcopy(audit_records)
    summary = scenarios["flat_d2_n240"]["methods"]["UStatisticDimension"]
    if mutation == "missing":
        summary["raw"].pop()
    elif mutation == "nonfinite":
        summary["raw"][0]["estimate"] = float("nan")
    elif mutation == "nonconverged":
        summary["raw"][0]["diagnostics"]["converged"] = False
    else:
        summary["mean"] += 1.0
    decision = _release_decision(catalog, scenarios)
    assert decision["passed"] is False
    assert decision["supported_scope"]["passed"] is False


def test_supported_scope_still_requires_other_estimators_accuracy(
    audit_records,
) -> None:
    from tools.release_audit import _dimension_summary

    catalog, scenarios = copy.deepcopy(audit_records)
    scenario = scenarios["flat_d2_n240"]
    values = scenario["methods"]["TwoNN"]["raw"]
    for value in values:
        value["estimate"] = 4.0
    scenario["methods"]["TwoNN"] = _dimension_summary(values, 2.0)
    decision = _release_decision(catalog, scenarios)
    assert decision["passed"] is False
    assert decision["supported_scope"]["passed"] is False


def test_larger_sample_accuracy_policy_has_an_explicit_method_scope() -> None:
    from tools.release_audit import LARGER_SAMPLE_ACCURACY_METHODS

    from believe14.registry import get_estimator

    assert {"UStatisticDimension"} == LARGER_SAMPLE_ACCURACY_METHODS
    assert get_estimator("UStatisticDimension").family == "estimation"


@pytest.mark.parametrize("failed_stage", [None, "calibration", "holdout", "integrity"])
@pytest.mark.parametrize("certificate_passed", [False, True])
def test_audit_requires_fresh_certificate_and_keeps_historical_outcomes(
    monkeypatch, failed_stage: str | None, certificate_passed: bool
) -> None:
    import json

    from tools import release_audit

    decisions = iter(
        {
            "passed": stage != failed_stage,
            "supported_scope": {
                "passed": not (stage == failed_stage == "integrity"),
            },
        }
        for stage in ("calibration", "holdout", "integrity")
    )
    monkeypatch.setattr(release_audit, "_catalog_audit", lambda *a, **kw: {})
    monkeypatch.setattr(release_audit, "_dimension_audit", lambda *a, **kw: {})
    monkeypatch.setattr(
        release_audit, "_release_decision", lambda *a, **kw: next(decisions)
    )
    monkeypatch.setattr(
        release_audit,
        "_run_ustatistic_certification",
        lambda: ({"complete": True}, {"passed": certificate_passed}),
    )
    result = release_audit.run_audit()
    assert result["historical_panel_gate"]["passed"] is (failed_stage is None)
    assert result["release_gate"]["passed"] is (
        failed_stage != "integrity" and certificate_passed
    )
    assert result["release_gate"]["historical_panel_passed"] is (failed_stage is None)
    assert result["release_gate"]["required_checks"]["passed"] is (
        failed_stage != "integrity"
    )
    json.dumps(result, allow_nan=False)


def test_audit_cli_cannot_publish_with_only_one_gate_passing(
    monkeypatch, tmp_path
) -> None:
    import json
    import sys

    from tools import release_audit

    output = tmp_path / "audit.json"
    result = {
        "release_gate": {"passed": False},
        "ustatistic_certification": {"passed": True},
    }
    monkeypatch.setattr(release_audit, "run_audit", lambda **kw: result)
    monkeypatch.setattr(sys, "argv", ["release_audit", "--output", str(output)])
    with pytest.raises(SystemExit, match="release audit failed"):
        release_audit.main()
    assert json.loads(output.read_text()) == result


def test_release_gate_rejects_contradictory_catalog_outputs(audit_records) -> None:
    catalog, scenarios = copy.deepcopy(audit_records)
    outputs = catalog["PCA"]["checked_numeric_outputs"]
    outputs[next(iter(outputs))]["finite"] = False
    assert _release_decision(catalog, scenarios)["passed"] is False


def test_prespecified_characterization_covers_broader_regimes() -> None:
    from tools.release_audit import _characterization_scenarios, _dimension_factories

    scenarios = _characterization_scenarios()
    assert len(scenarios) == 12
    assert {spec.n_samples for spec in scenarios} == {400, 800}
    assert {spec.dimension for spec in scenarios if spec.geometry == "flat"} == {
        5,
        10,
        15,
    }
    assert {spec.geometry for spec in scenarios} == {"flat", "gaussian", "anisotropic"}
    for spec in scenarios:
        for name, factory in _dimension_factories(
            5701, spec.ambient_dimension, max_dimension=max(5, spec.dimension + 3)
        ).items():
            model = factory()
            if "max_dimension" in model.get_params():
                bound = model.get_params()["max_dimension"]
                assert bound >= spec.truth
                assert bound > spec.truth or (
                    name == "UStatisticDimension" and bound == 15
                )


@pytest.mark.parametrize("seed", [5701, 5705])
def test_ustatistic_holdout_slopes_match_literal_all_partition_oracle(
    seed: int,
) -> None:
    from scipy.spatial.distance import cdist
    from tools.release_audit import _dimension_scenarios

    from believe14.estimation import UStatisticDimension

    spec = next(item for item in _dimension_scenarios() if item.name == "flat_d3_n240")
    data = spec.sample(np.random.default_rng(seed))
    fitted = UStatisticDimension(max_dimension=5, random_state=seed).fit(data)
    distances = cdist(data, data)
    nearest = distances.copy()
    np.fill_diagonal(nearest, np.inf)
    base_bandwidth = nearest.min(axis=1).mean()
    order = np.random.default_rng(seed).permutation(len(data))
    slopes = []
    for dimension in range(1, 6):
        xs, ys = [], []
        for divisor in range(1, 6):
            count = len(data) // divisor
            bandwidth = base_bandwidth * (
                len(data) / count * np.log(count) / np.log(len(data))
            ) ** (1 / dimension)
            groups = order[: count * divisor].reshape(count, divisor).T
            statistics = []
            for left in range(divisor):
                for right in range(left, divisor):
                    block = distances[np.ix_(groups[left], groups[right])]
                    if left == right:
                        squared = block[np.triu_indices(count, 1)] ** 2
                    else:
                        squared = block.ravel() ** 2
                    statistics.append(
                        np.maximum(1 - squared / bandwidth**2, 0).mean()
                        / bandwidth**dimension
                    )
            xs.append(np.log(bandwidth))
            ys.append(np.log(np.mean(statistics)))
        design = np.column_stack([np.ones(5), xs])
        roots = 1 / np.sqrt(np.arange(1, 6))
        coefficient = np.linalg.lstsq(
            design * roots[:, None], np.asarray(ys) * roots, rcond=None
        )[0]
        slopes.append(coefficient[1])
    np.testing.assert_allclose(fitted.slopes_, slopes, atol=1e-12, rtol=1e-11)
    assert fitted.dimension_ == 1 + np.argmin(np.abs(slopes))


def test_invalid_numbers_remain_archivable_and_fail_the_gate(audit_records) -> None:
    import json

    from tools.release_audit import _jsonable

    catalog, scenarios = copy.deepcopy(audit_records)
    scenarios["flat_d2_n240"]["methods"]["TwoNN"]["raw"][0]["estimate"] = float("nan")
    restored = json.loads(json.dumps(_jsonable(scenarios), allow_nan=False))
    assert restored["flat_d2_n240"]["methods"]["TwoNN"]["raw"][0]["estimate"] == "nan"
    assert _release_decision(catalog, restored)["passed"] is False
