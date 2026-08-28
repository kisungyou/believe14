from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial.distance import cdist
from scipy.special import i0
from sklearn.base import BaseEstimator, clone

from believe14.estimation import (
    CorrelationDimension,
    DANCo,
    LevinaBickelMLE,
    MiNDML,
    TwoNN,
    UStatisticDimension,
)
from believe14.estimation._common import (
    inverse_bessel_ratio,
    mind_log_likelihood,
    mind_score,
    norm_kl,
    von_mises_kl,
)

ESTIMATOR_CLASSES = (
    CorrelationDimension,
    TwoNN,
    LevinaBickelMLE,
    UStatisticDimension,
    MiNDML,
    DANCo,
)


def _literal_neighbor_distances(X: np.ndarray, count: int) -> np.ndarray:
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)
    return np.sort(distances, axis=1, kind="stable")[:, :count]


def test_classes_directly_inherit_base_estimator() -> None:
    for estimator_class in ESTIMATOR_CLASSES:
        assert estimator_class.__bases__ == (BaseEstimator,)


def test_public_contract_has_no_inapplicable_methods() -> None:
    for estimator in (
        CorrelationDimension(),
        TwoNN(),
        LevinaBickelMLE(),
        UStatisticDimension(),
        MiNDML(),
        DANCo(),
    ):
        assert not hasattr(estimator, "transform")
        assert not hasattr(estimator, "predict")
        assert not hasattr(estimator, "score")


def test_correlation_dimension_literal_pair_count_and_slope() -> None:
    X = np.array([[0.0], [1.0], [3.0], [7.0]])
    radii = np.array([1.5, 2.5, 4.5])
    estimator = CorrelationDimension(radii=radii).fit(X)

    # The six distances are 1, 2, 3, 4, 6, 7, so the strict counts are 1, 2, 4.
    correlation = np.array([1.0, 2.0, 4.0]) / 6.0
    x_values = np.log(radii)
    y_values = np.log(correlation)
    expected = np.sum((x_values - x_values.mean()) * (y_values - y_values.mean()))
    expected /= np.sum((x_values - x_values.mean()) ** 2)

    np.testing.assert_allclose(estimator.correlation_integral_, correlation)
    assert estimator.dimension_ == pytest.approx(expected)


def test_twonn_literal_empirical_cdf_regression() -> None:
    X = np.array([[0.0], [0.7], [2.0], [4.5], [8.0], [13.0]])
    estimator = TwoNN(discard_fraction=0.2).fit(X)
    distances = _literal_neighbor_distances(X, 2)
    ratios = np.sort(distances[:, 1] / distances[:, 0])
    retained = ratios[:4]
    x_values = np.log(retained)
    y_values = -np.log1p(-np.arange(1, 5) / 6.0)
    expected = float(x_values @ y_values / (x_values @ x_values))

    np.testing.assert_allclose(estimator.ratios_, ratios)
    assert estimator.dimension_ == pytest.approx(expected)


@pytest.mark.parametrize("bias_correction,numerator", [(False, 3.0), (True, 2.0)])
def test_levina_bickel_literal_local_formula(
    bias_correction: bool, numerator: float
) -> None:
    X = np.array(
        [[0.0, 0.0], [1.0, 0.1], [0.2, 1.4], [1.8, 1.0], [3.0, 0.4], [2.7, 2.2]]
    )
    estimator = LevinaBickelMLE(k_min=4, k_max=4, bias_correction=bias_correction).fit(
        X
    )
    distances = _literal_neighbor_distances(X, 4)
    expected_local = numerator / np.sum(
        np.log(distances[:, [3]] / distances[:, :3]), axis=1
    )

    np.testing.assert_allclose(estimator.local_dimensions_, expected_local)
    assert estimator.dimension_ == pytest.approx(float(np.mean(expected_local)))


def test_hein_audibert_literal_full_sample_u_statistic() -> None:
    X = np.column_stack(
        (
            np.linspace(0.0, 1.0, 10),
            np.array([0.0, 0.2, 0.05, 0.4, 0.1, 0.6, 0.3, 0.8, 0.55, 1.0]),
        )
    )
    estimator = UStatisticDimension(max_dimension=2, random_state=7).fit(X)
    distances = cdist(X, X)
    np.fill_diagonal(distances, np.inf)
    bandwidth = float(np.mean(np.min(distances, axis=1)))
    finite_distances = cdist(X, X)
    upper = finite_distances[np.triu_indices(X.shape[0], k=1)]
    expected = float(
        np.mean(np.maximum(1.0 - upper**2 / bandwidth**2, 0.0)) / bandwidth
    )

    assert estimator.base_bandwidth_ == pytest.approx(bandwidth)
    # Candidate dimension 1 and divisor r=1 occupy [0, 0].
    assert estimator.log_u_statistics_[0, 0] == pytest.approx(np.log(expected))


def test_mind_literal_ratios_likelihood_and_score() -> None:
    X = np.array(
        [[0.0, 0.0], [1.0, 0.2], [0.1, 1.7], [2.2, 0.9], [3.0, 2.1], [4.2, 0.3]]
    )
    estimator = MiNDML(n_neighbors=2, max_dimension=2).fit(X)
    distances = _literal_neighbor_distances(X, 3)
    ratios = distances[:, 0] / distances[:, 2]
    expected_likelihood = mind_log_likelihood(estimator.dimension_, ratios, 2)

    np.testing.assert_allclose(estimator.normalized_distances_, ratios)
    assert estimator.log_likelihood_ == pytest.approx(expected_likelihood)
    if 1.0 < estimator.dimension_ < 2.0:
        assert mind_score(estimator.dimension_, ratios, 2) == pytest.approx(
            0.0, abs=1e-5
        )


def test_danco_closed_form_components_match_identity_cases() -> None:
    assert norm_kl(2.75, 2.75, 10) == pytest.approx(0.0, abs=1e-9)
    assert von_mises_kl(1.2, 3.4, 1.2, 3.4) == pytest.approx(0.0, abs=1e-12)
    eta = 0.2
    assert inverse_bessel_ratio(eta) == pytest.approx(
        2.0 * eta + eta**3 + 5.0 * eta**5 / 6.0
    )
    with pytest.raises(ValueError, match="infinite concentration"):
        inverse_bessel_ratio(1.0)


def test_danco_norm_kl_matches_independent_numerical_integral() -> None:
    dimension_1, dimension_2, n_neighbors = 2.3, 4.1, 5

    # Under u=r**dimension_1, u has the Beta(1, k) density.  This
    # independently integrates the density ratio without using the production
    # alternating digamma sum.
    ratio = dimension_2 / dimension_1

    def integrand(unit_radius_power: float) -> float:
        if unit_radius_power in {0.0, 1.0}:
            return 0.0
        log_ratio = (
            np.log(dimension_1 / dimension_2)
            + (1.0 - ratio) * np.log(unit_radius_power)
            + (n_neighbors - 1.0)
            * (np.log1p(-unit_radius_power) - np.log1p(-(unit_radius_power**ratio)))
        )
        beta_density = n_neighbors * (1.0 - unit_radius_power) ** (n_neighbors - 1)
        return float(beta_density * log_ratio)

    expected, error = quad(integrand, 0.0, 1.0, epsabs=1e-11, epsrel=1e-11, limit=300)
    assert error < 1e-10
    assert norm_kl(dimension_1, dimension_2, n_neighbors) == pytest.approx(
        expected, abs=1e-10
    )


def test_danco_von_mises_kl_matches_independent_numerical_integral() -> None:
    mean_1, concentration_1 = 1.1, 2.4
    mean_2, concentration_2 = 1.8, 4.2

    def density(angle: float, mean: float, concentration: float) -> float:
        return float(
            np.exp(concentration * np.cos(angle - mean))
            / (2.0 * np.pi * i0(concentration))
        )

    def integrand(angle: float) -> float:
        first = density(angle, mean_1, concentration_1)
        second = density(angle, mean_2, concentration_2)
        return first * np.log(first / second)

    expected, error = quad(integrand, -np.pi, np.pi, epsabs=1e-12)
    assert error < 1e-9
    assert von_mises_kl(
        mean_1, concentration_1, mean_2, concentration_2
    ) == pytest.approx(expected, abs=1e-10)


def test_all_estimators_clone_and_fit_returns_self() -> None:
    rng = np.random.default_rng(814)
    X = rng.normal(size=(80, 3))
    estimators = (
        CorrelationDimension(),
        TwoNN(),
        LevinaBickelMLE(k_min=5, k_max=8),
        UStatisticDimension(max_dimension=3, random_state=4),
        MiNDML(n_neighbors=5, max_dimension=3),
        DANCo(n_neighbors=5, max_dimension=3, random_state=4),
    )
    for estimator in estimators:
        copied = clone(estimator)
        assert copied.fit(X) is copied
        assert isinstance(copied.dimension_, float)
        assert copied.diagnostics_.converged
        assert copied.n_features_in_ == 3
