"""Independent formula and contract tests for distance-based embeddings."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from believe14.nonlinear import (
    ClassicalMDS,
    FastMap,
    MetricMDS,
    SammonMapping,
)


def _raw_stress(target: np.ndarray, embedding: np.ndarray) -> float:
    residual = squareform(pdist(embedding)) - target
    return float(np.sum(np.triu(residual * residual, k=1)))


def _literal_sammon_objective(target: np.ndarray, embedding: np.ndarray) -> float:
    embedded = squareform(pdist(embedding))
    upper = np.triu_indices_from(target, k=1)
    return float(
        np.sum((target[upper] - embedded[upper]) ** 2 / target[upper])
        / np.sum(target[upper])
    )


def test_classical_mds_is_literal_double_centering() -> None:
    X = np.array(
        [[-1.0, 0.0], [0.0, 2.0], [2.0, -1.0], [3.0, 2.0]],
        dtype=np.float64,
    )
    distances = squareform(pdist(X))
    model = ClassicalMDS(2, dissimilarity="precomputed").fit(distances)
    centered = X - X.mean(axis=0)
    np.testing.assert_allclose(
        model.embedding_ @ model.embedding_.T,
        centered @ centered.T,
        atol=2e-12,
    )
    assert model.diagnostics_.residual_norm is not None
    assert model.diagnostics_.residual_norm < 1e-12
    assert model.diagnostics_.numerical_rank == 2


def test_classical_mds_numerical_rank_ignores_roundoff_eigenvalues() -> None:
    rng = np.random.default_rng(1401)
    X = rng.normal(size=(40, 2))
    model = ClassicalMDS(2).fit(X)
    assert model.diagnostics_.numerical_rank == 2


def test_classical_mds_rescales_extreme_representable_dissimilarities() -> None:
    X = np.array([[-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 2.0], [3.0, -2.0]])
    baseline = ClassicalMDS(2).fit(X)
    for factor in (1e-150, 1e150):
        changed = ClassicalMDS(2).fit(X * factor)
        np.testing.assert_allclose(
            squareform(pdist(changed.embedding_ / factor)),
            squareform(pdist(baseline.embedding_)),
            rtol=2e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            changed.eigenvalues_[:2] / factor / factor,
            baseline.eigenvalues_[:2],
            rtol=2e-12,
            atol=0.0,
        )
    with pytest.raises(FloatingPointError, match="overflows"):
        ClassicalMDS(2).fit(X * 1e300)


def test_classical_mds_rejects_nonmetric_requested_rank() -> None:
    distances = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    with pytest.raises(ValueError, match="positive numerical rank"):
        ClassicalMDS(2, dissimilarity="precomputed").fit(distances)


def test_metric_mds_reports_literal_raw_stress() -> None:
    rng = np.random.default_rng(10)
    X = rng.normal(size=(18, 4))
    target = squareform(pdist(X))
    model = MetricMDS(2, max_iter=100, tol=1e-8).fit(X)
    assert model.stress_ == pytest.approx(_raw_stress(target, model.embedding_))
    initial = ClassicalMDS(2).fit_transform(X)
    assert model.stress_ <= _raw_stress(target, initial) + 1e-10


def test_sammon_mapping_reports_paper_normalization() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(16, 3))
    target = squareform(pdist(X))
    model = SammonMapping(2, max_iter=80).fit(X)
    embedded = squareform(pdist(model.embedding_))
    upper = np.triu_indices_from(target, k=1)
    expected = np.sum((target[upper] - embedded[upper]) ** 2 / target[upper])
    expected /= np.sum(target[upper])
    assert model.stress_ == pytest.approx(expected, rel=1e-10)


def test_sammon_analytic_gradient_matches_literal_central_difference() -> None:
    training = np.array(
        [[-1.0, 0.0, 0.5], [0.2, 1.3, -0.4], [1.4, -0.8, 0.1], [2.0, 0.7, 1.1]]
    )
    target = squareform(pdist(training))
    embedding = np.array(
        [[-0.8, 0.1], [0.0, 0.9], [0.7, -0.6], [1.3, 0.4]],
        dtype=np.float64,
    )
    _, analytic = SammonMapping._objective_gradient(
        embedding.ravel(), target, n_components=2
    )
    finite_difference = np.empty_like(analytic)
    step = 1e-6
    flat = embedding.ravel()
    for coordinate in range(flat.size):
        forward = flat.copy()
        backward = flat.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        finite_difference[coordinate] = (
            _literal_sammon_objective(target, forward.reshape(-1, 2))
            - _literal_sammon_objective(target, backward.reshape(-1, 2))
        ) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-6, atol=2e-9)


def test_sammon_mapping_rejects_duplicate_observations() -> None:
    X = np.array([[0.0], [0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="strictly positive"):
        SammonMapping(1).fit(X)


def test_fastmap_is_exact_on_euclidean_rank_two_and_has_pivot_extension() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [-1.0, 2.0]])
    model = FastMap(2).fit(X)
    np.testing.assert_allclose(
        squareform(pdist(model.embedding_)), squareform(pdist(X)), atol=2e-12
    )
    np.testing.assert_allclose(model.transform(X), model.embedding_, atol=2e-12)


def test_precomputed_fastmap_accepts_query_to_training_dissimilarities() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    distances = squareform(pdist(X))
    model = FastMap(2, dissimilarity="precomputed").fit(distances)
    np.testing.assert_allclose(model.transform(distances), model.embedding_, atol=2e-12)
    with pytest.raises(ValueError, match="one column per fitted observation"):
        model.transform(np.ones((1, X.shape[0] - 1)))


@pytest.mark.parametrize(
    "estimator",
    [ClassicalMDS(), MetricMDS(max_iter=2), SammonMapping(max_iter=2)],
)
def test_transductive_distance_estimators_do_not_claim_transform(
    estimator: object,
) -> None:
    assert not hasattr(estimator, "transform")
