"""Probability, reproducibility, and numerical tests for t-SNE and PHATE."""

from __future__ import annotations

import numpy as np
import pytest

from believe14.nonlinear import PHATE, TSNE
from believe14.nonlinear._stochastic import _tsne_objective_gradient


def test_tsne_joint_probabilities_are_exactly_symmetric_and_normalized() -> None:
    X = np.random.default_rng(30).normal(size=(20, 4))
    model = TSNE(
        perplexity=5,
        init="random",
        random_state=7,
        early_exaggeration_iter=5,
        max_iter=20,
    ).fit(X)
    probabilities = model.joint_probabilities_
    np.testing.assert_allclose(probabilities, probabilities.T, atol=1e-15)
    np.testing.assert_array_equal(np.diag(probabilities), 0.0)
    assert probabilities.sum() == pytest.approx(1.0)
    conditional = model.conditional_probabilities_
    positive = conditional > 0.0
    safe_conditional = np.where(positive, conditional, 1.0)
    row_entropies = -np.sum(
        np.where(positive, conditional * np.log(safe_conditional), 0.0),
        axis=1,
    )
    np.testing.assert_allclose(row_entropies, np.log(5.0), atol=1e-8)
    np.testing.assert_allclose(model.perplexity_entropy_residuals_, 0.0, atol=1e-8)
    assert model.kl_divergence_ >= 0.0


def test_tsne_seed_replay_and_global_rng_isolation() -> None:
    X = np.random.default_rng(31).normal(size=(18, 3))
    np.random.seed(812)
    before = np.random.get_state()
    first = TSNE(
        perplexity=4,
        init="random",
        random_state=12,
        early_exaggeration_iter=3,
        max_iter=12,
    ).fit_transform(X)
    second = TSNE(
        perplexity=4,
        init="random",
        random_state=12,
        early_exaggeration_iter=3,
        max_iter=12,
    ).fit_transform(X)
    after = np.random.get_state()
    np.testing.assert_allclose(first, second)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_tsne_rejects_impossible_perplexity() -> None:
    X = np.arange(12.0).reshape(6, 2)
    with pytest.raises(ValueError, match=r"\[1, n_samples - 1\]"):
        TSNE(perplexity=6).fit(X)
    with pytest.raises(ValueError, match=r"\[1, n_samples - 1\]"):
        TSNE(perplexity=0.5).fit(X)


def test_tsne_rejects_unattainable_perplexity_from_distance_ties() -> None:
    X = np.array([[0.0], [-1.0], [1.0], [4.0]])
    with pytest.raises(ValueError, match="nearest distances are tied"):
        TSNE(perplexity=1.0).fit(X)


def test_tsne_analytic_gradient_matches_central_differences() -> None:
    rng = np.random.default_rng(32)
    embedding = rng.normal(scale=0.3, size=(5, 2)).ravel()
    weights = rng.uniform(0.1, 1.0, size=(5, 5))
    probabilities = (weights + weights.T) * 0.5
    np.fill_diagonal(probabilities, 0.0)
    probabilities /= probabilities.sum()
    _, analytic = _tsne_objective_gradient(embedding, probabilities, 2)
    step = 1e-6
    finite_difference = np.empty_like(embedding)
    for coordinate in range(embedding.size):
        forward = embedding.copy()
        backward = embedding.copy()
        forward[coordinate] += step
        backward[coordinate] -= step
        forward_objective = _tsne_objective_gradient(forward, probabilities, 2)[0]
        backward_objective = _tsne_objective_gradient(backward, probabilities, 2)[0]
        finite_difference[coordinate] = (forward_objective - backward_objective) / (
            2.0 * step
        )
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6, atol=1e-9)


def test_phate_operator_and_potential_distance_definitions() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    X = np.column_stack((np.cos(theta), np.sin(theta), theta / 4.0))
    model = PHATE(
        n_neighbors=4,
        decay=5.0,
        diffusion_time=4,
        mds_max_iter=50,
    ).fit(X)
    differences = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    bandwidth = np.sort(distances, axis=1, kind="stable")[:, 4]
    directed = np.exp(-((distances / bandwidth[:, None]) ** 5.0))
    expected_affinity = (directed + directed.T) * 0.5
    expected_operator = expected_affinity / expected_affinity.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(model.affinity_matrix_, expected_affinity, atol=1e-14)
    np.testing.assert_allclose(model.diffusion_operator_, expected_operator, atol=1e-14)
    np.testing.assert_allclose(model.diffusion_operator_.sum(axis=1), 1.0)
    potential = model.diffusion_potential_
    potential_differences = potential[:, None, :] - potential[None, :, :]
    expected = np.sqrt(np.sum(potential_differences**2, axis=2))
    np.testing.assert_allclose(model.potential_distances_, expected, atol=1e-12)
    np.testing.assert_allclose(model.potential_distances_, expected.T, atol=1e-12)
    np.testing.assert_array_equal(np.diag(model.potential_distances_), 0.0)
    assert model.stress_ == pytest.approx(model.diagnostics_.objective_value)


def test_phate_rejects_zero_adaptive_bandwidth() -> None:
    X = np.array([[0.0], [0.0], [0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="zero adaptive bandwidth"):
        PHATE(n_neighbors=2, decay=5.0).fit(X)


def test_stochastic_and_phate_estimators_are_transductive() -> None:
    assert not hasattr(TSNE(), "transform")
    assert not hasattr(PHATE(), "transform")
