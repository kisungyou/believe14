"""Graph quotient and stochastic nearest-neighbor objective checks."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.base import clone

from believe14.linear import LaplacianScore, NeighborhoodComponentsAnalysis
from believe14.linear._nca import _nca_loss_gradient


def test_nca_gradient_matches_independent_objective_on_each_entry():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(9, 4))
    y = np.repeat([0, 1, 2], 3)
    A = rng.normal(size=(2, 4))

    def literal(a):
        Z = X @ a.T
        success = 0.0
        for i in range(len(Z)):
            others = np.arange(len(Z)) != i
            weight = np.exp(-np.sum((Z[i] - Z[others]) ** 2, axis=1))
            success += weight[y[others] == y[i]].sum() / weight.sum()
        return -success / len(Z)

    f, gradient = _nca_loss_gradient(A.ravel(), X, y[:, None] == y)
    assert f == pytest.approx(literal(A))
    for j in range(A.size):
        perturbation = np.zeros_like(A)
        perturbation.flat[j] = 1e-6
        numerical = (literal(A + perturbation) - literal(A - perturbation)) / 2e-6
        assert numerical == pytest.approx(gradient[j], abs=3e-9)


@pytest.mark.parametrize("init", ["identity", "pca"])
def test_nca_learns_class_geometry_on_held_out_rows(init):
    rng = np.random.default_rng(351)
    y = np.tile([0, 1], 100)
    X = np.column_stack(
        (rng.normal(size=len(y)) * 4, y + rng.normal(size=len(y)) * 0.08)
    )
    model = NeighborhoodComponentsAnalysis(2, init=init, max_iter=150, tol=1e-7)
    fitted = clone(model).fit(X[:100], y[:100])
    train, test = fitted.transform(X[:100]), fitted.transform(X[100:])
    predictions = y[:100][cdist(test, train).argmin(axis=1)]
    assert (predictions == y[100:]).mean() >= 0.98
    assert fitted.expected_accuracy_ > 0.98 and fitted.diagnostics_.converged
    D = cdist(train, train, metric="sqeuclidean")
    np.fill_diagonal(D, np.inf)
    probs = np.exp(-(D - D.min(axis=1, keepdims=True)))
    probs /= probs.sum(axis=1, keepdims=True)
    assert fitted.expected_accuracy_ == pytest.approx(
        (probs * (y[:100, None] == y[:100])).sum() / 100
    )
    second = clone(model).fit(X[:100] + 32, 1 - y[:100])
    np.testing.assert_allclose(
        cdist(test, train),
        cdist(second.transform(X[100:] + 32), second.transform(X[:100] + 32)),
        rtol=2e-7,
        atol=2e-8,
    )
    assert len(fitted.get_feature_names_out()) == 2


def test_nca_labels_dimension_and_optimization_contract():
    X = np.random.default_rng(16).normal(size=(40, 4))
    with pytest.raises(ValueError, match="y is required"):
        NeighborhoodComponentsAnalysis().fit(X)
    with pytest.raises(ValueError, match="same-class"):
        NeighborhoodComponentsAnalysis().fit(X, np.arange(40))
    with pytest.raises(ValueError, match="nonconstant"):
        NeighborhoodComponentsAnalysis().fit(np.zeros_like(X), np.arange(40) % 3)
    with pytest.raises(ValueError, match="init"):
        NeighborhoodComponentsAnalysis(init="bad").fit(X, np.arange(40) % 3)
    m = NeighborhoodComponentsAnalysis(2, max_iter=1, tol=1e-14).fit(
        X, np.arange(40) % 3
    )
    assert not m.diagnostics_.converged and m.n_iter_ == 1
    assert m.diagnostics_.warnings


def test_laplacian_score_matches_path_graph_quadratic_forms_and_excludes_constants():
    # The second coordinate has amplitude .05. Distances strictly increase
    # along the line so the union of first-neighbor edges is a path.
    X = np.column_stack(
        ([0.0, 1.0, 2.1, 3.3, 4.6], [0.05, -0.05, 0.05, -0.05, 0.05], np.ones(5))
    )
    W = np.diag(np.ones(4), 1) + np.diag(np.ones(4), -1)
    D = np.diag(W.sum(axis=1))
    L = D - W
    expect = []
    for x in X[:, :2].T:
        g = x - np.average(x, weights=D.diagonal())
        expect.append(g @ L @ g / (g @ D @ g))
    m = LaplacianScore(1, n_neighbors=1).fit(X)
    np.testing.assert_allclose(m.scores_[:2], expect, rtol=1e-13)
    assert m.scores_[2] == 2.0 and not m.eligible_mask_[2]
    np.testing.assert_array_equal(m.get_support(indices=True), [0])
    np.testing.assert_array_equal(m.transform(X + 1), X[:, [0]] + 1)
    with pytest.raises(ValueError, match="nonconstant"):
        LaplacianScore(3, n_neighbors=1).fit(X)


def test_laplacian_score_heat_neighbors_and_stable_ties():
    from believe14.neighbors import build_neighbor_graph

    x = np.array([0.1, 0.4, 0.9, 1.2, 2.4, 2.8])
    X = np.column_stack((x, x))
    graph = build_neighbor_graph(X, n_neighbors=5)
    model = LaplacianScore(1, n_neighbors=3, weighting="heat", gamma=0.3)
    first = clone(model).fit(X)
    reused = clone(model).fit(X, neighbors=graph)
    np.testing.assert_array_equal(first.scores_, reused.scores_)
    np.testing.assert_array_equal(first.get_support(indices=True), [0])
    np.testing.assert_allclose(
        clone(model).fit(X + 100).scores_, first.scores_, rtol=1e-13
    )
    with pytest.raises(ValueError, match="disconnected"):
        LaplacianScore(1, n_neighbors=1).fit([[0.0], [0.1], [10.0], [10.1]])


def test_nca_singleton_label_contributes_zero_in_stochastic_accuracy():
    X = np.array([[0.0], [0.0], [0.0], [2.0]])
    y = np.array(["pair", "pair", "other", "singleton"])
    model = NeighborhoodComponentsAnalysis(1).fit(X, y)
    Z = model.transform(X)
    distances = cdist(Z, Z, metric="sqeuclidean")
    np.fill_diagonal(distances, np.inf)
    weight = np.exp(-(distances - distances.min(axis=1, keepdims=True)))
    expected = sum(weight[i, 1 - i] / weight[i].sum() for i in (0, 1)) / 4
    assert model.expected_accuracy_ == pytest.approx(expected)
    assert 0 < expected <= 0.25


def test_laplacian_score_reuses_larger_graph_and_matches_heat_oracle(monkeypatch):
    from believe14._core import graphs
    from believe14.neighbors import build_neighbor_graph

    rng = np.random.default_rng(76)
    X = rng.normal(size=(180, 8))
    k = 12
    distances = cdist(X, X, metric="sqeuclidean")
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    rows = np.arange(len(X))[:, None]
    W = np.zeros_like(distances)
    W[rows, neighbors] = np.exp(-0.35 * distances[rows, neighbors])
    W = np.maximum(W, W.T)
    D = np.diag(W.sum(axis=1))
    Z = X - np.average(X, axis=0, weights=D.diagonal())
    expected = (Z.T @ (D - W) @ Z).diagonal() / (Z.T @ D @ Z).diagonal()

    graph = build_neighbor_graph(X, n_neighbors=2 * k)

    def unexpected_search(*args):
        raise AssertionError("Cached graph fitting repeated the exact search.")

    monkeypatch.setattr(graphs, "exact_neighbors", unexpected_search)
    model = LaplacianScore(3, n_neighbors=k, weighting="heat", gamma=0.35)
    model.fit(X, neighbors=graph)
    np.testing.assert_allclose(model.scores_, expected, rtol=1e-13)
    np.testing.assert_array_equal(
        model.get_support(indices=True), np.sort(np.argsort(expected)[:3])
    )
