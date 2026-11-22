"""Independent neighborhood rankings and public graph reuse checks."""

import pickle

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

from believe14.metrics import continuity, trustworthiness
from believe14.neighbors import build_neighbor_graph
from believe14.nonlinear import (
    Isomap,
    LaplacianEigenmaps,
    LocallyLinearEmbedding,
    LocalTangentSpaceAlignment,
)


def test_neighbor_graph_stable_prefix_snapshot_and_serialization():
    x = np.array([[0.0], [1.0], [-1.0], [0.0], [2.0], [-2.0]])
    graph = build_neighbor_graph(x, 4)
    distances = cdist(x, x)
    np.fill_diagonal(distances, np.inf)
    expected = np.argsort(distances, axis=1, kind="stable")
    for k in (1, 3, 4):
        d, idx = pickle.loads(pickle.dumps(graph)).query(x.tolist(), k)
        np.testing.assert_array_equal(idx, expected[:, :k])
        np.testing.assert_array_equal(d, np.take_along_axis(distances, idx, axis=1))
        for a in (d, idx):
            with pytest.raises(ValueError):
                a.setflags(write=True)
    with pytest.raises(ValueError, match="more neighbors"):
        graph.query(x, 5)
    with pytest.raises(ValueError, match="different observations"):
        graph.query(x[::-1], 3)
    x[0] = 100
    with pytest.raises(ValueError, match="different observations"):
        graph.query(x, 2)


@pytest.mark.parametrize(
    "estimator",
    [Isomap, LocallyLinearEmbedding, LaplacianEigenmaps, LocalTangentSpaceAlignment],
)
def test_manifold_fits_reuse_exact_neighborhoods(estimator):
    x = np.random.default_rng(14).normal(size=(26, 4))
    graph = build_neighbor_graph(x, 13)
    direct = estimator(n_neighbors=8).fit_transform(x)
    cached = estimator(n_neighbors=8).fit_transform(x, neighbors=graph)
    np.testing.assert_allclose(cached @ cached.T, direct @ direct.T, atol=1e-11)
    with pytest.raises(ValueError, match="different observations"):
        estimator(n_neighbors=8).fit(x[::-1], neighbors=graph)


def test_fidelity_against_sklearn_in_both_directions():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(24, 5))
    z = rng.normal(size=(24, 2))
    for k in (1, 4, 11):
        assert trustworthiness(x, z, n_neighbors=k) == pytest.approx(
            sklearn_trustworthiness(x, z, n_neighbors=k)
        )
        assert continuity(x, z, n_neighbors=k) == pytest.approx(
            sklearn_trustworthiness(z, x, n_neighbors=k)
        )
        assert continuity(x, x, n_neighbors=k) == 1.0
    assert continuity(x, z) != trustworthiness(x, z)


def test_literal_tied_rank_penalty_and_domains():
    x = np.array([[0], [0], [1], [2], [4]])
    z = np.array([[3], [0], [1], [2], [4]])
    ranks = []
    for row in range(5):
        others = [j for j in range(5) if j != row]
        order = sorted(others, key=lambda j: abs(int(x[j, 0]) - int(x[row, 0])))
        neighbor = min(others, key=lambda j: abs(int(z[j, 0]) - int(z[row, 0])))
        ranks.append(1 + order.index(neighbor))
    assert trustworthiness(x, z, n_neighbors=1) == pytest.approx(
        1 - 2 * sum(r - 1 for r in ranks) / 30
    )
    with pytest.raises(ValueError, match="same observations"):
        continuity(x[:-1], z, n_neighbors=1)
    with pytest.raises(ValueError, match="half"):
        trustworthiness(x, z, n_neighbors=3)


def test_graph_recognizes_equal_values_across_signed_zero():
    X = np.array([[0.0, -0.0], [1.0, 2.0], [-1.0, 1.0]])
    graph = build_neighbor_graph(X, n_neighbors=2)
    query = X.copy()
    query[0] = [-0.0, +0.0]
    distances, indices = graph.query(query, n_neighbors=2)
    np.testing.assert_array_equal(indices, graph.indices)
    np.testing.assert_array_equal(distances, graph.distances)
