"""Independent classical-scaling and geodesic extension checks."""

import numpy as np
import pytest
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import clone
from sklearn.utils import get_tags

from believe14.neighbors import build_neighbor_graph
from believe14.nonlinear import ClassicalMDS, Isomap, LandmarkIsomap, LandmarkMDS


def test_mds_feature_projection_equals_landmark_principal_plane():
    rng = np.random.default_rng(78)
    X = rng.normal(size=(23, 5))
    query = rng.normal(size=(6, 5))
    idx = [1, 4, 8, 13, 16, 18, 22]
    model = LandmarkMDS(2, landmark_indices=idx).fit(X)
    center = X[idx].mean(axis=0)
    _, singular, vh = np.linalg.svd(X[idx] - center, full_matrices=False)
    reference = (np.vstack((X, query)) - center) @ vh[:2].T
    embedded = np.vstack((model.embedding_, model.transform(query)))
    np.testing.assert_allclose(
        embedded @ embedded.T, reference @ reference.T, atol=3e-14
    )
    np.testing.assert_allclose(model.eigenvalues_[:2], singular[:2] ** 2)
    np.testing.assert_allclose(model.transform(X), model.embedding_, atol=3e-14)
    np.testing.assert_array_equal(clone(model).landmark_indices, idx)


def test_mds_precomputed_feature_equivalence_and_all_landmark_limit():
    rng = np.random.default_rng(632)
    X = rng.normal(size=(18, 4))
    query = rng.normal(size=(4, 4))
    idx = [0, 3, 7, 9, 12, 15, 17]
    D = squareform(pdist(X))
    pre = LandmarkMDS(3, landmark_indices=idx, dissimilarity="precomputed").fit(D)
    raw = LandmarkMDS(3, landmark_indices=idx).fit(X)
    extended = np.vstack((pre.embedding_, pre.transform(cdist(query, X))))
    feature = np.vstack((raw.embedding_, raw.transform(query)))
    np.testing.assert_allclose(pdist(extended), pdist(feature), atol=2e-13)
    np.testing.assert_allclose(pre.transform(D), pre.embedding_, atol=1e-14)
    assert get_tags(pre).input_tags.pairwise and not get_tags(raw).input_tags.pairwise
    full = LandmarkMDS(3, n_landmarks=len(X)).fit_transform(X)
    np.testing.assert_allclose(pdist(full), pdist(ClassicalMDS(3).fit_transform(X)))


def test_mds_precomputed_non_euclidean_landmarks_and_query_formula():
    D = np.array(
        [
            [0.0, 1.0, 3.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [3.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    q = np.array([[2.0, 1.2, 1.7, 0.9]])
    model = LandmarkMDS(2, n_landmarks=4, dissimilarity="precomputed").fit(D)
    assert model.eigenvalues_[-1] < 0
    assert model.diagnostics_.warnings
    # Cross Gram products against the fitted axes give a sign-invariant oracle.
    H = np.eye(4) - np.ones((4, 4)) / 4
    _, vectors = np.linalg.eigh(-H @ (D**2) @ H / 2)
    keep = vectors[:, -2:]
    b = (np.mean(D**2, axis=0) - q**2) @ H / 2
    expected = b @ keep @ keep.T
    np.testing.assert_allclose(
        model.transform(q) @ model.embedding_.T, expected, atol=2e-14
    )
    for invalid in (q[:, :2], -q, q * np.nan):
        with pytest.raises(ValueError):
            model.transform(invalid)


def test_precomputed_distant_query_preserves_squared_distance_contrast():
    d = 2**52
    model = LandmarkMDS(1, n_landmarks=2, dissimilarity="precomputed")
    model.fit([[0.0, 3.0], [3.0, 0.0]])
    result = model.transform([[d, d + 1]])[0, 0]
    expected = (d**2 - (d + 1) ** 2) / 6
    np.testing.assert_allclose(abs(result), abs(expected), rtol=1e-15)
    model.fit([[0.0, 1e-150], [1e-150, 0.0]])
    np.testing.assert_allclose(model.transform([[1e300, 1e300]]), 0.0, atol=0.0)


def test_mds_feature_extrapolation_does_not_square_distant_points():
    landmarks = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    model = LandmarkMDS(2, n_landmarks=3).fit(landmarks)
    query = np.array([[1e200, -1e200]])
    got = model.transform(query)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(np.linalg.norm(got / 1e200), np.sqrt(2.0))


def test_isomap_landmark_paths_query_projection_and_graph_reuse():
    rng = np.random.default_rng(40)
    X = rng.normal(size=(25, 3))
    idx = [1, 5, 12, 14, 19, 22]
    k = 5
    graph = build_neighbor_graph(X, n_neighbors=k + 2)
    fitted = LandmarkIsomap(2, n_neighbors=k, landmark_indices=idx)
    embedding = fitted.fit_transform(X, neighbors=graph)
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    near = np.argsort(D, kind="stable", axis=1)[:, :k]
    A = np.full_like(D, np.inf)
    np.put_along_axis(A, near, np.take_along_axis(D, near, axis=1), axis=1)
    geodesics = shortest_path(np.minimum(A, A.T), directed=False)
    np.testing.assert_allclose(fitted.landmark_distances_, geodesics[:, idx])
    np.testing.assert_allclose(fitted.transform(X), embedding, atol=4e-14)
    # Independent projected cross-Gram formula for the grafted distances.
    query = rng.normal(size=(3, 3))
    edges = cdist(query, X)
    selected = np.argsort(edges, kind="stable", axis=1)[:, :k]
    qpaths = np.min(
        np.take_along_axis(edges, selected, axis=1)[:, :, None]
        + geodesics[:, idx][selected],
        axis=1,
    )
    H = np.eye(len(idx)) - 1 / len(idx)
    squared = geodesics[np.ix_(idx, idx)] ** 2
    _, V = np.linalg.eigh(-H @ squared @ H / 2)
    b = (squared.mean(axis=0) - qpaths**2) @ H / 2
    expected = b @ V[:, -2:] @ V[:, -2:].T
    np.testing.assert_allclose(
        fitted.transform(query) @ embedding[idx].T, expected, atol=1e-13
    )
    np.testing.assert_allclose(fitted.transform(query[:1]), fitted.transform(query)[:1])
    np.testing.assert_allclose(clone(fitted).fit_transform(X), embedding)
    assert fitted.landmark_distances_.shape == (len(X), len(idx))
    assert not hasattr(fitted, "geodesic_distances_")
    with pytest.raises(ValueError, match="different observations"):
        fitted.fit(X[::-1], neighbors=graph)


def test_isomap_all_landmarks_limit_and_line_interpolation():
    x = np.linspace(0, 1.5 * np.pi, 29)
    X = np.column_stack((np.cos(x), np.sin(x)))
    landmark = LandmarkIsomap(1, n_landmarks=len(X), n_neighbors=2).fit(X)
    reference = Isomap(1, n_neighbors=2).fit_transform(X)
    np.testing.assert_allclose(pdist(landmark.embedding_), pdist(reference), atol=1e-13)
    line = np.array([[0.0], [1.0], [2.1], [3.4], [4.8]])
    reduced = LandmarkIsomap(1, n_neighbors=2, landmark_indices=[0, 2, 4]).fit(line)
    np.testing.assert_allclose(pdist(reduced.embedding_), pdist(line))
    d = cdist(reduced.transform([[1.4]]), reduced.embedding_)
    np.testing.assert_allclose(d, cdist([[1.4]], line))


@pytest.mark.parametrize("cls", [LandmarkMDS, LandmarkIsomap])
@pytest.mark.parametrize(
    "params",
    [
        {"n_landmarks": 1},
        {"n_landmarks": True},
        {"landmark_indices": [0.0, 1.0, 2.0]},
        {"landmark_indices": [0, 0, 2]},
        {"landmark_indices": [-1, 0, 2]},
        {"landmark_indices": [0, 2, 15]},
        {"landmark_indices": [[0, 1, 2]]},
        {"n_landmarks": 3, "landmark_indices": [0, 1, 2]},
    ],
)
def test_landmark_selection_boundaries(cls, params):
    X = np.arange(12.0).reshape(-1, 1)
    with pytest.raises((TypeError, ValueError)):
        cls(n_components=1, **params).fit(X)


def test_default_selection_rank_and_graph_boundaries():
    X = np.arange(12.0).reshape(-1, 1)
    for cls in (LandmarkMDS, LandmarkIsomap):
        fitted = cls(1).fit(X)
        np.testing.assert_array_equal(fitted.landmark_indices_, [0, 3, 7, 11])
        assert fitted.n_landmarks_ == 4
        with pytest.raises(ValueError, match="positive numerical rank"):
            cls(2).fit(X)
    with pytest.raises(ValueError, match="distinct observations"):
        LandmarkIsomap(1, n_neighbors=2).fit([[0.0], [1.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="disconnected"):
        LandmarkIsomap(1, n_neighbors=1).fit([[0.0], [1.0], [10.0], [11.0]])


@pytest.mark.parametrize("n_neighbors", [1, 2, 3])
def test_isomap_distant_queries_preserve_small_graph_path_offsets(n_neighbors):
    line = np.array([[0.0], [1.0], [2.5], [4.75]])
    model = LandmarkIsomap(1, n_neighbors=n_neighbors, landmark_indices=[0, 3]).fit(
        line
    )
    query = np.array([[1e18], [-1e18], [1e300], [-1e300]])
    # Resolve the eigenvector sign from the fitted line orientation.
    sign = np.sign(model.embedding_[-1, 0] - model.embedding_[0, 0])
    np.testing.assert_allclose(
        model.transform(query) * sign, query - 4.75 / 2, atol=0.0, rtol=1e-14
    )
