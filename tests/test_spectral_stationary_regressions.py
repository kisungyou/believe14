"""Independent constraints and distance identities for graph spectral embeddings."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg
from scipy.spatial.distance import pdist, squareform

from believe14.nonlinear import DiffusionMap, LaplacianEigenmaps


def test_laplacian_excludes_constant_when_connected_graph_is_weakly_coupled() -> None:
    features = np.array([0.0, 0.1, 10.0, 10.1])[:, None]
    model = LaplacianEigenmaps(1, n_neighbors=3, gamma=1.0).fit(features)
    off_diagonal = ~np.eye(len(features), dtype=bool)
    assert np.all(model.affinity_matrix_[off_diagonal] > 0.0)
    weights = model.degree_ / model.degree_.sum()
    np.testing.assert_allclose(weights @ model.embedding_, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        model.embedding_.T @ (model.degree_[:, None] * model.embedding_),
        np.eye(1),
        atol=1e-13,
    )
    # Each pair has degree exp(-.01); the normalized mean-zero contrast has
    # coordinates +/-1/sqrt(4*degree), hence gap 1/sqrt(degree).
    expected_gap = np.exp(0.005)
    observed_gap = abs(model.embedding_[0, 0] - model.embedding_[2, 0])
    assert observed_gap == pytest.approx(expected_gap, abs=1e-13)
    assert model.diagnostics_.numerical_rank == 2
    assert model.diagnostics_.residual_norm < 1e-12


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_diffusion_excludes_stationary_mode_in_weakly_connected_kernel(
    alpha: float,
) -> None:
    features = np.array([0.0, 0.1, 10.0, 10.1])[:, None]
    model = DiffusionMap(1, gamma=1.0, alpha=alpha).fit(features)
    stationary = model.diffusion_degree_ / model.diffusion_degree_.sum()
    assert np.all(model.diffusion_operator_ > 0.0)
    np.testing.assert_allclose(stationary @ model.eigenvectors_, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        model.eigenvectors_.T @ (stationary[:, None] * model.eigenvectors_),
        np.eye(1),
        atol=1e-13,
    )
    # The two equally weighted clusters have stationary-normalized contrast
    # coordinates -1 and +1. Their nontrivial eigenvalue rounds to one.
    observed_gap = abs(model.embedding_[0, 0] - model.embedding_[2, 0])
    assert observed_gap == pytest.approx(2.0, abs=1e-13)
    np.testing.assert_allclose(model.transform(features), model.embedding_, atol=1e-12)
    assert model.diagnostics_.numerical_rank == 3
    assert model.diagnostics_.residual_norm < 1e-12


@pytest.mark.parametrize("diffusion_time", [0, 1, 3])
@pytest.mark.parametrize("geometry", ["ordinary", "equal_clusters", "unequal_clusters"])
def test_all_diffusion_coordinates_reproduce_transition_distance(
    diffusion_time: int, geometry: str
) -> None:
    if geometry == "equal_clusters":
        # Three almost isolated pairs yield a repeated numerical eigenvalue one.
        features = np.array([0.0, 0.1, 10.0, 10.1, 20.0, 20.1])[:, None]
    elif geometry == "unequal_clusters":
        # Unequal cluster sizes also exercise nonuniform stationary weights.
        features = np.array([0.0, 0.5, 1.0, 10.0, 10.5, 20.0, 20.5, 21.0, 21.5])[
            :, None
        ]
    else:
        features = np.random.default_rng(103).normal(size=(8, 3))
    model = DiffusionMap(
        len(features) - 1, gamma=1.0, alpha=0.7, diffusion_time=diffusion_time
    ).fit(features)
    stationary = model.diffusion_degree_ / model.diffusion_degree_.sum()
    transition = np.linalg.matrix_power(model.diffusion_operator_, diffusion_time)
    expected = squareform(pdist(transition / np.sqrt(stationary)))
    observed = squareform(pdist(model.embedding_))
    np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(stationary @ model.embedding_, 0.0, atol=1e-12)
    np.testing.assert_allclose(model.transform(features), model.embedding_, atol=2e-12)
    assert model.diagnostics_.numerical_rank == len(features) - 1


def test_laplacian_weak_cluster_geometry_survives_permutation() -> None:
    features = np.array([0.0, 0.1, 10.0, 10.1, 20.0, 20.1])[:, None]
    permutation = np.array([5, 2, 0, 3, 1, 4])
    first = LaplacianEigenmaps(2, n_neighbors=5).fit(features)
    second = LaplacianEigenmaps(2, n_neighbors=5).fit(features[permutation])
    recovered = second.embedding_[np.argsort(permutation)]
    np.testing.assert_allclose(pdist(first.embedding_), pdist(recovered), atol=1e-12)
    np.testing.assert_allclose(first.degree_ @ first.embedding_, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        first.embedding_.T @ (first.degree_[:, None] * first.embedding_),
        np.eye(2),
        atol=1e-12,
    )


@pytest.mark.parametrize("weighting", ["binary", "heat"])
def test_laplacian_ordinary_embedding_matches_generalized_eigenproblem(
    weighting: str,
) -> None:
    features = np.random.default_rng(108).normal(size=(18, 3))
    model = LaplacianEigenmaps(3, n_neighbors=7, weighting=weighting).fit(features)
    degree = np.diag(model.degree_)
    laplacian = degree - model.affinity_matrix_
    values, vectors = linalg.eigh(laplacian, degree)
    np.testing.assert_allclose(model.eigenvalues_, values[1:4], atol=1e-12)
    np.testing.assert_allclose(
        pdist(model.embedding_), pdist(vectors[:, 1:4]), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        laplacian @ model.embedding_,
        (degree @ model.embedding_) * model.eigenvalues_,
        atol=1e-12,
    )


def test_graph_spectral_embeddings_handle_two_observations() -> None:
    features = np.array([[0.0], [0.5]])
    laplacian = LaplacianEigenmaps(1, n_neighbors=1).fit(features)
    np.testing.assert_allclose(
        laplacian.degree_ @ laplacian.embedding_, 0.0, atol=1e-13
    )
    np.testing.assert_allclose(laplacian.eigenvalues_, [2.0], atol=1e-13)
    diffusion = DiffusionMap(1).fit(features)
    np.testing.assert_allclose(
        diffusion.transform(features), diffusion.embedding_, atol=1e-13
    )
    np.testing.assert_allclose(diffusion.embedding_.mean(axis=0), 0.0, atol=1e-13)
