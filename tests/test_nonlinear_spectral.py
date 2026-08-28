"""Formula, geometry, and failure tests for spectral manifold methods."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.exceptions import SkipTestWarning
from sklearn.utils.estimator_checks import check_estimator

from believe14.nonlinear import (
    DiffusionMap,
    Isomap,
    KernelPCA,
    LaplacianEigenmaps,
    LocallyLinearEmbedding,
    LocalTangentSpaceAlignment,
)


def test_kernel_pca_sklearn_checks() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SkipTestWarning)
        check_estimator(KernelPCA(kernel="linear"))


def _pairwise_squared(X: np.ndarray) -> np.ndarray:
    differences = X[:, None, :] - X[None, :, :]
    return np.sum(differences * differences, axis=2)


def test_linear_kernel_pca_matches_centered_gram() -> None:
    rng = np.random.default_rng(20)
    X = rng.normal(size=(12, 2))
    model = KernelPCA(2, kernel="linear").fit(X)
    centered = X - X.mean(axis=0)
    np.testing.assert_allclose(
        model.embedding_ @ model.embedding_.T,
        centered @ centered.T,
        atol=2e-12,
    )
    np.testing.assert_allclose(model.transform(X), model.embedding_, atol=2e-12)


def test_linear_kernel_pca_is_stable_under_large_translation() -> None:
    rng = np.random.default_rng(201)
    centered = rng.normal(size=(30, 4))
    shifted = centered + np.array([1e8, -2e8, 3e8, -4e8])
    reference = KernelPCA(3, kernel="linear").fit(centered)
    translated = KernelPCA(3, kernel="linear").fit(shifted)
    reference_gram = reference.embedding_ @ reference.embedding_.T
    translated_scores = translated.transform(shifted)
    np.testing.assert_allclose(
        translated.embedding_ @ translated.embedding_.T,
        reference_gram,
        rtol=2e-6,
        atol=3e-7,
    )
    np.testing.assert_allclose(
        translated_scores @ translated_scores.T,
        reference_gram,
        rtol=2e-6,
        atol=3e-7,
    )


def test_precomputed_kernel_pca_cross_kernel_extension() -> None:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(15, 3))
    kernel = X @ X.T
    model = KernelPCA(3, kernel="precomputed").fit(kernel)
    np.testing.assert_allclose(model.transform(kernel), model.embedding_, atol=2e-12)
    with pytest.raises(ValueError, match="one column"):
        model.transform(np.ones((2, 3)))


def test_diffusion_operator_is_markov_and_nystrom_recovers_training() -> None:
    rng = np.random.default_rng(22)
    X = rng.normal(size=(20, 3))
    model = DiffusionMap(2, gamma=0.7, alpha=1.0, diffusion_time=2).fit(X)
    np.testing.assert_allclose(model.diffusion_operator_.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.transform(X), model.embedding_, atol=2e-11)


def test_isomap_on_a_line_recovers_geodesic_distances() -> None:
    X = np.array([0.0, 0.25, 0.9, 1.7, 2.05, 3.2, 4.8, 5.4])[:, None]
    model = Isomap(1, n_neighbors=2).fit(X)
    expected = np.abs(X - X.T)
    np.testing.assert_allclose(model.geodesic_distances_, expected)
    assert model.diagnostics_.numerical_rank == 1
    observed = np.sqrt(_pairwise_squared(model.embedding_))
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_isomap_rejects_zero_length_neighbor_edges() -> None:
    X = np.array([[0.0], [0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="duplicate observations"):
        Isomap(1, n_neighbors=1).fit(X)


@pytest.mark.parametrize(
    "estimator",
    [
        Isomap(1, n_neighbors=1),
        LocallyLinearEmbedding(1, n_neighbors=2),
        LaplacianEigenmaps(1, n_neighbors=1),
        LocalTangentSpaceAlignment(1, n_neighbors=2),
    ],
)
def test_graph_methods_reject_disconnected_graphs(estimator: object) -> None:
    X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
    with pytest.raises(ValueError, match="disconnected"):
        estimator.fit(X)  # type: ignore[attr-defined]


def test_lle_weights_are_affine_and_embedding_solves_alignment() -> None:
    theta = np.linspace(0.0, 1.5 * np.pi, 30)
    X = np.column_stack((np.cos(theta), np.sin(theta), theta / 3.0))
    model = LocallyLinearEmbedding(2, n_neighbors=6).fit(X)
    np.testing.assert_allclose(model.reconstruction_weights_.sum(axis=1), 1.0)
    assert model.diagnostics_.residual_norm is not None
    assert model.diagnostics_.residual_norm < 1e-10


def test_laplacian_embedding_is_degree_orthonormal() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    X = np.column_stack((np.cos(theta), np.sin(theta)))
    model = LaplacianEigenmaps(2, n_neighbors=4, gamma=2.0).fit(X)
    weighted_gram = model.embedding_.T @ (model.degree_[:, None] * model.embedding_)
    np.testing.assert_allclose(weighted_gram, np.eye(2), atol=2e-12)


def test_ltsa_is_translation_invariant_up_to_embedding_geometry() -> None:
    rng = np.random.default_rng(24)
    X = rng.normal(size=(30, 3))
    first = LocalTangentSpaceAlignment(2, n_neighbors=7).fit_transform(X)
    second = LocalTangentSpaceAlignment(2, n_neighbors=7).fit_transform(X + 3.0)
    np.testing.assert_allclose(
        _pairwise_squared(first), _pairwise_squared(second), atol=2e-10
    )


def test_transductive_graph_estimators_have_no_transform() -> None:
    estimators = (
        Isomap(),
        LocallyLinearEmbedding(),
        LaplacianEigenmaps(),
        LocalTangentSpaceAlignment(),
    )
    assert all(not hasattr(estimator, "transform") for estimator in estimators)
