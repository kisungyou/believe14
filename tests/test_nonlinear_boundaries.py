"""Public API boundary, failure, and convergence tests for nonlinear methods."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import NotFittedError

from believe14.nonlinear import (
    PHATE,
    TSNE,
    ClassicalMDS,
    DiffusionMap,
    FastMap,
    Isomap,
    KernelPCA,
    LaplacianEigenmaps,
    LocallyLinearEmbedding,
    LocalTangentSpaceAlignment,
    MetricMDS,
    SammonMapping,
)


@pytest.fixture
def feature_matrix() -> np.ndarray:
    return np.random.default_rng(101).normal(size=(14, 3))


@pytest.mark.parametrize(
    "estimator",
    [
        ClassicalMDS(),
        MetricMDS(),
        SammonMapping(),
        FastMap(),
        KernelPCA(),
        Isomap(),
        LocallyLinearEmbedding(),
        LaplacianEigenmaps(),
        DiffusionMap(),
        LocalTangentSpaceAlignment(),
        TSNE(),
        PHATE(),
    ],
)
def test_feature_names_require_a_successful_fit(estimator: object) -> None:
    with pytest.raises(NotFittedError):
        estimator.get_feature_names_out()  # type: ignore[attr-defined]


@pytest.mark.parametrize("estimator", [FastMap(), KernelPCA(), DiffusionMap()])
def test_inductive_transforms_require_a_successful_fit(estimator: object) -> None:
    with pytest.raises(NotFittedError):
        estimator.transform(np.ones((2, 2)))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "estimator",
    [
        ClassicalMDS(2),
        MetricMDS(2, max_iter=3),
        SammonMapping(2, max_iter=3),
        FastMap(2),
        KernelPCA(2, kernel="linear"),
        Isomap(2, n_neighbors=6),
        LocallyLinearEmbedding(2, n_neighbors=6),
        LaplacianEigenmaps(2, n_neighbors=6),
        DiffusionMap(2),
        LocalTangentSpaceAlignment(2, n_neighbors=6),
        TSNE(
            2,
            perplexity=4,
            init="random",
            early_exaggeration_iter=0,
            max_iter=3,
            random_state=3,
        ),
        PHATE(2, n_neighbors=4, decay=3.0, diffusion_time=2, mds_max_iter=3),
    ],
)
def test_fit_transform_and_output_names(
    estimator: object, feature_matrix: np.ndarray
) -> None:
    embedding = estimator.fit_transform(feature_matrix)  # type: ignore[attr-defined]
    names = estimator.get_feature_names_out()  # type: ignore[attr-defined]
    assert embedding.shape == (feature_matrix.shape[0], 2)
    np.testing.assert_array_equal(
        names,
        [f"{estimator.__class__.__name__.lower()}{index}" for index in range(2)],
    )


@pytest.mark.parametrize(
    "estimator",
    [
        ClassicalMDS(dissimilarity="invalid"),  # type: ignore[arg-type]
        MetricMDS(dissimilarity="invalid"),  # type: ignore[arg-type]
        SammonMapping(dissimilarity="invalid"),  # type: ignore[arg-type]
        FastMap(dissimilarity="invalid"),  # type: ignore[arg-type]
    ],
)
def test_distance_estimators_reject_unknown_dissimilarity(
    estimator: object, feature_matrix: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="dissimilarity must be"):
        estimator.fit(feature_matrix)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "matrix",
    [
        np.ones((3, 2)),
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[1.0, 1.0], [1.0, 0.0]]),
        np.array([[0.0, -1.0], [-1.0, 0.0]]),
        np.array([[0.0, np.nan], [np.nan, 0.0]]),
    ],
)
def test_distance_estimators_reject_malformed_precomputed_matrix(
    matrix: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        ClassicalMDS(1, dissimilarity="precomputed").fit(matrix)


@pytest.mark.parametrize(
    "estimator",
    [
        MetricMDS(1, dissimilarity="precomputed", max_iter=2),
        SammonMapping(1, dissimilarity="precomputed", max_iter=2),
    ],
)
def test_iterative_mds_accepts_valid_precomputed_distances(estimator: object) -> None:
    distances = squareform(pdist(np.arange(6.0)[:, None]))
    result = estimator.fit_transform(distances)  # type: ignore[attr-defined]
    assert result.shape == (6, 1)


@pytest.mark.parametrize(
    "estimator",
    [
        MetricMDS(init="invalid"),  # type: ignore[arg-type]
        SammonMapping(init="invalid"),  # type: ignore[arg-type]
    ],
)
def test_iterative_mds_rejects_unknown_initialization(
    estimator: object, feature_matrix: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="init must be"):
        estimator.fit(feature_matrix)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("estimator", "error"),
    [
        (MetricMDS(max_iter=1.5), TypeError),  # type: ignore[arg-type]
        (MetricMDS(max_iter=0), ValueError),
        (MetricMDS(tol=0.0), ValueError),
        (SammonMapping(max_iter=1.5), TypeError),  # type: ignore[arg-type]
        (SammonMapping(max_iter=0), ValueError),
        (SammonMapping(tol=0.0), ValueError),
    ],
)
def test_iterative_mds_rejects_invalid_solver_controls(
    estimator: object, error: type[Exception], feature_matrix: np.ndarray
) -> None:
    with pytest.raises(error):
        estimator.fit(feature_matrix)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "estimator",
    [
        MetricMDS(init="random", random_state=0, max_iter=1, tol=1e-30),
        SammonMapping(init="random", random_state=0, max_iter=1, tol=1e-30),
    ],
)
def test_iterative_mds_reports_iteration_exhaustion(
    estimator: object, feature_matrix: np.ndarray
) -> None:
    fitted = estimator.fit(feature_matrix)  # type: ignore[attr-defined]
    assert fitted.diagnostics_.converged is False
    assert fitted.diagnostics_.n_iter == 1
    assert fitted.diagnostics_.warnings


@pytest.mark.parametrize(
    ("pivot_iterations", "error"),
    [(True, TypeError), (0, ValueError)],
)
def test_fastmap_rejects_invalid_pivot_iterations(
    pivot_iterations: int, error: type[Exception], feature_matrix: np.ndarray
) -> None:
    with pytest.raises(error):
        FastMap(pivot_iterations=pivot_iterations).fit(feature_matrix)


def test_fastmap_reports_residual_rank_exhaustion() -> None:
    X = np.arange(7.0)[:, None]
    model = FastMap(2).fit(X)
    assert model.n_active_components_ == 1
    assert model.pivot_distances_[1] == 0.0
    assert model.pivot_indices_[1, 0] == model.pivot_indices_[1, 1]
    assert model.diagnostics_.warnings


def test_fastmap_retains_a_small_but_numerically_resolved_axis() -> None:
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0e-7], [1.0, 1.0e-7]])
    model = FastMap(2).fit(X)
    assert model.n_active_components_ == 2
    assert model.pivot_distances_[1] > 0.0
    assert model.pivot_indices_[1, 0] != model.pivot_indices_[1, 1]


def test_fastmap_fit_transform_rejects_extra_fit_parameters(
    feature_matrix: np.ndarray,
) -> None:
    with pytest.raises(TypeError, match="fit parameters"):
        FastMap().fit_transform(feature_matrix, sample_weight=np.ones(14))


@pytest.mark.parametrize(
    ("estimator", "error"),
    [
        (KernelPCA(kernel="invalid"), ValueError),  # type: ignore[arg-type]
        (KernelPCA(gamma=0.0), ValueError),
        (KernelPCA(degree=True), TypeError),  # type: ignore[arg-type]
        (KernelPCA(degree=0), ValueError),
        (KernelPCA(coef0=True), TypeError),  # type: ignore[arg-type]
        (KernelPCA(coef0=np.inf), ValueError),
    ],
)
def test_kernel_pca_rejects_invalid_kernel_parameters(
    estimator: KernelPCA, error: type[Exception], feature_matrix: np.ndarray
) -> None:
    with pytest.raises(error):
        estimator.fit(feature_matrix)


@pytest.mark.parametrize("kernel", ["rbf", "poly"])
def test_kernel_pca_feature_kernels_have_training_nystrom_identity(
    kernel: str, feature_matrix: np.ndarray
) -> None:
    model = KernelPCA(2, kernel=kernel, gamma=0.5, degree=2, coef0=0.2).fit(
        feature_matrix
    )
    np.testing.assert_allclose(model.transform(feature_matrix), model.embedding_)


@pytest.mark.parametrize(
    "matrix",
    [np.ones((3, 2)), np.array([[1.0, 0.0], [2.0, 1.0]])],
)
def test_kernel_pca_rejects_malformed_precomputed_kernel(
    matrix: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        KernelPCA(1, kernel="precomputed").fit(matrix)


def test_kernel_pca_rejects_indefinite_and_rank_deficient_kernels() -> None:
    indefinite = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="indefinite"):
        KernelPCA(1, kernel="precomputed").fit(indefinite)
    with pytest.raises(ValueError, match="positive numerical rank"):
        KernelPCA(1, kernel="precomputed").fit(np.ones((4, 4)))


@pytest.mark.parametrize(
    ("estimator", "error"),
    [
        (DiffusionMap(gamma=0.0), ValueError),
        (DiffusionMap(alpha=True), TypeError),  # type: ignore[arg-type]
        (DiffusionMap(alpha=-0.1), ValueError),
        (DiffusionMap(alpha=1.1), ValueError),
        (DiffusionMap(diffusion_time=True), TypeError),  # type: ignore[arg-type]
        (DiffusionMap(diffusion_time=-1), ValueError),
    ],
)
def test_diffusion_map_rejects_invalid_parameters(
    estimator: DiffusionMap, error: type[Exception], feature_matrix: np.ndarray
) -> None:
    with pytest.raises(error):
        estimator.fit(feature_matrix)


def test_diffusion_map_rejects_zero_spectral_axis_and_zero_query_density(
    feature_matrix: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="eigenvalue is numerically zero"):
        DiffusionMap(1).fit(np.zeros((5, 2)))
    model = DiffusionMap(1, gamma=1.0).fit(feature_matrix)
    with pytest.raises(FloatingPointError, match="zero kernel density"):
        model.transform(np.full((1, feature_matrix.shape[1]), 1e3))


def test_diffusion_time_zero_is_an_explicit_valid_convention(
    feature_matrix: np.ndarray,
) -> None:
    model = DiffusionMap(2, diffusion_time=0).fit(feature_matrix)
    np.testing.assert_allclose(model.embedding_, model.eigenvectors_)


@pytest.mark.parametrize(
    ("estimator", "error"),
    [
        (LocallyLinearEmbedding(regularization=0.0), ValueError),
        (LaplacianEigenmaps(weighting="invalid"), ValueError),  # type: ignore[arg-type]
        (LaplacianEigenmaps(gamma=0.0), ValueError),
    ],
)
def test_local_spectral_methods_reject_invalid_controls(
    estimator: object, error: type[Exception], feature_matrix: np.ndarray
) -> None:
    with pytest.raises(error):
        estimator.fit(feature_matrix)  # type: ignore[attr-defined]


def test_local_spectral_degeneracy_failures_are_explicit() -> None:
    with pytest.raises(ValueError, match="zero scatter"):
        LocallyLinearEmbedding(1, n_neighbors=2).fit(np.zeros((5, 2)))
    line = np.repeat(np.arange(8.0)[:, None], 2, axis=1)
    with pytest.raises(ValueError, match="insufficient tangent rank"):
        LocalTangentSpaceAlignment(2, n_neighbors=3).fit(line)


def test_binary_laplacian_weighting_is_supported(
    feature_matrix: np.ndarray,
) -> None:
    model = LaplacianEigenmaps(2, n_neighbors=6, weighting="binary").fit(feature_matrix)
    selected = model.affinity_matrix_[model.affinity_matrix_ > 0.0]
    np.testing.assert_array_equal(selected, 1.0)


@pytest.mark.parametrize(
    ("estimator", "error"),
    [
        (TSNE(perplexity=3, early_exaggeration=0.5), ValueError),
        (TSNE(perplexity=3, max_iter=True), TypeError),  # type: ignore[arg-type]
        (TSNE(perplexity=3, max_iter=0), ValueError),
        (
            TSNE(perplexity=3, early_exaggeration_iter=True),  # type: ignore[arg-type]
            TypeError,
        ),
        (TSNE(perplexity=3, early_exaggeration_iter=-1), ValueError),
        (TSNE(perplexity=3, tol=0.0), ValueError),
        (TSNE(perplexity=3, init="invalid"), ValueError),  # type: ignore[arg-type]
    ],
)
def test_tsne_rejects_invalid_optimization_controls(
    estimator: TSNE, error: type[Exception], feature_matrix: np.ndarray
) -> None:
    with pytest.raises(error):
        estimator.fit(feature_matrix)


def test_tsne_pca_initialization_rank_and_degeneracy_fail_explicitly() -> None:
    line = np.arange(6.0)[:, None]
    with pytest.raises(ValueError, match="rank bound"):
        TSNE(2, perplexity=2, init="pca").fit(line)
    with pytest.raises(ValueError, match="degenerate"):
        TSNE(1, perplexity=4, init="pca").fit(np.zeros((5, 2)))


def test_tsne_entropy_endpoints_and_nonconvergence_are_truthful(
    feature_matrix: np.ndarray,
) -> None:
    maximum = TSNE(
        2,
        perplexity=feature_matrix.shape[0] - 1,
        init="random",
        early_exaggeration_iter=1,
        max_iter=1,
        random_state=2,
    ).fit(feature_matrix)
    np.testing.assert_allclose(
        maximum.conditional_probabilities_[
            ~np.eye(feature_matrix.shape[0], dtype=bool)
        ],
        1.0 / (feature_matrix.shape[0] - 1),
    )
    assert maximum.diagnostics_.converged is False
    assert maximum.diagnostics_.warnings

    minimum = TSNE(
        2,
        perplexity=1,
        init="random",
        early_exaggeration_iter=0,
        max_iter=1,
        random_state=2,
    ).fit(feature_matrix)
    np.testing.assert_array_equal(
        np.count_nonzero(minimum.conditional_probabilities_, axis=1), 1
    )


@pytest.mark.parametrize(
    ("estimator", "error"),
    [
        (PHATE(decay=0.0), ValueError),
        (PHATE(potential_floor=0.0), ValueError),
        (PHATE(potential_floor=1.0), ValueError),
        (PHATE(diffusion_time=True), TypeError),  # type: ignore[arg-type]
        (PHATE(diffusion_time=0), ValueError),
        (PHATE(mds_max_iter=1.5), TypeError),  # type: ignore[arg-type]
        (PHATE(mds_max_iter=0), ValueError),
        (PHATE(mds_tol=0.0), ValueError),
    ],
)
def test_phate_rejects_invalid_numerical_controls(
    estimator: PHATE, error: type[Exception], feature_matrix: np.ndarray
) -> None:
    with pytest.raises(error):
        estimator.fit(feature_matrix)


def test_phate_rejects_numerically_disconnected_affinity() -> None:
    clusters = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])[:, None]
    with pytest.raises(ValueError, match="numerically disconnected"):
        PHATE(1, n_neighbors=1, decay=4.0, diffusion_time=2).fit(clusters)


def test_phate_reports_floor_regularization_and_iteration_exhaustion(
    feature_matrix: np.ndarray,
) -> None:
    model = PHATE(
        2,
        n_neighbors=3,
        decay=3.0,
        diffusion_time=2,
        potential_floor=0.1,
        mds_max_iter=1,
        mds_tol=1e-30,
    ).fit(feature_matrix)
    assert model.diagnostics_.converged is False
    assert model.diagnostics_.n_iter == 1
    assert any("potential_floor" in item for item in model.diagnostics_.warnings)
    assert any("mds_max_iter" in item for item in model.diagnostics_.warnings)
