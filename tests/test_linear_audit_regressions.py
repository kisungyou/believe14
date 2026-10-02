"""Independent regression checks for leading PLS modes and affine restoration."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import hadamard

from believe14._core.linalg import restore_centering
from believe14.linear import (
    PCA,
    FactorAnalysis,
    FastICA,
    PLSRegression,
    ProbabilisticPCA,
)


def _orthogonal_design() -> tuple[np.ndarray, np.ndarray]:
    contrasts = np.vstack((hadamard(8).astype(float), np.zeros((1, 8))))
    X = contrasts[:, 1:3]
    Y = np.column_stack((0.5 * X[:, 1] + np.sqrt(0.75) * contrasts[:, 3], X[:, 0]))
    return X, Y


def _svd_pls_prediction(
    X: np.ndarray, Y: np.ndarray, components: int, *, scale: bool
) -> np.ndarray:
    """Independent PLS2 oracle using direct SVD and regression deflation."""

    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    x_scale = X.std(axis=0, ddof=1) if scale else np.ones(X.shape[1])
    y_scale = Y.std(axis=0, ddof=1) if scale else np.ones(Y.shape[1])
    X_initial = (X - x_mean) / x_scale
    Xr, Yr = X_initial.copy(), (Y - y_mean) / y_scale
    weights, x_loadings, y_loadings = [], [], []
    for _ in range(components):
        left, _, _ = np.linalg.svd(Xr.T @ Yr, full_matrices=False)
        w = left[:, 0]
        t = Xr @ w
        p, q = Xr.T @ t / (t @ t), Yr.T @ t / (t @ t)
        weights.append(w)
        x_loadings.append(p)
        y_loadings.append(q)
        Xr -= np.outer(t, p)
        Yr -= np.outer(t, q)
    W, P, Q = (
        np.column_stack(weights),
        np.column_stack(x_loadings),
        np.column_stack(y_loadings),
    )
    coefficients = W @ np.linalg.solve(P.T @ W, Q.T)
    return (X_initial @ coefficients) * y_scale + y_mean


@pytest.mark.parametrize("contamination", [0.0, 1e-12, 1e-8, 1e-4])
@pytest.mark.parametrize("scale", [False, True])
def test_pls_extracts_dominant_mode_with_tiny_initial_projection(
    contamination: float, scale: bool
) -> None:
    X, Y = _orthogonal_design()
    Y[:, 0] += contamination * X[:, 0]
    model = PLSRegression(1, scale=scale, tol=1e-12).fit(X, Y)
    Xs = (X - X.mean(axis=0)) / model.x_scale_
    Ys = (Y - Y.mean(axis=0)) / model.y_scale_
    cross = Xs.T @ Ys
    leading = np.linalg.svd(cross, compute_uv=False)[0]
    attained = np.linalg.norm(cross.T @ model.x_weights_[:, 0])
    assert model.diagnostics_.converged
    assert_allclose(attained, leading, rtol=1e-12)
    assert_allclose(
        model.predict(X), _svd_pls_prediction(X, Y, 1, scale=scale), atol=1e-12
    )


@pytest.mark.parametrize("scale", [False, True])
def test_pls_target_permutation_and_orthogonal_first_target(scale: bool) -> None:
    X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    Y = np.column_stack((X[:, 0] * X[:, 1], X[:, 0]))
    expected = np.column_stack((np.zeros(len(X)), X[:, 0]))
    for order in ([0, 1], [1, 0]):
        model = PLSRegression(1, scale=scale).fit(X, Y[:, order])
        assert model.diagnostics_.converged
        assert_allclose(model.predict(X), expected[:, order], atol=1e-14)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("seed", [90, 91, 92])
def test_pls_multiple_components_match_direct_svd_deflation(
    scale: bool, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 6))
    Y = X @ rng.normal(size=(6, 3)) + 0.2 * rng.normal(size=(40, 3))
    expected = _svd_pls_prediction(X, Y, 3, scale=scale)
    for order in ([0, 1, 2], [2, 0, 1]):
        model = PLSRegression(3, scale=scale, tol=1e-12).fit(X, Y[:, order])
        assert model.diagnostics_.converged
        assert_allclose(model.predict(X), expected[:, order], rtol=1e-10, atol=1e-11)


def test_pls_separates_nearly_tied_leading_singular_values() -> None:
    contrasts = np.vstack((hadamard(8).astype(float), np.zeros((1, 8))))
    X = contrasts[:, 1:3]
    first, second = 0.9, 0.9 - 1e-8
    Y = np.column_stack(
        (
            first * contrasts[:, 1] + np.sqrt(1 - first**2) * contrasts[:, 3],
            second * contrasts[:, 2] + np.sqrt(1 - second**2) * contrasts[:, 4],
        )
    ) @ np.array([[0.6, -0.8], [0.8, 0.6]])
    model = PLSRegression(1, tol=1e-12).fit(X, Y)
    assert model.diagnostics_.converged
    assert_allclose(abs(model.x_weights_[0, 0]), 1.0, atol=1e-12)
    assert abs(model.x_weights_[1, 0]) < 1e-6
    assert_allclose(
        model.predict(X), _svd_pls_prediction(X, Y, 1, scale=True), atol=1e-7
    )


@pytest.mark.parametrize("factor", [1e-70, 1e70])
def test_pls_cross_covariance_start_preserves_supported_scaling(factor: float) -> None:
    X, Y = _orthogonal_design()
    model = PLSRegression(1, scale=False).fit(X * factor, Y * factor)
    assert_allclose(
        model.predict(X * factor) / factor,
        _svd_pls_prediction(X, Y, 1, scale=False),
        atol=1e-12,
    )


def test_pls_retains_iteration_budget_and_true_zero_cross_covariance_failure() -> None:
    X, Y = _orthogonal_design()
    model = PLSRegression(1, max_iter=1).fit(X, Y)
    assert not model.diagnostics_.converged
    assert_array_equal(model.n_iter_, [1])
    assert model.diagnostics_.warnings
    contrasts = hadamard(4).astype(float)
    with pytest.raises(ValueError, match="zero cross-covariance"):
        PLSRegression(1, scale=False).fit(contrasts[:, 1:2], contrasts[:, 2:3])


@pytest.mark.parametrize(
    "model", [PCA(1), PCA(1, whiten=True), FastICA(1, random_state=0)]
)
def test_exact_inverse_preserves_representable_variation_at_large_offsets(
    model: object,
) -> None:
    X = np.array([[1e16], [1e16 + 2]])
    fitted = model.fit(X)
    assert_array_equal(fitted.inverse_transform(fitted.transform(X)), X)


@pytest.mark.parametrize("estimator", [FactorAnalysis, ProbabilisticPCA])
def test_gaussian_latent_reconstruction_restores_unrounded_mean(
    estimator: type,
) -> None:
    X = np.column_stack((1e16 + 2 * np.arange(4), [-1.0, 1.0, 1.0, -1.0]))
    model = estimator(1).fit(X)
    scores = np.array([[-1.0 / model.loadings_[0, 0]], [1.0 / model.loadings_[0, 0]]])
    expected = np.column_stack(([1e16 + 2, 1e16 + 4], [0.0, 0.0]))
    assert_array_equal(model.inverse_transform(scores), expected)


def test_pls_prediction_restores_unrounded_target_mean() -> None:
    X = np.array([[-1.0], [1.0]])
    y = np.array([1e16, 1e16 + 2])
    model = PLSRegression(1).fit(X, y)
    assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("estimator", [PCA, FastICA, FactorAnalysis, ProbabilisticPCA])
def test_inverse_restoration_agrees_with_ordinary_affine_map(estimator: type) -> None:
    X = np.random.default_rng(810).normal(size=(30, 4)) + 3
    model = estimator(2).fit(X)
    scores = np.random.default_rng(811).normal(size=(8, 2))
    if isinstance(model, PCA):
        reconstruction = scores @ model.components_
    elif isinstance(model, FastICA):
        reconstruction = scores @ model.mixing_.T
    else:
        reconstruction = scores @ model.loadings_.T
    assert_allclose(
        model.inverse_transform(scores), reconstruction + X.mean(axis=0), atol=1e-14
    )


def test_restoration_matches_decimal_through_cancellation_and_overflow() -> None:
    maximum = np.finfo(float).max
    tiny = np.nextafter(0.0, 1.0)
    triples = [
        (1e-300, 1e308, -1e308),
        (tiny, maximum, -maximum),
        (maximum, maximum, -maximum),
        (-maximum, -maximum, maximum),
        (maximum, np.spacing(maximum / 2), -maximum),
        (1.0, 1.0, 1e16),
        (-1.0, 1.0, 1e16),
        (0.0, 0.0, 0.0),
    ]
    values, offsets, references = np.asarray(triples).T
    with localcontext() as context:
        context.prec = 1200
        expected = [
            float(sum((Decimal.from_float(t) for t in triple), Decimal(0)))
            for triple in triples
        ]
    # One overflowing element must not erase a small residual in another row.
    assert_array_equal(restore_centering(values, references, offsets), expected)
    for triple, result in zip(triples, expected, strict=True):
        a, b, c = triple
        assert_array_equal(
            restore_centering(np.array([a]), np.array([c]), np.array([b])), [result]
        )


def test_restoration_rejects_truly_unrepresentable_results() -> None:
    maximum = np.finfo(float).max
    with pytest.raises(FloatingPointError, match="not representable"):
        restore_centering(np.array([maximum]), np.array([maximum]), np.array([0.0]))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_restoration_rejects_nonfinite_summands(value: float) -> None:
    with pytest.raises(FloatingPointError, match="finite summands"):
        restore_centering(np.array([value]), np.array([0.0]), np.array([0.0]))
