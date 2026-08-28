"""Stable linear-algebra helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


@dataclass(frozen=True, slots=True)
class SymmetricEigenResult:
    values: NDArray[np.float64]
    vectors: NDArray[np.float64]
    residual_norm: float


def stable_center(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a stable column mean and centered matrix.

    A reference shift preserves small, exactly representable variations around a
    large common offset.  The scaled fallback also covers data spanning opposite
    float64 extremes, where forming the reference differences would overflow.
    """

    reference = np.asarray(X[0], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            offsets = X - reference
        offset_scale = float(np.max(np.abs(offsets), initial=0.0))
        if offset_scale == 0.0:
            return reference.copy(), np.zeros_like(X, dtype=np.float64)
        offset_mean = (
            np.mean(offsets / offset_scale, axis=0, dtype=np.float64) * offset_scale
        )
        with np.errstate(over="raise", invalid="raise"):
            mean = reference + offset_mean
            centered = offsets - offset_mean
        return np.asarray(mean, dtype=np.float64), np.asarray(
            centered, dtype=np.float64
        )
    except FloatingPointError:
        scale = float(np.max(np.abs(X), initial=0.0))
        if scale == 0.0:
            return (
                np.zeros(X.shape[1], dtype=np.float64),
                np.zeros_like(X, dtype=np.float64),
            )
        scaled = X / scale
        scaled_reference = scaled[0]
        scaled_offsets = scaled - scaled_reference
        scaled_offset_mean = np.mean(scaled_offsets, axis=0, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            mean = (scaled_reference + scaled_offset_mean) * scale
            centered = (scaled_offsets - scaled_offset_mean) * scale
        return np.asarray(mean, dtype=np.float64), np.asarray(
            centered, dtype=np.float64
        )


def stable_mean(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute a reference-shifted column mean."""

    return stable_center(X)[0]


def centering_state(
    X: NDArray[np.float64], centered: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Capture the reference and offset mean used by :func:`stable_center`."""

    return X[0].copy(), -centered[0].copy()


def apply_centering(
    X: NDArray[np.float64],
    reference: NDArray[np.float64],
    offset_mean: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a fitted reference-shift centering state to query observations."""

    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray((X - reference) - offset_mean, dtype=np.float64)
    except FloatingPointError:
        scale = max(
            float(np.max(np.abs(X), initial=0.0)),
            float(np.max(np.abs(reference), initial=0.0)),
            float(np.max(np.abs(offset_mean), initial=0.0)),
        )
        if scale == 0.0:
            return np.zeros_like(X, dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(
                ((X / scale - reference / scale) - offset_mean / scale) * scale,
                dtype=np.float64,
            )


def canonicalize_columns(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Choose deterministic signs using each column's largest-magnitude entry."""

    result = np.array(matrix, dtype=np.float64, order="C", copy=True)
    if result.size == 0:
        return result
    pivots = np.argmax(np.abs(result), axis=0)
    signs = np.sign(result[pivots, np.arange(result.shape[1])])
    signs[signs == 0.0] = 1.0
    result *= signs
    return result


def centered_svd(
    X: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Center observations and compute an economy SVD."""

    _, centered = stable_center(X)
    scale = float(np.max(np.abs(centered), initial=0.0))
    scaled = centered if scale == 0.0 else centered / scale
    U, scaled_singular_values, Vt = linalg.svd(
        scaled, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    largest = float(np.max(scaled_singular_values, initial=0.0))
    if (
        scale > 0.0
        and largest > 0.0
        and (np.log(scale) + np.log(largest) > np.log(np.finfo(np.float64).max))
    ):
        raise FloatingPointError("The centered singular values overflow float64.")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        singular_values = scaled_singular_values * scale
    V = canonicalize_columns(Vt.T)
    signs = np.sum(V * Vt.T, axis=0)
    U = U * signs
    return centered, U, singular_values, V.T


def numerical_rank(
    singular_values: NDArray[np.float64],
    *,
    shape: tuple[int, int],
) -> int:
    """Compute the standard scale-aware floating-point numerical rank."""

    if singular_values.size == 0:
        return 0
    tolerance = max(shape) * np.finfo(np.float64).eps * singular_values[0]
    return int(np.count_nonzero(singular_values > tolerance))


def symmetric_eigh(
    matrix: NDArray[np.float64],
    *,
    n_components: int | None = None,
    largest: bool = True,
) -> SymmetricEigenResult:
    """Solve a symmetric eigenproblem with deterministic order and signs."""

    symmetric = (matrix + matrix.T) * 0.5
    values, vectors = linalg.eigh(symmetric, check_finite=False)
    order = np.argsort(values, kind="stable")
    if largest:
        order = order[::-1]
    if n_components is not None:
        order = order[:n_components]
    values = np.asarray(values[order], dtype=np.float64)
    vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
    residual = symmetric @ vectors - vectors * values
    denominator = max(1.0, float(linalg.norm(symmetric, ord=2)))
    residual_norm = float(linalg.norm(residual) / denominator)
    return SymmetricEigenResult(values, vectors, residual_norm)


def generalized_eigh(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    *,
    n_components: int,
    largest: bool = True,
) -> SymmetricEigenResult:
    """Solve a symmetric-definite generalized eigenproblem."""

    A_sym = (A + A.T) * 0.5
    B_sym = (B + B.T) * 0.5
    values, vectors = linalg.eigh(A_sym, B_sym, check_finite=False)
    order = np.argsort(values, kind="stable")
    if largest:
        order = order[::-1]
    order = order[:n_components]
    values = np.asarray(values[order], dtype=np.float64)
    vectors = canonicalize_columns(np.asarray(vectors[:, order], dtype=np.float64))
    residual = A_sym @ vectors - (B_sym @ vectors) * values
    denominator = max(
        1.0,
        float(linalg.norm(A_sym, ord=2))
        + float(np.max(np.abs(values))) * float(linalg.norm(B_sym, ord=2)),
    )
    return SymmetricEigenResult(
        values, vectors, float(linalg.norm(residual) / denominator)
    )


def condition_estimate(matrix: NDArray[np.float64]) -> float:
    """Return a two-norm condition estimate, including infinity for singular input."""

    return float(np.linalg.cond(matrix))
