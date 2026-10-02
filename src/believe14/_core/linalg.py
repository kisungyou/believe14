"""Stable linear-algebra helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

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


def restore_centering(
    centered: NDArray[np.float64],
    reference: NDArray[np.float64],
    offset_mean: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Restore a fitted mean without rounding its reference and offset first.

    Two compensated additions retain low-order variation, including when the
    offset cancels the reference. Rare overflowing partial sums use scalar
    accurate summation with opposite signs first, without rescaling away a
    representable small residual. Unrepresentable final values fail explicitly.
    """

    if not all(
        np.all(np.isfinite(value)) for value in (centered, reference, offset_mean)
    ):
        raise FloatingPointError("Restoring observations requires finite summands.")
    try:
        with np.errstate(over="raise", invalid="raise"):
            partial = centered + offset_mean
            offset_virtual = partial - centered
            partial_error = (centered - (partial - offset_virtual)) + (
                offset_mean - offset_virtual
            )
            result = partial + reference
            reference_virtual = result - partial
            result_error = (partial - (result - reference_virtual)) + (
                reference - reference_virtual
            )
            return np.asarray(result + (partial_error + result_error), dtype=np.float64)
    except FloatingPointError:
        values, offsets, references = np.broadcast_arrays(
            centered, offset_mean, reference
        )
        restored = np.empty(values.shape, dtype=np.float64)
        for index in np.ndindex(values.shape):
            a, b, c = (
                float(values[index]),
                float(offsets[index]),
                float(references[index]),
            )
            # Summing opposite signs first avoids fsum's intermediate-overflow
            # error when the final sum itself is representable.
            terms = (a, b, c)
            if (a < 0.0) == (b < 0.0) and (a < 0.0) != (c < 0.0):
                terms = (a, c, b)
            try:
                restored[index] = fsum(terms)
            except OverflowError as error:
                raise FloatingPointError(
                    "The restored observations are not representable in float64."
                ) from error
        return restored


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


def scale_squared(values: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    """Multiply by scale squared without an overflowing intermediate square."""

    mantissa, exponent = np.frexp(values)
    scale_mantissa, scale_exponent = np.frexp(scale)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return np.asarray(
            np.ldexp(
                mantissa * scale_mantissa * scale_mantissa,
                exponent + 2 * scale_exponent,
            ),
            dtype=np.float64,
        )


def eigh_without_constant(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve a symmetric eigenproblem constrained to the mean-zero subspace.

    Removing an eigenvector by index does not remove the constant mode when
    zero is repeated. Helmert contrasts impose that constraint before solving.
    """

    basis = linalg.helmert(matrix.shape[0], full=False).T
    reduced = basis.T @ matrix @ basis
    reduced = (reduced + reduced.T) * 0.5
    values, vectors = linalg.eigh(reduced, check_finite=False)
    return np.asarray(values, dtype=np.float64), canonicalize_columns(basis @ vectors)
