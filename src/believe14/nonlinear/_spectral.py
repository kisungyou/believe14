"""Constrained eigensystems for reversible graph operators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from believe14._core.linalg import canonicalize_columns


def eigh_without_stationary(
    symmetric: NDArray[np.float64],
    stationary: NDArray[np.float64],
    *,
    largest: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Diagonalize a symmetric operator orthogonally to its known stationary mode.

    A Householder reflection supplies an orthonormal basis of the complement.
    Imposing the constraint before diagonalization keeps the stationary mode
    out even when its eigenvalue cannot be numerically separated from others.
    The dense projection preserves the eigensolver's cubic time and quadratic
    storage bounds.
    """

    direction = stationary / linalg.norm(stationary)
    reflector = direction.copy()
    reflector[0] += np.copysign(1.0, direction[0])
    reflector /= linalg.norm(reflector)
    basis = np.eye(len(direction), dtype=np.float64)[:, 1:]
    basis -= 2.0 * np.outer(reflector, reflector[1:])
    restricted = basis.T @ symmetric @ basis
    restricted = (restricted + restricted.T) * 0.5
    values, vectors = linalg.eigh(restricted, check_finite=False)
    order = np.argsort(values, kind="stable")
    if largest:
        order = order[::-1]
    return (
        np.asarray(values[order], dtype=np.float64),
        canonicalize_columns(basis @ vectors[:, order]),
    )
