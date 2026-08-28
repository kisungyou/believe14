"""Small deterministic datasets shared by the executable method cards.

The generators deliberately avoid downloads.  They return fresh arrays so that
one method card cannot mutate data subsequently used by another card.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def latent_data(
    *, n_samples: int = 48, n_features: int = 6
) -> tuple[FloatArray, FloatArray]:
    """Return observations from a three-factor Gaussian model."""

    rng = np.random.default_rng(1401)
    latent = rng.normal(size=(n_samples, 3))
    loadings = rng.normal(size=(3, n_features))
    noise = 0.04 * rng.normal(size=(n_samples, n_features))
    return np.asarray(latent @ loadings + noise), latent


def source_data(*, n_samples: int = 72) -> tuple[FloatArray, FloatArray]:
    """Return a full-rank mixture of three non-Gaussian independent sources."""

    rng = np.random.default_rng(1402)
    sources = np.column_stack(
        (
            rng.laplace(size=n_samples),
            rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n_samples),
            np.sign(rng.normal(size=n_samples)) * rng.exponential(size=n_samples),
        )
    )
    sources -= sources.mean(axis=0)
    sources /= sources.std(axis=0)
    mixing = np.array(
        [[1.0, 0.3, -0.2], [0.2, 1.1, 0.4], [-0.5, 0.2, 0.9]],
        dtype=np.float64,
    )
    return np.asarray(sources @ mixing.T), np.asarray(sources)


def multiclass_data(*, samples_per_class: int = 16) -> tuple[FloatArray, IntArray]:
    """Return three compact classes in five-dimensional feature space."""

    rng = np.random.default_rng(1403)
    means = np.array(
        [
            [-2.0, 0.0, 0.5, 0.0, 0.2],
            [2.0, 0.2, -0.5, 0.0, -0.2],
            [0.0, 2.4, 0.0, 0.7, 0.0],
        ]
    )
    blocks = [
        mean + 0.35 * rng.normal(size=(samples_per_class, means.shape[1]))
        for mean in means
    ]
    labels = np.repeat(np.arange(3, dtype=np.int64), samples_per_class)
    return np.asarray(np.vstack(blocks)), labels


def paired_data(*, n_samples: int = 52) -> tuple[FloatArray, FloatArray]:
    """Return two noisy views of the same two-dimensional latent variables."""

    rng = np.random.default_rng(1404)
    latent = rng.normal(size=(n_samples, 2))
    x = latent @ np.array([[1.2, -0.3, 0.8, 0.1], [0.2, 1.0, -0.4, 0.7]])
    y = latent @ np.array([[0.9, 0.4, -0.5], [-0.2, 1.1, 0.6]])
    return (
        np.asarray(x + 0.04 * rng.normal(size=x.shape)),
        np.asarray(y + 0.04 * rng.normal(size=y.shape)),
    )


def single_index_data(*, n_samples: int = 60) -> tuple[FloatArray, FloatArray]:
    """Return predictors and a sliced-response target controlled by two directions."""

    rng = np.random.default_rng(1405)
    x = rng.normal(size=(n_samples, 6))
    index = x[:, 0] + 0.65 * x[:, 1]
    y = index + 0.15 * index**2 + 0.03 * rng.normal(size=n_samples)
    return np.asarray(x), np.asarray(y)


def circles(*, n_samples: int = 48) -> tuple[FloatArray, IntArray]:
    """Return two noiseless concentric circles and their ring labels."""

    if n_samples % 2:
        raise ValueError("n_samples must be even.")
    per_ring = n_samples // 2
    angles = 2.0 * np.pi * np.arange(per_ring) / per_ring
    inner = np.column_stack((np.cos(angles), np.sin(angles)))
    outer_angles = angles + np.pi / per_ring
    outer = 2.0 * np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    return np.asarray(np.vstack((inner, outer))), np.repeat([0, 1], per_ring)


def swiss_roll(*, n_samples: int = 54) -> tuple[FloatArray, FloatArray]:
    """Return a deterministic three-dimensional Swiss roll and its roll coordinate."""

    fraction = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    parameter = 1.5 * np.pi * (1.0 + 2.0 * fraction)
    height = 1.2 * ((np.arange(n_samples) * 17) % n_samples) / n_samples
    points = np.column_stack(
        (parameter * np.cos(parameter), height, parameter * np.sin(parameter))
    )
    points /= np.std(points, axis=0, where=np.ones_like(points, dtype=bool))
    return np.asarray(points), np.asarray(parameter)


def branching(*, points_per_branch: int = 14) -> tuple[FloatArray, IntArray]:
    """Return three separated rays sharing a small central neighborhood."""

    radii = np.linspace(0.12, 2.0, points_per_branch)
    directions = np.array(
        [[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0], [-0.5, -np.sqrt(3.0) / 2.0]]
    )
    points = np.vstack([radii[:, None] * direction for direction in directions])
    labels = np.repeat(np.arange(3, dtype=np.int64), points_per_branch)
    return np.asarray(points), labels


def digits_subset(*, samples_per_digit: int = 4) -> tuple[FloatArray, IntArray]:
    """Return a deterministic stratified subset of sklearn's bundled digits data."""

    # scikit-learn 1.9 still assigns to ndarray.shape internally; NumPy 2.5
    # deprecates that operation.  Keep documentation warnings strict while
    # narrowly suppressing this upstream loader implementation detail.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
            module=r"sklearn\.datasets\._base",
        )
        dataset = load_digits()
    selected = np.concatenate(
        [
            np.flatnonzero(dataset.target == digit)[:samples_per_digit]
            for digit in range(10)
        ]
    )
    return (
        np.asarray(dataset.data[selected], dtype=np.float64) / 16.0,
        np.asarray(dataset.target[selected], dtype=np.int64),
    )


def flat_data(
    *, intrinsic_dimension: int = 3, ambient_dimension: int = 6, n_samples: int = 72
) -> FloatArray:
    """Return a sample from a linear intrinsic-dimensional flat."""

    if not 1 <= intrinsic_dimension <= ambient_dimension:
        raise ValueError("Require 1 <= intrinsic_dimension <= ambient_dimension.")
    rng = np.random.default_rng(1406 + intrinsic_dimension)
    coordinates = rng.normal(size=(n_samples, intrinsic_dimension))
    basis, _ = np.linalg.qr(rng.normal(size=(ambient_dimension, intrinsic_dimension)))
    return np.asarray(coordinates @ basis.T)


def sphere(*, n_samples: int = 72) -> FloatArray:
    """Return approximately uniform deterministic points on the unit two-sphere."""

    index = np.arange(n_samples, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / n_samples
    angle = np.pi * (3.0 - np.sqrt(5.0)) * index
    radius = np.sqrt(1.0 - z**2)
    return np.asarray(
        np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    )


def curved_data(*, n_samples: int = 72) -> FloatArray:
    """Return a smooth one-dimensional helix embedded in three dimensions."""

    parameter = np.linspace(0.0, 4.0 * np.pi, n_samples)
    return np.asarray(
        np.column_stack(
            (np.cos(parameter), np.sin(parameter), parameter / (2.0 * np.pi))
        )
    )
