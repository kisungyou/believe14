"""Time selection and coarse diffusion for PHATE."""

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, special


def select_diffusion_time(
    transition: NDArray[np.float64], max_time: int
) -> tuple[int, NDArray[np.float64]]:
    """Select the interior break of two fitted lines on the diffusion entropy curve."""
    values = linalg.svdvals(transition)
    values = values[values > 0.0]
    times = np.arange(1, max_time + 1, dtype=np.float64)
    # Cancelling the largest log singular value before exponentiation avoids
    # overflows and preserves the same normalized spectral probabilities.
    log_weights = np.log(values) - np.log(values[0])
    entropy = np.empty_like(times)
    for i, time in enumerate(times):
        probabilities = special.softmax(time * log_weights)
        entropy[i] = np.sum(special.entr(probabilities))

    def deviation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        design = np.column_stack((x, np.ones(len(x))))
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(np.sum(np.abs(design @ coef - y)))

    errors = [
        deviation(times[: i + 1], entropy[: i + 1]) + deviation(times[i:], entropy[i:])
        for i in range(1, max_time - 1)
    ]
    return int(np.argmin(errors)) + 2, entropy


def landmark_diffusion(
    affinity: NDArray[np.float64], distances: NDArray[np.float64], n_landmarks: int
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]
]:
    """Aggregate affinity over a farthest-first Voronoi partition."""
    chosen = np.empty(n_landmarks, dtype=np.int64)
    chosen[0] = 0
    nearest = distances[0].copy()
    for i in range(1, n_landmarks):
        candidate = int(np.argmax(nearest))
        if nearest[candidate] == 0:
            raise ValueError("n_landmarks exceeds the number of distinct observations.")
        chosen[i] = candidate
        np.minimum(nearest, distances[candidate], out=nearest)
    labels = np.argmin(distances[:, chosen], axis=1).astype(np.int64)
    block = np.column_stack(
        [np.sum(affinity[:, labels == j], axis=1) for j in range(n_landmarks)]
    )
    sample_to_landmark = block / block.sum(axis=1, keepdims=True)
    landmark_to_sample = block.T / block.sum(axis=0)[:, None]
    return landmark_to_sample @ sample_to_landmark, sample_to_landmark, chosen, labels
