"""Exact graph construction and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from .distances import exact_neighbors, scaled_squares


@dataclass(frozen=True, slots=True)
class NeighborGraph:
    distances: NDArray[np.float64]
    weights: NDArray[np.float64]
    n_connected_components: int


def neighbor_graph(
    X: NDArray[np.float64],
    *,
    n_neighbors: int,
    weighting: Literal["binary", "heat"] = "binary",
    gamma: float | None = None,
    symmetrize: Literal["union", "mutual", "mean"] = "union",
) -> NeighborGraph:
    """Build a deterministic exact k-neighbor graph."""

    neighbor_distances, indices = exact_neighbors(X, n_neighbors)
    n_samples = X.shape[0]
    directed_distances = np.full((n_samples, n_samples), np.inf, dtype=np.float64)
    directed_weights = np.zeros((n_samples, n_samples), dtype=np.float64)
    rows = np.arange(n_samples)[:, None]
    directed_distances[rows, indices] = neighbor_distances
    if weighting == "binary":
        directed_weights[rows, indices] = 1.0
    else:
        if gamma is None or gamma <= 0.0:
            raise ValueError("gamma must be positive for heat-kernel weights.")
        directed_weights[rows, indices] = np.exp(
            -scaled_squares(neighbor_distances, multiplier=gamma)
        )

    if symmetrize == "union":
        weights = np.maximum(directed_weights, directed_weights.T)
        distances = np.minimum(directed_distances, directed_distances.T)
    elif symmetrize == "mutual":
        mutual = (directed_weights > 0.0) & (directed_weights.T > 0.0)
        weights = np.minimum(directed_weights, directed_weights.T) * mutual
        distances = np.maximum(directed_distances, directed_distances.T)
        distances[~mutual] = np.inf
    elif symmetrize == "mean":
        weights = (directed_weights + directed_weights.T) * 0.5
        distances = np.minimum(directed_distances, directed_distances.T)
    else:
        raise ValueError("symmetrize must be 'union', 'mutual', or 'mean'.")

    np.fill_diagonal(weights, 0.0)
    np.fill_diagonal(distances, 0.0)
    count = int(connected_components(csr_matrix(weights), directed=False)[0])
    return NeighborGraph(distances, weights, count)


def require_connected(graph: NeighborGraph) -> None:
    """Reject disconnected graphs instead of silently repairing them."""

    if graph.n_connected_components != 1:
        raise ValueError(
            "The neighborhood graph is disconnected; increase n_neighbors or "
            "change the graph parameters."
        )


def graph_shortest_paths(graph: NeighborGraph) -> NDArray[np.float64]:
    """Compute undirected all-pairs shortest paths after connectivity validation."""

    require_connected(graph)
    adjacency = np.where(np.isfinite(graph.distances), graph.distances, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    result = shortest_path(csr_matrix(adjacency), directed=False)
    return np.asarray(result, dtype=np.float64)
