# mypy: disallow-subclassing-any=False
"""Iterative distributional and diffusion-potential embeddings."""

from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from believe14._core.diagnostics import diagnostics
from believe14._core.distances import pairwise_distances, pairwise_squared_distances
from believe14._core.linalg import centered_svd
from believe14._core.validation import (
    make_rng,
    validate_features,
    validate_n_components,
    validate_n_neighbors,
    validate_positive_real,
)

from ._common import initialize_embedding, output_names, smacof


def _joint_probabilities(
    squared_distances: NDArray[np.float64], perplexity: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute exact symmetrized t-SNE probabilities by entropy matching."""

    n_samples = squared_distances.shape[0]
    target_entropy = np.log(perplexity)
    entropy_tolerance = 1e-8
    conditional = np.zeros_like(squared_distances)
    entropy_residuals = np.empty(n_samples, dtype=np.float64)
    all_indices = np.arange(n_samples)
    for row in range(n_samples):
        mask = all_indices != row
        distances = squared_distances[row, mask]
        shifted = distances - float(np.min(distances))
        minimum_mask = shifted == 0.0
        minimum_multiplicity = int(np.count_nonzero(minimum_mask))
        minimum_entropy = np.log(float(minimum_multiplicity))
        maximum_entropy = np.log(float(n_samples - 1))
        if target_entropy < minimum_entropy - entropy_tolerance:
            raise ValueError(
                f"perplexity={perplexity:g} is unattainable for row {row}: "
                f"{minimum_multiplicity} nearest distances are tied."
            )
        if abs(target_entropy - maximum_entropy) <= entropy_tolerance:
            probabilities = np.full(
                n_samples - 1, 1.0 / (n_samples - 1), dtype=np.float64
            )
        elif abs(target_entropy - minimum_entropy) <= entropy_tolerance:
            probabilities = minimum_mask.astype(np.float64)
            probabilities /= minimum_multiplicity
        else:
            beta = 1.0
            lower = 0.0
            upper = np.inf
            probabilities = np.empty_like(distances)
            for _ in range(100):
                unnormalized = np.exp(-beta * shifted)
                normalizer = float(np.sum(unnormalized, dtype=np.float64))
                probabilities = unnormalized / normalizer
                positive = probabilities > 0.0
                entropy = -float(
                    np.sum(
                        probabilities[positive] * np.log(probabilities[positive]),
                        dtype=np.float64,
                    )
                )
                difference = entropy - target_entropy
                if abs(difference) <= entropy_tolerance:
                    break
                if difference > 0.0:
                    lower = beta
                    beta = beta * 2.0 if np.isinf(upper) else (beta + upper) * 0.5
                else:
                    upper = beta
                    beta = (beta + lower) * 0.5
        positive = probabilities > 0.0
        achieved_entropy = -float(
            np.sum(
                probabilities[positive] * np.log(probabilities[positive]),
                dtype=np.float64,
            )
        )
        entropy_residuals[row] = achieved_entropy - target_entropy
        if abs(entropy_residuals[row]) > entropy_tolerance:
            raise ValueError(
                f"Could not match perplexity for row {row}; entropy residual "
                f"is {entropy_residuals[row]:.3e}."
            )
        conditional[row, mask] = probabilities
    joint = (conditional + conditional.T) / (2.0 * n_samples)
    np.fill_diagonal(joint, 0.0)
    joint /= np.sum(joint, dtype=np.float64)
    return (
        np.asarray(joint, dtype=np.float64),
        np.asarray(conditional, dtype=np.float64),
        entropy_residuals,
    )


def _tsne_objective_gradient(
    flat_embedding: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    n_components: int,
) -> tuple[float, NDArray[np.float64]]:
    """Return the exact dense symmetric t-SNE objective and gradient."""

    n_samples = probabilities.shape[0]
    embedding = flat_embedding.reshape(n_samples, n_components)
    squared = pairwise_squared_distances(embedding)
    numerator = 1.0 / (1.0 + squared)
    np.fill_diagonal(numerator, 0.0)
    partition = float(np.sum(numerator, dtype=np.float64))
    if partition <= 0.0 or not np.isfinite(partition):
        raise FloatingPointError("The t-SNE low-dimensional partition is invalid.")
    low_probabilities = numerator / partition
    positive = probabilities > 0.0
    objective = -float(
        np.sum(probabilities[positive] * np.log(numerator[positive]), dtype=np.float64)
    ) + np.log(partition)
    attraction_repulsion = (probabilities - low_probabilities) * numerator
    laplacian = -attraction_repulsion
    laplacian[np.diag_indices_from(laplacian)] = np.sum(
        attraction_repulsion, axis=1, dtype=np.float64
    )
    gradient = 4.0 * laplacian @ embedding
    return objective, np.asarray(gradient.ravel(), dtype=np.float64)


def _kl_divergence(
    embedding: NDArray[np.float64], probabilities: NDArray[np.float64]
) -> float:
    squared = pairwise_squared_distances(embedding)
    numerator = 1.0 / (1.0 + squared)
    np.fill_diagonal(numerator, 0.0)
    low = numerator / np.sum(numerator, dtype=np.float64)
    positive = probabilities > 0.0
    return float(
        np.sum(
            probabilities[positive] * np.log(probabilities[positive] / low[positive]),
            dtype=np.float64,
        )
    )


class TSNE(BaseEstimator):
    """Exact symmetric dense t-SNE optimized with analytic gradients."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        early_exaggeration_iter: int = 250,
        init: Literal["pca", "random"] = "pca",
        max_iter: int = 1000,
        tol: float = 1e-7,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize(
        self, features: NDArray[np.float64], n_components: int
    ) -> NDArray[np.float64]:
        if self.init == "random":
            rng = make_rng(self.random_state)
            return np.asarray(
                rng.normal(scale=1e-4, size=(features.shape[0], n_components)),
                dtype=np.float64,
            )
        if self.init != "pca":
            raise ValueError("init must be 'pca' or 'random'.")
        if n_components > min(features.shape[0] - 1, features.shape[1]):
            raise ValueError(
                "PCA initialization requires n_components no larger than the "
                "centered feature rank bound; use init='random'."
            )
        _, left, singular_values, _ = centered_svd(features)
        initial = left[:, :n_components] * singular_values[:n_components]
        maximum = float(np.max(np.abs(initial), initial=0.0))
        if maximum <= 0.0:
            raise ValueError("PCA initialization is degenerate.")
        scaled_initial = initial / maximum
        scale = float(np.std(scaled_initial, dtype=np.float64))
        if scale <= 0.0:
            raise ValueError("PCA initialization is degenerate.")
        return np.asarray(scaled_initial * (1e-4 / scale), dtype=np.float64)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> TSNE:
        """Fit the transductive t-SNE embedding."""

        del y
        features = validate_features(self, X, reset=True)
        n_components = validate_n_components(
            self.n_components, maximum=features.shape[0] - 1
        )
        perplexity = validate_positive_real(self.perplexity, name="perplexity")
        if perplexity < 1.0 or perplexity > features.shape[0] - 1:
            raise ValueError("perplexity must be in [1, n_samples - 1].")
        exaggeration = validate_positive_real(
            self.early_exaggeration, name="early_exaggeration"
        )
        if exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1.")
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, Integral):
            raise TypeError("max_iter must be an integer.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if isinstance(self.early_exaggeration_iter, bool) or not isinstance(
            self.early_exaggeration_iter, Integral
        ):
            raise TypeError("early_exaggeration_iter must be an integer.")
        if self.early_exaggeration_iter < 0:
            raise ValueError("early_exaggeration_iter must be nonnegative.")
        tolerance = validate_positive_real(self.tol, name="tol")
        feature_distances = pairwise_distances(features)
        row_scale = np.max(feature_distances, axis=1)
        if np.any(row_scale <= 0.0):
            raise ValueError(
                "t-SNE input is degenerate; at least two distinct observations "
                "are required."
            )
        normalized_squared = (feature_distances / row_scale[:, None]) ** 2
        probabilities, conditional, entropy_residuals = _joint_probabilities(
            normalized_squared, perplexity
        )
        initial = self._initialize(features, n_components)
        consumed = 0
        last_result: optimize.OptimizeResult | None = None
        early_iterations = min(int(self.early_exaggeration_iter), int(self.max_iter))
        if early_iterations:
            last_result = optimize.minimize(
                _tsne_objective_gradient,
                initial.ravel(),
                args=(probabilities * exaggeration, n_components),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": early_iterations,
                    "ftol": tolerance,
                    "gtol": tolerance,
                },
            )
            initial = np.asarray(last_result.x, dtype=np.float64).reshape(
                features.shape[0], n_components
            )
            consumed = int(last_result.nit)
        remaining = max(0, int(self.max_iter) - early_iterations)
        ran_standard_phase = remaining > 0
        if ran_standard_phase:
            last_result = optimize.minimize(
                _tsne_objective_gradient,
                initial.ravel(),
                args=(probabilities, n_components),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": remaining, "ftol": tolerance, "gtol": tolerance},
            )
            consumed += int(last_result.nit)
        if last_result is None:
            raise RuntimeError("t-SNE did not execute an optimization phase.")
        embedding = np.asarray(last_result.x, dtype=np.float64).reshape(
            features.shape[0], n_components
        )
        embedding -= np.mean(embedding, axis=0, keepdims=True)
        kl = _kl_divergence(embedding, probabilities)
        gradient = _tsne_objective_gradient(
            embedding.ravel(), probabilities, n_components
        )[1]
        gradient_norm = float(linalg.norm(gradient))
        converged = ran_standard_phase and bool(last_result.success)
        self.embedding_: NDArray[np.float64] = embedding
        self.joint_probabilities_ = probabilities
        self.conditional_probabilities_ = conditional
        self.perplexity_entropy_residuals_ = entropy_residuals
        self.kl_divergence_ = kl
        self.n_iter_ = consumed
        self.n_components_: int = n_components
        warning = () if converged else (str(last_result.message),)
        self.diagnostics_ = diagnostics(
            "exact_symmetric_tsne_lbfgsb",
            converged=converged,
            n_iter=consumed,
            residual_norm=gradient_norm,
            objective_value=kl,
            warnings=warning,
        )
        return self

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Fit and return the training coordinates."""

        return self.fit(X, y).embedding_.copy()

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("tsne", self.n_components_)


class PHATE(BaseEstimator):
    """PHATE using an adaptive alpha-decay kernel and potential distances."""

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 5,
        decay: float = 40.0,
        diffusion_time: int = 20,
        potential_floor: float = 1e-12,
        mds_max_iter: int = 300,
        mds_tol: float = 1e-6,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.decay = decay
        self.diffusion_time = diffusion_time
        self.potential_floor = potential_floor
        self.mds_max_iter = mds_max_iter
        self.mds_tol = mds_tol

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> PHATE:
        """Fit the transductive diffusion-potential embedding."""

        del y
        features = validate_features(self, X, reset=True)
        n_components = validate_n_components(
            self.n_components, maximum=features.shape[0] - 1
        )
        n_neighbors = validate_n_neighbors(
            self.n_neighbors, n_samples=features.shape[0]
        )
        decay = validate_positive_real(self.decay, name="decay")
        floor = validate_positive_real(self.potential_floor, name="potential_floor")
        if floor >= 1.0:
            raise ValueError("potential_floor must be smaller than 1.")
        if isinstance(self.diffusion_time, bool) or not isinstance(
            self.diffusion_time, Integral
        ):
            raise TypeError("diffusion_time must be an integer.")
        if self.diffusion_time < 1:
            raise ValueError("diffusion_time must be positive.")
        distances = pairwise_distances(features)
        ordered = np.sort(distances, axis=1, kind="stable")
        bandwidth = ordered[:, n_neighbors]
        scale = float(np.max(distances, initial=0.0))
        threshold = np.finfo(np.float64).eps * scale * 100.0
        if np.any(bandwidth <= threshold):
            raise ValueError(
                "PHATE encountered a zero adaptive bandwidth; duplicated or "
                "degenerate neighborhoods are not admissible."
            )
        ratios = distances / bandwidth[:, None]
        exponents = np.zeros_like(ratios)
        positive = ratios > 0.0
        log_exponents = decay * np.log(ratios[positive])
        maximum_log = np.log(np.finfo(np.float64).max)
        finite = log_exponents <= maximum_log
        selected = np.empty_like(log_exponents)
        selected[~finite] = np.inf
        selected[finite] = np.exp(log_exponents[finite])
        exponents[positive] = selected
        directed = np.exp(-exponents)
        affinity = (directed + directed.T) * 0.5
        count = int(connected_components(csr_matrix(affinity > 0.0), directed=False)[0])
        if count != 1:
            raise ValueError(
                "The PHATE alpha-decay affinity graph is numerically disconnected."
            )
        degree = np.sum(affinity, axis=1, dtype=np.float64)
        if np.any(degree <= 0.0):
            raise FloatingPointError("The PHATE affinity has a zero-degree row.")
        transition = affinity / degree[:, None]
        diffusion = np.linalg.matrix_power(transition, int(self.diffusion_time))
        clipped = diffusion < floor
        potential = -np.log(np.maximum(diffusion, floor))
        potential_distances = pairwise_distances(potential)
        initial = initialize_embedding(
            potential_distances,
            n_components,
            init="classical",
            random_state=None,
        )
        embedding, objective, n_iter, converged, residual = smacof(
            potential_distances,
            initial,
            max_iter=self.mds_max_iter,
            tol=self.mds_tol,
        )
        warnings: tuple[str, ...] = ()
        if np.any(clipped):
            warnings += (
                "Diffusion probabilities below potential_floor were explicitly "
                "regularized before the log transform.",
            )
        if not converged:
            warnings += ("PHATE metric MDS reached mds_max_iter.",)
        self.embedding_: NDArray[np.float64] = embedding
        self.affinity_matrix_ = affinity
        self.diffusion_operator_ = transition
        self.diffusion_potential_ = potential
        self.potential_distances_ = potential_distances
        self.stress_ = objective
        self.n_iter_ = n_iter
        self.n_neighbors_ = n_neighbors
        self.n_components_: int = n_components
        self.diagnostics_ = diagnostics(
            "alpha_decay+diffusion_potential+smacof",
            converged=converged,
            n_iter=n_iter,
            residual_norm=residual,
            objective_value=objective,
            warnings=warnings,
        )
        return self

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> NDArray[np.float64]:
        """Fit and return the training coordinates."""

        return self.fit(X, y).embedding_.copy()

    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> NDArray[np.object_]:
        """Return output-coordinate names."""

        del input_features
        check_is_fitted(self, "embedding_")
        return output_names("phate", self.n_components_)
