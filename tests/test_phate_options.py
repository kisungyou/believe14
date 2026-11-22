"""Literal entropy and coarse-graining checks for PHATE."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from believe14.nonlinear import PHATE
from believe14.nonlinear._phate_options import select_diffusion_time


def test_auto_time_matches_normalized_spectrum_and_two_line_knee():
    transition = np.diag([1.0, 0.9, 0.5, 0.2])
    t, entropy = select_diffusion_time(transition, 18)
    times = np.arange(1, 19)
    powers = np.diag(transition) ** times[:, None]
    probs = powers / powers.sum(axis=1, keepdims=True)
    exact = -(probs * np.log(probs)).sum(axis=1)
    np.testing.assert_allclose(entropy, exact, atol=1e-14)

    def error(x, y):
        return np.abs(y - np.polyval(np.polyfit(x, y, 1), x)).sum()

    costs = [
        error(times[: i + 1], exact[: i + 1]) + error(times[i:], exact[i:])
        for i in range(1, 17)
    ]
    assert t == 2 + np.argmin(costs)


def test_phate_landmark_transition_and_interpolation():
    x = np.random.default_rng(719).normal(size=(37, 3))
    model = PHATE(
        2,
        n_neighbors=5,
        decay=4,
        n_landmarks=9,
        diffusion_time="auto",
        max_diffusion_time=12,
    ).fit(x)
    selected = model.landmark_indices_
    np.testing.assert_array_equal(
        model.landmark_labels_, np.argmin(cdist(x, x[selected]), axis=1)
    )
    h = np.eye(9)[model.landmark_labels_]
    grouped = model.affinity_matrix_ @ h
    out = grouped / grouped.sum(axis=1, keepdims=True)
    back = grouped.T / grouped.sum(axis=0)[:, None]
    coarse = back @ out
    np.testing.assert_allclose(coarse, model.diffusion_operator_)
    np.testing.assert_allclose(out, model.landmark_transitions_)
    np.testing.assert_allclose(model.embedding_, out @ model.landmark_embedding_)
    assert model.diffusion_entropy_.shape == (12,)
    powered = np.linalg.matrix_power(coarse, model.diffusion_time_)
    np.testing.assert_allclose(
        model.diffusion_potential_, -np.log(np.maximum(powered, model.potential_floor))
    )
    assert model.potential_distances_.shape == (9, 9)
    full = PHATE(n_neighbors=5, decay=4, diffusion_time=3).fit_transform(x)
    model.set_params(n_landmarks=len(x), diffusion_time=3).fit(x)
    np.testing.assert_allclose(model.embedding_, full)
    assert (
        model.diffusion_entropy_
        is model.landmark_indices_
        is model.landmark_embedding_
        is None
    )


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"n_landmarks": True}, TypeError),
        ({"n_landmarks": 2}, ValueError),
        ({"n_landmarks": 50}, ValueError),
        ({"max_diffusion_time": 2}, ValueError),
        ({"max_diffusion_time": 5.5}, TypeError),
        ({"diffusion_time": "bad"}, TypeError),
    ],
)
def test_phate_option_domains(kwargs, exception):
    with pytest.raises(exception):
        PHATE(**kwargs).fit(np.arange(30.0).reshape(10, 3))
