"""Independent calculations for angular and neighbor-distance edge cases."""

from itertools import combinations

import numpy as np
import pytest

from believe14.estimation import CorrelationDimension, DANCo, MiNDML
from believe14.estimation._common import von_mises_kl


def test_flat_counts_rejected_before_regression():
    with pytest.raises(ValueError, match="nonpositive slope"):
        CorrelationDimension(radii=np.linspace(1.1, 1.9, 7)).fit(
            np.arange(3.0)[:, None]
        )


@pytest.mark.parametrize("scale", [1.0, 2.0**100, 2.0**-100])
def test_relative_radii_preserve_strictly_distinct_values(scale):
    points = np.array([0, 0.25, 0.5, 1.0, 3.0])
    radii = np.array([np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)])
    ds = np.array([abs(x - y) for x, y in combinations(points, 2)])
    counts = [np.count_nonzero(ds < r) for r in radii]
    expected = np.log(counts[1] / counts[0]) / np.log1p(
        (radii[1] - radii[0]) / radii[0]
    )
    fitted = CorrelationDimension(radii=radii * scale).fit((points * scale)[:, None])
    assert fitted.dimension_ == pytest.approx(expected, rel=2e-14)
    assert fitted.diagnostics_.residual_norm < 1e-15


@pytest.mark.parametrize("delta", [1e-5, 1e-7, 1e-8])
def test_nearly_parallel_neighbors_have_finite_danco_concentrations(delta):
    points = np.array(
        [
            [0, 0],
            [1, 0],
            [2, delta],
            [3, -2 * delta],
            [0.37, 10.1],
            [8.3, 7.7],
            [10, -4.0],
        ]
    )
    angles = np.array(
        [
            np.arctan2(abs(a[0] * b[1] - a[1] * b[0]), a @ b)
            for a, b in combinations(points[1:4], 2)
        ]
    )
    deficit = 4 / 9 * sum(np.sin((a - b) / 2) ** 2 for a, b in combinations(angles, 2))
    eta = np.sqrt(1 - deficit)
    expected = (1 + eta) / (eta * (3 - eta) * deficit)
    fitted = DANCo(n_neighbors=3, random_state=0).fit(points)
    assert fitted.local_angle_concentrations_[0] == pytest.approx(expected, rel=1e-12)
    assert np.isfinite(fitted.divergences_).all()


def test_large_concentration_kl_retains_order_one_information():
    # For fixed ratio 2 and equal means the two von Mises laws tend to
    # Gaussians with variances 1/k, so KL tends to (2-1-log(2))/2.
    assert von_mises_kl(0.7, 1e17, 0.7, 2e17) == pytest.approx(
        (1 - np.log(2)) / 2, rel=1e-12
    )


def test_mind_k_one_accepts_single_tie_by_literal_likelihood():
    points = np.array([[0, 0], [1, 0], [0, 1], [2, 3], [4, 0], [10, 10]], float)
    distances = np.sort(np.linalg.norm(points[:, None] - points, axis=2), axis=1)
    rho = distances[:, 1] / distances[:, 2]
    assert (rho == 1).sum() == 1
    d = np.clip(-len(rho) / np.sum(np.log(rho)), 1, 2)
    fitted = MiNDML(n_neighbors=1).fit(points)
    np.testing.assert_allclose(fitted.normalized_distances_, rho)
    assert fitted.dimension_ == pytest.approx(d)
    assert fitted.log_likelihood_ == pytest.approx(np.sum(np.log(d * rho ** (d - 1))))
