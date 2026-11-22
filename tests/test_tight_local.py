"""Independent first-contact solver for Amsaleg et al.'s local likelihood."""

import numpy as np
import pytest
from scipy.optimize import brentq

from believe14.estimation import (
    DANCo,
    LevinaBickelMLE,
    MiNDML,
    TightLocalEstimator,
    TwoNN,
)
from believe14.neighbors import build_neighbor_graph


def _first_contact_error(t, r, u, v):
    return np.linalg.norm((1 - t / r) * u - v) - t


def _first_contact_reference(X, k):
    values = []
    for query in X:
        distances = np.linalg.norm(X - query, axis=1)
        order = np.argsort(distances, kind="stable")[1 : k + 1]
        r = distances[order[-1]]
        terms = []
        for i in order:
            for j in order:
                if i == j:
                    continue
                for sign in (-1, 1):
                    u = sign * (X[i] - query)
                    v = X[j] - query
                    # Geometric first contact, in the original coordinates.
                    if distances[j] == r:
                        root = r
                    else:
                        root = brentq(_first_contact_error, 0, r, args=(r, u, v))
                    terms.append(np.log(root / r))
        values.append(-1 / np.mean(terms))
    return values


def test_tight_local_matches_first_contact_oracle_and_rigid_motion():
    rng = np.random.default_rng(417)
    X = rng.normal(size=(15, 4))
    model = TightLocalEstimator(n_neighbors=6).fit(X)
    np.testing.assert_allclose(
        model.local_dimensions_, _first_contact_reference(X, 6), rtol=1e-10
    )
    np.testing.assert_allclose(model.dimension_, np.mean(model.local_dimensions_))
    Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    permutation = rng.permutation(len(X))
    changed = TightLocalEstimator(n_neighbors=6).fit(
        (X[permutation] @ Q + 3) * 2.0**410
    )
    np.testing.assert_allclose(
        changed.local_dimensions_, model.local_dimensions_[permutation], rtol=1e-12
    )


@pytest.mark.parametrize(
    "estimator", [TightLocalEstimator(n_neighbors=8), LevinaBickelMLE(k_min=4, k_max=8)]
)
def test_pointwise_multiscale_reuse(estimator):
    X = np.random.default_rng(71).normal(size=(40, 3))
    direct = estimator.fit(X).local_dimensions_by_k_.copy()
    graph = build_neighbor_graph(X, n_neighbors=10)
    estimator.fit(X, neighbors=graph)
    np.testing.assert_array_equal(estimator.local_dimensions_by_k_, direct)
    assert direct.shape == (len(X), len(estimator.k_values_))
    np.testing.assert_allclose(estimator.local_dimensions_, direct.mean(axis=1))
    np.testing.assert_allclose(estimator.dimensions_by_k_, direct.mean(axis=0))
    np.testing.assert_allclose(estimator.dimension_, direct.mean())


def test_likelihood_surface_degeneracies_are_explicit():
    # One neighborhood has equal surface radii and therefore only unit contacts.
    X = np.array([[0.0, 0], [5.0, 0.0], [0.0, 5.0], [-3.0, -4.0]])
    with pytest.raises(ValueError, match="zero likelihood denominator"):
        TightLocalEstimator(n_neighbors=3).fit(X)
    X = np.array([[0.0, 0], [-1.0, 0], [1.0, 0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="zero adjusted radius"):
        TightLocalEstimator(n_neighbors=3).fit(X)
    with pytest.raises(ValueError, match="duplicate"):
        TightLocalEstimator(n_neighbors=3).fit(
            np.array([[0.0, 0], [0.0, 0], [1.0, 1], [0.0, 2.0]])
        )


@pytest.mark.parametrize(
    "estimator",
    [TightLocalEstimator(n_neighbors=12), LevinaBickelMLE(k_min=8, k_max=12)],
)
def test_local_estimates_distinguish_line_and_plane(estimator):
    rng = np.random.default_rng(751)
    line = np.column_stack([rng.uniform(0, 1, 100), np.zeros(100)])
    plane = rng.uniform(0, 1, (120, 2)) + 3
    local = estimator.fit(np.vstack([line, plane])).local_dimensions_
    assert 0.75 < np.median(local[:100]) < 1.35
    assert 1.55 < np.median(local[100:]) < 2.35


@pytest.mark.parametrize("bias_correction", [True, False])
def test_levina_pointwise_matrix_matches_scalar_likelihood(bias_correction):
    X = np.random.default_rng(149).uniform(size=(19, 3))
    model = LevinaBickelMLE(k_min=4, k_max=7, bias_correction=bias_correction).fit(X)
    expected = np.empty((len(X), 4))
    for i, point in enumerate(X):
        distances = np.sort(np.linalg.norm(X - point, axis=1))[1:]
        for col, k in enumerate(range(4, 8)):
            numerator = k - 2 if bias_correction else k - 1
            expected[i, col] = numerator / sum(
                np.log(distances[k - 1] / r) for r in distances[: k - 1]
            )
    np.testing.assert_array_equal(model.k_values_, [4, 5, 6, 7])
    np.testing.assert_allclose(model.local_dimensions_by_k_, expected)


@pytest.mark.parametrize(
    ("estimator", "output", "required"),
    [
        (TwoNN(), "ratios_", 2),
        (MiNDML(n_neighbors=4), "normalized_distances_", 5),
        (DANCo(n_neighbors=4, random_state=17), "divergences_", 5),
        (LevinaBickelMLE(k_min=3, k_max=4), "local_dimensions_by_k_", 4),
        (TightLocalEstimator(n_neighbors=4), "local_dimensions_", 4),
    ],
)
def test_precomputed_neighbors_match_and_validate_input(estimator, output, required):
    X = np.random.default_rng(88).normal(size=(30, 3))
    direct = estimator.fit(X)
    expected = getattr(direct, output).copy()
    dimension = direct.dimension_
    graph = build_neighbor_graph(X, n_neighbors=required)
    cached = estimator.fit(X, neighbors=graph)
    np.testing.assert_array_equal(getattr(cached, output), expected)
    assert cached.dimension_ == dimension
    with pytest.raises(ValueError, match="graph"):
        estimator.fit(X[::-1], neighbors=graph)
    with pytest.raises(ValueError, match="neighbors"):
        estimator.fit(X, neighbors=build_neighbor_graph(X, n_neighbors=required - 1))


def _one_dimensional_contacts(query, points):
    # On a line the appropriate moving interval endpoint reaches v first.
    # Use exact rational input differences, including across large offsets.
    from fractions import Fraction
    from math import log

    q = Fraction(query)
    offsets = [Fraction(point) - q for point in points]
    radius = max(map(abs, offsets))
    terms = []
    for i, u in enumerate(offsets):
        for j, v in enumerate(offsets):
            if i == j:
                continue
            for x in [u, -u]:
                delta = v - x
                contact = abs(delta) / (radius - x if delta > 0 else radius + x)
                terms.append(log(contact.numerator) - log(contact.denominator))
    return -1 / np.mean(terms)


def test_subneighborhood_separations_survive_extreme_scale_range():
    raw = np.array([0, 1e-240, 3.7e-240, 9.1e-240, 1e220, 1.15e220, 1.43e220, 1.69e220])
    k = 5
    distances = np.abs(raw[:, None] - raw)
    orders = np.argsort(distances, kind="stable")[:, 1 : k + 1]
    expected = [
        _one_dimensional_contacts(q, raw[order])
        for q, order in zip(raw, orders, strict=True)
    ]
    model = TightLocalEstimator(n_neighbors=k).fit(raw[:, None])
    np.testing.assert_allclose(model.local_dimensions_, expected, rtol=2e-12)


def test_tight_local_contacts_close_to_neighborhood_boundary_keep_precision():
    from believe14.estimation._tle import _adjusted_log_radii, _log_radial_deficits

    points = np.array([0.25, np.nextafter(1.0, 0.0), 1.0])[:, None]
    query = np.array([0.0])
    radii = points.ravel()
    deficits = _log_radial_deficits(query, points, radii)
    terms = _adjusted_log_radii(query, points, radii, deficits, reflected=False)
    simulated_gap = (1 - points[1, 0]) / (1 - points[0, 0])
    assert terms[0] == pytest.approx(np.log1p(-simulated_gap), rel=1e-15, abs=0)


def test_reflected_separations_exceeding_float_range_keep_scale_invariance():
    points = np.array([0.0, 0.45, 0.75, 1.03, 1.37])[:, None]
    ordinary = TightLocalEstimator(n_neighbors=4).fit(points)
    huge = TightLocalEstimator(n_neighbors=4).fit(points * 1e308)
    np.testing.assert_allclose(
        huge.local_dimensions_, ordinary.local_dimensions_, rtol=2e-14
    )


@pytest.mark.parametrize("k", [3, 5])
@pytest.mark.parametrize("use_graph", [False, True])
def test_subnormal_units_recover_geometry_and_neighbor_order(k, use_graph):
    X = np.array([[0, 0], [1, 1], [3, 1], [4, 3], [0, 4], [3, 6]], dtype=float)
    tiny = np.nextafter(0.0, 1.0)
    expected = _first_contact_reference(X, k)
    graph = build_neighbor_graph(X * tiny, n_neighbors=k) if use_graph else None
    model = TightLocalEstimator(n_neighbors=k).fit(X * tiny, neighbors=graph)
    np.testing.assert_allclose(model.local_dimensions_, expected, rtol=1e-10)


def test_safe_aggregation_of_extreme_finite_pointwise_dimensions():
    X = np.column_stack((np.eye(4), np.array([0, 1, 2, 4]) * 1.5e-154))
    model = TightLocalEstimator(n_neighbors=3).fit(X)
    assert np.isfinite(model.local_dimensions_).all()
    expected = sum(model.local_dimensions_ / len(X))
    assert expected > 4e307
    assert model.dimension_ == pytest.approx(expected)
