"""Dense BDI: the group-level ``bdi``/``recursive_bdi`` factorizations and the
Hamiltonian-level signed SVD behind ``decompose_horizontal_hamiltonian``."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from kak_tools._horizontal_bdi import (
    _special_orthogonal_givens, decompose_horizontal_hamiltonian, horizontal_generator_decomposition,
)
from kak_tools.dense_cartan import bdi, recursive_bdi
from kak_tools.paulie_bridge import labelled_matrix_basis, map_dla_to_irrep
from checks import (
    assert_mirrored_vertical, bdi_cartan, expected_evolution, physical_evolution, plane_rotation,
    random_special_orthogonal, reconstruct_recursive_factors, rotation, rotation_product, skew_plane,
)


# Group level: bdi and recursive_bdi ----------------------------------------------------

def _horizontal_element(p, q, angles, seed):
    rng = np.random.default_rng(seed)
    k = block_diag(random_special_orthogonal(p, rng), random_special_orthogonal(q, rng))
    return k @ bdi_cartan(angles, p, q) @ k.T


def _assert_factors(u, p, q, horizontal=True):
    k11, k12, theta, k21, k22 = bdi(
        u, p, q, is_horizontal=horizontal, validate=True
    )
    k1, k2 = block_diag(k11, k12), block_diag(k21, k22)
    np.testing.assert_allclose(k1 @ bdi_cartan(theta, p, q) @ k2, u, atol=2e-10)
    for block in (k11, k12, k21, k22):
        np.testing.assert_allclose(block.T @ block, np.eye(len(block)), atol=2e-10)
        np.testing.assert_allclose(np.linalg.det(block), 1.0, atol=2e-10)
    if horizontal:
        np.testing.assert_allclose(k1, k2.T, atol=2e-10)


@pytest.mark.parametrize("p,q", [(1, 4), (4, 1)])
def test_horizontal_unequal_blocks_and_signs(p, q):
    _assert_factors(_horizontal_element(p, q, np.linspace(0.3, 2.2, min(p, q)), 19), p, q)


@pytest.mark.parametrize("p,q,angles", [
    (3, 3, [0, 0, 0]), (3, 4, [np.pi, np.pi, np.pi]),
    (4, 3, [0, np.pi, np.pi / 2]), (3, 3, [.7, .7, .7]),
    (3, 4, [np.pi / 2] * 3), (4, 3, [np.pi - 1e-7, 1e-7, 0]),
])
def test_horizontal_resonances_and_degenerate_eigenspaces(p, q, angles):
    _assert_factors(_horizontal_element(p, q, angles, 42), p, q)


@pytest.mark.parametrize("p,q,angles", [
    (2, 2, [1.2, np.pi - 1.2 + 1e-9]),
    (5, 6, [np.arccos(.98), np.arccos(.95), np.arccos(.25), np.pi - np.arccos(.25) + 1e-9, np.arccos(-.3)]),
])
def test_horizontal_resonant_pair_with_small_cosine_gap(p, q, angles):
    # Angles theta and pi - theta + delta have sines differing by delta cos(theta)
    # but cosines differing by 2 cos(theta). Pairing the planes through the sine
    # SVD alone mixes them by O(eps / (delta cos theta)); with |cos theta| <= 1/2
    # the combined cosine spectrum is narrower than one, which a largest-gap
    # split that stops at that width leaves unresolved.
    _assert_factors(_horizontal_element(p, q, angles, 11), p, q)


@pytest.mark.parametrize("p,q,scale", [(2, 3, 1e-8), (4, 4, 1e-5)])
def test_horizontal_angles_near_identity(p, q, scale):
    rng = np.random.default_rng(7)
    cross = scale * rng.normal(size=(p, q))
    generator = np.block([
        [np.zeros((p, p)), cross],
        [-cross.T, np.zeros((q, q))],
    ])
    _assert_factors(expm(generator), p, q)


@pytest.mark.parametrize("p,q,tiny", [(3, 3, 1e-7), (3, 4, 1e-10), (4, 3, 1e-14)])
def test_weak_sine_plane_remains_separate_from_pi_kernel(p, q, tiny):
    # Unlike a spectrum uniformly near zero, this has a large sine singular
    # value and a weak one beside a pi plane with the opposite cosine sign.
    # A single SVD of the sine block mixes the weak direction and that kernel
    # by O(eps / tiny), despite the original group matrix being well-conditioned.
    left = plane_rotation(p, 0, 1, .3) @ plane_rotation(p, 1, 2, .4)
    right = plane_rotation(q, 0, 1, .5) @ plane_rotation(q, 1, 2, .6)
    k = block_diag(left, right)
    u = k @ bdi_cartan([tiny, np.pi, .7], p, q) @ k.T
    k11, k12, theta, k21, k22 = bdi(u, p, q, validate=True)
    reconstructed = block_diag(k11, k12) @ bdi_cartan(theta, p, q) @ block_diag(k21, k22)
    # No default relative tolerance that could hide errors on entries near one.
    relative_error = np.linalg.norm(reconstructed - u) / np.linalg.norm(u)
    assert relative_error < 2e-13
    np.testing.assert_allclose(block_diag(k11, k12), block_diag(k21, k22).T, rtol=0, atol=2e-13)


@pytest.mark.parametrize("p,q", [(2, 3), (3, 2), (3, 3)])
def test_general_csd_path_preserves_reconstruction(p, q):
    u = random_special_orthogonal(p + q, np.random.default_rng(7))
    _assert_factors(u, p, q, horizontal=False)


def test_general_path_rejects_wrong_shape_and_non_orthogonal_input():
    with pytest.raises(ValueError, match="size 4"):
        bdi(np.eye(5), 2, 2, is_horizontal=False)
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="orthogonal"):
        bdi(rng.normal(size=(4, 4)), 2, 2, is_horizontal=False, validate=True)
    nearly = expm(rng.normal(size=(4, 4)) * .1) @ np.diag([1 + 1e-3, 1., 1., 1.])
    with pytest.raises(ValueError, match="orthogonal"):
        bdi(nearly, 2, 2, is_horizontal=False, validate=True)


@pytest.mark.parametrize("validate", [False, True])
def test_horizontal_mode_rejects_a_general_group_element(validate):
    u = np.eye(5)
    u[:2, :2] = rotation(0.4)
    with pytest.raises(ValueError, match="is_horizontal=False"):
        bdi(u, 2, 3, validate=validate)


@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("kwargs", [
    {}, {"num_iter": 1}, {"return_all": True}, {"return_all": True, "num_iter": 1},
    {"first_is_horizontal": False}, {"first_is_horizontal": False, "return_all": True},
])
def test_recursive_bdi_options_reconstruct_at_every_level(n, kwargs):
    p, q = n // 2, n - n // 2
    u = _horizontal_element(p, q, np.resize([np.pi, 0.4, np.pi / 2], p), 23)
    result = recursive_bdi(u, n, validate=True, **kwargs)
    if kwargs.get("return_all"):
        assert min(result) == -1
        levels = [factors for level, factors in sorted(result.items()) if level >= 0]
        if "num_iter" in kwargs:
            assert len(levels) == kwargs["num_iter"] + 1
    else:
        levels = [result]
    for factors in levels:
        np.testing.assert_allclose(reconstruct_recursive_factors(factors, n), u, atol=2e-10)
    if "num_iter" not in kwargs:
        # The recursion depth grows with n, so only check that it ran to the end.
        assert max(end - start for _, start, end, kind in levels[-1] if kind.startswith("k")) <= 2


# Hamiltonian level: the signed SVD and its physical lift --------------------------------

@pytest.mark.parametrize("p,q,kind", [
    (1, 1, "zero"), (1, 2, "random"), (2, 1, "random"),
    (2, 5, "rank_one"), (5, 2, "random"), (4, 4, "degenerate"),
])
def test_signed_svd(p, q, kind):
    rng = np.random.default_rng(21)
    block = rng.normal(size=(p, q))
    if kind == "zero":
        block[:] = 0
    elif kind == "rank_one":
        block = np.outer(rng.normal(size=p), rng.normal(size=q))
    elif kind == "degenerate":
        left, _ = np.linalg.qr(block)
        right, _ = np.linalg.qr(rng.normal(size=(q, q)))
        block = left @ right.T * np.pi / 2
    h = np.block([[np.zeros((p, p)), block], [-block.T, np.zeros((q, q))]])
    k, rates, planes = horizontal_generator_decomposition(h, p)
    a = sum(skew_plane(p + q, i, j, rate) for (i, j), rate in zip(planes, rates, strict=True))
    np.testing.assert_allclose(k @ a @ k.T, h, atol=2e-13)
    np.testing.assert_allclose(k @ k.T, np.eye(p + q), atol=2e-13)
    np.testing.assert_allclose([np.linalg.det(k[:p, :p]), np.linalg.det(k[p:, p:])], 1, atol=2e-13)


@pytest.mark.parametrize("matrix", [np.eye(1), -np.eye(2), expm(np.array([
    [0., .2, -.7], [-.2, 0., 1.1], [.7, -1.1, 0.],
]))])
def test_givens_including_pi(matrix):
    reconstructed = np.eye(len(matrix))
    for (i, j), angle in _special_orthogonal_givens(matrix):
        reconstructed = reconstructed @ expm(skew_plane(len(matrix), i, j, angle))
    np.testing.assert_allclose(reconstructed, matrix, atol=2e-13)


@pytest.mark.parametrize("matrix", [np.diag([-1., 1., 1.]), np.array([[1., .1], [0., 1.]])])
def test_givens_rejects_reflections_and_non_orthogonal_input(matrix):
    # Every elimination has determinant one, so the leftover diagonal would
    # otherwise carry a silent reflection into the rebuilt matrix.
    with pytest.raises(ValueError, match="special orthogonal"):
        _special_orthogonal_givens(matrix)


@pytest.mark.parametrize("p", [1, 3, 5])
def test_time_independent_factorization_and_physical_phase(p):
    # A fixed Clifford family realizes so(6); select horizontal words for p<q,
    # p=q and p>q independently of the bridge's default partition.
    mapping, signs, classification = map_dla_to_irrep(["XI", "YI", "ZX", "ZY", "ZZ"], invol_kwargs={"p": 1})
    matrices = labelled_matrix_basis(mapping, signs, classification)
    words = [mapping[(i, j)] for i in range(p) for j in range(p, 6)]
    coefficients = np.random.default_rng(p).normal(size=len(words))
    h = sum(c * matrices[w] for c, w in zip(coefficients, words, strict=True))
    rotations = decompose_horizontal_hamiltonian(h, p, mapping, signs, time=0)
    assert decompose_horizontal_hamiltonian(h, p, mapping, signs, time=4.2) == rotations
    assert_mirrored_vertical(rotations)
    np.testing.assert_allclose(rotation_product(rotations, matrices, 4.2), expm(4.2 * h), atol=3e-12)
    for time in [-1.3, np.pi, 4.2]:
        np.testing.assert_allclose(
            physical_evolution(rotations, time, 2), expected_evolution(words, coefficients, time, 2), atol=3e-12
        )


def test_small_rate_survives_long_time_and_vertical_pruning():
    mapping, signs, classification = map_dla_to_irrep(["X", "Y"])
    matrices = labelled_matrix_basis(mapping, signs, classification)
    h = 1e-12 * matrices[mapping[(0, 1)]]
    rotations = decompose_horizontal_hamiltonian(h, 1, mapping, signs, time=0, tol=1e-8)
    np.testing.assert_allclose(rotation_product(rotations, matrices, 1e12), expm(1e12 * h), atol=2e-13)


@pytest.mark.parametrize("p", [0, 3, 1.5, True])
def test_invalid_partition(p):
    with pytest.raises(ValueError, match="partition"):
        horizontal_generator_decomposition(np.zeros((3, 3)), p)


def test_nonhorizontal_hamiltonian():
    h = skew_plane(4, 0, 1, 1)
    with pytest.raises(ValueError, match="not horizontal"):
        horizontal_generator_decomposition(h, 2)
