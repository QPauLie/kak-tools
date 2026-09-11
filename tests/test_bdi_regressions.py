"""Group-level regressions for horizontal and general BDI decompositions."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from kak_tools.dense_cartan import bdi, recursive_bdi


def _cartan(theta, p, q):
    generator = np.zeros((p + q, p + q))
    for i, angle in enumerate(theta):
        j = max(p, q) + i
        generator[i, j], generator[j, i] = angle, -angle
    return expm(generator)


def _orthogonal_block(size, rng):
    matrix = rng.normal(size=(size, size))
    return expm(matrix - matrix.T)


def _assert_factors(u, p, q, horizontal=True):
    k11, k12, theta, k21, k22 = bdi(
        u, p, q, is_horizontal=horizontal, validate=True
    )
    k1, k2 = block_diag(k11, k12), block_diag(k21, k22)
    np.testing.assert_allclose(k1 @ _cartan(theta, p, q) @ k2, u, atol=2e-10)
    for block in (k11, k12, k21, k22):
        np.testing.assert_allclose(block.T @ block, np.eye(len(block)), atol=2e-10)
        np.testing.assert_allclose(np.linalg.det(block), 1.0, atol=2e-10)
    if horizontal:
        np.testing.assert_allclose(k1, k2.T, atol=2e-10)


@pytest.mark.parametrize("p,q", [(1, 4), (4, 1)])
def test_horizontal_unequal_blocks_and_signs(p, q):
    rng = np.random.default_rng(19)
    k = block_diag(_orthogonal_block(p, rng), _orthogonal_block(q, rng))
    theta = np.linspace(0.3, 2.2, min(p, q))
    u = k @ _cartan(theta, p, q) @ k.T
    _assert_factors(u, p, q)


@pytest.mark.parametrize("p,q,angles", [
    (3, 3, [0, 0, 0]), (3, 4, [np.pi, np.pi, np.pi]),
    (4, 3, [0, np.pi, np.pi / 2]), (3, 3, [.7, .7, .7]),
    (3, 4, [np.pi / 2] * 3), (4, 3, [np.pi - 1e-7, 1e-7, 0]),
])
def test_horizontal_resonances_and_degenerate_eigenspaces(p, q, angles):
    rng = np.random.default_rng(42)
    k = block_diag(_orthogonal_block(p, rng), _orthogonal_block(q, rng))
    u = k @ _cartan(angles, p, q) @ k.T
    _assert_factors(u, p, q)


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
    def rotation(size, i, j, angle):
        block = np.eye(size)
        block[i, i] = block[j, j] = np.cos(angle)
        block[i, j], block[j, i] = np.sin(angle), -np.sin(angle)
        return block

    left = rotation(p, 0, 1, .3) @ rotation(p, 1, 2, .4)
    right = rotation(q, 0, 1, .5) @ rotation(q, 1, 2, .6)
    k = block_diag(left, right)
    u = k @ _cartan([tiny, np.pi, .7], p, q) @ k.T
    k11, k12, theta, k21, k22 = bdi(u, p, q, validate=True)
    reconstructed = block_diag(k11, k12) @ _cartan(theta, p, q) @ block_diag(k21, k22)
    # No default relative tolerance that could hide errors on entries near one.
    relative_error = np.linalg.norm(reconstructed - u) / np.linalg.norm(u)
    assert relative_error < 2e-13
    np.testing.assert_allclose(block_diag(k11, k12), block_diag(k21, k22).T, rtol=0, atol=2e-13)


@pytest.mark.parametrize("p,q", [(2, 3), (3, 2), (3, 3)])
def test_general_csd_path_preserves_reconstruction(p, q):
    rng = np.random.default_rng(7)
    u = _orthogonal_block(p + q, rng)
    _assert_factors(u, p, q, horizontal=False)


@pytest.mark.parametrize("validate", [False, True])
def test_horizontal_mode_rejects_a_general_group_element(validate):
    u = np.eye(5)
    angle = 0.4
    u[:2, :2] = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    with pytest.raises(ValueError, match="is_horizontal=False"):
        bdi(u, 2, 3, validate=validate)


@pytest.mark.parametrize("n", [5, 6])
def test_recursive_horizontal_factors_reconstruct(n):
    p, q = n // 2, n - n // 2
    rng = np.random.default_rng(23)
    k = block_diag(_orthogonal_block(p, rng), _orthogonal_block(q, rng))
    angles = np.resize([np.pi, 0.4, np.pi / 2], p)
    u = k @ _cartan(angles, p, q) @ k.T
    factors = recursive_bdi(u, n, validate=True)
    reconstructed = np.eye(n)
    for block, start, end, kind in factors:
        embedded = np.eye(n)
        width = end - start
        embedded[start:end, start:end] = (
            _cartan(block, width // 2, width - width // 2)
            if kind.startswith("a") else block
        )
        reconstructed = reconstructed @ embedded
    np.testing.assert_allclose(reconstructed, u, atol=2e-10)
