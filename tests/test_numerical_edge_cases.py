"""Degenerate spectra, endpoint, precision and input-contract regressions of the numerical types."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm
from scipy.stats import ortho_group, unitary_group

from kak_tools import numerical_decompositions as nd
from kak_tools.numerical_decompositions import cii_kak
from checks import (
    assert_close, assert_diagonal, assert_embedded, assert_orthogonal, assert_repeat, assert_skew_schur,
    assert_symplectic, assert_symplectic_diagonal, bdi_cartan, cii_sectors, random_group_element,
    random_sp_generator, rotation, unitary_in_so2n,
)


# A and AI: repeated eigenvalues of Delta -----------------------------------------

@pytest.mark.parametrize("phase", [1, -1, 1j], ids=["same", "negated", "quarter-turn"])
def test_a_kak_repeated_delta_eigenvalues(phase):
    # eig returns a non-orthonormal basis of the degenerate eigenspaces of
    # delta = u0 @ (phase u0)^dagger, which is a multiple of the identity.
    for seed in range(5):
        u0 = unitary_group.rvs(4, random_state=seed)
        matrix = block_diag(u0, phase * u0)
        k1, center, k2 = nd.a_kak(matrix, validate=False)
        assert_close(k1 @ center @ k2, matrix)
        assert_repeat(k1)
        assert_repeat(center, check=assert_diagonal, transform=np.conj)
        assert_repeat(k2)


@pytest.mark.parametrize("phases", [
    [0.3, 0.3, -0.3, 1.1],
    [0.3, 0.3, 0.3, -0.3, -0.3, 1.1],
    [0.7, 0.7, 0.7, 0.7, 0.1, 0.2],
    [0.3, 0.3, 0.5, 0.5, 0.9, 0.9, 1.2, 1.2],
    [np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2],
    [0.0] * 8,
], ids=["double", "triple", "quadruple", "four-pairs", "right-angles", "complex-typed-orthogonal"])
def test_ai_kak_repeated_phases(phases):
    # u @ u.T has repeated eigenvalues, for which LAPACK returns complex eigenvectors
    # that are neither conjugation-proportional nor orthonormal. Seed 10 is one of
    # the few frames where that also happens for the identity Delta of a real input.
    for seed in range(10, 20):
        frame = random_group_element("orthogonal", len(phases), np.random.default_rng(seed))
        matrix = frame @ np.diag(np.exp(1j * np.asarray(phases)))
        k1, center, k2 = nd.ai_kak(matrix)
        assert_close(k1 @ center @ k2, matrix)
        assert_orthogonal(k1)
        assert_diagonal(center)
        assert_orthogonal(k2)


# BD: Schur singletons and the Schur square root ------------------------------------

@pytest.mark.parametrize("angles,plus,minus,seed", [((0.4, 1.3), 1, 2, 8), ((0.4, 1.3, 2.2), 3, 2, 2)], ids=["isolated-axes", "odd-plus-run"])
def test_bd_schur_singletons(bd_schur_matrix, angles, plus, minus, seed):
    """Upstream regressions: isolated +/-1 Schur axes and an odd run of +1 axes."""
    matrix = bd_schur_matrix(angles, plus, minus, seed)
    k1, center, k2 = nd.bd_kak(matrix, validate=True)
    assert_close(k1 @ center @ k2, matrix)
    assert_repeat(k1, check=assert_orthogonal)
    assert_skew_schur(center)
    assert_repeat(k2, check=assert_orthogonal)


@pytest.mark.parametrize("plus", [0, 1], ids=["even", "odd"])
@pytest.mark.parametrize("deviation", [1e-9, 1e-6, 1e-4])
def test_bd_near_pi_rotation_beside_minus_one_axes(bd_schur_matrix, deviation, plus):
    # LAPACK may return the near-pi block between the two exact -1 axes; pairing
    # Schur entries by position then split the rotation in half.
    for seed in range(8):
        matrix = bd_schur_matrix([np.pi - deviation], plus, 2, seed)
        k1, center, k2 = nd.bd_kak(matrix, validate=True)
        assert_close(k1 @ center @ k2, matrix, atol=1e-13)
        assert_repeat(k1, check=assert_orthogonal, atol=1e-13)
        assert_skew_schur(center, atol=1e-13)
        assert_repeat(k2, check=assert_orthogonal, atol=1e-13)


@pytest.mark.parametrize("angle", [0.0, 1e-10, np.pi], ids=["identity", "tiny-rotation", "half-turn"])
def test_schur_sqrt_with_fixed_axis(angle):
    matrix = block_diag(rotation(angle), 1)
    square_root = nd.schur_sqrt(matrix)
    assert_close(square_root @ square_root, matrix, atol=1e-14)
    assert_orthogonal(square_root, atol=1e-14)


def test_schur_sqrt_pairs_minus_one_axes():
    square_root = nd.schur_sqrt(np.diag([-1.0, -1.0, 1.0]))
    assert_close(square_root, block_diag([[0, 1], [-1, 0]], 1), atol=1e-15)
    with pytest.raises(ValueError, match="unpaired"):
        nd.schur_sqrt(np.diag([-1.0, 1.0]))


# CII: structure, endpoints, precision and input contracts ----------------------------

def _cii_vertical(p, q, rng):
    """Random element of Sp(p) x Sp(q) in the sector presentation."""
    matrix = np.zeros((2 * (p + q), 2 * (p + q)), dtype=complex)
    for size, indices in zip((p, q), cii_sectors(p, q), strict=True):
        matrix[np.ix_(indices, indices)] = expm(random_sp_generator(size, rng)) if size else np.eye(0)
    return matrix


def _cii_cartan(p, q, angles):
    """The BDI(p, q) Cartan element doubled onto both symplectic halves."""
    block = bdi_cartan(angles, p, q)
    return block_diag(block, block)


def _assert_cii_factors(matrix, p, q, validate=True, atol=2e-11):
    original = matrix.copy()
    k1, center, k2 = cii_kak(matrix, p, q, validate=validate)
    np.testing.assert_array_equal(matrix, original)
    reconstructed = k1 @ center @ k2
    assert_close(reconstructed, matrix, atol)
    assert np.linalg.norm(reconstructed - matrix) <= atol * max(1, np.linalg.norm(matrix))
    for factor in (k1, center, k2):
        assert_symplectic(factor, atol)
        assert_close(np.linalg.det(factor), 1, 10 * atol)
    first, second = cii_sectors(p, q)
    for factor in (k1, k2):
        assert_close(factor[np.ix_(first, second)], 0, atol)
        assert_close(factor[np.ix_(second, first)], 0, atol)
    indices = np.arange(min(p, q))
    angles = np.arctan2(center[indices, max(p, q) + indices].real, center[indices, indices].real)
    assert_close(center, _cii_cartan(p, q, angles), atol)
    assert np.all(np.diff(angles) <= atol)
    return k1, center, k2


@pytest.mark.parametrize("p,q,angles,validate", [
    (3, 3, [0, 0, 0], True),
    (2, 4, [np.pi / 2, np.pi / 2], True),
    (4, 2, [-0.3, -1.1], False),
    (3, 4, [0, np.pi / 2, -0.7], True),
    (4, 2, [0.7, 0.7], True),
    (2, 4, [0.7, 0.7], False),
    (2, 4, [0.7, 0.7 + 1e-10], False),
    (3, 3, [0.7, 0.7 + 1e-14, 0.7 + 2e-14], True),
    (2, 4, [1e-9, 3e-9], True),
    (4, 2, [1e-9, 3e-9], False),
    (4, 3, [1e-9, 0.7, np.pi / 2 - 2e-9], False),
])
def test_cii_independent_vertical_factors_and_spectra(p, q, angles, validate):
    rng = np.random.default_rng(991 + 37 * p + 13 * q)
    matrix = _cii_vertical(p, q, rng) @ _cii_cartan(p, q, angles) @ _cii_vertical(p, q, rng)
    k1, center, k2 = _assert_cii_factors(matrix, p, q, validate)
    if 0 < max(np.abs(angles)) < 1e-8:
        first, second = cii_sectors(p, q)
        cross = matrix[np.ix_(first, second)]
        error = (k1 @ center @ k2 - matrix)[np.ix_(first, second)]
        assert np.linalg.norm(error) < 2e-13 + 2e-5 * np.linalg.norm(cross)


@pytest.mark.parametrize("p,q,validate", [(0, 0, True), (0, 3, False), (3, 0, True)])
def test_cii_empty_partitions(p, q, validate):
    matrix = _cii_vertical(p, q, np.random.default_rng(904))
    k1, center, k2 = _assert_cii_factors(matrix, p, q, validate)
    assert_close(center, np.eye(len(matrix)))
    assert not np.shares_memory(k1, matrix)
    assert not np.shares_memory(center, k2)


def test_cii_roundoff_split_symplectic_pairs():
    _assert_cii_factors(block_diag(rotation(0.4), rotation(0.4 + 1e-13)), 1, 1, atol=3e-13)


@pytest.mark.parametrize("p,q,validate", [(1, 3, True), (3, 1, False)])
def test_cii_long_time_symplectic_exponential(p, q, validate):
    matrix = expm(1e4 * random_sp_generator(p + q, np.random.default_rng(719)))
    _assert_cii_factors(matrix, p, q, validate, atol=3e-11)


@pytest.mark.parametrize("near_right_angle", [False, True])
def test_cii_entirely_weak_constrained_block(near_right_angle):
    angle = np.pi / 2 - 1e-14 if near_right_angle else 1e-14
    phases = [1j, 1, -1j, 1] if near_right_angle else [1, 1j, 1, -1j]
    matrix = np.diag(phases) @ _cii_cartan(1, 1, [angle])
    _, center, k2 = _assert_cii_factors(matrix, 1, 1)
    first, second = cii_sectors(1, 1)
    tolerance = 5e-15 * (np.cos(angle) if near_right_angle else np.sin(angle))
    # Product reconstruction alone can hide the lost weak direction in K2.
    assert_close(k2[np.ix_(first, second)], 0, tolerance)
    assert_close(k2[np.ix_(second, first)], 0, tolerance)
    assert_close(center, _cii_cartan(1, 1, [angle]), tolerance)


@pytest.mark.parametrize("p,q", [(True, 1), (1, np.bool_(False)), (-1, 1), (1, 1.5)])
def test_cii_invalid_partition(p, q):
    with pytest.raises(ValueError, match="nonnegative integers"):
        cii_kak(np.eye(4), p, q, validate=False)


@pytest.mark.parametrize("p,q,validate", [(0, 2, False), (1, 1, True)])
def test_cii_wrong_shape(p, q, validate):
    with pytest.raises(ValueError, match=r"2 \* \(p \+ q\)"):
        cii_kak(np.eye(3), p, q, validate=validate)


@pytest.mark.parametrize("entry,validate", [(np.nan, True), (np.inf, False), ("invalid", False)])
def test_cii_invalid_entries(entry, validate):
    matrix = np.eye(4, dtype=object if isinstance(entry, str) else float)
    matrix[0, 0] = entry
    with pytest.raises(ValueError, match="finite numeric"):
        cii_kak(matrix, 1, 1, validate=validate)


@pytest.mark.parametrize("p,q,matrix,message", [
    (0, 2, 2 * np.eye(4), "unitary"),
    (2, 0, np.diag([1j, 1, 1, 1]), "symplectic"),
    (1, 1, 2 * np.eye(4), "unitary"),
    (1, 1, np.diag([1j, 1, 1, 1]), "symplectic"),
])
def test_cii_input_group(p, q, matrix, message):
    with pytest.raises(ValueError, match=message):
        cii_kak(matrix, p, q)


# DIII and CI: the real eigenbasis at real, repeated and tiny eigenvalues -----------------

def _assert_embedded_factors(matrix, kind="diii", validate=True, atol=1e-10):
    with np.errstate(invalid="raise", divide="raise"):
        k1, center, k2 = getattr(nd, kind + "_kak")(matrix, validate=validate)
    assert_embedded(k1, atol)
    assert_embedded(k2, atol)
    (assert_skew_schur if kind == "diii" else assert_symplectic_diagonal)(center, atol)
    assert_close(k1 @ center @ k2, matrix, atol)


@pytest.mark.parametrize("size,seed", [(20, 154), (50, 95)])
def test_diii_random_phase_and_roundoff_endpoint_failures(size, seed):
    # Seed 154 annihilated a zero-phase vector; seed 95 overcounted the +1 space.
    matrix = ortho_group.rvs(size, random_state=seed)
    matrix[0] *= np.linalg.det(matrix)
    _assert_embedded_factors(matrix)


@pytest.mark.parametrize("minus_one", [False, True])
def test_diii_complex_eigenbasis_at_real_endpoints(monkeypatch, minus_one):
    mu = block_diag(rotation(np.pi / 2), 1) if minus_one else np.eye(3)
    n = len(mu)
    eigenvectors = np.block([[np.eye(n), np.eye(n)], [1j * np.eye(n), -1j * np.eye(n)]]) / np.sqrt(2)

    def degenerate_eig(matrix):
        eigenvalues = np.diag(matrix).astype(complex)
        assert_close(matrix @ eigenvectors, eigenvectors * eigenvalues)
        return eigenvalues, eigenvectors.copy()

    monkeypatch.setattr(nd, "eig", degenerate_eig)
    _assert_embedded_factors(block_diag(mu, mu.T))


@pytest.mark.parametrize("n,angle,rotate,validate", [
    (2, 1e-10, False, True),
    (3, 1e-7, False, True),
    (5, np.pi / 2 - 1e-10, False, True),
    (2, 1e-10, True, True),
    (5, 1e-10, True, False),
    (3, np.pi / 2 - 1e-10, True, True),
    (5, np.pi / 2 - 1e-10, True, False),
])
def test_diii_small_rotations_and_deflated_eigenvector_residuals(n, angle, rotate, validate):
    mu = np.eye(n)
    mu[:2, :2] = rotation(angle)
    matrix = block_diag(mu, mu.T)
    if rotate:
        matrix = unitary_in_so2n(n, 0) @ matrix @ unitary_in_so2n(n, 100)
    _assert_embedded_factors(matrix, validate=validate, atol=1e-13)


def test_diii_zero_bilinear_overlap():
    block = rotation(0.3)
    _assert_embedded_factors(block_diag(block, block.T))


def test_ci_degenerate_eigenbasis_with_zero_bilinear_overlap(monkeypatch):
    eigenvectors = np.eye(4, dtype=complex)
    eigenvectors[:2, :2] = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)

    def degenerate_eig(matrix):
        eigenvalues = np.diag(matrix).copy()
        assert_close(matrix @ eigenvectors, eigenvectors * eigenvalues)
        return eigenvalues, eigenvectors.copy()

    monkeypatch.setattr(nd, "eig", degenerate_eig)
    matrix = np.diag(np.exp(1j * np.array([0.3, 0.3, -0.3, -0.3])))
    _assert_embedded_factors(matrix, kind="ci")


def test_ci_phase_reference_ignores_tiny_leading_coordinate():
    # The deflated leading coordinate is about 1e-16; its phase is unreliable.
    diagonal = np.diag(np.exp(1j * np.array([0.3, 0.3, -0.3, -0.3])))
    _assert_embedded_factors(unitary_in_so2n(2, 145) @ diagonal, kind="ci")


def _ci_element(angles, seed):
    """K1 (D (+) D*) K2 with K in U(n) embedded in SO(2n) and Cartan angles ``angles``."""
    angles = np.asarray(angles, dtype=float)
    center = np.diag(np.exp(1j * np.concatenate([angles, -angles])))
    return unitary_in_so2n(len(angles), seed) @ center @ unitary_in_so2n(len(angles), seed + 500)


@pytest.mark.parametrize("angles", [
    (np.pi / 2, np.pi / 2),
    (np.pi, 0.0),
    (np.pi, np.pi),
    (np.pi / 2, -np.pi / 2),
    (np.pi, 0.3),
    (1e-12, -1e-12),
    (0.3, 0.3 + 1e-11),
    (0.3, 0.3 + 1e-8),
    (0.3, 0.3, 1.1, 1.1),
    (0.0, 0.0, 0.0, 0.0, 0.3, 0.3),
    (np.pi, np.pi, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.7, -0.7),
], ids=[
    "right-angles", "pi-and-zero", "two-pi", "opposite-right-angles", "pi-and-generic", "near-identity",
    "split-1e-11", "split-1e-8", "two-doubles", "quadruple-zero", "n-10-mixed",
])
def test_ci_real_and_nearly_repeated_delta_eigenvalues(angles):
    # A real (+-1) or nearly repeated eigenvalue of S S^T comes with complex LAPACK
    # eigenvectors that are neither conjugation-proportional nor orthonormal, and
    # whose imaginary parts may be roundoff; the real eigenbasis must not
    # normalize such parts into itself.
    for seed in range(10, 20):
        _assert_embedded_factors(_ci_element(angles, seed), kind="ci")


@pytest.mark.parametrize("n,seeds", [(4, range(10, 20)), (25, range(30, 40))])
def test_ci_vertical_input(n, seeds):
    # S in K = U(n) has Delta = I up to roundoff, a fully degenerate real eigenspace
    # for which LAPACK occasionally returns complex eigenvectors (seeds 13 and 30).
    for seed in seeds:
        _assert_embedded_factors(unitary_in_so2n(n, seed), kind="ci", atol=1e-9)
