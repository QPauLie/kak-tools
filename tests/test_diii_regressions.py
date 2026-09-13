"""The concrete CI/DIII eigenbasis, phase and endpoint failures."""

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.stats import ortho_group, unitary_group

from kak_tools import numerical_decompositions as nd
from test_numerical_decompositions import _close, _embedded, _skew_schur, _symplectic_diagonal


def _check(matrix, kind="diii", validate=True, atol=1e-10):
    with np.errstate(invalid="raise", divide="raise"):
        k1, center, k2 = getattr(nd, kind + "_kak")(matrix, validate=validate)
    _embedded(k1, atol)
    _embedded(k2, atol)
    (_skew_schur if kind == "diii" else _symplectic_diagonal)(center, atol)
    _close(k1 @ center @ k2, matrix, atol)


def _rotation(angle):
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def _vertical(n, seed):
    u = unitary_group.rvs(n, random_state=seed)
    return np.block([[u.real, u.imag], [-u.imag, u.real]])


@pytest.mark.parametrize("size,seed", [(20, 154), (50, 95)])
def test_diii_random_phase_and_roundoff_endpoint_failures(size, seed):
    # Seed 154 annihilated a zero-phase vector; seed 95 overcounted the +1 space.
    matrix = ortho_group.rvs(size, random_state=seed)
    matrix[0] *= np.linalg.det(matrix)
    _check(matrix)


@pytest.mark.parametrize("minus_one", [False, True])
def test_diii_complex_eigenbasis_at_real_endpoints(monkeypatch, minus_one):
    mu = block_diag(_rotation(np.pi / 2), 1) if minus_one else np.eye(3)
    n = len(mu)
    eigenvectors = np.block([[np.eye(n), np.eye(n)], [1j * np.eye(n), -1j * np.eye(n)]]) / np.sqrt(2)

    def degenerate_eig(matrix):
        eigenvalues = np.diag(matrix).astype(complex)
        _close(matrix @ eigenvectors, eigenvectors * eigenvalues)
        return eigenvalues, eigenvectors.copy()

    monkeypatch.setattr(nd, "eig", degenerate_eig)
    _check(block_diag(mu, mu.T))


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
    mu[:2, :2] = _rotation(angle)
    matrix = block_diag(mu, mu.T)
    if rotate:
        matrix = _vertical(n, 0) @ matrix @ _vertical(n, 100)
    _check(matrix, validate=validate, atol=1e-13)


def test_diii_zero_bilinear_overlap():
    rotation = _rotation(0.3)
    _check(block_diag(rotation, rotation.T))


def test_ci_degenerate_eigenbasis_with_zero_bilinear_overlap(monkeypatch):
    eigenvectors = np.eye(4, dtype=complex)
    eigenvectors[:2, :2] = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)

    def degenerate_eig(matrix):
        eigenvalues = np.diag(matrix).copy()
        _close(matrix @ eigenvectors, eigenvectors * eigenvalues)
        return eigenvalues, eigenvectors.copy()

    monkeypatch.setattr(nd, "eig", degenerate_eig)
    matrix = np.diag(np.exp(1j * np.array([0.3, 0.3, -0.3, -0.3])))
    _check(matrix, kind="ci")


def test_ci_phase_reference_ignores_tiny_leading_coordinate():
    # The deflated leading coordinate is about 1e-16; its phase is unreliable.
    diagonal = np.diag(np.exp(1j * np.array([0.3, 0.3, -0.3, -0.3])))
    _check(_vertical(2, 145) @ diagonal, kind="ci")


def _ci_element(angles, seed):
    """K1 (D (+) D*) K2 with K in U(n) embedded in SO(2n) and Cartan angles ``angles``."""
    angles = np.asarray(angles, dtype=float)
    center = np.diag(np.exp(1j * np.concatenate([angles, -angles])))
    return _vertical(len(angles), seed) @ center @ _vertical(len(angles), seed + 500)


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
        _check(_ci_element(angles, seed), kind="ci")


@pytest.mark.parametrize("n,seeds", [(4, range(10, 20)), (25, range(30, 40))])
def test_ci_vertical_input(n, seeds):
    # S in K = U(n) has Delta = I up to roundoff, a fully degenerate real eigenspace
    # for which LAPACK occasionally returns complex eigenvectors (seeds 13 and 30).
    for seed in seeds:
        _check(_vertical(n, seed), kind="ci", atol=1e-9)
