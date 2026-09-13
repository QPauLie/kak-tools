"""Hamiltonian factorization and its physical lift, without parameter sweeps."""

import numpy as np
import pytest
from scipy.linalg import expm

from kak_tools._horizontal_bdi import (
    _special_orthogonal_givens, decompose_horizontal_hamiltonian,
    horizontal_generator_decomposition,
)
from kak_tools.paulie_bridge import labelled_matrix_basis, map_dla_to_irrep


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
    a = np.zeros_like(h)
    for (i, j), rate in zip(planes, rates):
        a[i, j], a[j, i] = rate, -rate
    np.testing.assert_allclose(k @ a @ k.T, h, atol=2e-13)
    np.testing.assert_allclose(k @ k.T, np.eye(p + q), atol=2e-13)
    np.testing.assert_allclose([np.linalg.det(k[:p, :p]), np.linalg.det(k[p:, p:])], 1, atol=2e-13)


@pytest.mark.parametrize("matrix", [np.eye(1), -np.eye(2), expm(np.array([
    [0., .2, -.7], [-.2, 0., 1.1], [.7, -1.1, 0.],
]))])
def test_givens_including_pi(matrix):
    reconstructed = np.eye(len(matrix))
    for (i, j), angle in _special_orthogonal_givens(matrix):
        plane = np.zeros_like(matrix)
        plane[i, j], plane[j, i] = angle, -angle
        reconstructed = reconstructed @ expm(plane)
    np.testing.assert_allclose(reconstructed, matrix, atol=2e-13)


@pytest.mark.parametrize("p", [1, 3, 5])
def test_time_independent_factorization_and_physical_phase(p):
    # A fixed Clifford family realizes so(6); select horizontal words for p<q,
    # p=q and p>q independently of the bridge's default partition.
    mapping, signs, classification = map_dla_to_irrep(["XI", "YI", "ZX", "ZY", "ZZ"], invol_kwargs={"p": 1})
    matrices = labelled_matrix_basis(mapping, signs, classification)
    words = [mapping[(i, j)] for i in range(p) for j in range(p, 6)]
    coefficients = np.random.default_rng(p).normal(size=len(words))
    h = sum(c * matrices[w] for c, w in zip(coefficients, words))
    physical_h = sum(c * w.to_mat(wire_order=[0, 1]) for c, w in zip(coefficients, words))
    rotations = decompose_horizontal_hamiltonian(h, p, mapping, signs, time=0)
    assert decompose_horizontal_hamiltonian(h, p, mapping, signs, time=4.2) == rotations
    left = [(w, a) for w, a, kind in rotations if kind == "k1"]
    assert [(w, a) for w, a, kind in rotations if kind == "k2"] == [(w, -a) for w, a in reversed(left)]
    reconstructed = np.eye(6)
    for word, angle, kind in rotations:
        reconstructed = reconstructed @ expm(matrices[word] * angle * (4.2 if kind == "a0" else 1))
    np.testing.assert_allclose(reconstructed, expm(4.2 * h), atol=3e-12)
    for time in [-1.3, np.pi, 4.2]:
        physical = np.eye(4, dtype=complex)
        for word, angle, kind in rotations:
            physical = physical @ expm(1j * angle * (time if kind == "a0" else 1) * word.to_mat(wire_order=[0, 1]))
        np.testing.assert_allclose(physical, expm(1j * time * physical_h), atol=3e-12)


def test_small_rate_survives_long_time_and_vertical_pruning():
    mapping, signs, classification = map_dla_to_irrep(["X", "Y"])
    matrices = labelled_matrix_basis(mapping, signs, classification)
    h = 1e-12 * matrices[mapping[(0, 1)]]
    rotations = decompose_horizontal_hamiltonian(h, 1, mapping, signs, time=0, tol=1e-8)
    reconstructed = np.eye(3)
    for word, angle, kind in rotations:
        reconstructed = reconstructed @ expm(matrices[word] * angle * (1e12 if kind == "a0" else 1))
    np.testing.assert_allclose(reconstructed, expm(1e12 * h), atol=2e-13)


@pytest.mark.parametrize("p", [0, 3, 1.5, True])
def test_invalid_partition(p):
    with pytest.raises(ValueError, match="partition"):
        horizontal_generator_decomposition(np.zeros((3, 3)), p)


def test_nonhorizontal_hamiltonian():
    h = np.zeros((4, 4))
    h[0, 1], h[1, 0] = 1, -1
    with pytest.raises(ValueError, match="not horizontal"):
        horizontal_generator_decomposition(h, 2)
