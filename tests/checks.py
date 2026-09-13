"""Assertion helpers and matrix builders shared by the test modules.

The membership checks take ``atol`` as an absolute tolerance and no relative
one, so entries near one cannot hide errors.
"""

from functools import partial

import numpy as np
from pennylane.pauli import PauliWord
from scipy.linalg import expm
from scipy.stats import ortho_group, unitary_group

from kak_tools import as_pauli_words


# Closeness and group membership -----------------------------------------------

def assert_close(actual, expected, atol=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def symplectic_form(n):
    return np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])


def random_sp_generator(n, rng):
    """Random element of sp(n) in the ``[[A, B], [-B*, A*]]`` presentation."""
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a, b = a - a.conj().T, b + b.T
    return np.block([[a, b], [-b.conj(), a.conj()]]) / np.sqrt(4 * n)


def random_group_element(kind, n, rng):
    """Haar-random element of U(n) or SO(n), or a random Sp(n) element ``expm`` of a generator.

    ``ortho_group`` needs ``n >= 2``; :func:`random_special_orthogonal` covers ``n = 1``.
    """
    if kind == "unitary":
        return unitary_group.rvs(n, random_state=rng)
    if kind == "orthogonal":
        matrix = ortho_group.rvs(n, random_state=rng)
        matrix[0] *= np.linalg.det(matrix)
        return matrix
    return expm(random_sp_generator(n, rng))


def random_special_orthogonal(n, rng):
    """Random SO(n) element as the exponential of a random skew-symmetric matrix."""
    matrix = rng.normal(size=(n, n))
    return expm(matrix - matrix.T)


def unitary_in_so2n(n, seed):
    """Random U(n) element in its real embedding ``[[Re, Im], [-Im, Re]]`` in SO(2n)."""
    u = unitary_group.rvs(n, random_state=seed)
    return np.block([[u.real, u.imag], [-u.imag, u.real]])


def assert_unitary(x, atol=1e-8):
    assert np.isfinite(x).all()
    assert_close(x @ x.conj().T, np.eye(len(x)), atol)


def assert_orthogonal(x, atol=1e-8):
    assert_unitary(x, atol)
    assert_close(x.imag, 0, atol)
    assert_close(np.linalg.det(x), 1, atol)


def assert_symplectic(x, atol=1e-8):
    assert_unitary(x, atol)
    j = symplectic_form(len(x) // 2)
    assert_close(x.T @ j @ x, j, atol)


def assert_embedded(x, atol=1e-8):
    """Orthogonal and symplectic: the real embedding of a unitary."""
    assert_orthogonal(x, atol)
    assert_symplectic(x, atol)


def assert_diagonal(x, atol=1e-8):
    assert_unitary(x, atol)
    assert_close(x, np.diag(np.diag(x)), atol)


def assert_symplectic_diagonal(x, atol=1e-8):
    assert_diagonal(x, atol)
    assert_symplectic(x, atol)


def assert_blocks(x, p, check, atol=1e-8):
    """Block-diagonal with sizes ``p`` and ``len(x) - p``, each block passing ``check``."""
    assert_close(x[:p, p:], 0, atol)
    assert_close(x[p:, :p], 0, atol)
    check(x[:p, :p], atol=atol)
    check(x[p:, p:], atol=atol)


def assert_repeat(x, check=assert_unitary, transform=lambda a: a, atol=1e-8):
    """Block-diagonal ``block_diag(y, transform(y))`` with ``y`` passing ``check``."""
    n = len(x) // 2
    assert_blocks(x, n, check, atol)
    assert_close(x[n:, n:], transform(x[:n, :n]), atol)


def assert_schur(x, atol=1e-8):
    """Real Schur form of an SO(n) element: 2x2 rotation blocks and, for odd n, a +1 axis."""
    assert_orthogonal(x, atol)
    if len(x) % 2:
        # The lone +1 axis comes last; a rotation by less than atol also looks
        # isolated, so searching from the front could pick one of its axes.
        off_diagonal = x - np.diag(np.diag(x))
        isolated = (np.max(abs(off_diagonal), axis=0) <= atol) & (np.max(abs(off_diagonal), axis=1) <= atol)
        i = np.flatnonzero(isolated & (abs(np.diag(x) - 1) <= atol))[-1]
        keep = np.arange(len(x)) != i
        assert_schur(x[np.ix_(keep, keep)], atol)
        return
    diagonal, upper, lower = np.diag(x), np.diag(x, 1), np.diag(x, -1)
    assert_close(diagonal[::2], diagonal[1::2], atol)
    assert_close(upper, -lower, atol)
    assert_close(diagonal[::2] ** 2 + upper[::2] ** 2, 1, atol)
    assert_close(x, np.diag(diagonal) + np.diag(upper, 1) + np.diag(lower, -1), atol)


def assert_skew_schur(x, atol=1e-8):
    assert_repeat(x, assert_schur, np.transpose, atol)


def assert_cossin(x, p, q, atol=1e-8):
    """Cosine-sine form of a BDI(p, q) or AIII(p, q) Cartan element."""
    assert_orthogonal(x, atol)
    r, s = min(p, q), max(p, q)
    c, sine = x[:r, :r], x[:r, s:]
    assert_close(c, np.diag(np.diag(c)), atol)
    assert_close(sine, np.diag(np.diag(sine)), atol)
    expected = np.eye(p + q)
    expected[:r, :r] = expected[s:, s:] = c
    expected[:r, s:], expected[s:, :r] = sine, -sine
    assert_close(x, expected, atol)
    assert_close(np.diag(c) ** 2 + np.diag(sine) ** 2, 1, atol)


def cii_sectors(p, q):
    """Index sets of the Sp(p) and Sp(q) sectors inside the 2(p + q) symplectic presentation."""
    n = p + q
    return np.r_[0:p, n : n + p], np.r_[p:n, n + p : 2 * n]


def assert_sympl_blocks(x, p, q, atol=1e-8):
    first, second = cii_sectors(p, q)
    assert_close(x[np.ix_(first, second)], 0, atol)
    assert_close(x[np.ix_(second, first)], 0, atol)
    for indices in (first, second):
        if len(indices):
            assert_symplectic(x[np.ix_(indices, indices)], atol)


def assert_sympl_cossin(x, p, q, atol=1e-8):
    assert_repeat(x, check=partial(assert_cossin, p=p, q=q), atol=atol)


# Rotation and Cartan-matrix builders ----------------------------------------------

def rotation(angle):
    """The 2x2 rotation ``[[cos, sin], [-sin, cos]]``, the convention of every Schur block here."""
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def skew_plane(size, i, j, coefficient):
    """The so(size) generator ``coefficient * (E_ij - E_ji)``."""
    matrix = np.zeros((size, size))
    matrix[i, j], matrix[j, i] = coefficient, -coefficient
    return matrix


def plane_rotation(size, i, j, angle):
    """Rotation by ``angle`` in the (i, j) plane of R^size."""
    matrix = np.eye(size)
    matrix[np.ix_([i, j], [i, j])] = rotation(angle)
    return matrix


def bdi_cartan(angles, p, q):
    """Cartan element of BDI(p, q): rotations by ``angles`` in the planes ``(i, max(p, q) + i)``."""
    matrix = np.eye(p + q)
    for i, angle in enumerate(angles):
        axes = [i, max(p, q) + i]
        matrix[np.ix_(axes, axes)] = rotation(angle)
    return matrix


# Factor reconstruction ---------------------------------------------------------

def reconstruct_recursive_factors(factors, n):
    """Multiply out ``recursive_bdi`` factors ``(block, start, end, kind)`` into an n x n matrix."""
    reconstructed = np.eye(n)
    for block, start, end, kind in factors:
        embedded = np.eye(n)
        width = end - start
        embedded[start:end, start:end] = (
            bdi_cartan(block, width // 2, width - width // 2) if kind.startswith("a") else block
        )
        reconstructed = reconstructed @ embedded
    return reconstructed


def _scale(kind, time):
    return time if kind == "a0" else 1


def rotation_product(rotations, matrices, time):
    """``prod_k exp(M_k a_k s_k)`` over ``(word, angle, kind)`` with ``s_k = time`` for central ``a0`` rates."""
    size = len(next(iter(matrices.values())))
    product = np.eye(size)
    for word, angle, kind in rotations:
        product = product @ expm(matrices[word] * angle * _scale(kind, time))
    return product


def physical_evolution(rotations, time, n_qubits, parse=lambda word: word):
    """``prod_k exp(i a_k s_k P_k)`` on the qubit register, with ``parse`` mapping labels to Pauli words."""
    unitary = np.eye(2 ** n_qubits, dtype=complex)
    for word, angle, kind in rotations:
        unitary = unitary @ expm(1j * angle * _scale(kind, time) * parse(word).to_mat(wire_order=range(n_qubits)))
    return unitary


def pauli_hamiltonian(generators, coefficients, n_qubits):
    """The dense matrix of ``sum(c * P)`` on ``n_qubits`` qubits."""
    return sum(
        c * word.to_mat(wire_order=range(n_qubits))
        for c, word in zip(coefficients, as_pauli_words(generators), strict=True)
    )


def expected_evolution(generators, coefficients, time, n_qubits):
    """``exp(i time H)`` for ``H = sum(c * P)``, the physical convention of the bridge."""
    return expm(1j * time * pauli_hamiltonian(generators, coefficients, n_qubits))


def assert_equal_up_to_sign(actual, expected, atol=1e-13):
    """Pauli rotations lift SO(2) angles to the spin group only up to a global sign."""
    assert min(np.abs(actual - expected).max(), np.abs(actual + expected).max()) < atol


def assert_mirrored_vertical(rotations):
    """The ``k2`` rotations are the ``k1`` rotations reversed and negated, so K2 = K1^T exactly."""
    left = [(word, angle) for word, angle, kind in rotations if kind == "k1"]
    right = [(word, angle) for word, angle, kind in rotations if kind == "k2"]
    assert right == [(word, -angle) for word, angle in reversed(left)]


# Pauli word families --------------------------------------------------------------

def tfxy_words(n):
    """Generators of the transverse-field XY chain on ``n`` qubits: XX, YY couplings and Z fields."""
    return [PauliWord({i: p, i + 1: p}) for p in "XY" for i in range(n - 1)] + [
        PauliWord({i: "Z"}) for i in range(n)
    ]
