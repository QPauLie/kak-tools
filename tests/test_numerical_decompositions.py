"""Table-driven checks of every numerical Cartan type and their degenerate-spectrum regressions."""

from functools import partial

import numpy as np
import pytest
from scipy.linalg import block_diag, expm
from scipy.stats import ortho_group, unitary_group

from kak_tools import numerical_decompositions as nd


def _close(actual, expected, atol=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def _j(n):
    return np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])


def _generator(n, rng):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a, b = a - a.conj().T, b + b.T
    return np.block([[a, b], [-b.conj(), a.conj()]]) / np.sqrt(4 * n)


def _unitary(x, atol=1e-8):
    assert np.isfinite(x).all()
    _close(x @ x.conj().T, np.eye(len(x)), atol)


def _orthogonal(x, atol=1e-8):
    _unitary(x, atol)
    _close(x.imag, 0, atol)
    _close(np.linalg.det(x), 1, atol)


def _symplectic(x, atol=1e-8):
    _unitary(x, atol)
    j = _j(len(x) // 2)
    _close(x.T @ j @ x, j, atol)


def _embedded(x, atol=1e-8):
    _orthogonal(x, atol)
    _symplectic(x, atol)


def _diagonal(x, atol=1e-8):
    _unitary(x, atol)
    _close(x, np.diag(np.diag(x)), atol)


def _symplectic_diagonal(x, atol=1e-8):
    _diagonal(x, atol)
    _symplectic(x, atol)


def _blocks(x, p, check, atol=1e-8):
    _close(x[:p, p:], 0, atol)
    _close(x[p:, :p], 0, atol)
    check(x[:p, :p], atol=atol)
    check(x[p:, p:], atol=atol)


def _repeat(x, check=_unitary, transform=lambda a: a, atol=1e-8):
    n = len(x) // 2
    _blocks(x, n, check, atol)
    _close(x[n:, n:], transform(x[:n, :n]), atol)


def _schur(x, atol=1e-8):
    _orthogonal(x, atol)
    if len(x) % 2:
        # The lone +1 axis comes last; a rotation by less than atol also looks
        # isolated, so searching from the front could pick one of its axes.
        off_diagonal = x - np.diag(np.diag(x))
        isolated = (np.max(abs(off_diagonal), axis=0) <= atol) & (np.max(abs(off_diagonal), axis=1) <= atol)
        i = np.flatnonzero(isolated & (abs(np.diag(x) - 1) <= atol))[-1]
        keep = np.arange(len(x)) != i
        return _schur(x[np.ix_(keep, keep)], atol)
    diagonal, upper, lower = np.diag(x), np.diag(x, 1), np.diag(x, -1)
    _close(diagonal[::2], diagonal[1::2], atol)
    _close(upper, -lower, atol)
    _close(diagonal[::2] ** 2 + upper[::2] ** 2, 1, atol)
    _close(x, np.diag(diagonal) + np.diag(upper, 1) + np.diag(lower, -1), atol)


def _skew_schur(x, atol=1e-8):
    _repeat(x, _schur, np.transpose, atol)


def _cossin(x, p, q, atol=1e-8):
    _orthogonal(x, atol)
    r, s = min(p, q), max(p, q)
    c, sine = x[:r, :r], x[:r, s:]
    _close(c, np.diag(np.diag(c)), atol)
    _close(sine, np.diag(np.diag(sine)), atol)
    expected = np.eye(p + q)
    expected[:r, :r] = expected[s:, s:] = c
    expected[:r, s:], expected[s:, :r] = sine, -sine
    _close(x, expected, atol)
    _close(np.diag(c) ** 2 + np.diag(sine) ** 2, 1, atol)


def _sectors(p, q):
    n = p + q
    return np.r_[0:p, n : n + p], np.r_[p:n, n + p : 2 * n]


def _sympl_blocks(x, p, q, atol=1e-8):
    first, second = _sectors(p, q)
    _close(x[np.ix_(first, second)], 0, atol)
    _close(x[np.ix_(second, first)], 0, atol)
    for indices in (first, second):
        if len(indices):
            _symplectic(x[np.ix_(indices, indices)], atol)


def _sympl_cossin(x, p, q, atol=1e-8):
    _repeat(x, check=partial(_cossin, p=p, q=q), atol=atol)


def _matrix(kind, n, rng):
    if kind == "unitary":
        return unitary_group.rvs(n, random_state=rng)
    if kind == "orthogonal":
        matrix = ortho_group.rvs(n, random_state=rng)
        matrix[0] *= np.linalg.det(matrix)
        return matrix
    return expm(_generator(n, rng))


# kind: (group, doubled input, K check, Cartan-factor check); symplectic matrices of "size" n are 2n x 2n.
UNPARTITIONED = {
    "a": ("unitary", True, _repeat, partial(_repeat, check=_diagonal, transform=np.conj)),
    "ai": ("unitary", False, _orthogonal, _diagonal),
    "aii": ("unitary", False, _symplectic, partial(_repeat, check=_diagonal)),
    "bd": ("orthogonal", True, partial(_repeat, check=_orthogonal), _skew_schur),
    "diii": ("orthogonal", False, _embedded, _skew_schur),
    "c": ("symplectic", True, partial(_repeat, check=_symplectic), partial(_repeat, check=_symplectic_diagonal, transform=np.conj)),
    "ci": ("symplectic", False, _embedded, _symplectic_diagonal),
}
# kind: (group, K check(x, p, q), Cartan-factor check(x, p, q))
PARTITIONED = {
    "aiii": ("unitary", lambda x, p, q: _blocks(x, p, _unitary), _cossin),
    "bdi": ("orthogonal", lambda x, p, q: _blocks(x, p, _orthogonal), _cossin),
    "cii": ("symplectic", _sympl_blocks, _sympl_cossin),
}


def _element(kind, n, rng):
    group, doubled, _, _ = UNPARTITIONED[kind]
    matrix = _matrix(group, 2 * n if kind in {"aii", "diii"} else n, rng)
    return block_diag(matrix, _matrix(group, n, rng)) if doubled else matrix


@pytest.mark.parametrize("n,validate", [(3, True), (4, False), (7, True), (10, False)])
@pytest.mark.parametrize("kind", sorted(UNPARTITIONED))
def test_cartan_types(kind, n, validate):
    matrix = _element(kind, n, np.random.default_rng(1700 + n))
    k1, center, k2 = getattr(nd, kind + "_kak")(matrix, validate=validate)
    _close(k1 @ center @ k2, matrix)
    _, _, vertical, horizontal = UNPARTITIONED[kind]
    vertical(k1)
    horizontal(center)
    vertical(k2)


@pytest.mark.parametrize(
    "p,q,validate",
    [(1, 1, True), (1, 4, True), (4, 1, False), (0, 3, True), (3, 0, False), (5, 5, True), (2, 7, False), (7, 2, True)],
)
@pytest.mark.parametrize("kind", sorted(PARTITIONED))
def test_partitioned_cartan_types(kind, p, q, validate):
    group, vertical, horizontal = PARTITIONED[kind]
    matrix = _matrix(group, p + q, np.random.default_rng(2010 + p))
    k1, center, k2 = getattr(nd, kind + "_kak")(matrix, p, q, validate=validate)
    _close(k1 @ center @ k2, matrix)
    vertical(k1, p, q)
    horizontal(center, p, q)
    vertical(k2, p, q)


@pytest.mark.parametrize("kind", sorted(UNPARTITIONED))
def test_validate_rejects_elements_outside_the_group(kind):
    # Scaling leaves the block structure intact, so only validate can object, with
    # a ValueError that python -O cannot silence.
    matrix = 1.5 * _element(kind, 2, np.random.default_rng(3))
    with pytest.raises(ValueError):
        getattr(nd, kind + "_kak")(matrix, validate=True)


@pytest.mark.parametrize("kind", sorted(PARTITIONED))
def test_partitioned_validate_rejects_elements_outside_the_group(kind):
    group, _, _ = PARTITIONED[kind]
    matrix = _matrix(group, 4, np.random.default_rng(4))
    matrix[0] *= 1.5
    with pytest.raises(ValueError):
        getattr(nd, kind + "_kak")(matrix, 1, 3, validate=True)


@pytest.mark.parametrize("kind,shape", [
    ("ai", (2, 3)), ("aii", (3, 3)), ("diii", (3, 3)), ("ci", (5, 5)), ("a", (3, 3)), ("bd", (5, 5)), ("c", (6, 6)),
])
def test_wrong_shape_raises_value_error(kind, shape):
    with pytest.raises(ValueError):
        getattr(nd, kind + "_kak")(np.ones(shape))


def test_block_diagonal_input_required_for_doubled_types():
    with pytest.raises(ValueError, match="block-diagonal"):
        nd.a_kak(unitary_group.rvs(4, random_state=0))


@pytest.mark.parametrize("kind", sorted(PARTITIONED))
@pytest.mark.parametrize("p,q", [(True, 1), (1, np.bool_(False)), (-1, 1), (1, 1.5)])
def test_partition_sizes_must_be_nonnegative_integers(kind, p, q):
    with pytest.raises(ValueError, match="nonnegative integers"):
        getattr(nd, kind + "_kak")(np.eye(4), p, q, validate=False)


@pytest.mark.parametrize("kind", ["aiii", "bdi"])
def test_partition_must_match_the_matrix_size(kind):
    with pytest.raises(ValueError, match="size 4"):
        getattr(nd, kind + "_kak")(np.eye(5), 2, 2)


@pytest.mark.parametrize("phase", [1, -1, 1j], ids=["same", "negated", "quarter-turn"])
def test_a_kak_repeated_delta_eigenvalues(phase):
    # eig returns a non-orthonormal basis of the degenerate eigenspaces of
    # delta = u0 @ (phase u0)^dagger, which is a multiple of the identity.
    for seed in range(5):
        u0 = unitary_group.rvs(4, random_state=seed)
        matrix = block_diag(u0, phase * u0)
        k1, center, k2 = nd.a_kak(matrix, validate=False)
        _close(k1 @ center @ k2, matrix)
        _repeat(k1)
        _repeat(center, check=_diagonal, transform=np.conj)
        _repeat(k2)


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
        frame = _matrix("orthogonal", len(phases), np.random.default_rng(seed))
        matrix = frame @ np.diag(np.exp(1j * np.asarray(phases)))
        k1, center, k2 = nd.ai_kak(matrix)
        _close(k1 @ center @ k2, matrix)
        _orthogonal(k1)
        _diagonal(center)
        _orthogonal(k2)


@pytest.mark.parametrize("angles,plus,minus,seed", [((0.4, 1.3), 1, 2, 8), ((0.4, 1.3, 2.2), 3, 2, 2)], ids=["isolated-axes", "odd-plus-run"])
def test_bd_schur_singletons(bd_schur_matrix, angles, plus, minus, seed):
    """Upstream regressions: isolated +/-1 Schur axes and an odd run of +1 axes."""
    matrix = bd_schur_matrix(angles, plus, minus, seed)
    k1, center, k2 = nd.bd_kak(matrix, validate=True)
    _close(k1 @ center @ k2, matrix)
    _repeat(k1, check=_orthogonal)
    _skew_schur(center)
    _repeat(k2, check=_orthogonal)


@pytest.mark.parametrize("plus", [0, 1], ids=["even", "odd"])
@pytest.mark.parametrize("deviation", [1e-9, 1e-6, 1e-4])
def test_bd_near_pi_rotation_beside_minus_one_axes(bd_schur_matrix, deviation, plus):
    # LAPACK may return the near-pi block between the two exact -1 axes; pairing
    # Schur entries by position then split the rotation in half.
    for seed in range(8):
        matrix = bd_schur_matrix([np.pi - deviation], plus, 2, seed)
        k1, center, k2 = nd.bd_kak(matrix, validate=True)
        _close(k1 @ center @ k2, matrix, atol=1e-13)
        _repeat(k1, check=_orthogonal, atol=1e-13)
        _skew_schur(center, atol=1e-13)
        _repeat(k2, check=_orthogonal, atol=1e-13)


@pytest.mark.parametrize("angle", [0.0, 1e-10, np.pi], ids=["identity", "tiny-rotation", "half-turn"])
def test_schur_sqrt_with_fixed_axis(angle):
    c, s = np.cos(angle), np.sin(angle)
    matrix = block_diag([[c, s], [-s, c]], 1)
    square_root = nd.schur_sqrt(matrix)
    _close(square_root @ square_root, matrix, atol=1e-14)
    _orthogonal(square_root, atol=1e-14)


def test_schur_sqrt_pairs_minus_one_axes():
    square_root = nd.schur_sqrt(np.diag([-1.0, -1.0, 1.0]))
    _close(square_root, block_diag([[0, 1], [-1, 0]], 1), atol=1e-15)
    with pytest.raises(ValueError, match="unpaired"):
        nd.schur_sqrt(np.diag([-1.0, 1.0]))
