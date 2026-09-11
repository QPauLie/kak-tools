"""Representative, deterministic checks of every numerical Cartan type."""

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
    check(x[:p, :p], atol)
    check(x[p:, p:], atol)


def _repeat(x, check=_unitary, transform=lambda a: a, atol=1e-8):
    n = len(x) // 2
    _blocks(x, n, check, atol)
    _close(x[n:, n:], transform(x[:n, :n]), atol)


def _schur(x, atol=1e-8):
    _orthogonal(x, atol)
    if len(x) % 2:
        off_diagonal = x - np.diag(np.diag(x))
        isolated = (np.max(abs(off_diagonal), axis=0) <= atol) & (np.max(abs(off_diagonal), axis=1) <= atol)
        i = np.flatnonzero(isolated & (abs(np.diag(x) - 1) <= atol))[0]
        keep = np.arange(len(x)) != i
        _close(x[i, keep], 0, atol)
        _close(x[keep, i], 0, atol)
        return _schur(x[np.ix_(keep, keep)], atol)
    diagonal, upper, lower = np.diag(x), np.diag(x, 1), np.diag(x, -1)
    _close(diagonal[::2], diagonal[1::2], atol)
    _close(upper, -lower, atol)
    _close(diagonal[::2] ** 2 + upper[::2] ** 2, 1, atol)
    _close(x, np.diag(diagonal) + np.diag(upper, 1) + np.diag(lower, -1), atol)


def _skew_schur(x, atol=1e-8):
    _repeat(x, _schur, np.transpose, atol)


def _cossin(x, p, q):
    _orthogonal(x)
    r, s = min(p, q), max(p, q)
    c, sine = x[:r, :r], x[:r, s:]
    _close(c, np.diag(np.diag(c)))
    _close(sine, np.diag(np.diag(sine)))
    expected = np.eye(p + q)
    expected[:r, :r] = expected[s:, s:] = c
    expected[:r, s:], expected[s:, :r] = sine, -sine
    _close(x, expected)
    _close(np.diag(c) ** 2 + np.diag(sine) ** 2, 1)


def _matrix(kind, n, rng):
    if kind == "unitary":
        return unitary_group.rvs(n, random_state=rng)
    if kind == "orthogonal":
        matrix = ortho_group.rvs(n, random_state=rng)
        matrix[0] *= np.linalg.det(matrix)
        return matrix
    return expm(_generator(n, rng))


# One odd/small and one even/moderate case exercise both validation modes.
@pytest.mark.parametrize("n,validate", [(3, True), (10, False)])
@pytest.mark.parametrize("kind", ["a", "ai", "aii", "bd", "diii", "c", "ci"])
def test_cartan_types(kind, n, validate):
    rng = np.random.default_rng(1700 + n)
    group = "orthogonal" if kind in {"bd", "diii"} else "symplectic" if kind in {"c", "ci"} else "unitary"
    matrix = _matrix(group, 2 * n if kind in {"aii", "diii"} else n, rng)
    if kind in {"a", "bd", "c"}:
        matrix = block_diag(matrix, _matrix(group, n, rng))
    checks = {
        "a": (_repeat, partial(_repeat, check=_diagonal, transform=np.conj)),
        "ai": (_orthogonal, _diagonal),
        "aii": (_symplectic, partial(_repeat, check=_diagonal)),
        "bd": (partial(_repeat, check=_orthogonal), _skew_schur),
        "diii": (_embedded, _skew_schur),
        "c": (partial(_repeat, check=_symplectic), partial(_repeat, check=_symplectic_diagonal, transform=np.conj)),
        "ci": (_embedded, _symplectic_diagonal),
    }
    k1, center, k2 = getattr(nd, kind + "_kak")(matrix, validate=validate)
    _close(k1 @ center @ k2, matrix)
    vertical, horizontal = checks[kind]
    vertical(k1)
    horizontal(center)
    vertical(k2)


@pytest.mark.parametrize("kind", ["aiii", "bdi"])
@pytest.mark.parametrize("p,q,validate", [(1, 1, True), (1, 4, True), (4, 1, False), (0, 3, True), (3, 0, False)])
def test_partitioned_cartan_types(kind, p, q, validate):
    group = "unitary" if kind == "aiii" else "orthogonal"
    matrix = _matrix(group, p + q, np.random.default_rng(2010 + p))
    k1, center, k2 = getattr(nd, kind + "_kak")(matrix, p, q, validate=validate)
    _close(k1 @ center @ k2, matrix)
    check = _unitary if kind == "aiii" else _orthogonal
    _blocks(k1, p, check)
    _blocks(k2, p, check)
    _cossin(center, p, q)


@pytest.mark.parametrize("fixture_name", ["reference_matrix_bd", "reference_matrix_bd2"])
def test_bd_reference_schur_singletons(request, fixture_name):
    """Upstream regressions for singletons and an odd run of +1 Schur blocks."""
    matrix = request.getfixturevalue(fixture_name)
    k1, center, k2 = nd.bd_kak(matrix, validate=True)
    _close(k1 @ center @ k2, matrix)
    _repeat(k1, check=_orthogonal)
    _skew_schur(center)
    _repeat(k2, check=_orthogonal)


@pytest.mark.parametrize("angle", [0.0, 1e-10], ids=["identity", "tiny-rotation"])
def test_schur_sqrt_with_fixed_axis(angle):
    c, s = np.cos(angle), np.sin(angle)
    matrix = block_diag([[c, s], [-s, c]], 1)
    square_root = nd.schur_sqrt(matrix)
    _close(square_root @ square_root, matrix, atol=1e-14)
    _orthogonal(square_root, atol=1e-14)
