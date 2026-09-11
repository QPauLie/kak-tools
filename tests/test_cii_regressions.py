"""CII structure, endpoint, precision and input-contract regressions."""

import numpy as np
import pytest
from scipy.linalg import block_diag, expm

from kak_tools.numerical_decompositions import cii_kak
from test_numerical_decompositions import _close, _generator, _symplectic


def _sectors(p, q):
    n = p + q
    return np.r_[0:p, n:n + p], np.r_[p:n, n + p:2 * n]


def _vertical(p, q, rng):
    matrix = np.zeros((2 * (p + q), 2 * (p + q)), dtype=complex)
    for size, indices in zip((p, q), _sectors(p, q)):
        matrix[np.ix_(indices, indices)] = expm(_generator(size, rng)) if size else np.eye(0)
    return matrix


def _cartan(p, q, angles):
    block = np.eye(p + q)
    for i, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        axes = [i, max(p, q) + i]
        block[np.ix_(axes, axes)] = [[c, s], [-s, c]]
    return block_diag(block, block)


def _check(matrix, p, q, validate=True, atol=2e-11):
    original = matrix.copy()
    k1, center, k2 = cii_kak(matrix, p, q, validate=validate)
    np.testing.assert_array_equal(matrix, original)
    reconstructed = k1 @ center @ k2
    _close(reconstructed, matrix, atol)
    assert np.linalg.norm(reconstructed - matrix) <= atol * max(1, np.linalg.norm(matrix))
    for factor in (k1, center, k2):
        _symplectic(factor, atol)
        _close(np.linalg.det(factor), 1, 10 * atol)
    first, second = _sectors(p, q)
    for factor in (k1, k2):
        _close(factor[np.ix_(first, second)], 0, atol)
        _close(factor[np.ix_(second, first)], 0, atol)
    indices = np.arange(min(p, q))
    angles = np.arctan2(center[indices, max(p, q) + indices].real, center[indices, indices].real)
    _close(center, _cartan(p, q, angles), atol)
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
    matrix = _vertical(p, q, rng) @ _cartan(p, q, angles) @ _vertical(p, q, rng)
    k1, center, k2 = _check(matrix, p, q, validate)
    if 0 < max(np.abs(angles)) < 1e-8:
        first, second = _sectors(p, q)
        cross = matrix[np.ix_(first, second)]
        error = (k1 @ center @ k2 - matrix)[np.ix_(first, second)]
        assert np.linalg.norm(error) < 2e-13 + 2e-5 * np.linalg.norm(cross)


@pytest.mark.parametrize("p,q,validate", [(0, 0, True), (0, 3, False), (3, 0, True)])
def test_cii_empty_partitions(p, q, validate):
    matrix = _vertical(p, q, np.random.default_rng(904))
    k1, center, k2 = _check(matrix, p, q, validate)
    _close(center, np.eye(len(matrix)))
    assert not np.shares_memory(k1, matrix)
    assert not np.shares_memory(center, k2)


def test_cii_roundoff_split_symplectic_pairs():
    def rotation(t):
        return np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    _check(block_diag(rotation(0.4), rotation(0.4 + 1e-13)), 1, 1, atol=3e-13)


@pytest.mark.parametrize("p,q,validate", [(1, 3, True), (3, 1, False)])
def test_cii_long_time_symplectic_exponential(p, q, validate):
    matrix = expm(1e4 * _generator(p + q, np.random.default_rng(719)))
    _check(matrix, p, q, validate, atol=3e-11)


@pytest.mark.parametrize("near_right_angle", [False, True])
def test_cii_entirely_weak_constrained_block(near_right_angle):
    angle = np.pi / 2 - 1e-14 if near_right_angle else 1e-14
    phases = [1j, 1, -1j, 1] if near_right_angle else [1, 1j, 1, -1j]
    matrix = np.diag(phases) @ _cartan(1, 1, [angle])
    _, center, k2 = _check(matrix, 1, 1)
    first, second = _sectors(1, 1)
    tolerance = 5e-15 * (np.cos(angle) if near_right_angle else np.sin(angle))
    # Product reconstruction alone can hide the lost weak direction in K2.
    _close(k2[np.ix_(first, second)], 0, tolerance)
    _close(k2[np.ix_(second, first)], 0, tolerance)
    _close(center, _cartan(1, 1, [angle]), tolerance)


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
