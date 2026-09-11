"""Ordered analytic rotations, general-matrix fallback and input contracts."""

import numpy as np
import pytest
from scipy.linalg import expm
from kak_tools._pauli_rotations import reconstruct_from_pauli_rotations


def plane(size, i, j, coefficient):
    matrix = np.zeros((size, size))
    matrix[i, j], matrix[j, i] = coefficient, -coefficient
    return matrix


@pytest.mark.parametrize("time", [None, -2.3])
def test_order_and_time_scaling(time):
    basis = {"xy": plane(5, 0, 1, 2.), "yz": plane(5, 1, 3, -2.), "zw": plane(5, 3, 4, .75)}
    rotations = [("xy", .3, "k1"), ("yz", -.8, "a0"), ("zw", 1.9, "a"), ("xy", -.4, "k2")]
    expected = np.eye(5)
    for word, angle, kind in rotations:
        scale = time if kind == "a0" and time is not None else 1
        expected = expected @ expm(basis[word] * angle * scale)
    actual = reconstruct_from_pauli_rotations(iter(rotations), basis, np.int64(5), time)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(actual.T @ actual, np.eye(5), atol=2e-13)


@pytest.mark.parametrize("matrix", [
    np.array([[1e-12, 2.], [-2. + 1e-10, 0.]]),
    np.array([[0., 1j], [1j, 0.]]),
    plane(4, 0, 1, 2.) + plane(4, 1, 3, -.7),
])
def test_general_matrix_fallback_preserves_small_terms_and_inputs(matrix):
    before = matrix.copy()
    rotations = [("p", .7, "a0")]
    result = reconstruct_from_pauli_rotations(rotations, {"p": matrix}, len(matrix), -.4)
    np.testing.assert_allclose(result, expm(matrix * (.7 * -.4)), rtol=0, atol=2e-14)
    np.testing.assert_array_equal(matrix, before)
    assert rotations == [("p", .7, "a0")]
    np.testing.assert_array_equal(reconstruct_from_pauli_rotations([], {}, 2), np.eye(2))


@pytest.mark.parametrize("arguments", [
    {"irrep_size": 0}, {"irrep_size": True}, {"time": np.nan},
    {"pauli_rotations": [("p", .3j, "a")]},
    {"algebra_basis": {"p": np.eye(3)}},
    {"algebra_basis": {"p": np.full((2, 2), np.inf)}},
    {"algebra_basis": {"p": [["a"] * 2] * 2}},
    {"pauli_rotations": [("p", 1e308, "a0")], "time": 2.},
    {"algebra_basis": {"p": plane(2, 0, 1, 1e308)}, "time": 2.},
    {"algebra_basis": {"p": np.eye(2) * 1e308}, "time": 2.},
])
def test_invalid_inputs_and_scaled_overflow(arguments):
    inputs = dict(pauli_rotations=[("p", 2., "a0")], algebra_basis={"p": plane(2, 0, 1, 2.)}, irrep_size=2, time=1.)
    inputs.update(arguments)
    with pytest.raises(ValueError):
        reconstruct_from_pauli_rotations(**inputs)
