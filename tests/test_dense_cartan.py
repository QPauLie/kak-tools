"""Legacy recursive compilation through a real PennyLane circuit."""

import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm
from kak_tools import (
    lie_closure_pauli_words, map_simple_to_irrep, map_irrep_to_matrices,
    recursive_bdi, map_recursive_decomp_to_reducible,
)
from kak_tools.dense_cartan import group_matrix_to_reducible_str
from test_map_to_irrep import tfxy_words


@pytest.mark.parametrize("n", [2, 3])
def test_recursive_compilation_in_circuit(n):
    words = tfxy_words(n)
    basis = lie_closure_pauli_words(words)
    mapping, signs = map_simple_to_irrep(basis, words, n=2 * n, invol_type="BDI")
    matrices = map_irrep_to_matrices(mapping, signs, 2 * n, "BDI")
    coefficients = np.r_[np.ones(2 * (n - 1)), np.random.default_rng(n).normal(0, .3, n)]
    coefficients /= np.linalg.norm(coefficients)
    h = sum(c * matrices[w] for c, w in zip(coefficients, words, strict=True))
    epsilon = .01
    factors = recursive_bdi(expm(epsilon * h), 2 * n, validate=False)
    rotations = map_recursive_decomp_to_reducible(factors, mapping, signs, time=epsilon, validate=True)

    @qml.qnode(qml.device("default.qubit", wires=n))
    def circuit(time):
        qml.X(0)
        for word, angle, kind in reversed(rotations):
            coefficient = angle * (time if kind == "a0" else 1)
            qml.PauliRot(2 * coefficient, qml.pauli.pauli_word_to_string(word), wires=word.wires)
        return qml.probs()

    times = [-1.3, 0., 25.]
    actual = np.array([circuit(t) for t in times])
    physical_h = sum(c * w.to_mat(wire_order=range(n)) for c, w in zip(coefficients, words, strict=True))
    initial = np.zeros(2**n)
    initial[2**(n - 1)] = 1
    expected = np.array([np.abs(expm(-1j * t * physical_h) @ initial)**2 for t in times])
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_legacy_so2_angle_survives_a_sine_rounded_beyond_one():
    mapping = {(0, 1): ("XX", 1)}
    sine = np.nextafter(1.0, 2.0)
    [(word, angle)] = group_matrix_to_reducible_str(np.array([[0., sine], [-sine, 0.]]), 0, mapping).items()
    assert word == "XX" and np.isclose(angle, np.pi / 4)
    theta = 2.5
    block = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    [(_, angle)] = group_matrix_to_reducible_str(block, 0, mapping).items()
    assert np.isclose(angle, theta / 2)
