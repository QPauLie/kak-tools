"""Legacy recursive compilation through a real PennyLane circuit."""

import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm

from kak_tools import (
    lie_closure_pauli_words, map_irrep_to_matrices, map_recursive_decomp_to_reducible, map_simple_to_irrep,
    recursive_bdi,
)
from checks import assert_equal_up_to_sign, expected_evolution, tfxy_words


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

    # The rotations are the factors of U = R_1 ... R_N with R_k = exp(+i c_k P_k), so a
    # circuit applies them last to first, each as PauliRot(-2 c_k). The spin lift of the
    # irrep fixes the circuit only up to a global sign.
    def circuit(time):
        return qml.tape.QuantumScript([
            qml.PauliRot(-2 * angle * (time if kind == "a0" else 1), qml.pauli.pauli_word_to_string(word), wires=word.wires)
            for word, angle, kind in reversed(rotations)
        ])

    for time in [-1.3, 0., 25.]:
        actual = qml.matrix(circuit(time), wire_order=range(n))
        assert_equal_up_to_sign(actual, expected_evolution(words, coefficients, time, n), atol=1e-10)
