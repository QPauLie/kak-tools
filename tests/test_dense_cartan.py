"""Legacy recursive compilation through a real PennyLane/JAX circuit."""

import jax
import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm
from kak_tools import (
    lie_closure_pauli_words, map_simple_to_irrep, map_irrep_to_matrices,
    recursive_bdi, map_recursive_decomp_to_reducible,
)
from test_map_to_irrep import tfxy_words

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n", [2, 3])
def test_recursive_compilation_in_vectorized_circuit(n):
    words = tfxy_words(n)
    basis = lie_closure_pauli_words(words)
    mapping, signs = map_simple_to_irrep(basis, words, n=2 * n, invol_type="BDI")
    matrices = map_irrep_to_matrices(mapping, signs, 2 * n, "BDI")
    coefficients = np.r_[np.ones(2 * (n - 1)), np.random.default_rng(n).normal(0, .3, n)]
    coefficients /= np.linalg.norm(coefficients)
    h = sum(c * matrices[w] for c, w in zip(coefficients, words))
    epsilon = .01
    factors = recursive_bdi(expm(epsilon * h), 2 * n, validate=False)
    rotations = map_recursive_decomp_to_reducible(factors, mapping, signs, time=epsilon)

    @qml.qnode(qml.device("default.qubit", wires=n), interface="jax")
    def circuit(time):
        qml.X(0)
        for word, angle, kind in reversed(rotations):
            coefficient = angle * (time if kind == "a0" else 1)
            qml.PauliRot(2 * coefficient, qml.pauli.pauli_word_to_string(word), wires=word.wires)
        return qml.probs()

    times = np.array([-1.3, 0., 25.])
    actual = jax.jit(jax.vmap(circuit))(times)
    physical_h = sum(c * w.to_mat(wire_order=range(n)) for c, w in zip(coefficients, words))
    initial = np.zeros(2**n)
    initial[2**(n - 1)] = 1
    expected = np.array([np.abs(expm(-1j * t * physical_h) @ initial)**2 for t in times])
    np.testing.assert_allclose(actual, expected, atol=1e-10)
