"""Native PauLie classification and reusable KAK compilation of an open tfXY chain.

Run ``python notebooks/paulie_bridge_example.py`` after ``pip install -e .``.
The physical convention is exp(+i t H).
"""

import numpy as np
import pennylane as qml
from scipy.linalg import expm
from paulie import get_pauli_string
from paulie.common.pauli_string_factory import get_all_k_local

from kak_tools import kak_decomposition, pauli_string_to_word


n_qubits = 4
# Build each locality separately to include single-site fields at both boundaries.
generators = get_pauli_string([
    *get_all_k_local(n_qubits, ["XX", "YY"]),
    *get_all_k_local(n_qubits, ["Z"]),
])
classification = generators.get_class()
print("Generators:", [str(word) for word in generators])
print("Algebra:", classification.get_algebra())
print("Dimension:", classification.get_dla_dim())
print("Summands:", classification.get_subalgebras())
if classification.is_simple():
    print("Simple component:", classification.get_simple_component())
print("Classified basis:", classification.get_algebra_basis().shape)
print("Orthogonal size:", classification.get_orthogonal_size())

coefficients = np.random.default_rng(20250805).normal(size=len(generators))
coefficients /= np.linalg.norm(coefficients)
result = kak_decomposition(generators, coefficients, time=0.83)
print("Physical qubits / irrep size / partition:", result.n_qubits, result.irrep_size, result.partition)
print("Native classification:", result.classification.get_algebra())
print("Rotations / central rates:", len(result.pauli_rotations), len(result.cartan_angles))
print("Irrep reconstruction error:", result.reconstruction_error)

# Independently check exp(+i t H) on the physical qubits, including reused times.
# pauli_rotations lists the matrix product left to right; a circuit applies the
# gates in reversed order, which pennylane_ops does.
wire_order = list(range(n_qubits))
hamiltonian = sum(
    coefficient * pauli_string_to_word(word).to_mat(wire_order=wire_order)
    for word, coefficient in zip(generators, coefficients, strict=True)
)
for time in [0.0, -0.1, 0.83, 4.2]:
    physical = np.eye(2**n_qubits, dtype=complex)
    for word, coefficient, kind in result.pauli_rotations:
        angle = time * coefficient if kind == "a0" else coefficient
        physical = physical @ expm(1j * angle * word.to_mat(wire_order=wire_order))
    expected = expm(1j * time * hamiltonian)
    error = np.max(np.abs(physical - expected))
    np.testing.assert_allclose(physical, expected, rtol=0, atol=1e-8)
    circuit = qml.matrix(qml.tape.QuantumScript(result.pennylane_ops(time)), wire_order=wire_order)
    np.testing.assert_allclose(circuit, expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        result.reconstruct(time), expm(time * result.hamiltonian_irrep), rtol=0, atol=1e-8
    )
    print(f"t={time:5.2f}: physical reconstruction error {error:.2e}")
