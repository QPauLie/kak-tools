"""Public bridge evolution: representation, coefficients, and reusable gates."""

import numpy as np
import pennylane as qml
import pytest
from paulie.classifier.classification import Classification
from scipy.linalg import expm

from kak_tools import paulie_bridge
from kak_tools import as_pauli_words, kak_decomposition
from kak_tools._pauli_rotations import PauliRotation


TFXY = ["XX", "YY", "ZI", "IZ"]


def expand(patterns, n):
    return ["I" * start + pattern + "I" * (n - start - len(pattern))
            for pattern in patterns for start in range(n - len(pattern) + 1)]


def physical_evolution(result, time):
    out = np.eye(2 ** result.n_qubits, dtype=complex)
    for word, angle, kind in result.pauli_rotations:
        out = out @ expm(1j * angle * (time if kind == "a0" else 1)
                         * word.to_mat(wire_order=range(result.n_qubits)))
    return out


def expected_evolution(generators, coefficients, time, n):
    h = sum(c * w.to_mat(wire_order=range(n))
            for c, w in zip(coefficients, as_pauli_words(generators), strict=True))
    return expm(1j * time * h)


@pytest.mark.parametrize("patterns, n, m", [
    (["XX", "YY", "Z"], 2, 4),  # low-rank coincidence and sign gauge
    (["XX", "YY", "Z"], 3, 6),  # even algebra, odd BDI block widths
    (["XX", "XZ"], 4, 7),       # odd algebra and rectangular BDI
    (["XY"], 4, 4),             # a different model with the same algebra
])
def test_model_agnostic_compilation(patterns, n, m):
    generators = expand(patterns, n)
    coefficients = np.random.default_rng(n).normal(size=len(generators))
    result = kak_decomposition(generators, coefficients, time=.83)
    assert isinstance(result.classification, Classification)
    assert result.irrep_size == result.classification.get_orthogonal_size() == m
    assert result.partition == (m // 2, m - m // 2) and result.reconstruction_error < 1e-10
    assert len(result.cartan_angles) == m // 2
    assert {kind for _, _, kind in result.pauli_rotations} <= {"k1", "k2", "a0"}
    np.testing.assert_allclose(result.reconstruct(), result.unitary_irrep, atol=1e-11, rtol=0)
    np.testing.assert_allclose(physical_evolution(result, .83),
                              expected_evolution(generators, coefficients, .83, n), atol=1e-11, rtol=0)


@pytest.mark.parametrize("initial_time, time, coefficients", [
    (0, -2.5, [.3, -.8, .7, -1.2]),
    (np.pi / 4, 25, [1, 1, 1, 1]),  # uniform coefficients and resonance
    (.7, 0, [.3, -.8, .7, -1.2]),
    (0, 25, [0, 0, 0, 0]),
])
def test_reusable_compilation_has_exact_inverse_gates_and_physical_phase(initial_time, time, coefficients):
    result = kak_decomposition(TFXY, coefficients, time=initial_time)
    left = [(word, angle) for word, angle, kind in result.pauli_rotations if kind == "k1"]
    right = [(word, angle) for word, angle, kind in result.pauli_rotations if kind == "k2"]
    assert right == [(word, -angle) for word, angle in reversed(left)]
    np.testing.assert_allclose(physical_evolution(result, time),
                              expected_evolution(TFXY, coefficients, time, 2), atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.reconstruct(time), expm(time * result.hamiltonian_irrep),
                              atol=1e-10, rtol=0)


@pytest.mark.parametrize("generators, coefficients, n", [
    (TFXY, [.3, -.8, .7, -1.2], 2),
    (expand(["XX", "XZ"], 3), [.4, -1.1, .9, .2], 3),
    (["II"], [.72], 2),  # identity word becomes a global phase
])
def test_pennylane_ops_apply_the_rotations_in_circuit_order(generators, coefficients, n):
    result = kak_decomposition(generators, coefficients, time=.83)
    for time, ops in [(.83, result.pennylane_ops()), (-2.5, result.pennylane_ops(-2.5))]:
        assert all(isinstance(op, (qml.PauliRot, qml.GlobalPhase)) for op in ops)
        circuit = qml.matrix(qml.tape.QuantumScript(ops), wire_order=range(n))
        np.testing.assert_allclose(circuit, expected_evolution(generators, coefficients, time, n),
                                  atol=1e-11, rtol=0)
    # The list order is the matrix-product order; the circuit applies it reversed.
    for op, (word, c, kind) in zip(result.pennylane_ops(), reversed(result.pauli_rotations), strict=True):
        angle = c * (.83 if kind == "a0" else 1)
        assert float(op.data[0]) == pytest.approx(-2 * angle if len(word) else -angle)


@pytest.mark.parametrize("coefficients, expected", [(None, [2, 3, 3, 5]), ([1, 2, 3, 4, 2], [2, 6, 10, 20])])
def test_intrinsic_weights_multiply_external_weights_before_duplicate_aggregation(coefficients, expected):
    operators = [2 * (qml.X(0) @ qml.X(1)), 3 * (qml.Y(0) @ qml.Y(1)),
                 4 * qml.Z(0), 5 * qml.Z(1), -qml.Z(0)]
    result = kak_decomposition(operators, coefficients, time=.2)
    np.testing.assert_allclose(physical_evolution(result, .2),
                              expected_evolution(TFXY, expected, .2, 2), atol=1e-11, rtol=0)


def test_duplicate_sum_retains_small_terms_between_large_cancellations():
    result = kak_decomposition(["X", "X", "X", "Y"], [1e16, 1, -1e16, 2], time=.2)
    x, y = as_pauli_words(["X", "Y"])
    np.testing.assert_array_equal(result.hamiltonian_irrep, result.algebra_basis[x] + 2 * result.algebra_basis[y])
    np.testing.assert_allclose(physical_evolution(result, .2),
                              expected_evolution(["X", "Y"], [1, 2], .2, 1), atol=1e-13, rtol=0)


def test_default_cutoff_retains_small_vertical_angles_at_longer_times():
    result = kak_decomposition(["X", "Y"], [1e-12, 1])
    np.testing.assert_allclose(result.reconstruct(), result.unitary_irrep, atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.reconstruct(25), expm(25 * result.hamiltonian_irrep), atol=1e-13, rtol=0)


def test_validation_and_tolerance_options():
    result = kak_decomposition(TFXY, [.3, -.8, .7, -1.2], validate=False)
    assert result.reconstruction_error is None
    with pytest.raises(ValueError, match="do not recompose"):
        kak_decomposition(TFXY, [.3, -.8, .7, -1.2], atol=0.0)
    pruned = kak_decomposition(["X", "Y"], [1e-12, 1], tol=1e-8)
    assert {kind for _, _, kind in pruned.pauli_rotations} == {"a0"}
    np.testing.assert_allclose(pruned.reconstruct(25), expm(25 * pruned.hamiltonian_irrep), atol=1e-10, rtol=0)


@pytest.mark.parametrize("time, factor", [(0, 3.7), (np.pi, 1.5)])  # zero time and a 2*pi resonance
def test_validation_is_independent_of_the_compile_time(monkeypatch, time, factor):
    decompose = paulie_bridge.decompose_horizontal_hamiltonian

    def tampered(*args, **kwargs):
        return [PauliRotation(word, factor * rate if kind == "a0" else rate, kind)
                for word, rate, kind in decompose(*args, **kwargs)]

    monkeypatch.setattr(paulie_bridge, "decompose_horizontal_hamiltonian", tampered)
    with pytest.raises(ValueError, match=r"K1 A K1\^T"):
        kak_decomposition(TFXY, [1, 1, 1, 1], time=time)


def test_validation_accepts_a_correct_compilation_at_very_large_times():
    result = kak_decomposition(TFXY, [.31, .47, -.8, .2], time=1e8)
    assert result.reconstruction_error < 1e-13


@pytest.mark.parametrize("generator, coefficient, time", [("II", .72, -1.2), ("YIII", -2.3, 4.2), ("Z", 0, .3)])
def test_u1_evolution_preserves_global_phase_and_register(generator, coefficient, time):
    result = kak_decomposition(generator, [coefficient], time=0)
    assert result.n_qubits == len(generator) and result.irrep_size == 2
    assert set(result.mapping) == {(0, 1)}
    np.testing.assert_allclose(physical_evolution(result, time),
                              expected_evolution([generator], [coefficient], time, len(generator)),
                              atol=2e-13, rtol=0)


def test_zero_weights_keep_their_terms_in_the_generator_family():
    result = kak_decomposition([qml.X(0), 0 * qml.Y(0)], time=.3)
    assert result.classification.get_dla_dim() == result.irrep_size == 3
    np.testing.assert_allclose(physical_evolution(result, .3), expected_evolution(["X"], [1], .3, 1), atol=1e-14, rtol=0)
    for args in [([qml.X(0), qml.Y(0), 0 * qml.Z(0)],), (["X", "Y", "Z"], [1, 1, 0])]:
        with pytest.raises(ValueError, match="horizontal"):
            kak_decomposition(*args)


def test_default_partition_is_searched_and_the_failure_cause_is_named():
    generators, coefficients = ["IX", "IY", "XZ"], [.3, -.7, 1.1]
    result = kak_decomposition(generators, coefficients, time=.4)
    assert result.partition == (1, 3)
    np.testing.assert_allclose(physical_evolution(result, .4),
                              expected_evolution(generators, coefficients, .4, 2), atol=1e-13, rtol=0)
    with pytest.raises(ValueError, match=r"BDI\(2, 2\).*or partition"):
        kak_decomposition(generators, coefficients, invol_kwargs={"p": 2})
    with pytest.raises(ValueError, match="for any partition"):
        kak_decomposition(["XI", "YI", "IX", "IY"])
