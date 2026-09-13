"""The PauLie bridge: input normalization, classification and closure contracts, the compiled
evolution with its typed rotations and PennyLane gates, and the rotation reconstruction helper."""

import numpy as np
import pennylane as qml
import pytest
from paulie import get_pauli_string
from paulie.classifier.classification import Classification
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_collection import PauliStringCollection
from pauliebits import pauliebits
from pennylane.pauli import PauliWord
from scipy.linalg import expm

from kak_tools import (
    PauliRotation, as_pauli_collection, as_pauli_words, dla_pauli_basis, kak_decomposition,
    labelled_matrix_basis, lie_closure_pauli_words, map_dla_to_irrep, paulie_bridge, pauli_string_to_word,
    pauli_word_to_string, reconstruct_from_pauli_rotations,
)
from checks import assert_mirrored_vertical, expected_evolution, physical_evolution, skew_plane

TFXY = ["XX", "YY", "ZI", "IZ"]


def expand(patterns, n):
    return ["I" * start + pattern + "I" * (n - start - len(pattern))
            for pattern in patterns for start in range(n - len(pattern) + 1)]


def evolution(result, time):
    return physical_evolution(result.pauli_rotations, time, result.n_qubits)


# Input normalization and the classification/closure contracts ------------------------------

@pytest.mark.parametrize("text, expected", [("XYZI", {0: "X", 1: "Y", 2: "Z"}), ("IIII", {}), ("IIZ", {2: "Z"})])
def test_pauli_string_conversion_preserves_positions_and_roundtrips(text, expected):
    word = pauli_string_to_word(text)
    assert word == PauliWord(expected)
    assert str(pauli_word_to_string(word, len(text))) == text


@pytest.mark.parametrize("endians, collection", [(["big", "big"], False), (["little", "big"], True)])
def test_native_bit_endianness_and_container_do_not_change_the_closure(endians, collection):
    native = [PauliString(bits=pauliebits(bits, endian=endian))
              for bits, endian in zip(["10", "01"], endians, strict=True)]
    inputs = PauliStringCollection(native) if collection else native
    assert as_pauli_collection(inputs).get_class().get_dla_dim() == 3
    assert set(dla_pauli_basis(inputs)) == set(as_pauli_words(["X", "Y", "Z"]))


@pytest.mark.parametrize("generators, width", [
    (["XII", "YII"], 3),
    ([qml.X(0) @ qml.I(5), qml.Y(0) @ qml.I(5)], 6),
    ([qml.X(0), qml.Y(0), qml.X(0) @ qml.I(7)], 8),
])
def test_register_width_survives_padding_identity_and_duplicate_terms(generators, width):
    result = kak_decomposition(iter(generators))
    assert result.n_qubits == as_pauli_collection(generators).get_len() == width
    assert as_pauli_collection(generators, n_qubits=width + 1).get_len() == width + 1
    assert result.classification.get_dla_dim() == 3
    with pytest.raises(ValueError, match=f"require {width} qubits"):
        kak_decomposition(generators, n_qubits=1)


@pytest.mark.parametrize("wire", [-1, "ancilla", 1.5])
def test_identity_wires_do_not_bypass_validation(wire):
    with pytest.raises(ValueError, match="wire|Wire"):
        as_pauli_collection([qml.X(0) @ qml.I(wire)]).get_class()


def test_word_cannot_be_converted_to_a_smaller_register():
    with pytest.raises(ValueError, match="not an integer in range"):
        pauli_word_to_string(PauliWord({7: "X"}), 4)


def test_single_string_and_one_shot_iterators_are_not_split_or_consumed():
    assert as_pauli_words("XY") == [PauliWord({0: "X", 1: "Y"})]
    assert as_pauli_collection(iter(["XX", "YY", "ZI", "IZ"])).get_class().get_dla_dim() == 6
    assert len(map_dla_to_irrep(iter(["XX", "YY", "ZI", "IZ"]))[0]) == 6


@pytest.mark.parametrize("generators, message", [
    ([], "At least one"), (PauliStringCollection([]), "At least one"),
    ([qml.X(0) + qml.Y(0)], "single Pauli term"), (["AB"], "Pauli strings"), ([""], "Pauli strings"),
    ([1j * qml.X(0)], "finite real scalar"),
])
def test_invalid_generator_inputs_raise_clear_errors(generators, message):
    with pytest.raises(ValueError, match=message):
        kak_decomposition(generators)


@pytest.mark.parametrize("kwargs", [
    {"time": "1"}, {"time": 10**400}, {"time": 1j}, {"time": [1]}, {"time": np.inf},
    {"coefficients": [1]}, {"coefficients": [[1], [2]]}, {"coefficients": [1, np.inf]}, {"coefficients": [1, 1j]},
    {"tol": -1}, {"atol": np.nan}, {"n_qubits": 0}, {"n_qubits": True},
])
def test_invalid_numerical_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        kak_decomposition(["X", "Y"], **kwargs)


def test_compile_and_reconstruct_share_real_scalar_policy():
    result = kak_decomposition(["X", "Y"], time=.3 + 0j)
    np.testing.assert_allclose(result.reconstruct(np.float64(.3)), result.unitary_irrep, atol=1e-13, rtol=0)
    with pytest.raises(ValueError, match="finite real scalar"):
        result.reconstruct(1j)


def test_mapping_returns_native_classification_with_the_correct_matrix_presentation():
    mapping, signs, classification = map_dla_to_irrep(["XX", "YY", "ZI", "IZ"])
    assert isinstance(classification, Classification)
    assert classification.get_algebra() == "2*so(3)" and classification.get_dla_dim() == 6
    assert classification.get_orthogonal_size() == 4
    assert classification.get_algebra_basis().shape == (6, 6, 6)
    labelled = labelled_matrix_basis(mapping, signs, classification, n_qubits=2)
    assert np.stack(list(labelled.values())).shape == (6, 4, 4)


def test_classification_and_native_closure_are_independent_of_dimension_guessing():
    generators = ["XXII", "IXXI", "IIXX", "XZII", "IXZI", "IIXZ"]
    classification = get_pauli_string(generators).get_class()
    words = as_pauli_words(generators)
    basis = dla_pauli_basis(words)
    expected = {next(iter(op.pauli_rep)) for op in qml.lie_closure(words)}
    assert set(basis) == set(lie_closure_pauli_words(words, full_size=21)) == expected
    assert classification.get_dla_dim() == len(basis) == 21
    assert classification.get_algebra() == "so(7)"


@pytest.mark.parametrize("generators", [["X", "I"], ["XX", "YY", "ZZ", "XI", "IX"]])
def test_unsupported_algebras_name_their_summands_and_the_closure_helper(generators):
    classification = get_pauli_string(generators).get_class()
    assert classification.get_orthogonal_size() is None
    with pytest.raises(NotImplementedError, match=r"not \(isomorphic to\).*dla_pauli_basis") as info:
        kak_decomposition(generators)
    assert all(summand in str(info.value) for summand in classification.get_subalgebras())


def test_misclassified_dimension_is_reported_before_the_algebra_type(monkeypatch):
    # PauLie says 2*u(1) (dimension 2, no so(m) presentation) for a closure of 6 words.
    wrong = get_pauli_string(["X", "I"]).get_class()
    monkeypatch.setattr(PauliStringCollection, "get_class", lambda self: wrong)
    for call in [kak_decomposition, map_dla_to_irrep]:
        with pytest.raises(ValueError, match="closure produced 6"):
            call(["XI", "ZI", "XX"])


def test_mutated_native_classification_cannot_truncate_the_closure():
    generators = get_pauli_string(["XI", "ZI", "XX"])
    classification = generators.get_class()
    assert classification.get_dla_dim() == 6
    morphs = classification.get_morphs()
    morphs.clear()
    morphs.update(get_pauli_string(["XI", "ZI"]).get_class().get_morphs())
    assert generators.get_class().get_dla_dim() == 3
    basis = dla_pauli_basis(generators)
    assert set(basis) == set(as_pauli_words(["XI", "YI", "ZI", "XX", "YX", "ZX"]))


def test_incorrect_classifier_dimension_is_checked_after_full_closure(monkeypatch):
    smaller = get_pauli_string(["XI", "ZI"]).get_class()
    monkeypatch.setattr(PauliStringCollection, "get_class", lambda self: smaller)
    with pytest.raises(ValueError, match="closure produced 6"):
        dla_pauli_basis(["XI", "ZI", "XX"])
    with pytest.warns(UserWarning, match="closure produced 6"):
        basis = dla_pauli_basis(["XI", "ZI", "XX"], strict=False)
    assert set(basis) == set(as_pauli_words(["XI", "YI", "ZI", "XX", "YX", "ZX"]))


# The compiled evolution: representation, coefficients and reusable gates ----------------------

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
    np.testing.assert_allclose(evolution(result, .83),
                              expected_evolution(generators, coefficients, .83, n), atol=1e-11, rtol=0)


@pytest.mark.parametrize("initial_time, time, coefficients", [
    (0, -2.5, [.3, -.8, .7, -1.2]),
    (np.pi / 4, 25, [1, 1, 1, 1]),  # uniform coefficients and resonance
    (.7, 0, [.3, -.8, .7, -1.2]),
    (0, 25, [0, 0, 0, 0]),
])
def test_reusable_compilation_has_exact_inverse_gates_and_physical_phase(initial_time, time, coefficients):
    result = kak_decomposition(TFXY, coefficients, time=initial_time)
    assert_mirrored_vertical(result.pauli_rotations)
    np.testing.assert_allclose(evolution(result, time),
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
    np.testing.assert_allclose(evolution(result, .2), expected_evolution(TFXY, expected, .2, 2), atol=1e-11, rtol=0)


def test_duplicate_sum_retains_small_terms_between_large_cancellations():
    result = kak_decomposition(["X", "X", "X", "Y"], [1e16, 1, -1e16, 2], time=.2)
    x, y = as_pauli_words(["X", "Y"])
    np.testing.assert_array_equal(result.hamiltonian_irrep, result.algebra_basis[x] + 2 * result.algebra_basis[y])
    np.testing.assert_allclose(evolution(result, .2), expected_evolution(["X", "Y"], [1, 2], .2, 1), atol=1e-13, rtol=0)


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
    np.testing.assert_allclose(evolution(result, time),
                              expected_evolution([generator], [coefficient], time, len(generator)),
                              atol=2e-13, rtol=0)


def test_zero_weights_keep_their_terms_in_the_generator_family():
    result = kak_decomposition([qml.X(0), 0 * qml.Y(0)], time=.3)
    assert result.classification.get_dla_dim() == result.irrep_size == 3
    np.testing.assert_allclose(evolution(result, .3), expected_evolution(["X"], [1], .3, 1), atol=1e-14, rtol=0)
    for args in [([qml.X(0), qml.Y(0), 0 * qml.Z(0)],), (["X", "Y", "Z"], [1, 1, 0])]:
        with pytest.raises(ValueError, match="horizontal"):
            kak_decomposition(*args)


def test_default_partition_is_searched_and_the_failure_cause_is_named():
    generators, coefficients = ["IX", "IY", "XZ"], [.3, -.7, 1.1]
    result = kak_decomposition(generators, coefficients, time=.4)
    assert result.partition == (1, 3)
    np.testing.assert_allclose(evolution(result, .4), expected_evolution(generators, coefficients, .4, 2), atol=1e-13, rtol=0)
    with pytest.raises(ValueError, match=r"BDI\(2, 2\).*or partition"):
        kak_decomposition(generators, coefficients, invol_kwargs={"p": 2})
    with pytest.raises(ValueError, match="for any partition"):
        kak_decomposition(["XI", "YI", "IX", "IY"])


# reconstruct_from_pauli_rotations: ordered analytic rotations and the general fallback -----------

@pytest.mark.parametrize("time", [None, -2.3])
def test_order_and_time_scaling(time):
    basis = {"xy": skew_plane(5, 0, 1, 2.), "yz": skew_plane(5, 1, 3, -2.), "zw": skew_plane(5, 3, 4, .75)}
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
    skew_plane(4, 0, 1, 2.) + skew_plane(4, 1, 3, -.7),
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
    {"algebra_basis": {"p": skew_plane(2, 0, 1, 1e308)}, "time": 2.},
    {"algebra_basis": {"p": np.eye(2) * 1e308}, "time": 2.},
])
def test_invalid_inputs_and_scaled_overflow(arguments):
    inputs = {
        "pauli_rotations": [("p", 2., "a0")], "algebra_basis": {"p": skew_plane(2, 0, 1, 2.)},
        "irrep_size": 2, "time": 1.,
    }
    inputs.update(arguments)
    with pytest.raises(ValueError):
        reconstruct_from_pauli_rotations(**inputs)
