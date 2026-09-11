"""Input normalization and independent classification/closure contracts."""

import numpy as np
import pennylane as qml
import pytest
from paulie import get_pauli_string
from paulie.classifier.classification import Classification
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_collection import PauliStringCollection
from pauliebits import pauliebits
from pennylane.pauli import PauliWord

from kak_tools import (
    as_pauli_collection, as_pauli_words, labelled_matrix_basis,
    dla_pauli_basis, identify_algebra, kak_decomposition, lie_closure_pauli_words,
    map_dla_to_irrep, pauli_string_to_word, pauli_word_to_string,
)


@pytest.mark.parametrize("text, expected", [("XYZI", {0: "X", 1: "Y", 2: "Z"}), ("IIII", {}), ("IIZ", {2: "Z"})])
def test_pauli_string_conversion_preserves_positions_and_roundtrips(text, expected):
    word = pauli_string_to_word(text)
    assert word == PauliWord(expected)
    assert str(pauli_word_to_string(word, len(text))) == text


@pytest.mark.parametrize("endians, collection", [(["big", "big"], False), (["little", "big"], True)])
def test_native_bit_endianness_and_container_do_not_change_the_closure(endians, collection):
    native = [PauliString(bits=pauliebits(bits, endian=endian))
              for bits, endian in zip(["10", "01"], endians)]
    inputs = PauliStringCollection(native) if collection else native
    assert as_pauli_collection(inputs).get_class().get_dla_dim() == 3
    assert set(dla_pauli_basis(inputs)) == set(as_pauli_words(["X", "Y", "Z"]))


@pytest.mark.parametrize("generators, width", [
    (["XII", "YII"], 3),
    ([qml.X(0) @ qml.I(5), qml.Y(0) @ qml.I(5)], 6),
    ([qml.X(0), qml.Y(0), 0 * qml.I(7)], 8),
])
def test_register_width_survives_padding_identity_and_intrinsic_zero_terms(generators, width):
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
    ([0 * qml.Z(0)], "nonzero"), ([1j * qml.X(0)], "finite real scalar"),
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
    assert {(kind, size) for _, kind, size in identify_algebra(basis)} == {("so", 7), ("sp", 3)}


def test_unsupported_algebra_and_involution_are_explicit():
    assert get_pauli_string(["X", "I"]).get_class().get_orthogonal_size() is None
    with pytest.raises(NotImplementedError, match=r"not \(isomorphic to\)"):
        kak_decomposition(["X", "I"])
    with pytest.raises(NotImplementedError, match="involution"):
        kak_decomposition(["X", "Y"], involution="DIII")


@pytest.mark.parametrize("basis, convert", [
    (["XI", "YI"], True), (["XI", "YI", "XI"], True),
    (["IX", "IY", "IZ"], True), (["XI", "YI", "ZI"], False),
])
def test_supplied_basis_must_be_complete_distinct_and_contain_the_generators(basis, convert):
    words = list(map(pauli_string_to_word, basis)) if convert else basis
    with pytest.raises(ValueError, match="supplied DLA basis"):
        map_dla_to_irrep(["XI", "YI"], dla=words)


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
