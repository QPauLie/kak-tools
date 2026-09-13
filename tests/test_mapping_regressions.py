"""Mapping conventions and the distinct failures of the public matrix interfaces."""

import numpy as np
import pytest
from paulie import get_pauli_string
from pennylane.pauli import PauliWord
from scipy.linalg import expm

from kak_tools.dense_cartan import group_matrix_to_reducible
from kak_tools.map_to_irrep import (
    E, anticom_graph_irrep, irrep_dot, make_signs, map_matrix_to_reducible, map_simple_to_irrep,
)
from kak_tools.pauli_dlas import anticom_graph_pauli, lie_closure_pauli_words, split_pauli_algebra
from kak_tools.paulie_bridge import (
    as_pauli_words, kak_decomposition, labelled_matrix_basis,
    pauli_string_to_word,
)


@pytest.fixture
def so4_mapping():
    strings = {(0, 1): "XY", (0, 2): "ZI", (0, 3): "XX",
               (1, 2): "YY", (1, 3): "IZ", (2, 3): "YX"}
    mapping = {plane: pauli_string_to_word(word) for plane, word in strings.items()}
    signs = {(0, 1): 1, (0, 2): -1, (0, 3): 1, (1, 2): 1, (1, 3): -1, (2, 3): -1}
    return mapping, signs, get_pauli_string(["XX", "YY", "ZI", "IZ"]).get_class()


def test_standard_and_alternative_star_gauges_preserve_all_physical_brackets(so4_mapping):
    mapping, signs, classification = so4_mapping
    assert make_signs(mapping, 4, "BDI") == signs
    so3 = dict(zip([(0, 1), (0, 2), (1, 2)], as_pauli_words(["X", "Y", "Z"])))
    assert make_signs(so3, 3, "BDI") == {(0, 1): -1, (0, 2): 1, (1, 2): -1}
    gauge = [1, -1, 1, -1]
    alternative = {plane: complex(s * gauge[plane[0]] * gauge[plane[1]]) for plane, s in signs.items()}
    matrices = labelled_matrix_basis(mapping, alternative, classification, n_qubits=2)
    physical = np.stack([1j * word.to_mat(wire_order=range(2)) for word in mapping.values()])
    mapped = np.stack([matrices[word] for word in mapping.values()])
    brackets = physical[:, None] @ physical[None, :] - physical[None, :] @ physical[:, None]
    coefficients = np.einsum("kab,ijab->ijk", physical.conj(), brackets) / 4
    expected = np.einsum("ijk,kab->ijab", coefficients, mapped)
    np.testing.assert_array_equal(mapped[:, None] @ mapped[None, :] - mapped[None, :] @ mapped[:, None], expected)
    assert mapped.dtype.kind == "f"


@pytest.mark.parametrize("defect, message", [
    ("swap", "star commutator"), ("sign", "star commutators"), ("symbol", "PauliWord entries"),
    ("duplicate", "distinct|bijection"), ("plane", "rotation plane"),
    ("missing_sign", "sign"), ("invalid_sign", "sign"),
])
def test_public_matrix_basis_rejects_invalid_lie_maps(so4_mapping, defect, message):
    mapping, signs, classification = so4_mapping
    if defect == "swap":
        original = set(mapping.values())
        mapping[(1, 2)], mapping[(1, 3)] = mapping[(1, 3)], mapping[(1, 2)]
        assert set(mapping.values()) == original  # Still a complete bijection.
    elif defect == "sign":
        signs[(1, 2)] *= -1
    elif defect == "symbol":
        mapping[(0, 1)] = PauliWord({0: "A"})
    elif defect == "duplicate":
        mapping[(0, 2)] = mapping[(0, 1)]
    elif defect == "plane":
        del mapping[(1, 2)]
        del signs[(1, 2)]
    elif defect == "missing_sign":
        del signs[(1, 2)]
    else:
        signs[(1, 2)] = 2
    with pytest.raises(ValueError, match=message):
        labelled_matrix_basis(mapping, signs, classification)


def test_three_one_qubit_clifford_generators_do_not_form_a_faithful_so4_basis():
    x, y, z = as_pauli_words(["X", "Y", "Z"])
    collapsed = {(0, 1): x, (0, 2): y, (0, 3): z, (1, 2): z, (1, 3): y, (2, 3): x}
    with pytest.raises(ValueError, match="distinct PauliWords"):
        make_signs(collapsed, 4, "BDI")


def test_identity_u1_keeps_the_isolated_plane_and_accepts_another_gauge():
    graph = anticom_graph_irrep(2, "BDI", {})
    assert set(graph) == {(0, 1)} and graph.number_of_edges() == 0
    word = pauli_string_to_word("I")
    mapping = {(0, 1): word}
    assert make_signs(mapping, 2, "BDI") == {(0, 1): -1}
    labelled = labelled_matrix_basis(mapping, {(0, 1): 1}, get_pauli_string(["I"]).get_class())
    np.testing.assert_array_equal(labelled[word], E((0, 1), 2, "BDI"))


@pytest.mark.parametrize("options, planes", [
    ({"q": np.uint64(1)}, {(0, 2), (1, 2)}),
    ({"p": 1}, {(0, 1), (0, 2)}),
    ({"p": np.int64(2), "q": 1}, {(0, 2), (1, 2)}),
])
def test_partition_complements_agree_in_direct_mapping_and_bridge(options, planes):
    words = as_pauli_words(["X", "Y", "Z"])
    assert set(anticom_graph_irrep(np.int64(3), "BDI", options)) == planes
    mapping, signs = map_simple_to_irrep(words, horizontal_ops=words[:2], n=3, invol_type="BDI", invol_kwargs=options)
    assert set(mapping.values()) == set(words) and set(mapping) == set(signs)
    assert {plane for plane, word in mapping.items() if word in words[:2]} == planes
    result = kak_decomposition(["X", "Y"], [1, 2], time=.2, invol_kwargs=options)
    np.testing.assert_allclose(result.reconstruct(), result.unitary_irrep, atol=1e-13, rtol=0)


@pytest.mark.parametrize("options", [{"p": 0}, {"q": True}, {"p": 1.5}, {"p": 2, "q": 2}, {"other": 1}, {"q": np.uint64(2**64 - 1)}])
def test_direct_mapping_rejects_invalid_partitions(options):
    words = as_pauli_words(["X", "Y", "Z"])
    with pytest.raises(ValueError, match="BDI"):
        map_simple_to_irrep(words, horizontal_ops=words[:2], n=3, invol_type="BDI", invol_kwargs=options)


@pytest.fixture
def dot_mapping():
    x, y = as_pauli_words(["X", "Y"])
    return [x, y], {(0, 1): x, (0, 2): y}, {(0, 1): -1, (0, 2): 1}


@pytest.mark.parametrize("convention", ["separate", "paired"])
def test_irrep_dot_preserves_accepted_calling_conventions(dot_mapping, convention):
    words, mapping, signs = dot_mapping
    paired = {plane: (word, signs[plane]) for plane, word in mapping.items()}
    if convention == "separate":
        result = irrep_dot([.3, -1.7], words, mapping, signs, n=np.int64(3), invol_type="BDI")
    else:
        result = irrep_dot([.3, -1.7], words, paired, n=3)
    np.testing.assert_array_equal(result, -.3 * E((0, 1), 3, "BDI") - 1.7 * E((0, 2), 3, "BDI"))
    with pytest.raises(TypeError):
        irrep_dot([.3, -1.7], words, paired, 3, "BDI")  # n is keyword-only.


@pytest.mark.parametrize("defect", ["coefficient_count", "dimension", "missing_word", "missing_sign", "sign", "node"])
def test_irrep_dot_rejects_invalid_arguments(dot_mapping, defect):
    words, mapping, signs = dot_mapping
    coefficients, n = [1, 2], 3
    if defect == "coefficient_count":
        coefficients = [1]
    elif defect == "dimension":
        n = 1.5
    elif defect == "missing_word":
        del mapping[(0, 2)]
    elif defect == "missing_sign":
        del signs[(0, 2)]
    elif defect == "sign":
        signs[(0, 1)] = 2
    else:
        mapping[(0.0, 1)] = mapping.pop((0, 1))
    with pytest.raises(ValueError):
        irrep_dot(coefficients, words, mapping, signs, n=n, invol_type="BDI")


def test_empty_irrep_dot_returns_a_matrix():
    np.testing.assert_array_equal(irrep_dot([], [], {}, n=3, invol_type="BDI"), np.zeros((3, 3)))


@pytest.mark.parametrize("call", [
    lambda: anticom_graph_irrep(3, "unknown", {}),
    lambda: E((0, 1), 2, "unknown"),
    lambda: make_signs({(0, 1): pauli_string_to_word("X")}, 2, "DIII"),
    lambda: map_matrix_to_reducible(np.zeros((2, 2)), {}, {}, "AIII"),
    lambda: irrep_dot([], [], {}, n=2, invol_type="AIII"),
    lambda: map_simple_to_irrep(as_pauli_words(["X"]), as_pauli_words(["X"]), n=2, invol_type="AI"),
])
def test_unsupported_involutions_raise_not_implemented_instead_of_returning_none(call):
    with pytest.raises(NotImplementedError, match="Only BDI"):
        call()


@pytest.mark.parametrize("call", [
    lambda: anticom_graph_pauli(["X"]),
    lambda: split_pauli_algebra(["X"]),
    lambda: lie_closure_pauli_words(["X"]),
    lambda: map_simple_to_irrep(["X", "Y", "Z"], as_pauli_words(["X", "Y"]), n=3, invol_type="BDI"),
    lambda: map_simple_to_irrep(as_pauli_words(["X", "Y", "Z"]), ["X", "Y"], n=3, invol_type="BDI"),
    lambda: map_simple_to_irrep(as_pauli_words(["X", "Y", "Z"]), {(0, 1): "X"}, n=3, invol_type="BDI"),
])
def test_pauli_word_input_checks_raise_value_error_even_under_python_O(call):
    with pytest.raises(ValueError, match="PauliWords"):
        call()


def test_exact_pi_rotation_is_not_dropped_and_a_reflection_is_rejected():
    word = pauli_string_to_word("X")
    mapping, signs = {(0, 1): word}, {(0, 1): 1}
    [(actual, angle)] = group_matrix_to_reducible(-np.eye(2), 0, mapping, signs).items()
    assert actual == word and np.isclose(angle, np.pi / 2)
    np.testing.assert_allclose(expm(E((0, 1), 2, "BDI") * angle), -np.eye(2), atol=1e-15, rtol=0)
    assert not group_matrix_to_reducible(np.eye(2), 0, mapping, signs)
    with pytest.raises(ValueError, match="determinant -1"):
        group_matrix_to_reducible(np.diag([1., -1.]), 0, mapping, signs)
