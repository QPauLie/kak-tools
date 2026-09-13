"""Pauli-word closure and the so(n) mapping checked against PennyLane and dense commutators,
the mapping conventions and input contracts of the matrix interfaces, and the transverse-field
XY workflows checked against the qubit evolution."""

import os
import pathlib
import re
import subprocess
import sys
from itertools import combinations, product

import numpy as np
import pennylane as qml
import pytest
from paulie import get_pauli_string
from pennylane.pauli import PauliWord
from scipy.linalg import expm

from kak_tools import lie_closure_pauli_words, make_signs, map_irrep_to_matrices, map_simple_to_irrep
from kak_tools.dense_cartan import group_matrix_to_reducible, group_matrix_to_reducible_str
from kak_tools.full_workflows import (
    complete_workflow_tfXY, diagonalization_tfXY, minimal_workflow_tfXY, workflow_tfXY_known_algebra,
)
from kak_tools.map_to_irrep import (
    E, HorizontalEmbeddingError, anticom_graph_irrep, irrep_dot, map_horizontal_subgraph, map_matrix_to_reducible,
)
from kak_tools.pauli_dlas import anticom_graph_pauli, split_pauli_algebra
from kak_tools.paulie_bridge import as_pauli_words, kak_decomposition, labelled_matrix_basis, pauli_string_to_word
from kak_tools.tfxy_model import (
    _make_tfXY_coeffs, make_so_2n, make_so_2n_full_mapping, make_so_2n_full_mapping_str,
    make_tfXY_hamiltonian_irrep, make_tfXY_hamiltonian_qubits,
)
from checks import assert_equal_up_to_sign, pauli_hamiltonian, physical_evolution, rotation, tfxy_words


# Closure and the so(n) homomorphism -----------------------------------------------------

def assert_lie_homomorphism(basis, matrices, n_qubits):
    """Check ``[M_a, M_b]`` against the Pauli commutator of every pair of basis words."""
    size = len(next(iter(matrices.values())))
    for first, second in combinations(basis, 2):
        bracket = (1j * first).commutator(1j * second) / 1j
        bracket.simplify()
        expected = sum((c * matrices[w] for w, c in bracket.items()), np.zeros((size, size)))
        np.testing.assert_allclose(matrices[first] @ matrices[second] - matrices[second] @ matrices[first], expected)
    # Independent full structure constants also check the representation scale.
    dense = qml.structure_constants([matrices[w] / 1j for w in basis], matrix=True, is_orthogonal=False)
    np.testing.assert_allclose(dense, qml.structure_constants(basis))


def majorana_so_basis(m, rng):
    """Random so(m) Pauli representation: Jordan-Wigner Majoranas in random order and their pair products."""
    majoranas = [PauliWord(dict.fromkeys(range(j), "Z") | {j: p}) for j in range((m + 1) // 2) for p in "XY"][:m]
    majoranas = [majoranas[k] for k in rng.permutation(m)]
    return {(a, b): majoranas[a]._matmul(majoranas[b])[0] for a, b in combinations(range(m), 2)}


def random_generating_planes(p, q, rng):
    """Random connected spanning subgraph of K_{p,q}: its edges generate so(p + q)."""
    rows, columns = list(range(p)), list(range(p, p + q))
    tree_rows, tree_columns = [rows.pop(rng.integers(p))], [columns.pop(rng.integers(q))]
    planes = {(tree_rows[0], tree_columns[0])}
    while rows or columns:
        if rows and (not columns or rng.random() < .5):
            row = rows.pop(rng.integers(len(rows)))
            planes.add((row, tree_columns[rng.integers(len(tree_columns))]))
            tree_rows.append(row)
        else:
            column = columns.pop(rng.integers(len(columns)))
            planes.add((tree_rows[rng.integers(len(tree_rows))], column))
            tree_columns.append(column)
    planes |= {(i, j) for i in range(p) for j in range(p, p + q) if rng.random() < .2}
    return [planes.pop() for _ in range(len(planes))]


@pytest.mark.parametrize("generators", [
    [PauliWord({0: "X"}), PauliWord({0: "Y"})],
    [PauliWord({0: "X"}), PauliWord({1: "Y"})],
    tfxy_words(3),
])
def test_closure_matches_pennylane(generators):
    actual = lie_closure_pauli_words(generators)
    expected = [next(iter(op.pauli_rep)) for op in qml.lie_closure(generators)]
    assert len(actual) == len(set(actual))
    assert set(actual) == set(expected)


@pytest.mark.parametrize("n", [2, 3])
def test_mapping_preserves_all_commutators(n):
    generators = tfxy_words(n)
    basis = lie_closure_pauli_words(generators)
    assert len(basis) == n * (2 * n - 1)
    mapping, signs = map_simple_to_irrep(basis, generators, n=2 * n, invol_type="BDI")
    assert_lie_homomorphism(basis, map_irrep_to_matrices(mapping, signs, 2 * n, "BDI"), n)


@pytest.mark.parametrize("m", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_generating_sets_embed_horizontally_and_map_to_a_homomorphism(m, seed):
    rng = np.random.default_rng(1000 * m + seed)
    basis = majorana_so_basis(m, rng)
    p = int(rng.integers(1, m))
    horizontal = [basis[plane] for plane in random_generating_planes(p, m - p, rng)]
    planes = list(basis)
    ops = [basis[planes[k]] for k in rng.permutation(len(planes))]
    mapping, signs = map_simple_to_irrep(ops, horizontal, n=m, invol_type="BDI", invol_kwargs={"p": p})
    assert set(mapping.values()) == set(ops) and set(signs) == set(mapping)
    assert all(i < p <= j for (i, j), word in mapping.items() if word in horizontal)
    assert_lie_homomorphism(ops, map_irrep_to_matrices(mapping, signs, m, "BDI"), (m + 1) // 2)


def test_non_generating_horizontal_words_raise_deterministically():
    basis = majorana_so_basis(6, np.random.default_rng(0))
    ops = list(basis.values())
    disconnected = [basis[(0, 3)], basis[(1, 4)]]  # Two commuting words embed but span only so(2) + so(2).
    with pytest.raises(ValueError, match="do not generate so"):
        map_simple_to_irrep(ops, disconnected, n=6, invol_type="BDI")
    with pytest.raises(ValueError, match="ops must be the 15 Pauli words"):
        map_simple_to_irrep(ops[1:], [basis[plane] for plane in [(0, 3), (0, 4), (0, 5), (1, 3), (2, 3)]], n=6, invol_type="BDI")


def test_inconsistent_partial_mapping_is_rejected_as_a_non_representation():
    x, y, z1 = PauliWord({0: "X"}), PauliWord({0: "Y"}), PauliWord({1: "Z"})
    with pytest.raises(ValueError, match="do not represent so"):
        map_simple_to_irrep([x, y, z1], {(0, 2): x, (0, 3): y, (1, 2): z1}, n=4, invol_type="BDI")


@pytest.mark.parametrize("words, n, options", [
    (tfxy_words(6), 12, {"p": 5}),  # The TF-XY terms need six rows and six columns.
    ([PauliWord({0: p}) for p in "XYZ"], 3, {"p": 1}),  # A triangle is a row or a column of three.
    ([PauliWord({0: p}) for p in "XYZ"], 4, {"p": 2, "q": 2}),
    ([PauliWord({0: "X"}), PauliWord({1: "X"})], 2, {"p": 1}),  # Two commuting words need two rows.
    # A five-cycle of anticommutations is not a line graph of a bipartite graph.
    ([PauliWord({0: "X"}), PauliWord({0: "Z"}), PauliWord({0: "X", 1: "X"}), PauliWord({1: "Z"}),
      PauliWord({0: "Z", 1: "Y"})], 5, {"p": 2}),
])
def test_words_that_do_not_fit_the_horizontal_subspace_raise_before_any_search(words, n, options):
    with pytest.raises(HorizontalEmbeddingError, match="horizontal"):
        map_simple_to_irrep(words, words, n=n, invol_type="BDI", invol_kwargs=options)


def test_components_are_oriented_independently_to_fit_rows_and_columns():
    triangles = [PauliWord({q: p}) for q in (0, 1) for p in "XYZ"]
    mapping = map_horizontal_subgraph(anticom_graph_pauli(triangles), 4, 4)
    assert len(mapping) == 6 and all(i < 4 <= j for i, j in mapping)
    for qubit in (0, 1):
        planes = [plane for plane, word in mapping.items() if word.wires == qml.wires.Wires(qubit)]
        assert len(set(planes[0]) & set(planes[1]) & set(planes[2])) == 1
    with pytest.raises(HorizontalEmbeddingError, match="3 rows or 5 columns"):
        map_horizontal_subgraph(anticom_graph_pauli(triangles), 3, 5)


def test_partial_horizontal_mapping_is_completed_to_the_full_basis():
    n = 3
    mapping, signs = map_simple_to_irrep(make_so_2n(n), tfxy_words(n), n=2 * n, invol_type="BDI")
    # The XX couplings and Z fields alone generate the algebra (paper App. F.2).
    partial = {plane: word for plane, word in mapping.items() if "Y" not in word.values()}
    completed, completed_signs = map_simple_to_irrep(make_so_2n(n), partial, n=2 * n, invol_type="BDI")
    assert (completed, completed_signs) == (mapping, signs)
    with pytest.raises(ValueError, match="horizontal mapping"):
        map_simple_to_irrep(make_so_2n(n), {(0, 1): PauliWord({0: "X"})}, n=2 * n, invol_type="BDI")


def test_mapping_is_independent_of_the_hash_seed_and_orders_wires():
    """The plane assignment must not depend on set iteration order, and mapped words
    carry their wires in sorted order so ``word.wires`` can be used directly for gates."""
    script = (
        "import sys; sys.path.insert(0, 'tests')\n"
        "from checks import tfxy_words\n"
        "from kak_tools import lie_closure_pauli_words, map_simple_to_irrep\n"
        "for n in (2, 3, 4):\n"
        "    words = tfxy_words(n)\n"
        "    mapping, signs = map_simple_to_irrep(lie_closure_pauli_words(words), words, n=2 * n, invol_type='BDI')\n"
        "    assert all(list(w.wires) == sorted(w.wires) for w in mapping.values())\n"
        "    print(sorted((plane, str(w), signs[plane]) for plane, w in mapping.items()))\n"
    )
    outputs = set()
    for seed in ("0", "1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True,
            cwd=pathlib.Path(__file__).resolve().parents[1], env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


# Mapping conventions and the failures of the matrix interfaces --------------------------------

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
    so3 = dict(zip([(0, 1), (0, 2), (1, 2)], as_pauli_words(["X", "Y", "Z"]), strict=True))
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


def test_legacy_so2_angle_survives_a_sine_rounded_beyond_one():
    mapping = {(0, 1): ("XX", 1)}
    sine = np.nextafter(1.0, 2.0)
    [(word, angle)] = group_matrix_to_reducible_str(np.array([[0., sine], [-sine, 0.]]), 0, mapping).items()
    assert word == "XX" and np.isclose(angle, np.pi / 4)
    theta = 2.5
    [(_, angle)] = group_matrix_to_reducible_str(rotation(theta), 0, mapping).items()
    assert np.isclose(angle, theta / 2)


# The transverse-field XY model and its exported workflows ---------------------------------

def compressed_string_to_word(text):
    """Parse ``{left}A{zs}B{right}`` from :func:`make_so_2n_full_mapping_str`; numbers count identities and Zs."""
    left, first, zs, second, _ = re.fullmatch(r"(\d+)([XYZ])(?:(\d+)([XY]))?(\d+)", text).groups()
    word = {int(left): first}
    if second is not None:
        word |= dict.fromkeys(range(int(left) + 1, int(left) + 1 + int(zs)), "Z")
        word[int(left) + 1 + int(zs)] = second
    return PauliWord(word)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_tfxy_full_mapping_signs_follow_the_paper_and_make_signs(n):
    mapping, signs = make_so_2n_full_mapping(n)
    assert set(signs) == set(mapping) == set(combinations(range(2 * n), 2))
    assert signs == make_signs(mapping, 2 * n, "BDI")
    assert all(signs[(i, n + i)] == -1 and mapping[(i, n + i)] == PauliWord({i: "Z"}) for i in range(n))  # Eq. F14
    strings = make_so_2n_full_mapping_str(n)
    assert {plane: compressed_string_to_word(text) for plane, (text, _) in strings.items()} == mapping
    assert {plane: sign for plane, (_, sign) in strings.items()} == signs
    assert set(mapping.values()) == set(make_so_2n(n))
    assert_lie_homomorphism(list(mapping.values()), map_irrep_to_matrices(mapping, signs, 2 * n, "BDI"), n)


def test_tfxy_coefficients_are_normalized_seedable_and_validated():
    for choice in ("random", "random TF", "uniform"):
        alphas, betas, gammas = _make_tfXY_coeffs(4, choice, rng=3)
        assert alphas.shape == betas.shape == (3,) and gammas.shape == (4,)
        assert np.isclose(np.linalg.norm(np.concatenate([alphas, betas, gammas])), 1)
    first, second = _make_tfXY_coeffs(4, "random", rng=3), _make_tfXY_coeffs(4, "random", rng=3)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    _, _, coeffs = make_tfXY_hamiltonian_qubits(4, "random", rng=3)
    np.testing.assert_array_equal(coeffs, np.concatenate(first))
    for invalid in ("Random", None, np.ones(3)):
        with pytest.raises(ValueError, match="coefficients must be"):
            _make_tfXY_coeffs(4, invalid)


def qubit_hamiltonian(n, coefficients, rng):
    _, generators, coeffs = make_tfXY_hamiltonian_qubits(n, coefficients, rng)
    return pauli_hamiltonian(generators, coeffs, n)


def signed_sums(rates, n):
    """The spectrum of a free-fermion Hamiltonian with single-particle rates ``rates``."""
    return sorted(sum(s * r for s, r in zip(signs, rates, strict=True)) for signs in product([-1, 1], repeat=n))


@pytest.mark.parametrize("n", [2, 3])
def test_tfxy_irrep_hamiltonian_carries_the_factor_two_of_the_isomorphism(n):
    h_irrep = make_tfXY_hamiltonian_irrep(n, "random", rng=n)
    h_qubit = qubit_hamiltonian(n, "random", rng=n)
    np.testing.assert_allclose(h_irrep, -h_irrep.T, atol=0)
    # exp(t H_irrep) represents exp(i t H): the qubit spectrum consists of the signed sums of
    # the rates lambda, while H_irrep has eigenvalues +-2i lambda (Eq. F10).
    rates = np.sort(np.abs(np.linalg.eigvals(h_irrep).imag))[::2] / 2
    np.testing.assert_allclose(signed_sums(rates, n), np.linalg.eigvalsh(h_qubit), atol=1e-12)


@pytest.mark.parametrize("workflow, parse", [
    (minimal_workflow_tfXY, compressed_string_to_word),
    (complete_workflow_tfXY, lambda word: word),
    (workflow_tfXY_known_algebra, lambda word: word),
])
@pytest.mark.parametrize("n, t0", [(2, .83), (3, 2.)])
def test_exported_workflows_compile_the_qubit_evolution(workflow, parse, n, t0):
    rotations = workflow(n, t0, "random", rng=n)
    kinds = [kind for _, _, kind in rotations]
    assert {"a0"} <= set(kinds) <= {"k1", "k2", "a", "a0"} and kinds.count("a0") == n
    for time in (t0, 2.5 * t0):
        unitary = physical_evolution(rotations, time, n, parse=parse)
        assert_equal_up_to_sign(unitary, expm(1j * time * qubit_hamiltonian(n, "random", rng=n)))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_diagonalization_rates_reproduce_the_qubit_spectrum(n):
    rates = diagonalization_tfXY(n, .1, "random", rng=n)
    np.testing.assert_allclose(
        signed_sums(rates, n), np.linalg.eigvalsh(qubit_hamiltonian(n, "random", rng=n)), atol=1e-12
    )
