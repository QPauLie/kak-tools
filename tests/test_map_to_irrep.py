"""Pauli-word closure and so(n) mapping checked against PennyLane and dense commutators,
and the transverse-field XY workflows checked against the qubit evolution."""

import os
import pathlib
import subprocess
import sys
import re
from itertools import combinations, product
import numpy as np
import pennylane as qml
import pytest
from pennylane.pauli import PauliWord
from scipy.linalg import expm
from kak_tools import lie_closure_pauli_words, map_simple_to_irrep, map_irrep_to_matrices, make_signs
from kak_tools.map_to_irrep import HorizontalEmbeddingError
from kak_tools.pauli_dlas import anticom_graph_pauli
from kak_tools.tfxy_model import (
    _make_tfXY_coeffs, make_so_2n, make_so_2n_full_mapping, make_so_2n_full_mapping_str,
    make_tfXY_hamiltonian_irrep, make_tfXY_hamiltonian_qubits,
)
from kak_tools.full_workflows import (
    complete_workflow_tfXY, diagonalization_tfXY, minimal_workflow_tfXY, workflow_tfXY_known_algebra,
)


def tfxy_words(n):
    return [PauliWord({i: p, i + 1: p}) for p in "XY" for i in range(n - 1)] + [
        PauliWord({i: "Z"}) for i in range(n)
    ]


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
    from kak_tools.map_to_irrep import map_horizontal_subgraph

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
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    _, _, coeffs = make_tfXY_hamiltonian_qubits(4, "random", rng=3)
    np.testing.assert_array_equal(coeffs, np.concatenate(first))
    for invalid in ("Random", None, np.ones(3)):
        with pytest.raises(ValueError, match="coefficients must be"):
            _make_tfXY_coeffs(4, invalid)


def qubit_hamiltonian(n, coefficients, rng):
    _, generators, coeffs = make_tfXY_hamiltonian_qubits(n, coefficients, rng)
    return sum(c * g.to_mat(wire_order=range(n)) for c, g in zip(coeffs, generators))


def assert_equal_up_to_sign(actual, expected):
    """Pauli rotations lift SO(2) angles to the spin group only up to a global sign."""
    assert min(np.abs(actual - expected).max(), np.abs(actual + expected).max()) < 1e-13


@pytest.mark.parametrize("n", [2, 3])
def test_tfxy_irrep_hamiltonian_carries_the_factor_two_of_the_isomorphism(n):
    h_irrep = make_tfXY_hamiltonian_irrep(n, "random", rng=n)
    h_qubit = qubit_hamiltonian(n, "random", rng=n)
    np.testing.assert_allclose(h_irrep, -h_irrep.T, atol=0)
    # exp(t H_irrep) represents exp(i t H): the qubit spectrum consists of the signed sums of
    # the rates lambda, while H_irrep has eigenvalues +-2i lambda (Eq. F10).
    rates = np.sort(np.abs(np.linalg.eigvals(h_irrep).imag))[::2] / 2
    spectrum = sorted(sum(s * r for s, r in zip(signs, rates)) for signs in product([-1, 1], repeat=n))
    np.testing.assert_allclose(spectrum, np.linalg.eigvalsh(h_qubit), atol=1e-12)


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
        unitary = np.eye(2**n, dtype=complex)
        for word, coefficient, kind in rotations:
            angle = coefficient * (time if kind == "a0" else 1)
            unitary = unitary @ expm(1j * angle * parse(word).to_mat(wire_order=range(n)))
        assert_equal_up_to_sign(unitary, expm(1j * time * qubit_hamiltonian(n, "random", rng=n)))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_diagonalization_rates_reproduce_the_qubit_spectrum(n):
    rates = diagonalization_tfXY(n, .1, "random", rng=n)
    spectrum = sorted(sum(s * r for s, r in zip(signs, rates)) for signs in product([-1, 1], repeat=n))
    np.testing.assert_allclose(spectrum, np.linalg.eigvalsh(qubit_hamiltonian(n, "random", rng=n)), atol=1e-12)


def test_mapping_is_independent_of_the_hash_seed_and_orders_wires():
    """The plane assignment must not depend on set iteration order, and mapped words
    carry their wires in sorted order so ``word.wires`` can be used directly for gates."""
    script = (
        "import sys; sys.path.insert(0, 'tests')\n"
        "from test_map_to_irrep import tfxy_words\n"
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
