"""Legacy closure and mapping checked against PennyLane and dense commutators."""

from itertools import combinations
import numpy as np
import pennylane as qml
import pytest
from pennylane.pauli import PauliWord
from kak_tools import lie_closure_pauli_words, map_simple_to_irrep, map_irrep_to_matrices


def tfxy_words(n):
    return [PauliWord({i: p, i + 1: p}) for p in "XY" for i in range(n - 1)] + [
        PauliWord({i: "Z"}) for i in range(n)
    ]


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
    matrices = map_irrep_to_matrices(mapping, signs, 2 * n, "BDI")
    for first, second in combinations(basis, 2):
        bracket = (1j * first).commutator(1j * second) / 1j
        bracket.simplify()
        expected = sum((c * matrices[w] for w, c in bracket.items()), np.zeros((2 * n, 2 * n)))
        np.testing.assert_allclose(matrices[first] @ matrices[second] - matrices[second] @ matrices[first], expected)
    # Independent full structure constants also check the representation scale.
    dense = qml.structure_constants([matrices[w] / 1j for w in basis], matrix=True, is_orthogonal=False)
    np.testing.assert_allclose(dense, qml.structure_constants(basis))
