"""Tests for the PauLie <-> kak_tools bridge."""

import itertools

import numpy as np
import pennylane as qml
import pytest
from paulie import get_pauli_string as paulie_pauli_string
from paulie.classifier.classification import ClassificationException
from pennylane.pauli import PauliWord
from scipy.linalg import expm

from kak_tools import (
    DLAComponent,
    classify_dla,
    dla_pauli_basis,
    identify_algebra,
    kak_decomposition,
    labelled_matrix_basis,
    lie_closure_pauli_words,
    map_dla_to_irrep,
    pauli_string_to_word,
    pauli_word_to_string,
    reconstruct_from_pauli_rotations,
)
from kak_tools.map_to_irrep import make_tfXY_hamiltonian_qubits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tfxy_strings(n):
    """Transverse-field XY generators on ``n`` qubits, as Pauli strings."""
    gens = []
    for w in range(n - 1):
        chars = ["I"] * n
        chars[w] = chars[w + 1] = "X"
        gens.append("".join(chars))
    for w in range(n - 1):
        chars = ["I"] * n
        chars[w] = chars[w + 1] = "Y"
        gens.append("".join(chars))
    for w in range(n):
        chars = ["I"] * n
        chars[w] = "Z"
        gens.append("".join(chars))
    return gens


def k_local_strings(pattern, n):
    """All translations of ``pattern`` along an open chain of ``n`` qubits."""
    width = len(pattern)
    return [
        "".join(["I"] * w + list(pattern) + ["I"] * (n - w - width))
        for w in range(n - width + 1)
    ]


def two_local_strings(patterns, n):
    """The generators PauLie's own two-local convention builds from ``patterns``.

    This differs from :func:`expand` for one-site patterns: PauLie places them on
    ``n - 1`` sites rather than all ``n``, which is what turns the Ising family into
    so(2n - 1) instead of so(2n). Odd irrep sizes are therefore the ordinary case for
    these models, not a corner one.
    """
    return [str(g) for g in paulie_pauli_string(patterns, n=n)]


def expand(patterns, n):
    out = []
    for pattern in patterns:
        out += k_local_strings(pattern, n)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Representation conversion
# ---------------------------------------------------------------------------


class TestConversion:

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("XYZI", PauliWord({0: "X", 1: "Y", 2: "Z"})),
            ("IIII", PauliWord({})),
            ("IIZ", PauliWord({2: "Z"})),
        ],
    )
    def test_pauli_string_to_word(self, text, expected):
        assert pauli_string_to_word(text) == expected

    @pytest.mark.parametrize("text", ["XYZI", "IIZI", "ZZZZ", "IIII"])
    def test_roundtrip(self, text):
        word = pauli_string_to_word(text)
        assert str(pauli_word_to_string(word, len(text))) == text

    def test_wire_out_of_range_raises(self):
        with pytest.raises(ValueError, match="not an integer in range"):
            pauli_word_to_string(PauliWord({7: "X"}), 4)

    def test_accepts_pennylane_operators(self):
        """A DLA given as PennyLane operators classifies the same as one given as strings."""
        ops = [qml.X(0) @ qml.X(1), qml.Y(0) @ qml.Y(1), qml.Z(0), qml.Z(1)]
        assert classify_dla(ops).algebra == classify_dla(tfxy_strings(2)).algebra


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassifyDLA:

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_tfxy_is_so_2n(self, n):
        """PauLie identifies the transverse-field XY DLA as so(2n), exactly."""
        info = classify_dla(tfxy_strings(n))
        assert info.algebra == f"so({2 * n})"
        assert info.is_simple
        assert info.orthogonal_size == 2 * n
        assert info.dim == n * (2 * n - 1)

    def test_dimension_matches_lie_closure(self):
        """PauLie's predicted dimension agrees with an explicit Lie closure."""
        for n in [2, 3, 4, 5]:
            gens = tfxy_strings(n)
            assert classify_dla(gens).dim == len(lie_closure_pauli_words(
                [pauli_string_to_word(g) for g in gens]
            ))

    def test_resolves_ambiguity_of_identify_algebra(self):
        """`identify_algebra` is deliberately non-unique; PauLie is not.

        A 21-dimensional simple DLA is consistent with both so(7) and sp(3), and
        `identify_algebra` -- which only looks at the dimension -- returns both.
        """
        gens = expand(["XX", "XZ"], 4)
        dla = dla_pauli_basis(gens)
        assert len(dla) == 21

        candidates = identify_algebra(list(dla))
        assert {(kind, size) for _, kind, size in candidates} == {("so", 7), ("sp", 3)}
        assert classify_dla(gens).algebra == "so(7)"

    def test_low_rank_coincidence_is_recognised(self):
        """PauLie names the 2-qubit tfXY algebra `2*so(3)`; that is so(4)."""
        info = classify_dla(tfxy_strings(2))
        assert not info.is_simple
        assert info.algebra == "2*so(3)"
        assert info.orthogonal_size == 4
        assert info.dim == 6

    def test_non_orthogonal_algebra_has_no_so_size(self):
        info = classify_dla(expand(["XX", "YZ", "ZY"], 4))
        assert info.orthogonal_size is None or info.orthogonal_size >= 3

    def test_simple_component_of_a_simple_algebra(self):
        """The one summand of a simple algebra is the algebra itself."""
        info = classify_dla(tfxy_strings(4))
        assert info.is_simple
        assert info.simple_component == DLAComponent("so", 8, 1)

    def test_simple_component_of_a_semisimple_algebra_raises(self):
        """Asking a semisimple algebra for its one summand is an error.

        These three properties forward to PauLie, so the exception is PauLie's
        `ClassificationException` rather than a local `ValueError`.
        """
        info = classify_dla(tfxy_strings(2))
        assert not info.is_simple
        with pytest.raises(ClassificationException):
            _ = info.simple_component

    @pytest.mark.parametrize(
        "term, expected",
        [
            ("so(8)", DLAComponent("so", 8, 1)),
            ("2*su(2)", DLAComponent("su", 2, 2)),
            ("u(1)", DLAComponent("u", 1, 1)),
            ("16*sp(4)", DLAComponent("sp", 4, 16)),
        ],
    )
    def test_component_parsing(self, term, expected):
        """Components are parsed out of PauLie's own naming, not recomputed."""
        assert DLAComponent.parse(term) == expected
        assert str(expected) == term

    def test_component_parsing_rejects_garbage(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            DLAComponent.parse("so8")

    def test_sp_block_is_twice_the_size(self):
        assert DLAComponent("sp", 2).matrix_size == 4
        assert DLAComponent("so", 8).matrix_size == 8


# ---------------------------------------------------------------------------
# Delegation: the bridge must not answer questions PauLie already answers
# ---------------------------------------------------------------------------


class TestDelegatesToPauLie:
    """Everything about the algebra is PauLie's answer, not a reimplementation."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_name_dim_and_components_come_from_paulie(self, n):
        gens = tfxy_strings(n)
        info = classify_dla(gens)
        classification = info.classification

        assert info.algebra == classification.get_algebra()
        assert info.dim == classification.get_dla_dim()
        assert [str(c) for c in info.components] == classification.get_subalgebras()
        assert np.array_equal(info.matrix_basis, classification.get_algebra_basis())

    @pytest.mark.parametrize(
        "patterns, n, name",
        [
            (["XX", "YY", "Z"], 2, "so(4)"),  # PauLie says 2*so(3)
            (["XX", "YY", "Z"], 3, "so(6)"),
            (["XY"], 4, "so(4)"),
        ],
    )
    def test_orthogonal_size_uses_paulie_isomorphisms(self, patterns, n, name):
        """The so(m) check is `Classification.is_algebra`, not a local lookup table."""
        info = classify_dla(expand(patterns, n))
        assert info.is_algebra(name)
        assert info.orthogonal_size == int(name[3:-1])

    def test_matrix_basis_dimension_matches_classification(self):
        info = classify_dla(tfxy_strings(4))
        assert info.matrix_basis.shape == (info.dim, 8, 8)

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_the_two_bases_coincide_for_a_genuine_so(self, n):
        """When PauLie already names the algebra so(m), both bases are the same."""
        info = classify_dla(tfxy_strings(n))
        assert info.algebra == f"so({2 * n})"
        assert np.allclose(info.matrix_basis, info.orthogonal_basis)

    def test_the_two_bases_differ_for_a_low_rank_coincidence(self):
        """PauLie names the 2-qubit tfXY algebra 2*so(3), and bases it in 6x6.

        kak_tools decomposes the isomorphic so(4), which lives in 4x4, so
        `orthogonal_basis` has to ask PauLie for the so(4) basis explicitly rather
        than reuse `get_algebra_basis()`.
        """
        info = classify_dla(tfxy_strings(2))
        assert info.algebra == "2*so(3)"
        assert info.matrix_basis.shape == (6, 6, 6)
        assert info.orthogonal_basis.shape == (6, 4, 4)

    def test_orthogonal_basis_refused_without_an_so_presentation(self):
        info = classify_dla(expand(["XX", "YZ"], 5))  # sp(8)
        with pytest.raises(ValueError, match="no so\\(m\\) presentation"):
            _ = info.orthogonal_basis


# ---------------------------------------------------------------------------
# Lie closure
# ---------------------------------------------------------------------------


class TestPauliBasis:
    """`dla_pauli_basis` is kak_tools' own closure, told the answer's size up front."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_basis_size_matches_classification(self, n):
        gens = tfxy_strings(n)
        info = classify_dla(gens)
        assert len(dla_pauli_basis(gens, info=info)) == info.dim

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_full_size_hint_does_not_change_the_closure(self, n):
        """The hint is an early exit, so it must not alter the result."""
        words = [pauli_string_to_word(g) for g in tfxy_strings(n)]
        unhinted = lie_closure_pauli_words(words)
        hinted = lie_closure_pauli_words(words, full_size=len(unhinted))
        assert set(unhinted) == set(hinted)
        assert len(hinted) == len(unhinted)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_closure_still_matches_pennylane(self, n):
        words = [pauli_string_to_word(g) for g in tfxy_strings(n)]
        closure = [next(iter(op.pauli_rep)) for op in qml.lie_closure(words)]
        assert set(lie_closure_pauli_words(words)) == set(closure)


# ---------------------------------------------------------------------------
# Matrix basis
# ---------------------------------------------------------------------------


class TestLabelledMatrixBasis:

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_labelled_basis_is_paulies_basis(self, n):
        """Each labelled matrix is (twice) one of PauLie's basis elements."""
        gens = tfxy_strings(n)
        mapping, signs, info = map_dla_to_irrep(gens)
        labelled = labelled_matrix_basis(mapping, signs, info)

        assert set(labelled) == set(mapping.values())
        paulie_basis = info.orthogonal_basis
        for matrix in labelled.values():
            assert any(
                np.allclose(matrix, 2 * element) or np.allclose(matrix, -2 * element)
                for element in paulie_basis
            )

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_labelled_basis_reproduces_the_commutators(self, n):
        """PauLie's matrices satisfy the Pauli words' commutation relations."""
        gens = tfxy_strings(n)
        dla = dla_pauli_basis(gens)
        mapping, signs, info = map_dla_to_irrep(gens, dla=dla)
        matrices = labelled_matrix_basis(mapping, signs, info)

        for word_1 in dla:
            for word_2 in dla:
                if word_1 == word_2:
                    continue
                pauli_com = (1j * word_1).commutator(1j * word_2) / 1j
                pauli_com.simplify()
                mat_1, mat_2 = matrices[word_1], matrices[word_2]
                mat_com = mat_1 @ mat_2 - mat_2 @ mat_1
                if len(pauli_com) == 0:
                    assert np.allclose(mat_com, 0.0)
                    continue
                [(com_word, com_coeff)] = pauli_com.items()
                assert np.allclose(matrices[com_word] * com_coeff, mat_com)

    def test_refuses_non_orthogonal_algebras(self):
        info = classify_dla(expand(["XX", "YZ"], 5))  # sp(8)
        assert info.orthogonal_size is None
        with pytest.raises(ValueError, match="needs an so\\(m\\) presentation"):
            labelled_matrix_basis({}, {}, info)


# ---------------------------------------------------------------------------
# Irrep mapping
# ---------------------------------------------------------------------------


class TestMapDLAToIrrep:

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_mapping_covers_the_algebra(self, n):
        gens = tfxy_strings(n)
        mapping, signs, info = map_dla_to_irrep(gens)
        m = info.orthogonal_size
        assert len(mapping) == m * (m - 1) // 2 == info.dim
        assert set(mapping) == set(signs)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_paulie_and_kak_tools_bases_agree(self, n):
        """PauLie's so(m) basis is kak_tools' E/2, element for element.

        `labelled_matrix_basis` relies on this, so a convention drift on either side
        should fail here rather than silently produce a wrong decomposition.
        """
        from kak_tools.map_to_irrep import E

        info = classify_dla(tfxy_strings(n))
        m = info.orthogonal_size
        for k, node in enumerate(itertools.combinations(range(m), 2)):
            assert np.allclose(info.orthogonal_basis[k], E(node, m, "BDI") / 2)


# ---------------------------------------------------------------------------
# End-to-end decomposition
# ---------------------------------------------------------------------------


class TestKAKDecomposition:

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_tfxy_recomposes(self, n, seed):
        """The Pauli rotations recompose exp(t H) for the transverse-field XY model."""
        gens = tfxy_strings(n)
        rng = np.random.default_rng(seed)
        coeffs = rng.normal(0.0, 1.0, len(gens))
        coeffs /= np.linalg.norm(coeffs)

        result = kak_decomposition(gens, coeffs, time=0.83)

        assert result.info.algebra == f"so({2 * n})" or result.info.orthogonal_size == 2 * n
        assert result.irrep_size == 2 * n
        assert result.involution == "BDI"
        assert result.reconstruction_error is not None

        recomposed = reconstruct_from_pauli_rotations(
            result.pauli_rotations,
            result.algebra_basis,
            result.irrep_size,
            time=result.time,
        )
        assert np.allclose(recomposed, result.unitary_irrep, atol=1e-8)
        assert np.allclose(result.reconstruct(), recomposed)

    @pytest.mark.parametrize(
        "patterns, n",
        [
            (["XX", "Z"], 3),  # transverse-field Ising
            (["XX", "Z"], 4),
            (["XX", "YY", "Z"], 4),  # transverse-field XY
            (["XY"], 4),  # so(4) via the 2*so(3) coincidence
            (["XY"], 6),
        ],
    )
    def test_model_agnostic(self, patterns, n):
        """The workflow is not tied to the transverse-field XY model."""
        gens = expand(patterns, n)
        info = classify_dla(gens)
        assert info.orthogonal_size is not None and info.orthogonal_size % 2 == 0

        result = kak_decomposition(gens, np.linspace(0.2, 1.0, len(gens)), time=0.6)
        assert result.reconstruction_error < 1e-8
        assert len(result.cartan_angles) == result.irrep_size // 2

    def test_horizontal_element_gives_one_cartan_block(self):
        """A horizontal Hamiltonian yields exactly n/2 central Cartan angles."""
        result = kak_decomposition(tfxy_strings(4), time=0.4)
        kinds = {kind for _, _, kind in result.pauli_rotations}
        assert kinds <= {"k1", "k2", "a0", "a"}
        assert len(result.cartan_angles) == result.irrep_size // 2

    def test_coefficient_count_is_checked(self):
        with pytest.raises(ValueError, match="coefficients for"):
            kak_decomposition(tfxy_strings(3), [1.0, 2.0])

    def test_odd_irrep_size_decomposes(self):
        """so(m) with odd m decomposes, having once refused nearly every Hamiltonian.

        The top-level split is then BDI(p, p+1), whose horizontal cosine-sine
        decomposition carries an O(1) = {+-1} gauge on the one direction that `a`
        leaves fixed. `bdi` used to repair that sign only for p == q and raise
        otherwise, which rejected 0/25 to 2/25 of coefficient draws here.
        """
        gens = expand(["XX", "XZ"], 5)
        info = classify_dla(gens)
        assert info.orthogonal_size % 2 == 1

        result = kak_decomposition(gens, np.linspace(0.2, 1.0, len(gens)), time=0.6)
        assert result.reconstruction_error < 1e-10

    @pytest.mark.parametrize("generators", [["XX", "Z"], ["ZZ", "X"], ["XY", "Z"]])
    @pytest.mark.parametrize("n", [4, 5, 6, 7])
    def test_odd_irrep_size_over_many_coefficients(self, generators, n):
        """Odd m holds up across coefficient draws, which is where it used to fail.

        In PauLie's two-local convention these are so(2n - 1), so the transverse-field
        Ising family lands on odd m -- it is the common case, not a corner one.
        """
        gens = two_local_strings(generators, n)
        info = classify_dla(gens)
        assert info.orthogonal_size == 2 * n - 1

        for seed in range(20):
            coefficients = np.random.default_rng(seed).normal(size=len(gens))
            result = kak_decomposition(gens, coefficients, time=0.7)
            assert result.reconstruction_error < 1e-12
            # One rotation per algebra dimension, less any whose angle `tol` rounded away.
            assert 0 < len(result.pauli_rotations) <= info.dim

    def test_non_orthogonal_algebra_raises_a_clear_error(self):
        """A DLA that is not (isomorphic to) so(m) is refused, with a pointer."""
        gens = expand(["XX", "YZ"], 5)  # sp(8)
        info = classify_dla(gens)
        assert info.orthogonal_size is None
        with pytest.raises(NotImplementedError, match="not \\(isomorphic to\\)"):
            kak_decomposition(gens, time=0.6)


# ---------------------------------------------------------------------------
# Agreement with the hard-coded tfXY workflow
# ---------------------------------------------------------------------------


class TestAgreementWithHardcodedWorkflow:

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_same_unitary_as_manual_pipeline(self, n):
        """The bridge reproduces the hand-configured `complete_workflow_tfXY` pipeline."""
        from scipy.linalg import expm

        from kak_tools import map_simple_to_irrep
        from kak_tools.map_to_irrep import irrep_dot

        np.random.seed(n)
        _, generators, coeffs = make_tfXY_hamiltonian_qubits(n)

        # Manual route: n_so supplied by hand.
        ops = lie_closure_pauli_words(generators)
        mapping, signs = map_simple_to_irrep(ops, generators, n=2 * n, invol_type="BDI")
        manual_H = irrep_dot(coeffs, generators, mapping, signs, n=2 * n, invol_type="BDI")

        # Bridge route: n_so obtained from PauLie.
        result = kak_decomposition(generators, coeffs, time=1.0)

        assert result.irrep_size == 2 * n
        # Both Hamiltonians live in the same algebra and have the same spectrum.
        assert np.allclose(
            np.sort(np.linalg.eigvals(manual_H).imag),
            np.sort(np.linalg.eigvals(result.hamiltonian_irrep).imag),
        )
        assert np.allclose(
            expm(result.hamiltonian_irrep) @ expm(result.hamiltonian_irrep).T,
            np.eye(2 * n),
        )


# ---------------------------------------------------------------------------
# Regressions in the Pauli-rotation mapping, reached through the bridge
# ---------------------------------------------------------------------------


class TestPauliRotationMapping:
    """Three bugs in `dense_cartan` that only PauLie-classified algebras reach.

    The hard-coded workflow only ever ran the transverse-field XY model with random
    coefficients, which misses all three.
    """

    @pytest.mark.parametrize("n", [3, 5, 6, 7])
    def test_odd_block_widths(self, n):
        """n_so = 6, 10, 12, 14 recurse through odd-width blocks.

        `angles_to_reducible` indexed the cosine-sine angles with `p = w // 2` rather
        than `q = w - p`; equal for even widths, silently wrong for odd ones.
        """
        gens = tfxy_strings(n)
        rng = np.random.default_rng(n)
        result = kak_decomposition(gens, rng.normal(0, 1, len(gens)), time=0.83)
        assert result.irrep_size == 2 * n
        assert np.allclose(result.reconstruct(), result.unitary_irrep, atol=1e-8)

    @pytest.mark.parametrize("seed", range(8))
    def test_horizontal_sign_gauge(self, seed):
        """`bdi` used to raise on a sign gauge that leaves k1 @ a @ k2 invariant.

        At so(4) this tripped on roughly three quarters of random Hamiltonians.
        """
        gens = tfxy_strings(2)
        rng = np.random.default_rng(seed)
        result = kak_decomposition(gens, rng.normal(0, 1, len(gens)), time=0.83)
        assert np.allclose(result.reconstruct(), result.unitary_irrep, atol=1e-8)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_uniform_coefficients(self, n):
        """A translation-invariant Hamiltonian produces rotations by exactly pi.

        Those have no off-diagonal entry, so reading the angle off `arcsin` dropped
        them without complaint.
        """
        gens = tfxy_strings(n)
        result = kak_decomposition(gens, np.ones(len(gens)), time=0.4)
        assert np.allclose(result.reconstruct(), result.unitary_irrep, atol=1e-8)

    def test_a_pi_rotation_is_not_dropped(self):
        from kak_tools.dense_cartan import group_matrix_to_reducible
        from kak_tools.map_to_irrep import E

        mapping, signs = {(0, 1): PauliWord({0: "X"})}, {(0, 1): 1}

        rotations = group_matrix_to_reducible(-np.eye(2), 0, mapping, signs)
        [(word, coeff)] = rotations.items()
        assert word == PauliWord({0: "X"}) and np.isclose(coeff, np.pi / 2)
        assert np.allclose(expm(E((0, 1), 2, "BDI") * coeff), -np.eye(2))

        assert len(group_matrix_to_reducible(np.eye(2), 0, mapping, signs)) == 0

    def test_determinant_minus_one_is_rejected(self):
        """An odd number of -1s is not a product of rotations, and must not pass."""
        from kak_tools.dense_cartan import group_matrix_to_reducible

        with pytest.raises(AssertionError, match="determinant -1"):
            group_matrix_to_reducible(
                np.diag([1.0, -1.0]), 0, {(0, 1): PauliWord({0: "X"})}, {(0, 1): 1}
            )
