"""Bridge from PauLie classification to Pauli-word BDI decompositions.

The pipeline classifies the DLA, independently closes its Pauli words, maps them
onto so(m) rotation planes, and factors the horizontal Hamiltonian. Even and odd
m, explicit BDI(p, q) partitions, and PauLie's low-rank so(m) isomorphisms are
supported. Without an explicit partition the balanced one is tried first and the
unbalanced ones afterwards, since some generator sets are horizontal only there.

Cartan rates are independent of time, including zero time and resonances.
Other classical types are available through the matrix-level routines in
:mod:`kak_tools.numerical_decompositions`. PauLie requires Python >= 3.12.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
from paulie.classifier.classification import Classification
from pennylane.pauli import PauliWord
from scipy.linalg import expm

from ._horizontal_bdi import decompose_horizontal_hamiltonian
from ._pauli_inputs import (
    as_pauli_collection, as_pauli_words, pauli_string_to_word,
    pauli_word_to_string, prepare_pauli_inputs, words_to_collection,
)
from ._pauli_rotations import PauliRotation, reconstruct_from_pauli_rotations
from ._validation import finite_real_scalar, nonnegative_tolerance, require, resolve_bdi_partition
from .map_to_irrep import (
    HorizontalEmbeddingError, _validate_so_mapping, map_irrep_to_matrices, map_simple_to_irrep,
)

__all__ = [
    "KAKResult",
    "PauliRotation",
    "as_pauli_collection",
    "as_pauli_words",
    "dla_pauli_basis",
    "kak_decomposition",
    "labelled_matrix_basis",
    "map_dla_to_irrep",
    "pauli_string_to_word",
    "pauli_word_to_string",
    "reconstruct_from_pauli_rotations",
]


def dla_pauli_basis(generators, n_qubits: int | None = None, strict: bool = True) -> list[PauliWord]:
    """Compute the complete native closure and convert it to PennyLane words.

    Classification comes directly from PauLie on normalized generators. Its
    dimension is checked after closure; it never stops discovery early.
    ``strict=False`` warns instead of raising for a dimension mismatch.
    """
    collection = as_pauli_collection(generators, n_qubits)
    return _native_pauli_basis(collection, collection.get_class(), strict=strict)


def _native_pauli_basis(collection, classification, strict=True):
    """Close under the original generators' adjoints, then check the dimension.

    By the Jacobi identity, this invariant span is the generated Lie algebra.
    Iterating the growing list visits every new word exactly once.
    """
    generators = list(collection.get())
    native_basis = generators.copy()
    seen = set(native_basis)
    for word in native_basis:
        for generator in generators:
            commutator = generator.adjoint_map(word)
            if commutator is not None and commutator not in seen:
                seen.add(commutator)
                native_basis.append(commutator)
    target = classification.get_dla_dim()
    if len(native_basis) != target:
        message = (
            f"The Lie closure produced {len(native_basis)} Pauli words but PauLie "
            f"classified the DLA as {classification.get_algebra()} of dimension {target}."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)
    return [pauli_string_to_word(word) for word in native_basis]


def labelled_matrix_basis(mapping, signs, classification: Classification, n_qubits: int | None = None) -> dict:
    """Map Pauli words to their signed so(m) generator matrices.

    ``mapping`` and ``signs`` come from :func:`map_dla_to_irrep`. Each word at
    plane ``(i, j)`` gets ``2 * sign * (E_ij - E_ji)``, kak_tools' generator
    normalization. The complete Lie-map bijection and the wire labels are
    checked; ``n_qubits`` also bounds the wires. Invalid conventions raise ValueError.
    """
    m = classification.get_orthogonal_size()
    if m is None:
        raise ValueError(
            f"labelled_matrix_basis needs an so(m) presentation; PauLie classified this "
            f"DLA as {classification.get_algebra()}."
        )
    signs = _validate_so_mapping(mapping, signs, m)
    as_pauli_words(mapping.values(), n_qubits=n_qubits)
    return map_irrep_to_matrices(mapping, signs, m, "BDI")


def _orthogonal_size(basis, classification: Classification) -> int:
    """Return the m with m(m-1)/2 Pauli words in the closure, rejecting other algebras."""
    size = classification.get_orthogonal_size()
    if size is None:
        raise NotImplementedError(
            f"PauLie classified this DLA as {classification.get_algebra()} with summands "
            f"{classification.get_subalgebras()}, which is not (isomorphic to) a single so(m); "
            "kak_tools can only build a Pauli-word irrep mapping for so(m) with a BDI "
            "involution. Compute the closure with `kak_tools.dla_pauli_basis(generators)`, "
            "split it with `kak_tools.split_pauli_algebra`, and choose horizontal generators "
            "within each component, or use the matrix-level routines in "
            "`kak_tools.numerical_decompositions` directly."
        )
    # The closure already matched PauLie's dimension; the rotation planes must
    # match it too before words are paired with them.
    require(
        size * (size - 1) // 2 == len(basis),
        f"so({size}) has {size * (size - 1) // 2} rotation planes but the Lie closure "
        f"has {len(basis)} Pauli words.",
    )
    return size


def _irrep_mapping(words, n_qubits, invol_kwargs):
    """Classify, close and map normalized words onto signed BDI rotation planes.

    Returns ``(classification, irrep_size, partition, mapping, signs)``. The
    closure is verified against PauLie's dimension before the so(m) size is read
    off it, so a misclassified dimension is reported as such. Without
    ``invol_kwargs`` the balanced partition is tried first (it is
    parameter-optimal) and every other ``p < m // 2`` afterwards, since a
    generator set may be horizontal only for an unbalanced BDI(p, q);
    complementary partitions are equivalent.
    """
    collection = words_to_collection(words, n_qubits)
    classification = collection.get_class()
    basis = _native_pauli_basis(collection, classification)
    irrep_size = _orthogonal_size(basis, classification)
    if invol_kwargs is None:
        half = irrep_size // 2
        partitions = [(p, irrep_size - p) for p in [half, *range(1, half)]]
    else:
        partitions = [resolve_bdi_partition(irrep_size, invol_kwargs)]

    for p, q in partitions:
        try:
            mapping, signs = map_simple_to_irrep(
                basis, horizontal_ops=words, n=irrep_size,
                invol_type="BDI", invol_kwargs={"p": p, "q": q},
            )
        except HorizontalEmbeddingError as exc:
            failure = exc
            continue
        return classification, irrep_size, (p, q), mapping, signs

    if invol_kwargs is None:
        message = (
            f"The Pauli generators cannot all be embedded in the horizontal subspace of "
            f"BDI(p, q) for any partition p + q = {irrep_size}: the DLA's Pauli words admit "
            f"no so({irrep_size}) rotation-plane presentation with these generators "
            "horizontal. Supply a compatible generator set."
        )
    else:
        message = (
            f"The Pauli generators cannot all be embedded in the horizontal subspace of "
            f"BDI({p}, {q}). Supply a compatible generator set or partition, or omit "
            "invol_kwargs to search the partitions."
        )
    raise ValueError(message) from failure


def map_dla_to_irrep(generators, n_qubits: int | None = None, invol_kwargs: dict | None = None):
    """Return ``(mapping, signs, classification)`` using PauLie's native result.

    The generators must be horizontal for the BDI partition, which is searched
    when ``invol_kwargs`` is omitted (see :func:`kak_decomposition`). Register
    width is inferred from the generators unless ``n_qubits`` is supplied.
    """
    words, _, n_qubits = prepare_pauli_inputs(generators, n_qubits=n_qubits)
    classification, _, _, mapping, signs = _irrep_mapping(words, n_qubits, invol_kwargs)
    return mapping, signs, classification


@dataclass
class KAKResult:
    """Compiled Pauli-word KAK decomposition and its defining irrep.

    ``pauli_rotations`` lists the factors of the left-to-right matrix product
    ``U = R_1 R_2 ... R_N = K1 A K2`` with ``R_k = exp(i * c_k * P_k)``; the
    physical convention is ``exp(+i * time * H)``. The ``PauliRotation(word,
    coefficient, kind)`` records of kind ``k1``/``k2`` are vertical rotations and
    those of kind ``a0`` carry the time-independent Cartan rates, the only
    coefficients to multiply by the evolution time. A circuit applies the gates
    in reversed list order, each as ``qml.PauliRot(-2 * c_k, P_k)``; see
    :meth:`pennylane_ops`.

    ``classification`` is PauLie's native Classification; ``n_qubits`` records
    the physical register independently of its irreducible representation and
    ``partition`` the BDI(p, q) sizes used. ``mapping`` and ``signs`` relate irrep
    planes to Pauli words; ``algebra_basis`` contains their signed matrices.
    ``hamiltonian_irrep`` is H in that basis and ``unitary_irrep = expm(time * H)``.
    ``reconstruction_error`` is the time-independent Lie-algebra error
    ``max|K1 A K1^T - H|`` of the compiled rotations, or None when validation
    was disabled.
    """

    classification: Classification
    n_qubits: int
    irrep_size: int
    partition: tuple[int, int]
    pauli_rotations: list[PauliRotation]
    time: float
    hamiltonian_irrep: np.ndarray
    unitary_irrep: np.ndarray
    mapping: dict = field(repr=False, default_factory=dict)
    signs: dict = field(repr=False, default_factory=dict)
    algebra_basis: dict = field(repr=False, default_factory=dict)
    reconstruction_error: float | None = None

    @property
    def cartan_angles(self) -> list:
        """list: Central ``(PauliWord, rate)`` pairs; multiply rates by evolution time."""
        return [(pw, angle) for pw, angle, kind in self.pauli_rotations if kind == "a0"]

    def reconstruct(self, time: float | None = None) -> np.ndarray:
        """Recompose the evolution, optionally reusing this compilation at a new time."""
        return reconstruct_from_pauli_rotations(
            self.pauli_rotations, self.algebra_basis, self.irrep_size,
            time=self.time if time is None else time,
        )

    def pennylane_ops(self, time: float | None = None) -> list:
        """PennyLane gates whose circuit is ``exp(i * time * H)`` on the physical register.

        The rotations are emitted in reversed list order as ``PauliRot(-2 * c)``,
        with ``a0`` rates multiplied by ``time`` (default: the compile time). A
        rotation with the identity word becomes a ``GlobalPhase``.
        """
        time = self.time if time is None else finite_real_scalar(time, "time")
        ops = []
        for word, coefficient, kind in reversed(self.pauli_rotations):
            angle = coefficient * (time if kind == "a0" else 1.0)
            if len(word) == 0:
                ops.append(qml.GlobalPhase(-angle))
            else:
                ops.append(qml.PauliRot(-2 * angle, "".join(word[w] for w in word.wires), wires=word.wires))
        return ops

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        err = "not validated" if self.reconstruction_error is None else f"{self.reconstruction_error:.2e}"
        return (
            f"KAKResult(algebra={self.classification.get_algebra()}, "
            f"BDI{self.partition}, n={self.irrep_size}, rotations={len(self.pauli_rotations)}, "
            f"reconstruction_error={err})"
        )


def _validate_rotations(pauli_rotations, algebra_basis, irrep_size, hamiltonian, unitary, time, atol):
    """Check ``K1 A K1^T == H`` and then the recomposed ``exp(time * H)``; return the first error.

    The ``k2`` sequence inverts ``k1`` by construction, so a group-level
    comparison alone is vacuous at zero time and blind to rate errors that are
    multiples of ``2 pi / time``. Both comparisons carry rounding of order
    ``eps`` times the magnitude of what is exponentiated, hence the scaled
    tolerances: a fixed one rejects correct compilations once ``|time * H|``
    reaches about ``atol / eps``.
    """
    k1 = reconstruct_from_pauli_rotations(
        [rotation for rotation in pauli_rotations if rotation[2] == "k1"], algebra_basis, irrep_size
    )
    cartan = np.zeros((irrep_size, irrep_size))
    for word, rate, kind in pauli_rotations:
        if kind == "a0":
            cartan = cartan + rate * algebra_basis[word]
    scale = max(1.0, float(np.abs(hamiltonian).max()))
    error = float(np.abs(k1 @ cartan @ k1.T - hamiltonian).max())
    require(
        np.isfinite(error) and error <= atol * scale,
        f"The Pauli rotations do not recompose H: max |K1 A K1^T - H| = {error:.3e} > {atol * scale:.1e}.",
    )

    recomposed = reconstruct_from_pauli_rotations(pauli_rotations, algebra_basis, irrep_size, time=time)
    group_error = float(np.abs(recomposed - unitary).max())
    group_atol = atol * max(1.0, abs(time) * scale)
    require(
        np.isfinite(group_error) and group_error <= group_atol,
        f"The Pauli rotations do not recompose exp({time} * H): max error "
        f"{group_error:.3e} > {group_atol:.1e}.",
    )
    return error


def kak_decomposition(
    generators,
    coefficients=None,
    time: float = 1.0,
    n_qubits: int | None = None,
    invol_kwargs: dict | None = None,
    validate: bool = True,
    atol: float | None = None,
    tol: float | None = None,
) -> KAKResult:
    """Compile ``exp(time * H)`` into Pauli rotations using PauLie's classification.

    Generators accepted by :func:`as_pauli_words` must be individual Pauli terms;
    sums are rejected. Every listed generator belongs to the generator family,
    whatever its weight: intrinsic PennyLane factors and the external
    ``coefficients`` (one finite real per input term, default one) only shape
    ``H = sum(factor * coefficient * P)``, with duplicate words summed. A zero
    weight therefore keeps its term in the family, so ``[X, Y, 0 * Z]`` and
    ``([X, Y, Z], [1, 1, 0])`` both compile within the algebra generated by X, Y
    and Z. ``n_qubits`` is inferred when omitted; an Identity factor on a term
    also widens the register.

    Only BDI on an so(m) presentation is supported. ``invol_kwargs`` fixes the
    p/q partition; otherwise the balanced partition is tried first and the other
    ``p < m // 2`` afterwards, and the result records the ``partition`` used. The
    horizontal Hamiltonian determines time-independent Cartan rates, reusable via
    the result's ``reconstruct(time=...)`` and ``pennylane_ops(time=...)``.

    ``validate`` checks the compiled rotations at the Lie-algebra level,
    ``max|K1 A K1^T - H| <= atol * max(1, |H|)``, which is independent of time
    and hence meaningful at zero time and at resonances, and additionally the
    recomposed group element against ``expm(time * H)`` with the tolerance scaled
    by ``max(1, |time * H|)``. ``atol`` defaults to ``1e-10 * irrep_size**2`` for
    accumulated Givens error. ``tol`` optionally drops small vertical angles, an
    approximation that validation measures; central rates are always retained so
    longer evolution times remain valid. Both tolerances must be nonnegative and
    finite when supplied.

    Raises ValueError for invalid inputs or failed validation, and
    NotImplementedError for unsupported algebras.
    """
    time = finite_real_scalar(time, "time")
    tol = nonnegative_tolerance(tol, "tol")
    atol = nonnegative_tolerance(atol, "atol")
    words, coefficients, n_qubits = prepare_pauli_inputs(generators, coefficients, n_qubits)
    classification, irrep_size, partition, mapping, signs = _irrep_mapping(words, n_qubits, invol_kwargs)

    algebra_basis = labelled_matrix_basis(mapping, signs, classification, n_qubits=n_qubits)
    hamiltonian = np.zeros((irrep_size, irrep_size))
    for coeff, word in zip(coefficients, words, strict=True):
        hamiltonian = hamiltonian + coeff * algebra_basis[word]

    pauli_rotations = decompose_horizontal_hamiltonian(
        hamiltonian, partition[0], mapping, signs, time=time, tol=tol
    )
    unitary = expm(time * hamiltonian)

    if atol is None:
        # Givens factorization uses O(irrep_size**2) rotations; allow for their
        # accumulated numerical error rather than expecting machine precision.
        atol = 1e-10 * irrep_size**2

    error = None
    if validate:
        error = _validate_rotations(
            pauli_rotations, algebra_basis, irrep_size, hamiltonian, unitary, time, atol
        )

    return KAKResult(
        classification=classification,
        n_qubits=n_qubits,
        irrep_size=irrep_size,
        partition=partition,
        pauli_rotations=pauli_rotations,
        time=time,
        hamiltonian_irrep=hamiltonian,
        unitary_irrep=unitary,
        mapping=mapping,
        signs=signs,
        algebra_basis=algebra_basis,
        reconstruction_error=error,
    )
