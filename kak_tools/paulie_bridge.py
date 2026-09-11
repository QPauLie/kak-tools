"""Bridge from PauLie classification to Pauli-word BDI decompositions.

The pipeline classifies the DLA, independently closes its Pauli words, maps them
onto so(m) rotation planes, and factors the horizontal Hamiltonian. Even and odd
m, explicit BDI(p, q) partitions, and PauLie's low-rank so(m) isomorphisms are
supported. The default balanced partition is not searched automatically: the
Pauli basis must admit a rotation-plane bijection with all generators horizontal.

Cartan rates are independent of time, including zero time and resonances.
Other classical types are available through the matrix-level routines in
:mod:`kak_tools.numerical_decompositions`. PauLie requires Python >= 3.12.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from paulie.classifier.classification import Classification
from paulie.common.algebra_basis import get_so_basis
from pennylane.pauli import PauliWord
from scipy.linalg import expm

from ._horizontal_bdi import decompose_horizontal_hamiltonian
from ._pauli_inputs import (
    as_pauli_collection, as_pauli_words, pauli_string_to_word,
    pauli_word_to_string, prepare_pauli_inputs,
)
from ._validation import finite_real_scalar, nonnegative_tolerance, resolve_bdi_partition
from ._pauli_rotations import reconstruct_from_pauli_rotations as _reconstruct_rotations
from .map_to_irrep import _validate_so_mapping, map_simple_to_irrep

__all__ = [
    "DLAComponent",
    "DLAInfo",
    "KAKResult",
    "as_pauli_collection",
    "as_pauli_words",
    "classify_dla",
    "dla_pauli_basis",
    "kak_decomposition",
    "labelled_matrix_basis",
    "map_dla_to_irrep",
    "pauli_string_to_word",
    "pauli_word_to_string",
    "reconstruct_from_pauli_rotations",
]


_COMPONENT_RE = re.compile(r"^(?:(?P<multiplicity>\d+)\*)?(?P<type>[a-z]+)\((?P<size>\d+)\)$")


@dataclass(frozen=True)
class DLAComponent:
    """One ``multiplicity * type(size)`` summand reported by PauLie.

    ``size`` is n in so(n), su(n), sp(n), or u(n), before matrix-size conversion.
    """

    type: str
    size: int
    multiplicity: int = 1

    @classmethod
    def parse(cls, term: str) -> DLAComponent:
        """Parse PauLie's ``[k*]name(size)`` format; raise ValueError for invalid terms."""
        match = _COMPONENT_RE.match(term.replace(" ", ""))
        if match is None:
            raise ValueError(f"Cannot parse {term!r} as a PauLie algebra component.")
        return cls(
            type=match["type"],
            size=int(match["size"]),
            multiplicity=int(match["multiplicity"] or 1),
        )

    @property
    def matrix_size(self) -> int:
        """Size of one defining-representation block; sp(n) uses 2n matrices."""
        return 2 * self.size if self.type == "sp" else self.size

    def __str__(self) -> str:
        core = f"{self.type}({self.size})"
        return core if self.multiplicity == 1 else f"{self.multiplicity}*{core}"


@dataclass(frozen=True)
class DLAInfo:
    """PauLie classification and qubit count, adapted to kak_tools' irrep conventions."""

    classification: Classification
    n_qubits: int

    @property
    def algebra(self) -> str:
        """str: PauLie's name for the algebra, e.g. ``"so(8)"`` or ``"u(1)+2*su(2)"``."""
        return str(self.classification.get_algebra())

    @property
    def dim(self) -> int:
        """int: Dimension of the full dynamical Lie algebra."""
        return int(self.classification.get_dla_dim())

    @property
    def components(self) -> tuple[DLAComponent, ...]:
        """Summands of PauLie's classified algebra."""
        return tuple(
            DLAComponent.parse(term) for term in self.classification.get_subalgebras()
        )

    @property
    def matrix_basis(self) -> np.ndarray:
        """PauLie's basis in its classified presentation.

        For example, ``2*so(3)`` yields shape (6, 6, 6). Use :attr:`orthogonal_basis`
        for the isomorphic so(4) presentation with shape (6, 4, 4).
        """
        return self.classification.get_algebra_basis()

    @property
    def orthogonal_basis(self) -> np.ndarray:
        """PauLie's so(m) basis, ordered by ``combinations(range(m), 2)``.

        Uses :attr:`orthogonal_size` and raises ValueError without an so(m) presentation.
        """
        m = self.orthogonal_size
        if m is None:
            raise ValueError(
                f"PauLie classified this DLA as {self.algebra}, which has no so(m) "
                "presentation, so there is no orthogonal basis to build."
            )
        return get_so_basis(m)

    def is_algebra(self, algebra: str) -> bool:
        """Test equality up to PauLie's low-rank algebra isomorphisms."""
        return bool(self.classification.is_algebra(algebra))

    @property
    def is_simple(self) -> bool:
        """bool: Whether the algebra is a single simple factor."""
        return bool(self.classification.is_simple())

    @property
    def simple_component(self) -> DLAComponent:
        """Unique simple summand; PauLie raises ClassificationException if not simple."""
        return DLAComponent.parse(self.classification.get_simple_component())

    @property
    def orthogonal_size(self) -> int | None:
        """The m of an isomorphic so(m) algebra, or None.

        PauLie also resolves low-rank coincidences, such as 2*so(3) = so(4).
        """
        return self.classification.get_orthogonal_size()

    def __str__(self) -> str:
        return self.algebra


def classify_dla(generators, n_qubits: int | None = None) -> DLAInfo:
    """Classify generators accepted by :func:`as_pauli_words` through PauLie.

    The register size is inferred unless ``n_qubits`` is supplied. Classification
    uses the generators, not just the dimension of their algebra.
    """
    inputs = prepare_pauli_inputs(generators, n_qubits)
    return _classification_for(inputs, inputs.collection())


def dla_pauli_basis(
    generators,
    info: DLAInfo | None = None,
    n_qubits: int | None = None,
    strict: bool = True,
) -> list[PauliWord]:
    """Return the complete Pauli-word Lie closure, checked against PauLie.

    Native ``PauliString.adjoint_map`` operations close the original generators;
    only the final basis is converted to PennyLane PauliWords. Classification checks
    the result's dimension and never stops the closure early.

    ``info`` supplies metadata to verify against a fresh classification and a default
    register size. With ``strict=False``, inconsistent metadata or closure dimensions
    warn instead of raising. Explicit register conflicts always raise ValueError.
    Use ``info.matrix_basis`` for a defining-representation matrix basis.
    """
    inputs = prepare_pauli_inputs(
        generators, n_qubits if n_qubits is not None else (info.n_qubits if info else None)
    )
    collection = inputs.collection()
    verified_info = _classification_for(inputs, collection, info, strict=strict)
    return _native_pauli_basis(collection, verified_info, strict=strict)


def _classification_for(inputs, collection, cached=None, strict=True):
    """Classify this call's native collection, then check any supplied metadata."""
    if cached is not None and cached.n_qubits != inputs.n_qubits:
        raise ValueError(
            f"n_qubits={inputs.n_qubits} conflicts with cached DLAInfo on "
            f"{cached.n_qubits} qubits. Reclassify for the requested register."
        )
    actual = DLAInfo(collection.get_class(), inputs.n_qubits)
    if cached is not None and (
        actual.dim != cached.dim or not actual.is_algebra(cached.algebra)
    ):
        message = (
            f"Cached DLAInfo describes {cached.algebra} (dimension {cached.dim}), "
            f"but these generators produce {actual.algebra} (dimension {actual.dim})."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message + " Computing the complete basis of the actual generators.",
                      UserWarning, stacklevel=3)
    return actual


def _native_pauli_basis(collection, info, strict=True):
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
    target = info.dim
    if len(native_basis) != target:
        message = (
            f"The Lie closure produced {len(native_basis)} Pauli words but PauLie "
            f"classified the DLA as {info.algebra} of dimension {target}."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)
    return [pauli_string_to_word(word) for word in native_basis]


def labelled_matrix_basis(mapping, signs, info: DLAInfo, validate: bool = True) -> dict:
    """Map Pauli words to signed matrices in PauLie's so(m) basis.

    ``mapping`` and ``signs`` come from :func:`map_dla_to_irrep`. Each word gets
    ``2 * sign * basis[k]``, matching kak_tools' generator normalization.
    With ``validate=True``, check the complete Lie-map bijection, register,
    basis shape and rotation-plane ordering. Invalid conventions raise ValueError.
    """
    m = info.orthogonal_size
    if m is None:
        raise ValueError(
            f"labelled_matrix_basis needs an so(m) presentation; PauLie classified this "
            f"DLA as {info.algebra}."
        )
    # Match get_so_basis's upper-triangle ordering.
    index = {node: k for k, node in enumerate(zip(*np.triu_indices(m, k=1)))}

    if validate:
        signs = _validate_so_mapping(mapping, signs, m)
        as_pauli_words(mapping.values(), n_qubits=info.n_qubits)

    basis = info.orthogonal_basis
    if validate:
        if basis.shape != (len(index), m, m):
            raise ValueError(
                f"PauLie's basis has shape {basis.shape}, expected {(len(index), m, m)}."
            )
        for node, k in index.items():
            if not np.isclose(basis[k][node], 1.0):
                raise ValueError(
                    f"PauLie's so({m}) basis element {k} does not generate the rotation "
                    f"in plane {node}; the basis orderings have drifted apart."
                )

    return {
        word: 2.0 * signs[node] * basis[index[node]].real
        for node, word in mapping.items()
    }

#: The only involution kak_tools can build a Pauli-word irrep mapping for.
PAULI_LEVEL_INVOLUTION = "BDI"


def _resolve_orthogonal(info: DLAInfo, involution: str | None) -> tuple[int, str]:
    """Resolve the so(m) size and BDI involution, rejecting unsupported algebras."""
    involution = involution or PAULI_LEVEL_INVOLUTION
    if involution != PAULI_LEVEL_INVOLUTION:
        raise NotImplementedError(
            f"kak_tools can only build a Pauli-word irrep mapping for the "
            f"{PAULI_LEVEL_INVOLUTION} involution, not {involution!r}."
        )

    size = info.orthogonal_size
    if size is None:
        raise NotImplementedError(
            f"PauLie classified this DLA as {info.algebra}, which is not (isomorphic to) "
            "a single so(m). kak_tools can only build a Pauli-word irrep mapping for "
            "so(m) with a BDI involution. Split the algebra into its components with "
            "`kak_tools.split_pauli_algebra` and decompose them separately, or use the "
            "matrix-level routines in `kak_tools.numerical_decompositions` directly."
        )
    return size, involution


def map_dla_to_irrep(
    generators,
    dla: Sequence[PauliWord] | None = None,
    info: DLAInfo | None = None,
    n_qubits: int | None = None,
    involution: str | None = None,
    invol_kwargs: dict | None = None,
):
    """Return ``(mapping, signs, info)`` with the irrep size supplied by PauLie.

    Generators accepted by :func:`as_pauli_words` must be horizontal. ``dla`` may
    supply the complete distinct PauliWord basis; otherwise its closure is computed.
    ``info`` is checked against a fresh classification and supplies a default register
    size when ``n_qubits`` is omitted. The default involution is BDI; ``invol_kwargs``
    can specify its p/q partition, for example ``{"p": 3}``.

    The mapping sends irrep index pairs to Pauli words, accompanied by their signs.
    """
    inputs = prepare_pauli_inputs(
        generators, n_qubits if n_qubits is not None else (info.n_qubits if info else None)
    )
    collection = inputs.collection()
    info = _classification_for(inputs, collection, info)
    irrep_size, involution = _resolve_orthogonal(info, involution)
    p, q = resolve_bdi_partition(irrep_size, invol_kwargs)
    if dla is None:
        dla = _native_pauli_basis(collection, info)
    else:
        dla = list(dla)
        if (len(dla) != info.dim or not all(isinstance(word, PauliWord) for word in dla)
                or len(set(dla)) != len(dla)):
            raise ValueError(f"The supplied DLA basis must contain {info.dim} distinct PauliWords.")
        # Apply the same register checks to supplied basis elements as to generators.
        as_pauli_words(dla, inputs.n_qubits)
        if not set(inputs.words).issubset(dla):
            raise ValueError("The supplied DLA basis does not contain all generators.")
    mapping, signs = _map_verified_basis(inputs.words, dla, irrep_size, involution, p, q)
    return mapping, signs, info


def _map_verified_basis(words, basis, irrep_size, involution, p, q):
    """Map already normalized data without re-entering public conversion APIs."""
    try:
        mapping, signs = map_simple_to_irrep(
            list(basis), horizontal_ops=words, n=irrep_size,
            invol_type=involution, invol_kwargs={"p": p, "q": q},
        )
    except StopIteration as exc:
        raise ValueError(
            f"The Pauli generators cannot all be embedded in the horizontal "
            f"subspace of BDI({p}, {q}). Supply a compatible generator set or partition."
        ) from exc
    if set(mapping.values()) != set(basis):
        raise ValueError("The supplied DLA basis does not match the completed irrep mapping.")
    return mapping, signs


@dataclass
class KAKResult:
    """Compiled Pauli-word KAK decomposition and its defining irrep.

    ``pauli_rotations`` contains (PauliWord, angle, kind) triples: ``k1``/``k2``
    are vertical rotations; ``a`` and ``a0`` are Cartan factors. Only ``a0`` rates
    are scaled by evolution time.
    ``matrix_factors`` contains (matrix, start, end, kind) group factors at ``time``;
    matrices embed in [start:end, start:end], with a full central matrix.

    ``mapping`` and ``signs`` relate irrep planes to Pauli words; ``algebra_basis``
    contains their signed matrices. ``unitary_irrep = expm(time * hamiltonian_irrep)``.
    ``reconstruction_error`` is the max-abs Pauli-rotation reconstruction error,
    or None when validation was disabled.
    """

    info: DLAInfo
    involution: str
    irrep_size: int
    pauli_rotations: list
    time: float
    hamiltonian_irrep: np.ndarray
    unitary_irrep: np.ndarray
    matrix_factors: list = field(repr=False, default_factory=list)
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

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        err = "not validated" if self.reconstruction_error is None else f"{self.reconstruction_error:.2e}"
        return (
            f"KAKResult(algebra={self.info.algebra}, involution={self.involution}, "
            f"n={self.irrep_size}, rotations={len(self.pauli_rotations)}, "
            f"reconstruction_error={err})"
        )


def reconstruct_from_pauli_rotations(
    pauli_rotations, algebra_basis, irrep_size, time=None
) -> np.ndarray:
    """Recompose ordered ``(PauliWord, angle, kind)`` rotations in the irrep.

    ``algebra_basis`` is the labelled matrix basis; ``time`` scales only ``a0`` rates.
    """
    return _reconstruct_rotations(pauli_rotations, algebra_basis, irrep_size, time=time)


def kak_decomposition(
    generators,
    coefficients=None,
    time: float = 1.0,
    n_qubits: int | None = None,
    involution: str | None = None,
    invol_kwargs: dict | None = None,
    validate: bool = True,
    atol: float | None = None,
    tol: float | None = None,
) -> KAKResult:
    """Compile ``exp(time * H)`` into Pauli rotations using PauLie's classification.

    Generators accepted by :func:`as_pauli_words` must be individual horizontal Pauli
    terms. ``coefficients`` supplies one finite real weight per input term before
    deduplication, multiplying intrinsic PennyLane coefficients; duplicates are summed.
    The default weights are one. Generator sums are rejected.

    ``n_qubits`` is inferred when omitted. Only BDI on an so(m) presentation is
    supported; ``invol_kwargs`` may specify its p/q partition. The horizontal
    Hamiltonian determines time-independent Cartan rates, reusable via the result's
    ``reconstruct(time=...)`` method.

    ``validate`` checks the Pauli-rotation reconstruction against the matrix
    exponential. ``atol`` defaults to ``1e-10 * irrep_size**2`` for accumulated Givens
    error. ``tol`` optionally drops small vertical angles; central rates are always
    retained so longer evolution times remain valid. Both tolerances must be
    nonnegative and finite when supplied.

    Raises ValueError for invalid inputs or failed reconstruction, and
    NotImplementedError for unsupported algebras or involutions.
    """
    inputs = prepare_pauli_inputs(generators, n_qubits)
    time = finite_real_scalar(time, "time")
    tol = nonnegative_tolerance(tol, "tol")
    atol = nonnegative_tolerance(atol, "atol")
    words, coefficients = inputs.hamiltonian_terms(coefficients)
    collection = inputs.collection()
    info = _classification_for(inputs, collection)
    irrep_size, involution = _resolve_orthogonal(info, involution)
    p, q = resolve_bdi_partition(irrep_size, invol_kwargs)
    basis = _native_pauli_basis(collection, info)
    mapping, signs = _map_verified_basis(words, basis, irrep_size, involution, p, q)

    # Label PauLie's orthogonal presentation with the verified Pauli words.
    algebra_basis = labelled_matrix_basis(mapping, signs, info)
    hamiltonian = np.zeros((irrep_size, irrep_size))
    for coeff, word in zip(coefficients, words):
        hamiltonian = hamiltonian + coeff * algebra_basis[word]

    with np.errstate(over="ignore", invalid="ignore"):
        scaled_hamiltonian = time * hamiltonian
    if not np.isfinite(scaled_hamiltonian).all():
        raise ValueError("The time-scaled Hamiltonian must contain finite values.")
    unitary = expm(scaled_hamiltonian)
    pauli_rotations, matrix_factors = decompose_horizontal_hamiltonian(
        hamiltonian, p, mapping, signs, time=time, tol=tol
    )

    if atol is None:
        # Givens factorization uses O(irrep_size**2) rotations; allow for their
        # accumulated numerical error rather than expecting machine precision.
        atol = 1e-10 * irrep_size**2

    error = None
    if validate:
        recomposed = reconstruct_from_pauli_rotations(
            pauli_rotations, algebra_basis, irrep_size, time=time
        )
        error = float(np.abs(recomposed - unitary).max())
        if not np.isfinite(error) or error > atol:
            raise ValueError(
                f"The Pauli rotations do not recompose exp({time} * H): max error "
                f"{error:.3e} > {atol:.1e}."
            )

    return KAKResult(
        info=info,
        involution=involution,
        irrep_size=irrep_size,
        pauli_rotations=pauli_rotations,
        time=time,
        hamiltonian_irrep=hamiltonian,
        unitary_irrep=unitary,
        matrix_factors=matrix_factors,
        mapping=mapping,
        signs=signs,
        algebra_basis=algebra_basis,
        reconstruction_error=error,
    )
