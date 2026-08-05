"""Bridge between `PauLie <https://github.com/QPauLie/PauLie>`__ and ``kak_tools``.

``kak_tools`` computes KAK (Cartan) decompositions of Lie algebras spanned by Pauli
words. To do so it needs to know *which* algebra it is looking at: the irrep size
``n`` and an involution type. Until now that information had to be supplied by hand
(``n_so = 2 * n`` is hard-coded in :mod:`kak_tools.full_workflows`) or guessed from
the dimension alone by :func:`kak_tools.identify_algebra`, which is deliberately
non-unique -- ``dim = 28`` is consistent with ``so(8)`` *and* with ``2 x so(7)``,
and the function returns every candidate.

``PauLie`` settles the question: it classifies the dynamical Lie algebra of a set of
Pauli-string generators exactly and in polynomial time, returning a decomposition
such as ``"so(8)"`` or ``"u(1)+2*su(2)"``. This module wires the two together:

    generators -> PauLie classification -> (type, n, multiplicity, dim)
               -> Pauli-word DLA basis   (size known exactly in advance)
               -> kak_tools irrep mapping / KAK decomposition
               -> Pauli rotations

Typical use::

    from kak_tools.paulie_bridge import kak_decomposition

    n = 4
    gens = ([f"{'I'*w}XX{'I'*(n-w-2)}" for w in range(n - 1)]
            + [f"{'I'*w}YY{'I'*(n-w-2)}" for w in range(n - 1)]
            + [f"{'I'*w}Z{'I'*(n-w-1)}" for w in range(n)])
    result = kak_decomposition(gens, time=0.5)
    print(result.info.algebra)          # 'so(8)'
    print(len(result.pauli_rotations))  # sequence of Pauli rotations implementing exp(t H)

**Scope.** ``kak_tools`` builds its Pauli-word irrep mapping for ``so(m)`` with a ``BDI``
involution, so that is what :func:`kak_decomposition` covers; the low-rank coincidences
(``su(2) = so(3)``, ``2*su(2) = so(4)``, ``sp(2) = so(5)``, ``su(4) = so(6)``, ...) are
recognised, so a DLA that PauLie names ``2*so(3)`` is still decomposed as ``so(4)``.
Odd ``m`` is currently unreliable -- the top-level split is then ``BDI(p, p+1)``, whose
horizontal cosine-sine decomposition has a gauge freedom that
:func:`kak_tools.dense_cartan.bdi` does not fix; :func:`kak_decomposition` raises a
``NotImplementedError`` explaining this rather than returning a wrong answer. Even ``m``,
which is what qubit models with a free-fermionic DLA produce, is fully supported.
Anything else is refused with a message naming the algebra PauLie found; the matrix-level
routines in :mod:`kak_tools.numerical_decompositions` cover the other classical types.

``paulie`` is a hard dependency of this module (Python >= 3.12).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from paulie.classifier.classification import Classification
from paulie.common.algebra_basis import get_so_basis
from paulie.common.pauli_string_collection import PauliStringCollection
from paulie.common.pauli_string_factory import get_pauli_string
from pennylane.pauli import PauliWord
from scipy.linalg import expm

from .dense_cartan import map_recursive_decomp_to_reducible, recursive_bdi
from .map_to_irrep import map_simple_to_irrep
from .pauli_dlas import get_simple_dim, lie_closure_pauli_words

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


# ---------------------------------------------------------------------------
# Representation conversion: PauLie PauliString <-> PennyLane PauliWord
# ---------------------------------------------------------------------------


def pauli_string_to_word(pauli_string) -> PauliWord:
    """Convert a PauLie ``PauliString`` into a PennyLane ``PauliWord``.

    Position ``i`` of the Pauli string is identified with wire ``i``; identity factors
    are dropped, as is PennyLane's convention.

    Args:
        pauli_string (paulie.common.pauli_string_bitarray.PauliString or str): The Pauli
            string, e.g. ``"XYIZ"``.

    Returns:
        pennylane.pauli.PauliWord: The equivalent Pauli word, e.g. ``X(0) @ Y(1) @ Z(3)``.
    """
    text = str(pauli_string)
    return PauliWord({i: char for i, char in enumerate(text) if char != "I"})


def pauli_word_to_string(pauli_word: PauliWord, n_qubits: int):
    """Convert a PennyLane ``PauliWord`` into a PauLie ``PauliString`` of length ``n_qubits``.

    Args:
        pauli_word (pennylane.pauli.PauliWord): The Pauli word. All of its wires must be
            integers in ``range(n_qubits)``.
        n_qubits (int): Length of the resulting Pauli string.

    Returns:
        paulie.common.pauli_string_bitarray.PauliString: The equivalent Pauli string.

    Raises:
        ValueError: If a wire is not an integer in ``range(n_qubits)``.
    """
    chars = ["I"] * n_qubits
    for wire, pauli in pauli_word.items():
        if not isinstance(wire, (int, np.integer)) or not 0 <= wire < n_qubits:
            raise ValueError(
                f"Cannot convert {pauli_word}: wire {wire!r} is not an integer in "
                f"range({n_qubits}). Relabel the wires to 0, ..., n_qubits - 1 first."
            )
        chars[int(wire)] = pauli
    return get_pauli_string("".join(chars))


def _infer_n_qubits(generators) -> int:
    """Infer the number of qubits spanned by a collection of generators."""
    if isinstance(generators, PauliStringCollection):
        return len(generators.get()[0])

    max_wire = -1
    max_len = 0
    for gen in generators:
        if isinstance(gen, PauliWord):
            word = gen
        elif isinstance(gen, str):
            max_len = max(max_len, len(gen))
            continue
        elif hasattr(gen, "pauli_rep") and gen.pauli_rep is not None:
            word = next(iter(gen.pauli_rep))
        else:  # PauLie PauliString
            max_len = max(max_len, len(gen))
            continue
        if len(word):
            max_wire = max(max_wire, max(int(w) for w in word))
    return max(max_wire + 1, max_len)


def as_pauli_words(generators, n_qubits: int | None = None) -> list[PauliWord]:
    """Normalise a generator set to a list of PennyLane ``PauliWord``s.

    Accepts PennyLane ``PauliWord``s, PennyLane operators with a Pauli representation,
    plain strings such as ``"XXII"``, PauLie ``PauliString``s and PauLie
    ``PauliStringCollection``s.

    Args:
        generators: The generators, in any of the formats listed above.
        n_qubits (int, optional): Ignored; present for signature symmetry with
            :func:`as_pauli_collection`.

    Returns:
        list[pennylane.pauli.PauliWord]: The generators as Pauli words, order preserved
        and duplicates removed.
    """
    if isinstance(generators, PauliStringCollection):
        generators = generators.get()

    words: list[PauliWord] = []
    for gen in generators:
        if isinstance(gen, PauliWord):
            word = gen
        elif isinstance(gen, str):
            word = pauli_string_to_word(gen)
        elif hasattr(gen, "pauli_rep") and gen.pauli_rep is not None:
            # A general PennyLane operator such as qml.X(0) @ qml.X(1)
            word = next(iter(gen.pauli_rep))
        else:  # PauLie PauliString (or anything that stringifies to one)
            word = pauli_string_to_word(gen)
        if word not in words:
            words.append(word)
    return words


def as_pauli_collection(generators, n_qubits: int | None = None):
    """Normalise a generator set to a PauLie ``PauliStringCollection``.

    Args:
        generators: The generators, in any format accepted by :func:`as_pauli_words`.
        n_qubits (int, optional): Length of the Pauli strings. Inferred from the
            generators when not given.

    Returns:
        paulie.common.pauli_string_collection.PauliStringCollection: The collection,
        with every string padded to ``n_qubits``.
    """
    if isinstance(generators, PauliStringCollection):
        if n_qubits is None or len(generators.get()[0]) == n_qubits:
            return generators
        generators = generators.get()

    if n_qubits is None:
        n_qubits = _infer_n_qubits(generators)

    words = as_pauli_words(generators)
    strings = [str(pauli_word_to_string(word, n_qubits)) for word in words]
    return get_pauli_string(strings)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


_COMPONENT_RE = re.compile(r"^(?:(?P<multiplicity>\d+)\*)?(?P<type>[a-z]+)\((?P<size>\d+)\)$")


@dataclass(frozen=True)
class DLAComponent:
    """One ``multiplicity x type(size)`` summand of a dynamical Lie algebra.

    These are not computed here: they are parsed straight out of PauLie's own
    ``Classification.get_subalgebras()``, which is the decomposition PauLie reports.

    Attributes:
        type (str): One of ``"u"``, ``"so"``, ``"su"``, ``"sp"``.
        size (int): The ``n`` in ``so(n)``, ``su(n)``, ``sp(n)``, ``u(n)``.
        multiplicity (int): How many isomorphic copies of this algebra occur.
    """

    type: str
    size: int
    multiplicity: int = 1

    @classmethod
    def parse(cls, term: str) -> DLAComponent:
        """Parse one term of a PauLie algebra name, e.g. ``"2*su(2)"`` or ``"so(8)"``.

        Args:
            term (str): A single summand of ``Classification.get_algebra()``.

        Returns:
            DLAComponent: The parsed component.

        Raises:
            ValueError: If the term is not in PauLie's ``[k*]name(size)`` format.
        """
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
        """int: Size of the matrices in the defining irrep of a single copy.

        Mirrors the block sizing in ``paulie.common.algebra_basis.get_algebras_basis``,
        where an ``sp(n)`` block is ``2n x 2n``.
        """
        return 2 * self.size if self.type == "sp" else self.size

    def __str__(self) -> str:
        core = f"{self.type}({self.size})"
        return core if self.multiplicity == 1 else f"{self.multiplicity}*{core}"


@dataclass(frozen=True)
class DLAInfo:
    """Thin adapter over PauLie's ``Classification``, in kak_tools' vocabulary.

    Every question about the algebra itself is answered by PauLie: the name comes from
    ``Classification.get_algebra()``, the dimension from ``get_dla_dim()``, the summands
    from ``get_subalgebras()``, the matrix basis from ``get_algebra_basis()`` and the
    isomorphism tests from ``is_algebra()``. Nothing here re-derives them. What this
    class adds is the one thing PauLie has no reason to know about, namely which irrep
    size and involution ``kak_tools`` should be configured with.

    Attributes:
        classification (paulie.classifier.classification.Classification): PauLie's
            classification object, exposed so callers can reach the rest of its API.
        n_qubits (int): Number of qubits the generators act on.
    """

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
        """tuple[DLAComponent, ...]: The summands, as PauLie splits them."""
        return tuple(
            DLAComponent.parse(term) for term in self.classification.get_subalgebras()
        )

    @property
    def matrix_basis(self) -> np.ndarray:
        """numpy.ndarray: PauLie's basis of the algebra *as PauLie names it*.

        This is ``Classification.get_algebra_basis()`` verbatim, so it follows the
        classified presentation: a DLA reported as ``2*so(3)`` gets a block-diagonal
        ``(6, 6, 6)`` basis, not the ``(6, 4, 4)`` basis of the isomorphic ``so(4)``.
        Use :attr:`orthogonal_basis` for the presentation ``kak_tools`` decomposes in.
        """
        return self.classification.get_algebra_basis()

    @property
    def orthogonal_basis(self) -> np.ndarray:
        """numpy.ndarray: PauLie's ``so(m)`` basis, for the ``m`` of :attr:`orthogonal_size`.

        Built with ``paulie.common.algebra_basis.get_so_basis``. This differs from
        :attr:`matrix_basis` exactly when PauLie names the algebra through one of its
        low-rank coincidences (``2*so(3)`` rather than ``so(4)``), where the two
        presentations live in different dimensions. The ``k``-th matrix generates the
        rotation in the plane ``combinations(range(m), 2)[k]``, which is the node
        ordering ``kak_tools.map_simple_to_irrep`` uses.

        Raises:
            ValueError: If the DLA has no ``so(m)`` presentation.
        """
        m = self.orthogonal_size
        if m is None:
            raise ValueError(
                f"PauLie classified this DLA as {self.algebra}, which has no so(m) "
                "presentation, so there is no orthogonal basis to build."
            )
        return get_so_basis(m)

    def is_algebra(self, algebra: str) -> bool:
        """Whether the DLA equals ``algebra``, up to PauLie's low-rank isomorphisms.

        Args:
            algebra (str): An algebra name such as ``"so(4)"``.

        Returns:
            bool: PauLie's verdict.
        """
        return bool(self.classification.is_algebra(algebra))

    @property
    def is_simple(self) -> bool:
        """bool: Whether the algebra is a single simple factor."""
        components = self.components
        return len(components) == 1 and components[0].multiplicity == 1

    @property
    def simple_component(self) -> DLAComponent:
        """DLAComponent: The unique simple summand.

        Raises:
            ValueError: If the algebra is not simple.
        """
        if not self.is_simple:
            raise ValueError(
                f"The DLA {self.algebra} is not simple; it has components "
                f"{[str(c) for c in self.components]}. Decompose the components "
                "separately, or use `split_pauli_algebra` to obtain them."
            )
        return self.components[0]

    @property
    def orthogonal_size(self) -> int | None:
        """int or None: The ``m`` for which this DLA is (isomorphic to) ``so(m)``.

        ``so(m)`` is simple for ``m >= 3, m != 4``, but the classification of the
        classical Lie algebras has a handful of low-rank coincidences, and PauLie
        reports whichever presentation its canonical graph produces. The transverse-field
        XY model on two qubits, for example, classifies as ``2*so(3)`` -- which *is*
        ``so(4)``, and which ``kak_tools`` is perfectly happy to decompose with a BDI
        involution.

        The candidate ``m`` is fixed by the dimension, and the verdict is PauLie's own
        :meth:`~paulie.classifier.classification.Classification.is_algebra`, which
        canonicalises both sides through its table of low-rank isomorphisms.

        Returns:
            int or None: ``m`` if an ``so(m)`` presentation exists, else ``None``.
        """
        # so(m) has dimension m (m - 1) / 2, so m is fixed by the dimension.
        is_so_dimension, m = get_simple_dim("so", self.dim)
        if not is_so_dimension:
            return None
        if m == 2:
            # PauLie's isomorphism table does not list so(2) = u(1), so check by name.
            return 2 if self.algebra == "u(1)" else None
        return m if self.is_algebra(f"so({m})") else None

    def __str__(self) -> str:
        return self.algebra


def classify_dla(generators, n_qubits: int | None = None) -> DLAInfo:
    """Classify the dynamical Lie algebra of a set of Pauli operators, using PauLie.

    This converts ``kak_tools``-flavoured input (PennyLane Pauli words and operators,
    plain Pauli strings) into a PauLie ``PauliStringCollection`` and hands back its
    ``Classification``. The classification itself is entirely PauLie's -- see
    :meth:`paulie.common.pauli_string_collection.PauliStringCollection.get_class`.

    Unlike :func:`kak_tools.identify_algebra`, which only sees the dimension and returns
    every candidate consistent with it, this gives a single answer.

    Args:
        generators: The generators, in any format accepted by :func:`as_pauli_words`.
        n_qubits (int, optional): Number of qubits. Inferred from the generators when
            not given.

    Returns:
        DLAInfo: PauLie's classification, wrapped for use by ``kak_tools``.

    **Example**

    >>> classify_dla(["XXI", "IXX", "YYI", "IYY", "ZII", "IZI", "IIZ"]).algebra
    'so(6)'
    """
    if n_qubits is None:
        n_qubits = _infer_n_qubits(generators)
    collection = as_pauli_collection(generators, n_qubits)
    return DLAInfo(classification=collection.get_class(), n_qubits=n_qubits)


# ---------------------------------------------------------------------------
# Pauli-word basis of the DLA
# ---------------------------------------------------------------------------


def dla_pauli_basis(
    generators,
    info: DLAInfo | None = None,
    n_qubits: int | None = None,
    strict: bool = True,
) -> list[PauliWord]:
    """Return a Pauli-word basis of the DLA, using PauLie's dimension as a target.

    The closure itself is :func:`kak_tools.lie_closure_pauli_words`; the only thing added
    here is that PauLie already knows the answer's size, so it is passed as that
    function's ``full_size`` hint and the iteration stops the moment the algebra is
    complete instead of running one more full sweep to discover that nothing new appears.

    Note this is the *Pauli-word* basis, which is what
    :func:`kak_tools.map_simple_to_irrep` consumes. For the matrix basis of the algebra
    in its defining representation, use ``info.matrix_basis`` (PauLie's
    ``get_algebra_basis``) or :func:`labelled_matrix_basis`.

    Args:
        generators: The generators, in any format accepted by :func:`as_pauli_words`.
        info (DLAInfo, optional): A previously computed classification. Recomputed when
            not given.
        n_qubits (int, optional): Number of qubits. Inferred when not given.
        strict (bool): Whether to raise if the closure size disagrees with PauLie's
            predicted dimension. Set to ``False`` to downgrade this to a warning.

    Returns:
        list[pennylane.pauli.PauliWord]: A basis of the DLA, of length ``info.dim``.

    Raises:
        ValueError: If ``strict`` and the closure size differs from ``info.dim``.
    """
    if info is None:
        info = classify_dla(generators, n_qubits)
    words = as_pauli_words(generators)
    basis = lie_closure_pauli_words(words, full_size=info.dim)

    if len(basis) != info.dim:
        message = (
            f"The Lie closure produced {len(basis)} Pauli words but PauLie classified "
            f"the DLA as {info.algebra} of dimension {info.dim}. This usually means "
            "the generators span a centre (identity-like) direction that PauLie counts "
            "differently, or that the closure did not converge."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning)
    return basis


def _so_node_index(m: int) -> dict:
    """Position of each ``(i, j)`` rotation plane in PauLie's ``so(m)`` basis.

    Built with the same ``numpy.triu_indices(m, k=1)`` call that
    ``paulie.common.algebra_basis.get_so_basis`` lays its matrices out along, so the two
    cannot drift apart. It is also the order ``kak_tools.map_simple_to_irrep`` enumerates
    its nodes in.
    """
    rows, cols = np.triu_indices(m, k=1)
    return {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}


def labelled_matrix_basis(mapping, signs, info: DLAInfo, validate: bool = True) -> dict:
    """Label PauLie's basis matrices with the Pauli words they correspond to.

    PauLie's ``get_algebra_basis`` returns a basis of the algebra in its defining
    representation but has no reason to know which Pauli word each element came from;
    ``kak_tools.map_simple_to_irrep`` produces exactly that correspondence but no
    matrices. Putting the two together gives a basis that is usable on both sides.

    ``kak_tools`` scales its generators by two relative to PauLie's ``E_ij - E_ji``, so
    the matrix returned for a Pauli word is ``2 * sign * basis[k]``.

    Args:
        mapping (dict): Irrep index pairs to Pauli words, from :func:`map_dla_to_irrep`.
        signs (dict): The accompanying signs.
        info (DLAInfo): PauLie's classification, supplying the basis.
        validate (bool): Whether to check that the two packages' conventions agree.

    Returns:
        dict[pennylane.pauli.PauliWord, numpy.ndarray]: The labelled basis.

    Raises:
        ValueError: If ``validate`` and the bases do not line up.
    """
    m = info.orthogonal_size
    if m is None:
        raise ValueError(
            f"labelled_matrix_basis needs an so(m) presentation; PauLie classified this "
            f"DLA as {info.algebra}."
        )
    basis = info.orthogonal_basis
    index = _so_node_index(m)

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


# ---------------------------------------------------------------------------
# Involutions
# ---------------------------------------------------------------------------

#: The only involution kak_tools can build a Pauli-word irrep mapping for.
PAULI_LEVEL_INVOLUTION = "BDI"


def _resolve_orthogonal(info: DLAInfo, involution: str | None) -> tuple[int, str]:
    """Resolve the irrep size and involution for the Pauli-level pipeline.

    Args:
        info (DLAInfo): PauLie's classification.
        involution (str or None): Requested involution, or ``None`` for the default.

    Returns:
        tuple[int, str]: The irrep size ``m`` and the involution label.

    Raises:
        NotImplementedError: If the algebra or involution has no Pauli-level mapping.
    """
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


# ---------------------------------------------------------------------------
# Mapping into an irrep
# ---------------------------------------------------------------------------


def map_dla_to_irrep(
    generators,
    dla: Sequence[PauliWord] | None = None,
    info: DLAInfo | None = None,
    n_qubits: int | None = None,
    involution: str | None = None,
    invol_kwargs: dict | None = None,
):
    """Map a Pauli-word DLA onto an irrep, with the irrep size taken from PauLie.

    This wraps :func:`kak_tools.map_simple_to_irrep`, replacing its hand-supplied ``n``
    with PauLie's exact classification.

    Args:
        generators: The generators, in any format accepted by :func:`as_pauli_words`.
            They are treated as the *horizontal* operators of the decomposition.
        dla (Sequence[pennylane.pauli.PauliWord], optional): Pauli-word basis of the
            DLA. Computed via :func:`dla_pauli_basis` when not given.
        info (DLAInfo, optional): A previously computed classification.
        n_qubits (int, optional): Number of qubits. Inferred when not given.
        involution (str, optional): Involution type. Defaults to the type implied by the
            classification (``"BDI"`` for ``so(n)``).
        invol_kwargs (dict, optional): Extra involution parameters, e.g. ``{"p": 3}``.

    Returns:
        tuple: ``(mapping, signs, info)`` where ``mapping`` sends irrep index pairs to
        Pauli words, ``signs`` carries the relative signs, and ``info`` is the
        classification used.
    """
    if info is None:
        info = classify_dla(generators, n_qubits)
    irrep_size, involution = _resolve_orthogonal(info, involution)

    if dla is None:
        dla = dla_pauli_basis(generators, info=info, n_qubits=n_qubits)

    horizontal_ops = as_pauli_words(generators)
    mapping, signs = map_simple_to_irrep(
        list(dla),
        horizontal_ops=horizontal_ops,
        n=irrep_size,
        invol_type=involution,
        invol_kwargs=invol_kwargs,
    )
    return mapping, signs, info


# ---------------------------------------------------------------------------
# Full decomposition
# ---------------------------------------------------------------------------


@dataclass
class KAKResult:
    """Result of a full PauLie-informed KAK decomposition.

    Attributes:
        info (DLAInfo): PauLie's classification of the DLA.
        involution (str): The Cartan involution used.
        irrep_size (int): Size of the irrep matrices, i.e. the ``n`` in ``so(n)``.
        pauli_rotations (list): The decomposition, as ``(PauliWord, angle, kind)``
            triples. ``kind`` is ``"k1"``/``"k2"`` for vertical factors and ``"a0"``,
            ``"a"`` for Cartan-subalgebra factors.
        time (float): The evolution time the Hamiltonian was scaled by.
        hamiltonian_irrep (numpy.ndarray): The Hamiltonian in the irrep.
        unitary_irrep (numpy.ndarray): ``expm(time * hamiltonian_irrep)``.
        matrix_factors (list): The raw output of :func:`kak_tools.recursive_bdi`.
        mapping (dict): Irrep index pairs to Pauli words.
        signs (dict): Relative signs accompanying ``mapping``.
        algebra_basis (dict): Pauli words to algebra elements -- PauLie's
            ``get_algebra_basis`` matrices, labelled via ``mapping``.
        reconstruction_error (float or None): Max-abs error of recomposing
            ``unitary_irrep`` from ``pauli_rotations``; ``None`` if not validated.
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
        """list: The ``(PauliWord, angle)`` pairs of the central Cartan factor."""
        return [(pw, angle) for pw, angle, kind in self.pauli_rotations if kind == "a0"]

    def reconstruct(self) -> np.ndarray:
        """numpy.ndarray: Recompose ``unitary_irrep`` from the Pauli rotations."""
        return reconstruct_from_pauli_rotations(
            self.pauli_rotations, self.algebra_basis, self.irrep_size, time=self.time
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
    """Recompose the irrep unitary from a sequence of Pauli rotations.

    Used to verify a decomposition: the result should equal the unitary that was
    decomposed.

    Args:
        pauli_rotations (list): ``(PauliWord, angle, kind)`` triples.
        algebra_basis (dict): Pauli words to algebra elements, as returned by
            :func:`labelled_matrix_basis` (and carried on ``KAKResult.algebra_basis``).
        irrep_size (int): Size of the irrep matrices.
        time (float, optional): Evolution time, if the ``"a0"`` angles were divided by it.

    Returns:
        numpy.ndarray: The recomposed matrix.
    """
    out = np.eye(irrep_size)
    for word, angle, kind in pauli_rotations:
        scale = time if (kind == "a0" and time is not None) else 1.0
        out = out @ expm(algebra_basis[word] * angle * scale)
    return out


def kak_decomposition(
    generators,
    coefficients=None,
    time: float = 1.0,
    n_qubits: int | None = None,
    involution: str | None = None,
    invol_kwargs: dict | None = None,
    validate: bool = True,
    atol: float | None = None,
    tol: float | None = 1e-8,
) -> KAKResult:
    """Decompose ``exp(time * H)`` into Pauli rotations, with the algebra fixed by PauLie.

    This is the model-agnostic counterpart of
    :func:`kak_tools.full_workflows.complete_workflow_tfXY`: instead of assuming the
    transverse-field XY model and its ``so(2n)`` algebra, it asks PauLie what the DLA
    of the given generators actually is and configures the decomposition accordingly.

    The generators are taken to be the Hamiltonian terms and are treated as *horizontal*
    operators, i.e. ``H`` lies in the ``m`` subspace of the Cartan decomposition. That is
    the setting in which the KAK decomposition yields a fixed-depth circuit.

    Args:
        generators: The Hamiltonian terms, in any format accepted by
            :func:`as_pauli_words`.
        coefficients (Sequence[float], optional): Coefficients of the Hamiltonian terms.
            Defaults to all ones.
        time (float): Evolution time; the decomposed unitary is ``expm(time * H)``.
        n_qubits (int, optional): Number of qubits. Inferred when not given.
        involution (str, optional): Involution type. Defaults to the one implied by the
            classification.
        invol_kwargs (dict, optional): Extra involution parameters.
        validate (bool): Whether to recompose the unitary from the Pauli rotations and
            check the result.
        atol (float, optional): Tolerance for the validation check. Defaults to
            ``1e-10 * irrep_size ** 2``, since the recursion multiplies ``O(irrep_size)``
            orthogonal factors.
        tol (float, optional): Coefficient cutoff passed to
            :func:`kak_tools.map_recursive_decomp_to_reducible`.

    Returns:
        KAKResult: The decomposition and everything needed to interpret it.

    Raises:
        ValueError: If ``validate`` and the Pauli rotations do not recompose the unitary.
        NotImplementedError: If the classified algebra has no Pauli-level involution
            implemented in ``kak_tools``.
    """
    words = as_pauli_words(generators)
    if coefficients is None:
        coefficients = np.ones(len(words))
    coefficients = np.asarray(coefficients, dtype=float)
    if len(coefficients) != len(words):
        raise ValueError(
            f"Got {len(coefficients)} coefficients for {len(words)} distinct generators."
        )

    info = classify_dla(words, n_qubits)
    irrep_size, involution = _resolve_orthogonal(info, involution)

    dla = dla_pauli_basis(words, info=info)
    mapping, signs, info = map_dla_to_irrep(
        words,
        dla=dla,
        info=info,
        involution=involution,
        invol_kwargs=invol_kwargs,
    )

    # The algebra elements come from PauLie's get_algebra_basis, labelled with the Pauli
    # words that map_simple_to_irrep matched them to.
    algebra_basis = labelled_matrix_basis(mapping, signs, info)
    hamiltonian = np.zeros((irrep_size, irrep_size))
    for coeff, word in zip(coefficients, words):
        hamiltonian = hamiltonian + coeff * algebra_basis[word]

    unitary = expm(time * hamiltonian)
    try:
        matrix_factors = recursive_bdi(unitary, irrep_size, validate=False, return_all=False)
    except ValueError as exc:
        if irrep_size % 2:
            raise NotImplementedError(
                f"PauLie classified this DLA as so({irrep_size}), which has an odd irrep "
                "size, so the top-level BDI split is BDI(p, p+1) with p != q. The "
                "horizontal cosine-sine decomposition then has a continuous O(q - p) "
                "gauge freedom that `kak_tools.dense_cartan.bdi` does not currently fix, "
                "and the k1 = k2.T relation fails for most Hamiltonians. Even irrep "
                "sizes -- which is what qubit models with a free-fermionic DLA give -- "
                "are fully supported."
            ) from exc
        raise

    pauli_rotations = map_recursive_decomp_to_reducible(
        matrix_factors, mapping, signs, time=time, tol=tol, validate=False
    )

    if atol is None:
        # The recursion multiplies O(irrep_size) orthogonal factors, so the achievable
        # accuracy degrades with the irrep size rather than staying at machine epsilon.
        atol = 1e-10 * irrep_size**2

    error = None
    if validate:
        recomposed = reconstruct_from_pauli_rotations(
            pauli_rotations, algebra_basis, irrep_size, time=time
        )
        error = float(np.abs(recomposed - unitary).max())
        if error > atol:
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
