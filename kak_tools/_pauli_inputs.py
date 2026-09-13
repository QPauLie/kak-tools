"""Normalize Pauli inputs once, retaining coefficient and register semantics."""

import math

import numpy as np
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_collection import PauliStringCollection
from paulie.common.pauli_string_factory import get_pauli_string
from pennylane.pauli import PauliWord

from ._validation import (
    finite_real_scalar,
    integer,
    positive_integer,
    real_coefficients,
)


def _wire_width(wires):
    width = 0
    for wire in wires:
        width = max(width, integer(wire, f"Wire {wire!r}", minimum=0) + 1)
    return width


def _word_width(word):
    if any(pauli not in ("I", "X", "Y", "Z") for pauli in word.values()):
        raise ValueError("PauliWord entries must contain only I, X, Y, Z.")
    return _wire_width(word)


def pauli_string_to_word(pauli_string) -> PauliWord:
    """Convert a plain string or PauLie PauliString, preserving wire positions."""
    if not isinstance(pauli_string, (str, PauliString)):
        raise TypeError("Expected a Pauli string or PauLie PauliString.")
    text = str(pauli_string)
    if not text or any(char not in "IXYZ" for char in text):
        raise ValueError("Pauli strings must be nonempty and contain only I, X, Y, Z.")
    return PauliWord({i: char for i, char in enumerate(text) if char != "I"})


def pauli_word_to_string(pauli_word: PauliWord, n_qubits: int):
    """Convert a PauliWord to a big-endian PauLie string on n_qubits wires."""
    n_qubits = positive_integer(n_qubits, "n_qubits")
    if not isinstance(pauli_word, PauliWord):
        raise TypeError("Expected a PennyLane PauliWord.")
    if _word_width(pauli_word) > n_qubits:
        raise ValueError(
            f"Cannot convert {pauli_word}: wire is not an integer in range({n_qubits})."
        )
    chars = ["I"] * n_qubits
    for wire, pauli in pauli_word.items():
        chars[int(wire)] = pauli
    return get_pauli_string("".join(chars))


def _parse_generators(generators, n_qubits):
    """Read each generator once as a word and its intrinsic weight.

    Wires on explicit Identity operators count towards the register width even
    though ``pauli_rep`` drops them.
    """
    if isinstance(generators, (str, PauliString, PauliWord)) or hasattr(generators, "pauli_rep"):
        generators = [generators]
    terms, factors = [], []
    width = 0
    for gen in generators:
        factor = 1.0
        if isinstance(gen, PauliWord):
            word = gen
        elif isinstance(gen, (str, PauliString)):
            word = pauli_string_to_word(gen)
            width = max(width, len(gen))
        elif getattr(gen, "pauli_rep", None) is not None:
            width = max(width, _wire_width(gen.wires))
            if len(gen.pauli_rep) != 1:
                raise ValueError(
                    "Each generator must contain a single Pauli term; sums and zero "
                    "operators are not supported. Supply individual Pauli terms and coefficients."
                )
            word, factor = next(iter(gen.pauli_rep.items()))
            factor = finite_real_scalar(factor, "Pauli coefficient")
        else:
            raise TypeError("Generators must be Pauli strings, PauliWords or single Pauli operators.")
        width = max(width, _word_width(word))
        terms.append(word)
        factors.append(factor)
    if not terms:
        raise ValueError("At least one Pauli generator is required.")
    n_qubits = max(width, 1) if n_qubits is None else positive_integer(n_qubits, "n_qubits")
    if width > n_qubits:
        raise ValueError(f"Generators require {width} qubits, but n_qubits={n_qubits}.")
    return terms, factors, n_qubits


def prepare_pauli_inputs(generators, coefficients=None, n_qubits=None):
    """Return ``(words, weights, n_qubits)`` with distinct words in first-occurrence order.

    Every listed generator belongs to the family, whatever its weight. Each
    weight is the intrinsic PennyLane factor times the external coefficient for
    that input term (default one); the weights of identical words are summed
    with :func:`math.fsum`, so small terms survive large cancellations.
    """
    terms, factors, n_qubits = _parse_generators(generators, n_qubits)
    if coefficients is None:
        coefficients = np.ones(len(terms))
    else:
        coefficients = real_coefficients(coefficients, len(terms))
    contributions = {}
    for word, factor, coefficient in zip(terms, factors, coefficients, strict=True):
        weighted = factor * coefficient
        if not math.isfinite(weighted):
            raise ValueError("Combined Hamiltonian coefficients must be finite.")
        contributions.setdefault(word, []).append(weighted)
    try:
        weights = [math.fsum(values) for values in contributions.values()]
    except OverflowError as exc:
        raise ValueError("Combined Hamiltonian coefficients must be finite.") from exc
    return list(contributions), weights, n_qubits


def words_to_collection(words, n_qubits):
    """Build one canonical native collection, preserving padding and bit endianness."""
    return PauliStringCollection([pauli_word_to_string(word, n_qubits) for word in words])


def as_pauli_words(generators, n_qubits: int | None = None) -> list[PauliWord]:
    """Normalize single Pauli terms, preserving order and removing duplicates.

    Supports strings, PauLie strings/collections, PennyLane PauliWords and real
    multiples of single Pauli operators. Sums are rejected. Weights, including
    zero, do not change the generator family. n_qubits, when given, validates
    the width.
    """
    return prepare_pauli_inputs(generators, n_qubits=n_qubits)[0]


def as_pauli_collection(generators, n_qubits: int | None = None):
    """Return a deduplicated, padded, big-endian PauLie collection.

    Re-encoding native strings handles little/mixed-endian inputs independently
    of the endian assumptions in older pauliebits C extensions.
    """
    words, _, n_qubits = prepare_pauli_inputs(generators, n_qubits=n_qubits)
    return words_to_collection(words, n_qubits)
