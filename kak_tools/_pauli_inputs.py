"""Normalize Pauli inputs once, retaining coefficient and register semantics."""

from dataclasses import dataclass
import math

import numpy as np
from paulie.common.pauli_string_bitarray import PauliString
from paulie.common.pauli_string_collection import PauliStringCollection
from paulie.common.pauli_string_factory import get_pauli_string
from pennylane.pauli import PauliWord

from ._validation import finite_real_scalar, positive_integer, real_coefficients


def _wire_width(wires):
    width = 0
    for wire in wires:
        if (isinstance(wire, (bool, np.bool_))
                or not isinstance(wire, (int, np.integer)) or wire < 0):
            raise ValueError(
                f"Wire {wire!r} is not an integer in range(n_qubits). "
                "Relabel the wires to 0, ..., n_qubits - 1 first."
            )
        width = max(width, int(wire) + 1)
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


def _materialize_generators(generators):
    if isinstance(generators, PauliStringCollection):
        items = list(generators.get())
    elif isinstance(generators, (str, PauliString, PauliWord)) or hasattr(generators, "pauli_rep"):
        items = [generators]
    else:
        items = list(generators)
    if not items:
        raise ValueError("At least one Pauli generator is required.")
    return items


@dataclass(frozen=True)
class PauliInputs:
    """Per-call input data; original terms and weights precede deduplication.

    No cache is kept across calls on mutable PennyLane/PauLie objects.
    """

    terms: tuple[PauliWord, ...]
    weights: tuple[float, ...]
    n_qubits: int

    @property
    def words(self):
        words = list(dict.fromkeys(w for w, c in zip(self.terms, self.weights) if c != 0))
        if not words:
            raise ValueError("At least one nonzero Pauli generator is required.")
        return words

    def collection(self):
        """Build one canonical native collection, preserving padding and bit endianness."""
        return PauliStringCollection([
            pauli_word_to_string(word, self.n_qubits) for word in self.words
        ])

    def hamiltonian_terms(self, coefficients):
        """Apply external weights in input order, then aggregate identical words."""
        if coefficients is None:
            coefficients = np.ones(len(self.terms))
        else:
            coefficients = real_coefficients(coefficients, len(self.terms))
        contributions = {}
        for word, factor, coefficient in zip(self.terms, self.weights, coefficients):
            if factor != 0:
                weighted = factor * coefficient
                if not math.isfinite(weighted):
                    raise ValueError("Combined Hamiltonian coefficients must be finite.")
                contributions.setdefault(word, []).append(weighted)
        if not contributions:
            raise ValueError("At least one nonzero Pauli generator is required.")
        try:
            weights = [math.fsum(values) for values in contributions.values()]
        except OverflowError as exc:
            raise ValueError("Combined Hamiltonian coefficients must be finite.") from exc
        return list(contributions), weights


def prepare_pauli_inputs(generators, n_qubits=None):
    """Read generators once, including wires on explicit Identity operators."""
    words, weights = [], []
    width = 0
    for gen in _materialize_generators(generators):
        weight = 1.0
        if isinstance(gen, PauliWord):
            word = gen
        elif isinstance(gen, (str, PauliString)):
            word = pauli_string_to_word(gen)
            width = max(width, len(gen))
        elif getattr(gen, "pauli_rep", None) is not None:
            # pauli_rep drops Identity wires. Capture them before inspecting the word,
            # also for intrinsic zero terms which still declare the register width.
            width = max(width, _wire_width(gen.wires))
            if len(gen.pauli_rep) != 1:
                raise ValueError(
                    "Each generator must contain a single Pauli term; sums and zero "
                    "operators are not supported. Supply individual Pauli terms and coefficients."
                )
            word, weight = next(iter(gen.pauli_rep.items()))
            weight = finite_real_scalar(weight, "Pauli coefficient")
        else:
            raise TypeError("Generators must be Pauli strings, PauliWords or single Pauli operators.")
        width = max(width, _word_width(word))
        words.append(word)
        weights.append(weight)
    n_qubits = max(width, 1) if n_qubits is None else positive_integer(n_qubits, "n_qubits")
    if width > n_qubits:
        raise ValueError(f"Generators require {width} qubits, but n_qubits={n_qubits}.")
    return PauliInputs(tuple(words), tuple(weights), n_qubits)


def as_pauli_words(generators, n_qubits: int | None = None) -> list[PauliWord]:
    """Normalize single Pauli terms, preserving order and removing duplicates.

    Supports strings, PauLie strings/collections, PennyLane PauliWords and real
    multiples of single Pauli operators. Sums are rejected. Nonzero weights do
    not change the generator algebra. n_qubits, when given, validates the width.
    """
    return prepare_pauli_inputs(generators, n_qubits).words


def as_pauli_collection(generators, n_qubits: int | None = None):
    """Return a deduplicated, padded, big-endian PauLie collection.

    Re-encoding native strings handles little/mixed-endian inputs independently
    of the endian assumptions in older pauliebits C extensions.
    """
    return prepare_pauli_inputs(generators, n_qubits).collection()
