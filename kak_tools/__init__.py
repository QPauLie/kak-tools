"""KAK decomposition tools.

Recursive Cartan decompositions of dense matrices, the matrix-level KAK routines for
the classical Cartan types, and a Pauli-level BDI compiler backed by PauLie.

The public names below are imported lazily on first access (PEP 562), so the
numpy-only routines can be used without paying for PennyLane and PauLie:

>>> from kak_tools import bdi                  # loads dense_cartan only
>>> from kak_tools import kak_decomposition    # loads the PauLie bridge
"""

import importlib

__all__ = [
    # Pauli-level BDI compiler backed by PauLie (https://github.com/QPauLie/PauLie)
    "kak_decomposition",
    "KAKResult",
    "PauliRotation",
    "map_dla_to_irrep",
    "labelled_matrix_basis",
    "dla_pauli_basis",
    "as_pauli_words",
    "as_pauli_collection",
    "pauli_string_to_word",
    "pauli_word_to_string",
    "reconstruct_from_pauli_rotations",
    # Dense BDI decompositions
    "bdi",
    "recursive_bdi",
    "group_matrix_to_reducible",
    "map_recursive_decomp_to_reducible",
    # Matrix-level KAK decompositions by Cartan type
    "a_kak",
    "ai_kak",
    "aii_kak",
    "aiii_kak",
    "bd_kak",
    "bdi_kak",
    "c_kak",
    "ci_kak",
    "cii_kak",
    "diii_kak",
    # Mapping between Pauli algebras and their matrix irreps
    "map_simple_to_irrep",
    "map_irrep_to_matrices",
    "map_matrix_to_reducible",
    "irrep_dot",
    "make_signs",
    "E",
    # Pauli-word Lie closure
    "lie_closure_pauli_words",
    "split_pauli_algebra",
    "anticom_graph_pauli",
]

# Submodule that defines each public name.
_SUBMODULES = {
    "paulie_bridge": (
        "kak_decomposition",
        "KAKResult",
        "map_dla_to_irrep",
        "labelled_matrix_basis",
        "dla_pauli_basis",
        "as_pauli_words",
        "as_pauli_collection",
        "pauli_string_to_word",
        "pauli_word_to_string",
        "reconstruct_from_pauli_rotations",
    ),
    "_pauli_rotations": ("PauliRotation",),
    "dense_cartan": (
        "bdi",
        "recursive_bdi",
        "group_matrix_to_reducible",
        "map_recursive_decomp_to_reducible",
    ),
    "numerical_decompositions": (
        "a_kak",
        "ai_kak",
        "aii_kak",
        "aiii_kak",
        "bd_kak",
        "bdi_kak",
        "c_kak",
        "ci_kak",
        "cii_kak",
        "diii_kak",
    ),
    "map_to_irrep": (
        "map_simple_to_irrep",
        "map_irrep_to_matrices",
        "map_matrix_to_reducible",
        "irrep_dot",
        "make_signs",
        "E",
    ),
    "pauli_dlas": ("lie_closure_pauli_words", "split_pauli_algebra", "anticom_graph_pauli"),
}
_MODULE_OF = {name: module for module, names in _SUBMODULES.items() for name in names}


def __getattr__(name):
    try:
        module = _MODULE_OF[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # resolve each name once
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
