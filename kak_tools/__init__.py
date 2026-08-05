"""Init file for the source files of the KAK tools."""

from .pauli_dlas import (
    is_int,
    identify_algebra,
    split_pauli_algebra,
    get_simple_dim,
    lie_closure_pauli_words,
    anticom_graph_pauli,
)
from .map_to_irrep import (
    map_simple_to_irrep,
    map_irrep_to_matrices,
    map_matrix_to_reducible,
    irrep_dot,
    make_signs,
    make_so_2n,
    make_so_2n_full_mapping,
    make_so_2n_full_mapping_str,
    make_so_2n_horizontal_mapping,
    make_tfXY_hamiltonian_irrep,
    make_tfXY_hamiltonian_qubits,
)
from .dense_cartan import (
    bdi,
    recursive_bdi,
    group_matrix_to_reducible,
    map_recursive_decomp_to_reducible,
    map_recursive_decomp_to_matrices,
    map_recursive_decomp_to_reducible_str,
    round_mult_recursive_decomp_str,
)
from .numerical_decompositions import (
    a_kak,
    ai_kak,
    aii_kak,
    aiii_kak,
    bd_kak,
    bdi_kak,
    diii_kak,
    c_kak,
    ci_kak,
    cii_kak,  # to do
    sympl_eig,
)

# Bridge to PauLie (https://github.com/QPauLie/PauLie), which classifies the dynamical
# Lie algebra of a Pauli generator set exactly.
from .paulie_bridge import (
    DLAComponent,
    DLAInfo,
    KAKResult,
    as_pauli_collection,
    as_pauli_words,
    classify_dla,
    dla_pauli_basis,
    kak_decomposition,
    labelled_matrix_basis,
    map_dla_to_irrep,
    pauli_string_to_word,
    pauli_word_to_string,
    reconstruct_from_pauli_rotations,
)
