"""Fixed-depth Hamiltonian simulation of the transverse-field XY model (paper App. F).

Every workflow compiles ``exp(i * t0 * H)`` for the qubit Hamiltonian ``H`` of
:func:`kak_tools.tfxy_model.make_tfXY_hamiltonian_qubits` into ``(word, coefficient, kind)``
triples whose ordered product of ``exp(i * coefficient * word)`` factors reproduces it;
the ``a0`` coefficients are rates that are multiplied by the evolution time.
"""

import numpy as np
from scipy.linalg import expm

from .dense_cartan import (
    bdi, recursive_bdi, map_recursive_decomp_to_reducible, map_recursive_decomp_to_reducible_str,
)
from .map_to_irrep import irrep_dot, map_simple_to_irrep
from .pauli_dlas import lie_closure_pauli_words
from .tfxy_model import (
    make_so_2n, make_so_2n_full_mapping_str, make_tfXY_hamiltonian_irrep,
    make_tfXY_hamiltonian_qubits,
)


def minimal_workflow_tfXY(n, t0, coefficients="random", rng=None):
    """Run the fixed-depth Hamiltonian simulation compilation algorithm in its minimal
    form, i.e., with the highest degree of hardcoded mappings and analytical pre-computation.
    The words are the compressed strings of :func:`make_so_2n_full_mapping_str`."""

    H = make_tfXY_hamiltonian_irrep(n, coefficients, rng)

    U = expm(t0 * H)
    recursive_decomp = recursive_bdi(U, 2 * n, validate=False, return_all=False)

    mapping = make_so_2n_full_mapping_str(n)
    return map_recursive_decomp_to_reducible_str(
        recursive_decomp,
        mapping,
        time=t0,
        tol=None,
    )


def complete_workflow_tfXY(n, t0, coefficients="random", rng=None):
    H, generators, coeffs = make_tfXY_hamiltonian_qubits(n, coefficients, rng)
    algebra = lie_closure_pauli_words(generators, verbose=False)

    n_so = 2 * n  # The "n" in so(n)
    mapping, signs = map_simple_to_irrep(
        algebra, horizontal_ops=generators, n=n_so, invol_type="BDI"
    )
    H_irrep = irrep_dot(coeffs, generators, mapping, signs, n=n_so, invol_type="BDI")

    U = expm(t0 * H_irrep)
    recursive_decomp = recursive_bdi(U, n_so, validate=False, return_all=False)

    return map_recursive_decomp_to_reducible(
        recursive_decomp, mapping, signs, time=t0, validate=False
    )


def workflow_tfXY_known_algebra(n, t0, coefficients="random", rng=None):
    H, generators, coeffs = make_tfXY_hamiltonian_qubits(n, coefficients, rng)
    algebra = make_so_2n(n)

    n_so = 2 * n  # The "n" in so(n)
    mapping, signs = map_simple_to_irrep(
        algebra, horizontal_ops=generators, n=n_so, invol_type="BDI"
    )
    H_irrep = irrep_dot(coeffs, generators, mapping, signs, n=n_so, invol_type="BDI")

    U = expm(t0 * H_irrep)
    recursive_decomp = recursive_bdi(U, n_so, validate=False, return_all=False)

    return map_recursive_decomp_to_reducible(
        recursive_decomp, mapping, signs, time=t0, validate=False
    )


def diagonalization_tfXY(n, t0, coefficients="random", rng=None):
    """Return the Pauli-rotation rates of the Cartan factor of ``exp(i * t0 * H)``.

    The qubit spectrum of ``H`` consists of all signed sums of these rates."""
    H = make_tfXY_hamiltonian_irrep(n, coefficients, rng)

    U = expm(t0 * H)
    theta = bdi(U, n, n, is_horizontal=True, validate=False, compute_u=False, compute_vh=False)

    return np.array(theta) / (2 * t0)
