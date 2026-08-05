import numpy as np
from scipy.linalg import expm

from kak_tools import (
    make_tfXY_hamiltonian_irrep,
    make_tfXY_hamiltonian_qubits,
    bdi,
    recursive_bdi,
    make_so_2n,
    make_so_2n_full_mapping_str,
    map_recursive_decomp_to_reducible,
    map_recursive_decomp_to_reducible_str,
    map_irrep_to_matrices,
    map_simple_to_irrep,
    irrep_dot,
    lie_closure_pauli_words,
)
import time


def minimal_workflow_tfXY(n, t0, coefficients="random"):
    """Run the fixed-depth Hamiltonian simulation compilation algorithm in its minimal
    form, i.e., with the highest degree of hardcoded mappings and analytical pre-computation."""

    H = make_tfXY_hamiltonian_irrep(n, coefficients)

    U = expm(t0 * H)
    recursive_decomp = recursive_bdi(U, 2 * n, validate=False, return_all=False)

    mapping = make_so_2n_full_mapping_str(n)
    pauli_decomp = map_recursive_decomp_to_reducible_str(
        recursive_decomp,
        mapping,
        time=t0,
        tol=None,
    )
    return pauli_decomp


def complete_workflow_tfXY(n, t0, coefficients="random"):
    H, generators, coeffs = make_tfXY_hamiltonian_qubits(n, coefficients=coefficients)

    # start = time.process_time()
    algebra = lie_closure_pauli_words(generators, verbose=False)
    # end = time.process_time()
    # print(f"Close algebra: {end-start}")

    n_so = 2 * n  # The "n" in so(n)
    so_dim = (n_so**2 - n_so) // 2

    # start = time.process_time()
    mapping, signs = map_simple_to_irrep(
        algebra, horizontal_ops=generators, n=n_so, invol_type="BDI"
    )
    H_irrep = irrep_dot(coeffs, generators, mapping, signs, n=n_so, invol_type="BDI")
    # end = time.process_time()
    # print(f"Map algebra and H: {end-start}")

    # start = time.process_time()
    U = expm(t0 * H_irrep)
    recursive_decomp = recursive_bdi(U, n_so, validate=False, return_all=False)
    # end = time.process_time()
    # print(f"Matrix exp and recursive decomp: {end-start}")

    # start = time.process_time()
    pauli_decomp = map_recursive_decomp_to_reducible(
        recursive_decomp, mapping, signs, time=t0, validate=False
    )
    # end = time.process_time()
    # print(f"Mapping decomp back: {end-start}")
    return pauli_decomp


def workflow_tfXY_known_algebra(n, t0, coefficients="random"):
    H, generators, coeffs = make_tfXY_hamiltonian_qubits(n, coefficients=coefficients)

    # start = time.process_time()
    algebra = make_so_2n(n)
    # end = time.process_time()
    # print(f"Close algebra: {end-start}")

    n_so = 2 * n  # The "n" in so(n)
    so_dim = (n_so**2 - n_so) // 2

    # start = time.process_time()
    mapping, signs = map_simple_to_irrep(
        algebra, horizontal_ops=generators, n=n_so, invol_type="BDI"
    )
    H_irrep = irrep_dot(coeffs, generators, mapping, signs, n=n_so, invol_type="BDI")
    # end = time.process_time()
    # print(f"Map algebra and H: {end-start}")

    # start = time.process_time()
    U = expm(t0 * H_irrep)
    recursive_decomp = recursive_bdi(U, n_so, validate=False, return_all=False)
    # end = time.process_time()
    # print(f"Matrix exp and recursive decomp: {end-start}")

    # start = time.process_time()
    pauli_decomp = map_recursive_decomp_to_reducible(
        recursive_decomp, mapping, signs, time=t0, validate=False
    )
    # end = time.process_time()
    # print(f"Mapping decomp back: {end-start}")
    return pauli_decomp


def paulie_workflow(generators, coefficients=None, t0=1.0, validate=True):
    """Model-agnostic counterpart of ``complete_workflow_tfXY``.

    ``complete_workflow_tfXY`` hard-codes the algebra of the transverse-field XY model
    (``n_so = 2 * n``, ``invol_type="BDI"``). This version asks PauLie to classify the
    dynamical Lie algebra of whatever generators it is given and configures the
    decomposition from that answer, so it works for any generator set whose DLA PauLie
    identifies as an ``so(n)``.

    Args:
        generators: Hamiltonian terms, as Pauli words, Pauli strings such as ``"XXII"``,
            or a PauLie ``PauliStringCollection``.
        coefficients (Sequence[float], optional): Hamiltonian coefficients. Defaults to
            all ones.
        t0 (float): Evolution time.
        validate (bool): Whether to check that the Pauli rotations recompose the unitary.

    Returns:
        list: ``(PauliWord, angle, kind)`` triples, matching the output format of
        ``complete_workflow_tfXY``.
    """
    from .paulie_bridge import kak_decomposition

    result = kak_decomposition(generators, coefficients, time=t0, validate=validate)
    return result.pauli_rotations


def diagonalization_tfXY(n, t0, coefficients="random"):
    H = make_tfXY_hamiltonian_irrep(n, coefficients=coefficients)

    U = expm(t0 * H)
    theta = bdi(U, n, n, is_horizontal=True, validate=False, compute_u=False, compute_vh=False)

    return np.array(theta) / (2 * t0)
