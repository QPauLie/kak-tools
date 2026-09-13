"""Hamiltonian-level BDI factorization with time-independent Cartan rates.

Factoring the off-diagonal block of a real skew-symmetric Hamiltonian avoids
recovering a logarithm from ``exp(time * H)``. In particular, the singular values
retain their full magnitude at resonances and the same rotations work at any time.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, svd

from ._pauli_rotations import PauliRotation
from ._validation import finite_real_scalar, nonnegative_tolerance


def _cartan_matrix(theta, p, q):
    """Canonical CS rotation of BDI(p, q): plane ``(i, max(p, q) + i)`` turns by ``theta[i]``."""
    matrix = np.eye(p + q)
    first = np.arange(min(p, q))
    second = first + max(p, q)
    matrix[first, first] = matrix[second, second] = np.cos(theta)
    matrix[first, second], matrix[second, first] = np.sin(theta), -np.sin(theta)
    return matrix


def horizontal_generator_decomposition(hamiltonian, p):
    """Return ``(K, rates, planes)`` with ``H = K @ A @ K.T``.

    ``H`` must be real, skew-symmetric and horizontal for BDI(p, n-p).
    ``K`` belongs to SO(p) x SO(n-p). The only nonzero upper-triangular
    entries of ``A`` are its signed ``rates`` at the corresponding ``planes``.
    The planes are ``(i, max(p, n-p) + i)`` for ``i < min(p, n-p)``.

    The returned rates describe the Hamiltonian, not wrapped group angles.
    Both square and rectangular partitions, including rank-deficient blocks,
    use the same construction.
    """
    hamiltonian = np.asarray(hamiltonian)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("The Hamiltonian must be a square matrix.")
    if np.iscomplexobj(hamiltonian) or not np.isfinite(hamiltonian).all():
        raise ValueError("The Hamiltonian must contain finite real values.")
    hamiltonian = np.asarray(hamiltonian, dtype=float)
    n = len(hamiltonian)
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, np.integer)) or not 0 < p < n:
        raise ValueError("The BDI partition p must be an integer with 0 < p < n.")
    p = int(p)
    q = n - p
    scale = max(1.0, float(np.max(np.abs(hamiltonian))))
    tolerance = 64 * np.finfo(float).eps * max(1, n) * scale
    if not np.allclose(hamiltonian, -hamiltonian.T, rtol=0, atol=tolerance):
        raise ValueError("The Hamiltonian must be skew-symmetric.")
    if (
        np.max(np.abs(hamiltonian[:p, :p])) > tolerance
        or np.max(np.abs(hamiltonian[p:, p:])) > tolerance
    ):
        raise ValueError(f"The Hamiltonian is not horizontal for BDI({p}, {q}).")

    left, rates, right_transpose = svd(hamiltonian[:p, p:], full_matrices=True)
    right = right_transpose.T.copy()
    rank = min(p, q)
    # The canonical CSA has its unused coordinates between the paired axes.
    if q > p:
        right = right[:, list(range(rank, q)) + list(range(rank))]
    # Put both diagonal blocks in SO, carrying each reflection into one rate.
    if np.linalg.det(left) < 0:
        left[:, 0] *= -1
        rates[0] *= -1
    if np.linalg.det(right) < 0:
        right[:, q - rank] *= -1
        rates[0] *= -1

    planes = [(i, max(p, q) + i) for i in range(rank)]
    return block_diag(left, right), rates, planes


def _special_orthogonal_givens(matrix, start=0):
    """Express an SO matrix as ordered adjacent-plane rotations.

    A rotation angle ``theta`` means ``exp(theta * (E_ij - E_ji))``.
    Left Givens elimination makes the diagonal positive, including exact pi
    rotations. Inverting those eliminations in their original order gives the
    requested matrix. No matrix logarithm or angle unwrapping is needed.
    """
    remainder = np.array(matrix, dtype=float, copy=True)
    rotations = []
    n = len(remainder)
    for column in range(n - 1):
        for row in range(n - 1, column, -1):
            upper, lower = row - 1, row
            a, b = remainder[upper, column], remainder[lower, column]
            radius = np.hypot(a, b)
            if radius == 0 or (b == 0 and a >= 0):
                continue
            cosine, sine = a / radius, b / radius
            old_upper, old_lower = remainder[upper].copy(), remainder[lower].copy()
            remainder[upper] = cosine * old_upper + sine * old_lower
            remainder[lower] = -sine * old_upper + cosine * old_lower
            rotations.append(((start + upper, start + lower), -float(np.arctan2(b, a))))
    # Every elimination has determinant one, so an orthogonal input leaves
    # diag(1, ..., 1, det); a reflection or a non-orthogonal input would
    # otherwise be rebuilt silently as a different matrix.
    if not np.allclose(remainder, np.eye(n), atol=1e-10, rtol=0):
        raise ValueError("The Givens factorization requires a special orthogonal matrix.")
    return rotations


def decompose_horizontal_hamiltonian(hamiltonian, p, mapping, signs, time=1.0, tol=None):
    """Return the Pauli rotations that compile ``exp(time * H)``.

    ``mapping`` and ``signs`` use kak_tools' convention
    ``i * PauliWord -> 2 * sign * (E_ij - E_ji)``. The central ``a0`` coefficients
    are Hamiltonian rates: multiply them by the evolution time when applying the
    circuit. They are never divided by time, so a result computed at zero time
    remains reusable at any other time; ``time`` only has to keep ``time * rate``
    finite.

    The right sequence is exactly the reverse, negated left sequence. This
    preserves the inverse also in the physical Pauli representation, where
    independently factoring two equal SO matrices could choose different lifts.

    ``tol`` omits small vertical Pauli angles; this is an approximation. Cartan
    rates are retained, including zeros, so small Hamiltonian coefficients do
    not disappear when a compiled circuit is reused at longer times.
    """
    time = finite_real_scalar(time, "time")
    tol = nonnegative_tolerance(tol, "tol")

    k, rates, planes = horizontal_generator_decomposition(hamiltonian, p)
    givens = _special_orthogonal_givens(k[:p, :p])
    givens += _special_orthogonal_givens(k[p:, p:], start=p)
    if not np.isfinite(time * rates).all():
        raise ValueError("Evolution time times a Cartan rate exceeds the finite float range.")

    left = []
    for plane, angle in givens:
        coefficient = angle / (2 * signs[plane])
        if tol is None or abs(coefficient) >= tol:
            left.append(PauliRotation(mapping[plane], coefficient, "k1"))
    cartan = [
        PauliRotation(mapping[plane], float(rate / (2 * signs[plane])), "a0")
        for plane, rate in zip(planes, rates, strict=True)
    ]
    right = [PauliRotation(word, -coefficient, "k2") for word, coefficient, _ in reversed(left)]
    return left + cartan + right
