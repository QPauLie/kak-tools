"""Reconstruct products of Pauli rotations in a matrix representation."""

import math
from typing import NamedTuple

import numpy as np
from scipy.linalg import expm

from ._validation import finite_real_scalar as _finite_real_scalar, positive_integer


class PauliRotation(NamedTuple):
    """One factor ``exp(i * coefficient * word)`` of a compiled evolution.

    ``kind`` is ``"k1"`` or ``"k2"`` for the vertical factors and ``"a0"`` for
    the central Cartan factors, whose coefficients are rates that the caller
    multiplies by the evolution time. Unpacks like a plain ``(word, coefficient, kind)``
    tuple.
    """

    word: object
    coefficient: float
    kind: str


def _rotation_plane(matrix):
    """Recognize an exact real multiple of E_ij - E_ji, without tolerances."""
    if not np.isrealobj(matrix):
        return None
    rows, cols = np.nonzero(matrix)
    if len(rows) != 2:
        return None
    i, j = int(rows[0]), int(cols[0])
    if i == j or rows[1] != j or cols[1] != i:
        return None
    coefficient = matrix[i, j].item()
    if matrix[j, i].item() != -coefficient:
        return None
    return i, j, float(coefficient)


def reconstruct_from_pauli_rotations(
    pauli_rotations, algebra_basis, irrep_size, time=None
):
    """Recompose a matrix from ordered ``(word, angle, kind)`` rotations.

    ``time`` scales only rotations of kind ``"a0"``, when supplied. Real
    antisymmetric basis matrices supported on a single coordinate plane are
    exponentiated analytically by updating two columns of the accumulated
    matrix. All other finite basis matrices use :func:`scipy.linalg.expm`.

    Raises:
        ValueError: If the size is not a positive integer, an angle or time is
            not a finite real scalar, or a used basis matrix has the wrong
            shape or contains nonfinite or nonnumeric values.
    """
    irrep_size = positive_integer(irrep_size, "irrep_size")
    if time is not None:
        time = _finite_real_scalar(time, "time")

    out = np.eye(irrep_size)
    prepared = {}
    for word, angle, kind in pauli_rotations:
        angle = _finite_real_scalar(angle, "angle")
        scale = time if kind == "a0" and time is not None else 1.0
        scaled_angle = angle * scale
        if not math.isfinite(scaled_angle):
            raise ValueError("The scaled rotation angle must be finite.")

        if word not in prepared:
            matrix = np.asarray(algebra_basis[word])
            if matrix.shape != (irrep_size, irrep_size):
                raise ValueError(
                    f"Basis matrix for {word!r} has shape {matrix.shape}; expected "
                    f"{(irrep_size, irrep_size)}."
                )
            if matrix.dtype.kind not in "biufc" or not np.isfinite(matrix).all():
                raise ValueError(
                    f"Basis matrix for {word!r} must contain finite numeric values."
                )
            prepared[word] = matrix, _rotation_plane(matrix)
        matrix, plane = prepared[word]

        if scaled_angle == 0.0:
            continue
        if plane is None:
            with np.errstate(over="ignore", invalid="ignore"):
                exponent = matrix * scaled_angle
            if not np.isfinite(exponent).all():
                raise ValueError("The scaled basis matrix must contain finite values.")
            out = out @ expm(exponent)
            continue

        i, j, coefficient = plane
        theta = coefficient * scaled_angle
        if not math.isfinite(theta):
            raise ValueError("The scaled rotation angle must be finite.")
        cosine, sine = np.cos(theta), np.sin(theta)
        column_i = out[:, i].copy()
        column_j = out[:, j].copy()
        out[:, i] = cosine * column_i - sine * column_j
        out[:, j] = sine * column_i + cosine * column_j
    return out
