"""Shared numerical input checks without Pauli or quantum-framework dependencies."""

import numpy as np


def require(condition, message):
    """Raise ValueError unless ``condition`` holds; unlike ``assert`` it survives ``python -O``."""
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=None, maximum=None):
    """Return ``int(value)`` for a genuine integer (bools excluded) within the given bounds."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def require_square(matrix, name="matrix", size=None):
    """Return ``matrix`` as an array after checking it is square, numeric and finite."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if size is not None and matrix.shape[0] != size:
        raise ValueError(f"{name} must have size {size}, got {matrix.shape[0]}.")
    if matrix.dtype.kind not in "biufc" or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite numeric values.")
    return matrix


def finite_real_scalar(value, name):
    """Accept finite numeric scalars, including complex values with zero imaginary part."""
    message = f"{name} must be a finite real scalar."
    try:
        scalar = np.asarray(value)
        if scalar.ndim != 0 or scalar.dtype.kind not in "biufc":
            raise ValueError(message)
        if np.iscomplexobj(scalar) and scalar.imag != 0:
            raise ValueError(message)
        result = float(scalar.real)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(result):
        raise ValueError(message)
    return result


def nonnegative_tolerance(value, name):
    """Validate a finite nonnegative tolerance, or preserve an explicit None."""
    if value is None:
        return None
    result = finite_real_scalar(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def resolve_bdi_partition(size, options=None):
    """Resolve positive BDI(p, q) sizes, inferring either omitted side."""
    size = positive_integer(size, "BDI size")
    options = {} if options is None else dict(options)
    if set(options) - {"p", "q"}:
        raise ValueError("BDI invol_kwargs may only contain p and q.")
    options = {
        name: positive_integer(value, f"BDI partition {name}")
        for name, value in options.items()
    }
    p = options.get("p", size - options["q"] if "q" in options else size // 2)
    q = options.get("q", size - p)
    if p < 1 or q < 1 or p + q != size:
        raise ValueError(f"BDI requires positive integer p and q with p + q = {size}.")
    return p, q


def real_coefficients(values, count):
    """Validate one finite real coefficient for every original input term."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != count:
        raise ValueError(f"Expected one-dimensional coefficients for {count} input generators.")
    return np.asarray([finite_real_scalar(value, "coefficient") for value in values])
