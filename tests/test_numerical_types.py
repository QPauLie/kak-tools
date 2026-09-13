"""Table-driven checks of every numerical Cartan type and its input contracts."""

from functools import partial

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.stats import unitary_group

from kak_tools import numerical_decompositions as nd
from checks import (
    assert_blocks, assert_close, assert_cossin, assert_diagonal, assert_embedded, assert_orthogonal,
    assert_repeat, assert_skew_schur, assert_sympl_blocks, assert_sympl_cossin, assert_symplectic,
    assert_symplectic_diagonal, assert_unitary, random_group_element,
)

# kind: (group, doubled input, K check, Cartan-factor check); symplectic matrices of "size" n are 2n x 2n.
UNPARTITIONED = {
    "a": ("unitary", True, assert_repeat, partial(assert_repeat, check=assert_diagonal, transform=np.conj)),
    "ai": ("unitary", False, assert_orthogonal, assert_diagonal),
    "aii": ("unitary", False, assert_symplectic, partial(assert_repeat, check=assert_diagonal)),
    "bd": ("orthogonal", True, partial(assert_repeat, check=assert_orthogonal), assert_skew_schur),
    "diii": ("orthogonal", False, assert_embedded, assert_skew_schur),
    "c": (
        "symplectic", True, partial(assert_repeat, check=assert_symplectic),
        partial(assert_repeat, check=assert_symplectic_diagonal, transform=np.conj),
    ),
    "ci": ("symplectic", False, assert_embedded, assert_symplectic_diagonal),
}
# kind: (group, K check(x, p, q), Cartan-factor check(x, p, q))
PARTITIONED = {
    "aiii": ("unitary", lambda x, p, q: assert_blocks(x, p, assert_unitary), assert_cossin),
    "bdi": ("orthogonal", lambda x, p, q: assert_blocks(x, p, assert_orthogonal), assert_cossin),
    "cii": ("symplectic", assert_sympl_blocks, assert_sympl_cossin),
}


def _element(kind, n, rng):
    group, doubled, _, _ = UNPARTITIONED[kind]
    matrix = random_group_element(group, 2 * n if kind in {"aii", "diii"} else n, rng)
    return block_diag(matrix, random_group_element(group, n, rng)) if doubled else matrix


@pytest.mark.parametrize("n,validate", [(3, True), (4, False), (7, True), (10, False)])
@pytest.mark.parametrize("kind", sorted(UNPARTITIONED))
def test_cartan_types(kind, n, validate):
    matrix = _element(kind, n, np.random.default_rng(1700 + n))
    k1, center, k2 = getattr(nd, kind + "_kak")(matrix, validate=validate)
    assert_close(k1 @ center @ k2, matrix)
    _, _, vertical, horizontal = UNPARTITIONED[kind]
    vertical(k1)
    horizontal(center)
    vertical(k2)


@pytest.mark.parametrize(
    "p,q,validate",
    [(1, 1, True), (1, 4, True), (4, 1, False), (0, 3, True), (3, 0, False), (5, 5, True), (2, 7, False), (7, 2, True)],
)
@pytest.mark.parametrize("kind", sorted(PARTITIONED))
def test_partitioned_cartan_types(kind, p, q, validate):
    group, vertical, horizontal = PARTITIONED[kind]
    matrix = random_group_element(group, p + q, np.random.default_rng(2010 + p))
    k1, center, k2 = getattr(nd, kind + "_kak")(matrix, p, q, validate=validate)
    assert_close(k1 @ center @ k2, matrix)
    vertical(k1, p, q)
    horizontal(center, p, q)
    vertical(k2, p, q)


@pytest.mark.parametrize("kind", sorted(UNPARTITIONED))
def test_validate_rejects_elements_outside_the_group(kind):
    # Scaling leaves the block structure intact, so only validate can object, with
    # a ValueError that python -O cannot silence.
    matrix = 1.5 * _element(kind, 2, np.random.default_rng(3))
    with pytest.raises(ValueError):
        getattr(nd, kind + "_kak")(matrix, validate=True)


@pytest.mark.parametrize("kind", sorted(PARTITIONED))
def test_partitioned_validate_rejects_elements_outside_the_group(kind):
    group, _, _ = PARTITIONED[kind]
    matrix = random_group_element(group, 4, np.random.default_rng(4))
    matrix[0] *= 1.5
    with pytest.raises(ValueError):
        getattr(nd, kind + "_kak")(matrix, 1, 3, validate=True)


@pytest.mark.parametrize("kind,shape", [
    ("ai", (2, 3)), ("aii", (3, 3)), ("diii", (3, 3)), ("ci", (5, 5)), ("a", (3, 3)), ("bd", (5, 5)), ("c", (6, 6)),
])
def test_wrong_shape_raises_value_error(kind, shape):
    with pytest.raises(ValueError):
        getattr(nd, kind + "_kak")(np.ones(shape))


def test_block_diagonal_input_required_for_doubled_types():
    with pytest.raises(ValueError, match="block-diagonal"):
        nd.a_kak(unitary_group.rvs(4, random_state=0))


@pytest.mark.parametrize("kind", sorted(PARTITIONED))
@pytest.mark.parametrize("p,q", [(True, 1), (1, np.bool_(False)), (-1, 1), (1, 1.5)])
def test_partition_sizes_must_be_nonnegative_integers(kind, p, q):
    with pytest.raises(ValueError, match="nonnegative integers"):
        getattr(nd, kind + "_kak")(np.eye(4), p, q, validate=False)


@pytest.mark.parametrize("kind", ["aiii", "bdi"])
def test_partition_must_match_the_matrix_size(kind):
    with pytest.raises(ValueError, match="size 4"):
        getattr(nd, kind + "_kak")(np.eye(5), 2, 2)
