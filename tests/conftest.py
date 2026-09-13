"""Seeded BD inputs whose Schur form has prescribed rotation blocks and fixed axes."""

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.stats import ortho_group

from checks import rotation


@pytest.fixture
def bd_schur_matrix():
    """Factory for ``block_diag(o1, o2)`` in SO(n) x SO(n) with a designed ``delta = o1 @ o2.T``.

    ``delta`` is conjugated from rotations by ``angles`` followed by ``plus``
    fixed +1 axes and an even number ``minus`` of fixed -1 axes, so its real
    Schur form contains exactly those 2x2 blocks and 1x1 singletons.
    """

    def build(angles, plus, minus, seed):
        rng = np.random.default_rng(seed)
        delta = block_diag(*[rotation(t) for t in angles], np.eye(plus), -np.eye(minus))
        frame, o2 = (ortho_group.rvs(len(delta), random_state=rng) for _ in range(2))
        frame[0] *= np.linalg.det(frame)
        o2[0] *= np.linalg.det(o2)
        return block_diag(frame @ delta @ frame.T @ o2, o2)

    return build
