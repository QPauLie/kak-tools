"""Cosine-sine decomposition with coupled compact-symplectic bases.

The ordinary complex CSD resolves small sine and cosine values without forming
normal equations. Its right singular subspaces are then given symplectic bases;
the images of these bases fix both left factors together. The final right factor
is derived from that common choice, including at zero and right-angle limits.
"""

import numpy as np
from scipy.linalg import cossin


def _partner(vector):
    """The second column of a symplectic pair: -J conjugate(vector)."""
    n = len(vector) // 2
    return np.concatenate((-vector[n:].conj(), vector[:n].conj()))


def _project_complement(vectors, chosen):
    """Reorthogonalize against complete symplectic pairs without mutating inputs."""
    residual = vectors.copy()
    if chosen:
        basis = np.stack(chosen, axis=1)
        for _ in range(2):
            residual -= basis @ (basis.conj().T @ residual)
    return residual


def _pivoted_columns(candidates, count, partner=None):
    """Choose ``count`` orthonormal vectors from the candidate columns by global pivoting.

    Each choice is the largest remaining residual, reorthogonalized against the
    full basis and completed by ``partner(vector)`` when given; all candidate
    residuals are then updated only against the new pair to keep cubic cost.
    Pivoting on the largest residual means that a candidate which is mostly
    roundoff, such as the imaginary part of a nearly real eigenvector, is never
    normalized into the basis while an independent direction remains. Returns
    the pivot column indices and the chosen unit vectors.
    """
    tolerance = 64 * np.finfo(float).eps * max(1, len(candidates))
    chosen, indices, vectors = [], [], []
    residuals = candidates.copy()
    for _ in range(count):
        lengths = np.linalg.norm(residuals, axis=0)
        pivot = int(np.argmax(lengths))
        if not lengths[pivot] > tolerance:
            raise ValueError("The candidate vectors do not span enough independent directions.")
        vector = _project_complement(residuals[:, pivot], chosen)
        vector /= np.linalg.norm(vector)
        pair = [vector] if partner is None else [vector, partner(vector)]
        indices.append(pivot)
        vectors.append(vector)
        chosen.extend(pair)
        residuals = _project_complement(residuals, pair)
    return indices, vectors


def _paired_right_basis(candidates):
    """Choose a fixed number of symplectic pairs by global column pivoting.

    In exact arithmetic, partner formation and projection preserve CS subspaces.
    For unitary candidates of size 2p, after k complete pairs the squared
    residual norms sum to 2(p-k). The largest remaining norm is therefore at
    least sqrt((p-k)/p); no angle-gap or multiplicity threshold is needed.
    """
    _, first_columns = _pivoted_columns(candidates, len(candidates) // 2, _partner)
    return np.stack(first_columns, axis=1)


def _paired_left_basis(images, slots, n):
    """Normalize constrained images first, then complete their symplectic complement.

    Large images are processed before small ones. Reorthogonalizing a small
    image can then affect reconstruction only through its small CS coefficient.
    The image cutoff scales with the strongest image, so an entirely weak block
    is not discarded solely for its absolute scale. Numerically zero images leave
    a free direction, which is filled by a projected coordinate pair.
    """
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    chosen, filled = [], set()
    strengths = np.linalg.norm(images, axis=0)
    tolerance = 64 * np.finfo(float).eps * max(1, 2 * n)
    image_tolerance = tolerance * np.max(strengths)

    def add_pair(vector, slot):
        companion = _partner(vector)
        out[:, slot], out[:, slot + n] = vector, companion
        chosen.extend((vector, companion))
        filled.add(slot)

    for j in np.argsort(-strengths, kind="stable"):
        vector = _project_complement(images[:, j], chosen)
        length = np.linalg.norm(vector)
        if length > image_tolerance:
            add_pair(vector / length, slots[j])

    for slot in range(n):
        if slot in filled:
            continue
        residuals = _project_complement(np.eye(2 * n, dtype=complex), chosen)
        lengths = np.linalg.norm(residuals, axis=0)
        pivot = int(np.argmax(lengths))
        if lengths[pivot] <= tolerance:
            raise ValueError("Could not complete the symplectic CS basis.")
        add_pair(residuals[:, pivot] / lengths[pivot], slot)
    return out


def compact_symplectic_csd(matrix, p, q):
    """Return K1, F, K2 for positive CII(p, q) partitions.

    Input and output use the coordinate order [p, q, p, q], with F the real
    repeated CS block in that order. Each K preserves the p/p and q/q sectors.
    Shapes and group membership are validated by the public cii_kak wrapper.
    """
    n, r = p + q, min(p, q)
    p_indices = np.r_[0:p, n:n + p]
    q_indices = np.r_[p:n, n + p:2 * n]
    order = np.r_[p_indices, q_indices]
    grouped = matrix[np.ix_(order, order)]
    top = grouped[:2 * p, :2 * p]
    bottom = grouped[2 * p:, :2 * p]

    _, _, (right_p, _) = cossin(
        grouped, p=2 * p, q=2 * p, separate=True, swap_sign=True, compute_u=False
    )
    candidates = right_p.conj().T
    right_basis = _paired_right_basis(candidates)
    top_images, bottom_images = top @ right_basis, bottom @ right_basis
    angles = np.arctan2(
        np.linalg.norm(bottom_images, axis=0),
        np.linalg.norm(top_images, axis=0),
    )
    order = np.argsort(-angles, kind="stable")
    angles = angles[order]
    top_images, bottom_images = top_images[:, order], bottom_images[:, order]
    left_p = _paired_left_basis(top_images, range(p), p)
    left_q = _paired_left_basis(-bottom_images[:, :r], range(q - r, q), q)

    k1 = np.zeros_like(matrix, dtype=complex)
    k1[np.ix_(p_indices, p_indices)] = left_p
    k1[np.ix_(q_indices, q_indices)] = left_q
    cartan = np.eye(2 * n)
    for i, angle in enumerate(angles[:r]):
        cosine, sine = np.cos(angle), np.sin(angle)
        for offset in (0, n):
            plane = np.array([i, max(p, q) + i]) + offset
            cartan[np.ix_(plane, plane)] = [[cosine, sine], [-sine, cosine]]

    k2 = cartan.T @ k1.conj().T @ matrix
    return k1, cartan, k2
