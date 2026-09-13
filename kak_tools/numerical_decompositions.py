"""Matrix-level KAK decompositions of the classical Cartan types (paper App. E).

Every ``*_kak`` returns ``k1, a, k2`` with ``k1 @ a @ k2`` equal to the input,
the K factors in the symmetric subgroup and ``a`` in the Cartan subgroup of the
type. ``validate=True`` re-checks those properties and raises ValueError.
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag, cossin, det, eig, schur

from ._symplectic_csd import (
    _pivoted_columns,
    _project_complement,
    compact_symplectic_csd,
)
from ._validation import integer, require, require_square


def J_n(n):
    eye = np.eye(n)
    z = np.zeros((n, n))
    return np.block([[z, eye], [-eye, z]])


def _check_kak(g, k1, a, k2, *, real=False, J=None, blocks=None, delta=None, atol=1e-6):
    """Raise ValueError unless ``k1 @ a @ k2`` reconstructs ``g`` with K factors in the subgroup.

    ``real`` demands real K factors, ``J`` symplectic ones and ``blocks`` (index
    sets) K factors that do not mix the sectors. ``delta`` additionally checks
    the horizontal decomposition ``delta = k1 @ a @ a @ k1^dagger``. The
    tolerance is loose because the eigenbases lose accuracy for n > 75.
    """
    dim = len(g)
    for k in (k1, k2):
        if real:
            require(np.allclose(k.imag, 0.0, atol=atol), "The K factors are not real.")
        require(np.allclose(k @ k.conj().T, np.eye(dim), atol=atol), "The K factors are not unitary.")
        if J is not None:
            require(np.allclose(J @ k.conj() @ J.T, k, atol=atol), "The K factors are not symplectic.")
        for i, first in enumerate(blocks or []):
            for second in blocks[i + 1 :]:
                require(
                    np.allclose(k[np.ix_(first, second)], 0.0, atol=atol)
                    and np.allclose(k[np.ix_(second, first)], 0.0, atol=atol),
                    "The K factors mix the sectors.",
                )
    if delta is not None:
        require(
            np.allclose(delta, k1 @ a @ a @ k1.conj().T, atol=atol),
            "The Cartan factor is not a horizontal decomposition of delta.",
        )
    require(np.allclose(g, k1 @ a @ k2, atol=atol), "K1 A K2 does not reconstruct the input.")


def _check_cs(f, p, q, atol=1e-6):
    """Raise ValueError unless ``f`` is in CS(p, q): plane ``(i, max(p, q) + i)`` turns by theta_i."""
    r, s = min(p, q), max(p, q)
    cosine, sine = np.diag(f[:r, :r]), np.diag(f[:r, s:])
    expected = np.eye(p + q, dtype=f.dtype)
    expected[:r, :r] = expected[s:, s:] = np.diag(cosine)
    expected[:r, s:], expected[s:, :r] = np.diag(sine), -np.diag(sine)
    require(
        np.allclose(f, expected, atol=atol) and np.allclose(cosine**2 + sine**2, 1.0, atol=atol),
        "The Cartan factor is not a cosine-sine matrix.",
    )


def _partition_sizes(p, q, kind):
    """Return the partition sizes as ints, rejecting bools, negatives and non-integers."""
    try:
        return integer(p, "p", minimum=0), integer(q, "q", minimum=0)
    except ValueError:
        raise ValueError(f"{kind} partition sizes p and q must be nonnegative integers.") from None


def _diagonal_blocks(g, kind):
    """Return the two diagonal blocks of the block-diagonal input of a doubled type."""
    n, remainder = divmod(len(g), 2)
    require(remainder == 0, f"{kind} requires an even-dimensional block-diagonal input.")
    require(
        np.allclose(g[:n, n:], 0.0) and np.allclose(g[n:, :n], 0.0),
        f"{kind} requires a block-diagonal input.",
    )
    return g[:n, :n], g[n:, n:]


def _eigenbasis(mat, count, real, partner=None):
    """Select ``count`` orthonormal eigenvectors of ``mat`` by largest-residual pivoting.

    For a symmetric unitary matrix the real and imaginary parts of every
    eigenvector are real eigenvectors for the same eigenvalue (Lemma 14), so a
    real basis is drawn from all those parts at once instead of deciding per
    eigenvector whether it is proportional to its conjugate. ``partner``
    completes each choice by its symplectic companion (Lemma 15), which is an
    eigenvector as well. Returns the eigenvalues and unit vectors of the choices.
    """
    eigvals, eigvecs = eig(mat)
    if real:
        eigvecs = np.concatenate([eigvecs.real, eigvecs.imag], axis=1)
    indices, vectors = _pivoted_columns(eigvecs, count, partner)
    return eigvals[np.asarray(indices) % len(mat)], vectors


def real_eig(mat):
    """Real orthogonal eigenbasis of a symmetric unitary matrix (Lemma 14)."""
    eigvals, vectors = _eigenbasis(mat, len(mat), real=True)
    return eigvals, np.column_stack(vectors)


def sympl_eig(mat, J, subspace):
    """Symplectic eigenbasis ``(J v*, v)`` of a unitary matrix (Lemma 15).

    ``subspace="horizontal"`` implies that the matrix is skew-symplectic, so
    ``J v*`` shares the eigenvalue of ``v``; ``subspace="vertical"`` implies
    that the matrix is symplectic, so ``J v*`` has the conjugate eigenvalue.
    """
    n = len(mat) // 2
    eigvals, vectors = _eigenbasis(mat, n, real=False, partner=lambda v: J @ v.conj())
    partners = eigvals.conj() if subspace == "vertical" else eigvals
    eigvecs = np.column_stack([J @ v.conj() for v in vectors] + vectors)
    return np.concatenate([partners, eigvals]), eigvecs


def sympl_real_eig_ci(mat, J):
    """Real symplectic eigenbasis ``(J v, v)`` of ``S @ S.T`` for ``S`` in Sp(n).

    Every real eigenvector ``v`` with eigenvalue ``lambda`` has the real partner
    ``J v`` with eigenvalue ``conj(lambda)`` (paper App. E, CI).
    """
    n = len(mat) // 2
    eigvals, vectors = _eigenbasis(mat, n, real=True, partner=lambda v: J @ v)
    eigvecs = np.column_stack([J @ v for v in vectors] + vectors)
    return np.concatenate([eigvals.conj(), eigvals]), eigvecs


def sympl_real_eig_diii(mat, J):
    assert mat.shape[0] % 2 == 0
    n = mat.shape[0] // 2
    eigvals, eigvecs = eig(mat)
    endpoint_tolerance = 32 * np.finfo(eigvals.real.dtype).eps * mat.shape[0]
    quadruples = np.zeros_like(eigvecs)
    quadruple_eigvals = np.zeros_like(eigvals)
    minus_one_eigvecs = []
    plus_one_eigvecs = []
    chosen = []

    def real_unit_vector(vec, basis):
        vec = _project_complement(vec, basis)
        length = norm(vec)
        if not np.isfinite(length) or length <= endpoint_tolerance:
            raise ValueError("DIII could not complete an independent real symplectic basis.")
        return vec / length

    d = 0
    while len(chosen) < 2 * n:
        if not eigvecs.shape[1]:
            raise ValueError("DIII could not complete the real symplectic basis.")
        # Near +/-1, small residuals of used eigenvectors can survive projection.
        # Complete the basis from its strongest remaining direction, not a
        # sequential residual that may only contain eigenvector roundoff.
        pivot = int(np.argmax(norm(eigvecs, axis=0)))
        eigvecs[:, [0, pivot]] = eigvecs[:, [pivot, 0]]
        eigvals[[0, pivot]] = eigvals[[pivot, 0]]
        vec0 = eigvecs[:, 0]
        _norm = norm(vec0)
        if not np.isfinite(_norm) or _norm <= endpoint_tolerance:
            raise ValueError("DIII has insufficient independent eigenvectors for a complete basis.")
        vec0 /= _norm
        eigenvalue = eigvals[0]
        if min(abs(eigenvalue - 1), abs(eigenvalue + 1)) <= endpoint_tolerance:
            # A numerically real eigenvalue may have a complex LAPACK eigenvector
            # in a repeated +/-1 space. Its real and imaginary parts belong to
            # that same real invariant space; each contributes a J-pair, not an
            # independent complex quartet. Use the stronger part for stability.
            vec0 = max((vec0.real, vec0.imag), key=norm)
            vec0 = real_unit_vector(vec0, chosen)
            vecs = [J @ vec0, vec0]

            # store real symplectic eigenvectors
            if eigenvalue.real > 0:
                plus_one_eigvecs.extend(vecs)
            else:
                minus_one_eigvecs.extend(vecs)
            remove_vecs = vecs
        else:
            if len(chosen) + 4 > 2 * n:
                raise ValueError("DIII has no room for an independent complex eigenvalue quartet.")
            overlap = np.dot(vec0, vec0)
            # Rotate the phase so real and imaginary parts are orthogonal.
            # Multiplication by the angle itself annihilates vectors at phase zero.
            new_vec = np.exp(-0.5j * np.angle(overlap)) * vec0
            vec2 = real_unit_vector(new_vec.real, chosen)
            vec3 = real_unit_vector(new_vec.imag, chosen + [vec2, J @ vec2])
            vec0 = J @ vec2
            vec1 = J @ vec3
            quadruples[:, 2 * d] = vec0
            quadruples[:, 2 * d + 1] = vec1
            quadruples[:, 2 * d + n] = vec2
            quadruples[:, 2 * d + n + 1] = vec3
            quadruple_eigvals[2 * d] = quadruple_eigvals[2 * d + n + 1] = eigenvalue
            quadruple_eigvals[2 * d + 1] = quadruple_eigvals[2 * d + n] = np.conj(eigenvalue)

            d += 1
            remove_vecs = [vec0, vec1, vec2, vec3]

        eigvecs = eigvecs[:, 1:]
        eigvals = eigvals[1:]
        chosen.extend(remove_vecs)
        eigvecs = _project_complement(eigvecs, remove_vecs)

    two_f = len(minus_one_eigvecs)
    if two_f % 4:
        raise ValueError("DIII requires the -1 eigenspace to have dimension divisible by four.")
    f = two_f // 2

    main_diag = np.real(quadruple_eigvals)
    if f > 0:
        minus_one_eigvecs = np.stack(minus_one_eigvecs)
        quadruples[:, 2 * d : 2 * d + f] = minus_one_eigvecs[::2].T
        quadruples[:, 2 * d + n : 2 * d + f + n] = minus_one_eigvecs[1::2].T
        main_diag[2 * d : 2 * d + f] = -1
        main_diag[2 * d + n : 2 * d + f + n] = -1
    if n - (2 * d + f) > 0:
        plus_one_eigvecs = np.stack(plus_one_eigvecs)
        quadruples[:, 2 * d + f : n] = plus_one_eigvecs[::2].T
        quadruples[:, 2 * d + n + f : 2 * n] = plus_one_eigvecs[1::2].T
        main_diag[2 * d + f : n] = 1
        main_diag[2 * d + f + n : 2 * n] = 1

    upper_diag = -np.imag(quadruple_eigvals)
    if n % 2:
        upper_diag[1:n:2] = 0
        upper_diag[n + 1 :: 2] = 0
    else:
        upper_diag[1::2] = 0
    upper_diag = upper_diag[: 2 * n - 1]
    mu = np.diag(main_diag) + np.diag(upper_diag, k=1) - np.diag(upper_diag, k=-1)

    if not np.allclose(quadruples.T @ quadruples, np.eye(2 * n), atol=4 * endpoint_tolerance, rtol=0):
        raise ValueError("DIII could not form an orthonormal real symplectic basis.")

    return mu, quadruples


def _schur_blocks(t):
    """Split a raw real Schur form into its 1x1 and 2x2 diagonal blocks.

    LAPACK leaves the subdiagonal exactly zero outside 2x2 blocks, so the test
    is exact even when a small rotation's cosine rounds to one. It is not valid
    after permuting the matrix, which can move upper-triangle roundoff below
    the diagonal.
    """
    blocks, i = [], 0
    while i < len(t):
        size = 2 if i + 1 < len(t) and t[i + 1, i] != 0 else 1
        blocks.append(list(range(i, i + size)))
        i += size
    return blocks


def schur_sqrt(u):
    """Real square root of a block-diagonal Schur matrix, halving every rotation angle.

    A nonzero subdiagonal entry marks a 2x2 rotation block, a -1 axis pairs with
    the following axis as a rotation by pi, and +1 axes are their own root.
    """
    sqrt = np.copy(u)
    i = 0
    while i < len(u):
        if i + 1 < len(u) and (u[i + 1, i] != 0 or u[i, i] < 0):
            # atan2 also covers exact +/-I and preserves nearby nonzero angles.
            theta = np.arctan2(u[i, i + 1], u[i, i]) / 2
            sqrt[i : i + 2, i : i + 2] = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            i += 2
        else:
            i += 1
    require(np.allclose(sqrt @ sqrt, u), "The Schur matrix has an unpaired -1 axis or coupling between blocks.")
    return sqrt


def _cs_decomposition(u, p, q):
    """CSD ``u = k1 @ f @ k2`` with ``f`` in CS(p, q), pairing axis i with axis max(p, q) + i.

    For p > q scipy pairs the first q axes of the p sector with the q sector;
    exchanging the first q with the following p - q columns (rows / rows and
    columns) of k1 (k2 / f) moves f into our CS(p, q) (paper App. E, AIII).
    """
    # Note that the argument p of cossin is the same as for this function, but q *is not the same*.
    k1, f, k2 = cossin(u, p=p, q=p, swap_sign=True, separate=False)
    if p > q:
        k1[:, :p] = np.roll(k1[:, :p], q - p, axis=1)
        k2[:p] = np.roll(k2[:p], q - p, axis=0)
        f[:, :p] = np.roll(f[:, :p], q - p, axis=1)
        f[:p] = np.roll(f[:p], q - p, axis=0)
    return k1, f, k2


def ai_kak(u, validate=True):
    """Decompose a unitary matrix as ``O1 @ D @ O2`` with O1, O2 in SO(n) and D diagonal."""
    u = require_square(u, "AI input")
    evals, o1 = real_eig(u @ u.T)

    if det(o1) < 0:
        o1[:, 0] *= -1

    d = np.diag(np.sqrt(evals))
    o2 = np.conj(d) @ o1.T @ u
    if det(o2) < 0:
        # Instead of guaranteeing the correct determinant while taking the square root,
        # we correct it after the fact
        o2[0] *= -1
        d[0] *= -1

    if validate:
        _check_kak(u, o1, d, o2, real=True, delta=u @ u.T)

    return o1, d, o2


def aii_kak(u, validate=True):
    """Decompose a unitary matrix as ``S1 @ D @ S2`` with S1, S2 in Sp(n) and D repeat-diagonal."""
    u = require_square(u, "AII input")
    dim = len(u)
    require(dim % 2 == 0, "AII requires an even-dimensional input.")
    J = J_n(dim // 2)

    Delta = u @ J @ u.T @ J.T
    eigvals, s1 = sympl_eig(Delta, J, "horizontal")
    d = np.diag(np.sqrt(eigvals))
    s2 = np.conj(d) @ s1.conj().T @ u

    if validate:
        _check_kak(u, s1, d, s2, J=J, delta=Delta)

    return s1, d, s2


def aiii_kak(u, p, q, validate=True):
    """Decompose a unitary matrix as ``K1 @ F @ K2`` with K in U(p) x U(q) and F in CS(p, q)."""
    p, q = _partition_sizes(p, q, "AIII")
    u = require_square(u, "AIII input", size=p + q)
    if p == 0 or q == 0:
        return u, np.eye(p + q), np.eye(p + q)
    k1, f, k2 = _cs_decomposition(u, p, q)

    if validate:
        _check_kak(u, k1, f, k2, blocks=[np.arange(p), np.arange(p, p + q)])
        _check_cs(f, p, q)

    return k1, f, k2


def bdi_kak(o, p, q, validate=True):
    """Decompose a special orthogonal matrix as ``K1 @ F @ K2`` with K in SO(p) x SO(q), F in CS(p, q)."""
    p, q = _partition_sizes(p, q, "BDI")
    o = require_square(o, "BDI input", size=p + q)
    if p == 0 or q == 0:
        return o, np.eye(p + q), np.eye(p + q)
    k1, f, k2 = _cs_decomposition(o, p, q)

    # Reflect the first paired axis of every O(p) or O(q) block with determinant
    # -1 and absorb the reflections into f, which stays in CS(p, q) because the
    # four block determinants multiply to det(o) = 1 (paper Eq. E17).
    for axis, sector in ((0, slice(0, p)), (max(p, q), slice(p, None))):
        if det(k1[sector, sector]) < 0:
            k1[:, axis] *= -1
            f[axis] *= -1
        if det(k2[sector, sector]) < 0:
            k2[axis] *= -1
            f[:, axis] *= -1

    if validate:
        _check_kak(o, k1, f, k2, real=True, blocks=[np.arange(p), np.arange(p, p + q)])
        _check_cs(f, p, q)

    return k1, f, k2


def diii_kak(o, validate=True):
    """Decompose a special orthogonal matrix as ``U1 @ A @ U2`` with U in U(n), A = mu (+) mu^T."""
    o = require_square(o, "DIII input")
    dim = len(o)
    require(dim % 2 == 0, "DIII requires an even-dimensional input.")
    n = dim // 2
    J = J_n(n)

    Delta = o @ J @ o.T @ J.T
    # A_squared is mu^2 (+) mu^2^T
    A_squared, u1 = sympl_real_eig_diii(Delta, J)
    mu = schur_sqrt(A_squared[:n, :n])
    A = block_diag(mu, mu.T)
    u2 = A.T @ u1.conj().T @ o

    if validate:
        _check_kak(o, u1, A, u2, real=True, J=J, delta=Delta)

    return u1, A, u2


def ci_kak(s, validate=True):
    """Decompose a compact symplectic matrix as ``U1 @ D @ U2`` with U in U(n), D = D0 (+) D0*."""
    s = require_square(s, "CI input")
    dim = len(s)
    require(dim % 2 == 0, "CI requires an even-dimensional input.")
    n = dim // 2
    J = J_n(n)

    Delta = s @ s.T
    D_slash_squared, u1 = sympl_real_eig_ci(Delta, J)
    D = np.sqrt(D_slash_squared[:n])
    A = np.diag(np.concatenate([D, D.conj()]))
    u2 = A.conj().T @ u1.conj().T @ s

    if validate:
        _check_kak(s, u1, A, u2, real=True, J=J, delta=Delta)

    return u1, A, u2


def cii_kak(s, p, q, validate=True):
    """Decompose a compact symplectic matrix as ``K1 @ F @ K2`` for CII(p, q).

    The input has complex dimension ``2 * (p + q)`` and coordinate order
    ``[p, q, p, q]``. Each K preserves the paired p and q sectors. F contains
    two identical real cosine-sine blocks, with unused coordinates between
    their paired axes. Unequal partitions, repeated angles, zero angles and
    right angles are supported. If a partition is empty, F is identity.

    A complex CSD supplies the joint sine/cosine subspaces. Symplectic bases
    are chosen within those subspaces, and the same choice determines both
    left blocks and the final right factor. ``validate`` checks input group
    membership as well as reconstruction and the returned factor structures.
    The input array is never modified.
    """
    p, q = _partition_sizes(p, q, "CII")
    n = p + q
    s = require_square(s, "CII input")
    require(s.shape == (2 * n, 2 * n), "CII requires a square matrix of size 2 * (p + q).")
    s = np.asarray(s, dtype=complex)
    identity, J = np.eye(2 * n), J_n(n)
    if validate:
        require(np.allclose(s.conj().T @ s, identity), "CII requires a unitary input matrix.")
        require(np.allclose(J @ s.conj() @ J.T, s), "CII requires a symplectic input matrix.")
    if p == 0 or q == 0:
        return s.copy(), identity, identity.copy()

    k1, cartan, k2 = compact_symplectic_csd(s, p, q)
    if validate:
        sectors = [np.r_[0:p, n : n + p], np.r_[p:n, n + p : 2 * n]]
        _check_kak(s, k1, cartan, k2, J=J, blocks=sectors)
        require(
            np.allclose(cartan, block_diag(cartan[:n, :n], cartan[:n, :n])),
            "The CII Cartan factor does not repeat its cosine-sine block.",
        )
        _check_cs(cartan[:n, :n], p, q)
    return k1, cartan, k2


def a_kak(u, validate=True):
    """Decompose ``U (+) U'`` as ``(U1 (+) U1)(D (+) D*)(U2 (+) U2)`` with U1, U2 unitary."""
    u = require_square(u, "A input")
    u_a, u_b = _diagonal_blocks(u, "A")

    delta = u_a @ u_b.conj().T
    # delta is normal, so its complex Schur form is diagonal in a unitary basis;
    # an eigendecomposition need not return one for repeated eigenvalues.
    D_squared, u1 = schur(delta, output="complex")
    D = np.sqrt(np.diag(D_squared))
    u2 = np.diag(D) @ u1.conj().T @ u_b
    doubled_u1, doubled_u2 = block_diag(u1, u1), block_diag(u2, u2)
    doubled_D = np.diag(np.concatenate([D, D.conj()]))

    if validate:
        _check_kak(u, doubled_u1, doubled_D, doubled_u2)

    return doubled_u1, doubled_D, doubled_u2


def bd_kak(o, validate=True):
    """Decompose ``O (+) O'`` as ``(O1 (+) O1)(mu (+) mu^T)(O2 (+) O2)`` with O1, O2 in SO(n)."""
    o = require_square(o, "BD input")
    o_a, o_b = _diagonal_blocks(o, "BD")

    delta = o_a @ o_b.T
    mu_squared, o1 = schur(delta, output="real")
    # Order the raw Schur blocks as rotations, -1 axes, +1 axes, so that the
    # even number of -1 axes pairs into rotations by pi wherever LAPACK placed
    # them, and rebuild mu^2 from the blocks alone to drop the roundoff between
    # them, which a permutation could move onto the subdiagonal.
    blocks = _schur_blocks(mu_squared)
    blocks.sort(key=lambda block: 0 if len(block) == 2 else 1 if mu_squared[block[0], block[0]] < 0 else 2)
    o1 = o1[:, [i for block in blocks for i in block]]
    mu_squared = block_diag(*(mu_squared[np.ix_(block, block)] for block in blocks))

    if det(o1) < 0:
        # A coordinate reflection preserves Schur blocks, including a lone axis.
        o1[:, 0] *= -1
        mu_squared[0, :] *= -1
        mu_squared[:, 0] *= -1

    mu = schur_sqrt(mu_squared)
    o2 = mu @ o1.T @ o_b
    doubled_o1, doubled_o2 = block_diag(o1, o1), block_diag(o2, o2)
    doubled_mu = block_diag(mu, mu.T)

    if validate:
        _check_kak(o, doubled_o1, doubled_mu, doubled_o2, real=True)

    return doubled_o1, doubled_mu, doubled_o2


def c_kak(s, validate=True):
    """Decompose ``S (+) S'`` as ``(S1 (+) S1)(D (+) D^dagger)(S2 (+) S2)`` with S1, S2 in Sp(n)."""
    s = require_square(s, "C input")
    s_a, s_b = _diagonal_blocks(s, "C")
    m = len(s_a)
    require(m % 2 == 0, "C requires blocks of even dimension.")
    n = m // 2
    J = J_n(n)

    delta = s_a @ s_b.conj().T
    D_slash_squared, s1 = sympl_eig(delta, J, "vertical")
    D = np.sqrt(D_slash_squared[:n])
    D_slash = np.concatenate([D, D.conj()])
    s2 = np.diag(D_slash) @ s1.conj().T @ s_b
    doubled_s1, doubled_s2 = block_diag(s1, s1), block_diag(s2, s2)
    doubled_D_slash = np.diag(np.concatenate([D_slash, D_slash.conj()]))

    if validate:
        _check_kak(s, doubled_s1, doubled_D_slash, doubled_s2, J=block_diag(J, J))

    return doubled_s1, doubled_D_slash, doubled_s2
