"""Dense implementations of Cartan decompositions."""

import numpy as np
from itertools import combinations
from pennylane.pauli import PauliSentence
from scipy.linalg import block_diag, cossin, expm, det


def _cosine_resolved_svd(a, b, d):
    """Keep weak sine directions separate from opposite-cosine eigenspaces.

    Horizontal orthogonality implies ``a @ b == b @ d``. Consequently,
    eigenspaces of the symmetric cosine blocks with disjoint spectra cannot
    couple through ``b``. Resolving those spaces before its SVD avoids an
    O(eps / small_sine) rotation between near-zero and near-pi planes.

    Use gaps in the actual combined spectrum, not a fixed sign threshold:
    nearly repeated cosine eigenvalues must remain together. Within a cosine
    interval of width at most one, the sine SVD still resolves small angles
    near the identity without relying on their almost identical cosines.
    """
    ca, left_cosine = np.linalg.eigh((a + a.T) / 2)
    cd, right_cosine = np.linalg.eigh((d + d.T) / 2)
    combined = np.sort(np.concatenate((ca, cd)))
    if combined[-1] - combined[0] <= 1.0:
        return np.linalg.svd(b, full_matrices=True)

    def spectral_intervals(values, lower=-np.inf, upper=np.inf):
        if values[-1] - values[0] <= 1.0:
            yield lower, upper
            return
        split = int(np.argmax(np.diff(values))) + 1
        boundary = (values[split - 1] + values[split]) / 2
        yield from spectral_intervals(values[:split], lower, boundary)
        yield from spectral_intervals(values[split:], boundary, upper)

    paired, left_free, right_free = [], [], []
    for lower, upper in spectral_intervals(combined):
        left_space = left_cosine[:, (ca >= lower) & (ca < upper)]
        right_space = right_cosine[:, (cd >= lower) & (cd < upper)]
        local_left, local_sine, local_right_t = np.linalg.svd(
            left_space.T @ b @ right_space, full_matrices=True
        )
        local_left = left_space @ local_left
        local_right = right_space @ local_right_t.T
        paired.extend(
            (value, local_left[:, i], local_right[:, i])
            for i, value in enumerate(local_sine)
        )
        left_free.extend(local_left[:, i] for i in range(len(local_sine), local_left.shape[1]))
        right_free.extend(local_right[:, i] for i in range(len(local_sine), local_right.shape[1]))

    paired.sort(key=lambda item: item[0], reverse=True)
    left = np.column_stack([item[1] for item in paired] + left_free)
    right = np.column_stack([item[2] for item in paired] + right_free)
    return left, np.asarray([item[0] for item in paired]), right.T


def _horizontal_bdi(u, p, q):
    """Diagonalize a horizontal group element, including repeated CS angles.

    The off-diagonal block determines the sine planes. Inside repeated singular
    subspaces its symmetric diagonal blocks determine the cosine signs. The sine
    kernel is resolved separately, including pi rotations and fixed directions.
    Starting with the sine block avoids ill-conditioned cosine eigenvectors near
    the identity, where distinct angles have nearly indistinguishable cosines.
    """
    u = np.asarray(u)
    n = p + q
    if u.shape != (n, n) or min(p, q) < 1:
        raise ValueError("BDI requires positive block sizes and a matrix of size p + q.")
    if np.iscomplexobj(u):
        if not np.allclose(u.imag, 0.0, atol=1e-12):
            raise ValueError("BDI requires a real special orthogonal matrix.")
        u = u.real
    # Verify the horizontal symmetry independently of the optional diagnostics:
    # setting the second factor to the first factor's transpose is valid only here.
    a, b, d = u[:p, :p], u[:p, p:], u[p:, p:]
    orthogonality_error = u.T @ u - np.eye(n)
    if not (
        np.allclose(orthogonality_error, 0.0, atol=1e-10, rtol=0.0)
        and np.allclose(a, a.T, atol=1e-10, rtol=1e-10)
        and np.allclose(d, d.T, atol=1e-10, rtol=1e-10)
        and np.allclose(u[p:, :p], -b.T, atol=1e-10, rtol=1e-10)
    ):
        raise ValueError(
            "u is not the exponential of a horizontal BDI element; "
            "pass is_horizontal=False for a general group element."
        )

    left, sine, right_t = _cosine_resolved_svd(a, b, d)
    right = right_t.T
    spectral_tol = 64 * np.finfo(float).eps * n
    nonzero = int(np.count_nonzero(sine > spectral_tol))
    left_paired, right_paired, angles = [], [], []
    start = 0
    while start < nonzero:
        end = start + 1
        while end < nonzero and sine[start] - sine[end] <= spectral_tol:
            end += 1
        lg, rg = left[:, start:end], right[:, start:end]
        cosine_block = (lg.T @ a @ lg + rg.T @ d @ rg) / 2
        cosine, rotation = np.linalg.eigh((cosine_block + cosine_block.T) / 2)
        lg, rg = lg @ rotation, rg @ rotation
        sg = np.diag(rotation.T @ np.diag(sine[start:end]) @ rotation)
        left_paired.extend(lg[:, i] for i in range(end - start))
        right_paired.extend(rg[:, i] for i in range(end - start))
        angles.extend(np.arctan2(sg, np.clip(cosine, -1.0, 1.0)))
        start = end

    # At zero sine, independently choose the -1 and +1 cosine eigenspaces.
    # The -1 multiplicities must agree: every pi rotation uses one axis per block.
    lk, rk = left[:, nonzero:], right[:, nonzero:]
    ak, dk = lk.T @ a @ lk, rk.T @ d @ rk
    ca, la = np.linalg.eigh((ak + ak.T) / 2)
    cd, rd = np.linalg.eigh((dk + dk.T) / 2)
    lk, rk = lk @ la, rk @ rd
    negative_a, negative_d = int(np.count_nonzero(ca < 0)), int(np.count_nonzero(cd < 0))
    if negative_a != negative_d:
        raise ValueError("Horizontal BDI blocks have incompatible cosine eigenspaces.")
    paired = min(len(ca), len(cd))
    left_paired.extend(lk[:, i] for i in range(paired))
    right_paired.extend(rk[:, i] for i in range(paired))
    angles.extend([np.pi] * negative_a + [0.0] * (paired - negative_a))
    left_free = [lk[:, i] for i in range(paired, len(ca))]
    right_free = [rk[:, i] for i in range(paired, len(cd))]

    r = min(p, q)
    if len(angles) != r:
        raise ValueError("Horizontal BDI blocks do not admit the requested paired planes.")
    k11 = np.column_stack(left_paired + left_free)
    k12 = np.column_stack(right_free + right_paired)
    theta = np.asarray(angles)
    # Move each block into SO without changing the represented group element.
    d11 = -1.0 if det(k11) < 0 else 1.0
    d12 = -1.0 if det(k12) < 0 else 1.0
    k11[:, 0] *= d11
    k12[:, q - r] *= d12
    theta[0] *= d11 * d12

    k1 = block_diag(k11, k12)
    cartan = _cartan_matrix(theta, p, q)
    if not np.allclose(k1 @ cartan @ k1.T, u, atol=1e-10, rtol=1e-10):
        raise ValueError("The horizontal BDI factors do not reconstruct the input matrix.")
    return k11, k12, theta, k11.T.copy(), k12.T.copy()


def _cartan_matrix(theta, p, q):
    """Canonical CS rotation with free axes between the paired axes."""
    matrix = np.eye(p + q)
    first = np.arange(min(p, q))
    second = first + max(p, q)
    matrix[first, first] = matrix[second, second] = np.cos(theta)
    matrix[first, second], matrix[second, first] = np.sin(theta), -np.sin(theta)
    return matrix


def bdi(u, p, q, is_horizontal=True, validate=True, **kwargs):
    """Return ``k11, k12, theta, k21, k22`` for BDI(p, q).

    The K factors have diagonal blocks in SO(p) and SO(q). ``theta`` gives
    the CS rotations between axes i and max(p, q) + i. Horizontal input
    requires K2 = K1.T; otherwise use the general cosine-sine decomposition.
    ``compute_u=False`` returns only the CS angles.
    """
    if validate:
        assert u.shape == (p + q, p + q)
    if kwargs.get("compute_u", True) is False:
        return cossin(u, p=p, q=p, swap_sign=True, separate=True, **kwargs)[1]
    if is_horizontal:
        return _horizontal_bdi(u, p, q)
    (k11, k12), theta, (k21, k22) = cossin(u, p=p, q=p, swap_sign=True, separate=True)
    if p > q:
        k11 = np.roll(k11, q - p, axis=1)
        k21 = np.roll(k21, q - p, axis=0)

    # Transfer block reflections into the first signed Cartan angle.
    d11, d12, d21, d22 = (det(k) for k in (k11, k12, k21, k22))
    assert np.isclose(d11 * d12 * d21 * d22, 1.0)
    k11[:, 0] *= d11
    k12[:, q - min(p, q)] *= d12
    k21[0] *= d21
    k22[q - min(p, q)] *= d22
    theta[0] *= d11 * d12
    if d11 * d21 < 0:
        theta[0] += np.pi

    if validate:
        k1, k2 = block_diag(k11, k12), block_diag(k21, k22)
        cartan = _cartan_matrix(theta, p, q)
        assert np.allclose(k1 @ cartan @ k2, u)
        assert np.allclose([det(k) for k in (k11, k12, k21, k22, cartan)], 1.0)
    return k11, k12, theta, k21, k22


def embed(op, start, end, n):
    mat = np.eye(n, dtype=op.dtype)
    mat[start:end, start:end] = op
    return mat


def recursive_bdi(U, n, num_iter=None, first_is_horizontal=True, validate=True, return_all=False):
    p = n // 2
    q = n - p
    k11, k12, theta, k21, k22 = bdi(U, p, q, is_horizontal=first_is_horizontal, validate=validate)
    ops = {
        -1: [(U, 0, n, None)],
        0: [
            (k11, 0, p, "k1"),
            (k12, p, n, "k1"),
            (theta, 0, n, "a0"),
            (k21, 0, p, "k2"),
            (k22, p, n, "k2"),
        ],
    }
    current_ops = ops[0]
    _iter = 0

    decomposed_something = True
    while decomposed_something:
        decomposed_something = False
        new_ops = []

        for op, start, end, _type in current_ops:
            _n = end - start
            if _n <= 1:
                continue
            if _type.startswith("a") or _n == 2:
                # CSA element
                new_ops.append((op, start, end, _type))
                if _type == "a0" and first_is_horizontal:
                    # Exploit horizontalness
                    break
                continue
            _p = _n // 2
            _q = _n - _p
            k11, k12, theta, k21, k22 = bdi(op, _p, _q, is_horizontal=False, validate=validate)
            new_ops.extend(
                [
                    (k11, start, start + _p, "k1"),
                    (k12, start + _p, end, "k1"),
                    (theta, start, end, "a"),
                    (k21, start, start + _p, "k2"),
                    (k22, start + _p, end, "k2"),
                ]
            )
            decomposed_something = True

        _iter += 1
        if return_all and first_is_horizontal:
            # Exploit horizontalness
            new_ops.extend(
                (
                    ((-op if _type.startswith("a") else op.T), start, end, _type)
                    for op, start, end, _type in new_ops[:-1][::-1]
                )
            )
            ops[_iter] = new_ops
        current_ops = new_ops
        if _iter == num_iter:
            break

    if return_all:
        return ops
    if first_is_horizontal:
        current_ops.extend(
            (
                ((-op if _type.startswith("a") else op.T), start, end, _type)
                for op, start, end, _type in current_ops[:-1][::-1]
            )
        )
    return current_ops


def angles_to_reducible(theta, s, e, mapping, signs):
    """Map the cosine-sine angles of a block spanning ``[s, e)`` to Pauli rotations.

    The block is split by ``bdi``/``recursive_bdi`` into ``p = (e - s) // 2`` and
    ``q = (e - s) - p``, and the resulting Cartan element rotates index ``i`` against
    index ``i + q`` (not ``i + p``). The two agree whenever the block width is even;
    for odd widths ``q = p + 1`` and using ``p`` silently assigns the angles to the
    wrong Pauli words.
    """
    p = (e - s) // 2
    q = (e - s) - p
    op = {
        mapping[(s + i, s + q + i)]: th / 2 / signs[(s + i, s + q + i)]
        for i, th in enumerate(theta)
    }
    return PauliSentence(op)


def angles_to_reducible_str(theta, s, e, mapping):
    """String-mapping variant of ``angles_to_reducible``; see there for the ``q`` offset."""
    p = (e - s) // 2
    q = (e - s) - p
    op = {
        (pw_sign := mapping[(s + i, s + q + i)])[0]: th / 2 / pw_sign[1]
        for i, th in enumerate(theta)
    }
    return op


def group_matrix_to_reducible(matrix, start, mapping, signs, tol=1e-10):
    """Map a (SO(n)) group element composed of commuting Given's rotations
    into commuting Pauli rotations on the reducible representation given by mapping & signs.

    The rotation planes are read off from the non-zero off-diagonal entries, and the angle
    of the plane spanned by ``(i, j)`` is ``arctan2(matrix[i, j], matrix[i, i])``. Indices
    that are left over span a diagonal block of :math:`\\pm 1`; a pair of ``-1`` entries
    there is a rotation by :math:`\\pi`, which has no off-diagonal entry to be discovered
    by but is emitted all the same. (Reading the angle off ``arcsin`` of the off-diagonal
    entry alone silently drops those factors, which happens whenever an angle lands
    exactly on :math:`\\pi` -- for instance for a translation-invariant Hamiltonian with
    uniform coefficients.)
    """
    op = {}
    seen_ids = set()
    for i, j in zip(*np.where(np.abs(matrix) > tol)):
        if i >= j:
            continue
        assert i not in seen_ids and j not in seen_ids, f"{matrix}"
        m_ii = matrix[i, i]
        m_jj = matrix[j, j]
        assert np.isclose(np.sign(m_ii), np.sign(m_jj)) or np.allclose(
            [m_ii, m_jj], 0.0
        ), f"{m_ii}, {m_jj}"
        angle = float(np.arctan2(matrix[i, j], m_ii))
        op[mapping[(start + i, start + j)]] = angle / 2 / signs[(start + i, start + j)]
        seen_ids |= {i, j}

    # Whatever is left is a diagonal block of +-1. Every pair of -1s is a rotation by pi.
    flipped = [
        i for i in range(len(matrix)) if i not in seen_ids and matrix[i, i] < 0
    ]
    assert len(flipped) % 2 == 0, (
        f"An odd number of -1 entries ({len(flipped)}) is left over, so this matrix has "
        f"determinant -1 and is not a product of rotations:\n{matrix}"
    )
    for i, j in zip(flipped[::2], flipped[1::2]):
        op[mapping[(start + i, start + j)]] = np.pi / 2 / signs[(start + i, start + j)]

    return PauliSentence(op)


def group_matrix_to_reducible_str(matrix, start, mapping):
    """Convert an SO(2) block using the legacy string mapping and angle branch."""
    assert matrix.shape == (2, 2)
    angle = np.arcsin(matrix[0, 1])
    if matrix[0, 0] < 0:
        angle = np.pi - angle
    word, sign = mapping[(start, start + 1)]
    return {word: angle / 2 / sign}


def map_recursive_decomp_to_reducible(
    recursive_decomp, mapping, signs, time=None, tol=1e-8, validate=False
):
    """Map the result of recursive_bdi back to a series of Pauli rotations in the reducible
    representation specified by mapping & signs.

    If the group element that was decomposed with recursive_bdi was not exp(H) but some
    rescaled variant exp(t H), the parameter t should be provided as the ``time`` parameter
    to this function.
    """

    pauli_decomp = []
    if validate:
        from .map_to_irrep import E

        inv_mapping = {val: key for key, val in mapping.items()}
    n = max([k[1] for k in mapping]) + 1
    for mat, s, e, t in recursive_decomp:
        if t.startswith("a"):
            ps = angles_to_reducible(mat, s, e, mapping, signs)
        else:
            ps = group_matrix_to_reducible(mat, s, mapping, signs)
        if t == "a0" and time is not None:
            ps = ps / time
        if tol is not None:
            ps.simplify(tol=tol)
        pauli_decomp.extend(((pw, coeff, t) for pw, coeff in ps.items()))

        # validate to check some properties. Not required for the actual computation
        if validate:
            assert all(pw1.commutes_with(pw2) for pw1, pw2 in combinations(ps.keys(), r=2))
            if not t.startswith("a"):
                assert len(ps) == 2
            if t == "a0" and time is not None:
                ps = ps * time
            rec_mat = np.eye(n)
            for pw, coeff in ps.items():
                i, j = inv_mapping[pw]
                rec_mat = rec_mat @ expm(E((i, j), n, "BDI") * signs[(i, j)] * coeff)
            if not np.allclose(mat, rec_mat):
                print(np.round(mat, 4))
                print(np.round(rec_mat, 4))
                raise ValueError(
                    "The decomposition into Pauli rotations did not correctly reproduce the matrix."
                )

    return pauli_decomp


def round_angles_to_irreducible_mat(theta, start, end, n, tol):
    p = (end - start) // 2
    theta = np.where(np.abs(np.sin(theta)) > tol, theta, 0)
    return embed(_cartan_matrix(theta, p, end - start - p), start, end, n)


def round_mat_to_irreducible_mat(mat, start, end, n, tol):
    k = np.where((np.abs(mat) > tol) + (np.abs(mat) < (1 - tol)), mat, np.eye(len(mat)))
    return embed(k, start, end, n)


def round_mult_recursive_decomp_str(recursive_decomp, time, n_so, tol=1e-8):
    out = np.eye(n_so)
    for matrix in _iter_recursive_matrices(recursive_decomp, time, n_so, tol):
        out @= matrix
    return out


def _iter_recursive_matrices(recursive_decomp, time, n_so, tol):
    for mat, s, e, t in recursive_decomp:
        if t.startswith("a"):
            if t == "a0" and time is not None:
                mat = mat / time
            mat = round_angles_to_irreducible_mat(mat, s, e, n_so, tol)
        else:
            mat = round_mat_to_irreducible_mat(mat, s, e, n_so, tol)

        yield mat


def map_recursive_decomp_to_matrices(recursive_decomp, time, n_so, tol=1e-8):
    return list(_iter_recursive_matrices(recursive_decomp, time, n_so, tol))


def map_recursive_decomp_to_reducible_str(recursive_decomp, mapping, time=None, tol=1e-8):
    """Map the result of recursive_bdi back to a series of Pauli rotations in the reducible
    representation specified by mapping & signs.

    If the group element that was decomposed with recursive_bdi was not exp(H) but some
    rescaled variant exp(t H), the parameter t should be provided as the ``time`` parameter
    to this function.
    """
    pauli_decomp = []
    for mat, s, e, t in recursive_decomp:
        if t.startswith("a"):
            ps = angles_to_reducible_str(mat, s, e, mapping)
        else:
            ps = group_matrix_to_reducible_str(mat, s, mapping)
        if t == "a0" and time is not None:
            ps = {key: val / time for key, val in ps.items()}
        if tol is not None:
            ps = {key: val for key, val in ps.items() if np.abs(val) >= tol}
        pauli_decomp.extend(((pw, coeff, t) for pw, coeff in ps.items()))

    return pauli_decomp
