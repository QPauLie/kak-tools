from itertools import combinations, product
from collections.abc import Mapping
import networkx as nx
import numpy as np

import pennylane as qml
from pennylane.pauli import PauliWord, PauliSentence
from pennylane import X, Y, Z
from .pauli_dlas import anticom_graph_pauli
from ._validation import (
    finite_real_scalar, positive_integer, real_coefficients, resolve_bdi_partition,
)


def _anticom_graph_bdi(n, invol_kwargs):
    """Horizontal rotation-plane anticommutation graph for BDI(p, q)."""
    p, _ = resolve_bdi_partition(n, invol_kwargs)
    edges_hor = [((i, j), (i, l)) for i in range(p) for j in range(p, n) for l in range(j + 1, n)]
    edges_hor += [((i, j), (k, j)) for i in range(p) for j in range(p, n) for k in range(i + 1, p)]
    horizontal_graph = nx.Graph()
    # Include the isolated generator of BDI(1, 1).
    horizontal_graph.add_nodes_from((i, j) for i in range(p) for j in range(p, n))
    horizontal_graph.add_edges_from(edges_hor)
    return horizontal_graph


def _anticom_graph_diii(n, invol_kwargs):
    assert n % 2 == 0
    assert set(invol_kwargs) == set()
    m = n // 2
    nodes = [
        (i, j, _type, sign)
        for _type in "AB"
        for sign in "+-"
        for i, j in combinations(range(m), r=2)
    ]
    nodes_hor = [(i, j, _type, "-") for _type in "AB" for i, j in combinations(range(m), r=2)]

    return _anticom_graph_from_labels(nodes), _anticom_graph_from_labels(nodes_hor)


def _anticom_graph_aiii(n, invol_kwargs):
    assert set(invol_kwargs).issubset({"p", "q"})
    if n % 2 or invol_kwargs.get("p", None) != invol_kwargs.get("q", None):
        raise NotImplementedError("BDI currently only is supported with p=q")
    m = n // 2
    nodes = [(i, j, t) for t in "XYZ" for i, j in combinations(range(m), r=2) if t != Z or i == j]
    nodes_hor = [(i, j, t) for t in "XY" for i, j in product(range(m), range(m, n))]

    return _anticom_graph_from_labels(nodes), _anticom_graph_from_labels(nodes_hor)


def _anticom_graph_from_labels(nodes):
    """Connect DIII/AIII basis labels whose first two entries share an index."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        (a, b) for a, b in combinations(graph.nodes(), 2) if a[0] in b or a[1] in b
    )
    return graph


def anticom_graph_irrep(n, invol_type=None, invol_kwargs=None):
    """Create an anticommutation graph for an irrep of a simple algebra,
    in a basis adapted to a given involution type."""
    if invol_type == "BDI":
        return _anticom_graph_bdi(n, invol_kwargs)

    elif invol_type == "DIII":
        return _anticom_graph_diii(n, invol_kwargs)

    elif invol_type == "AIII":
        return _anticom_graph_aiii(n, invol_kwargs)

    raise NotImplementedError("Only BDI, DIII and AIII are implemented.")


def map_horizontal_subgraph(pauli_graph, horizontal_graph):
    """Initiate a mapping between irrep elements and Pauli words by identifying
    a subgraph in the horizontal anticommutation graph of the former that is isomorphic
    to the anticommutation graph of the latter."""
    graph_matcher = nx.algorithms.isomorphism.GraphMatcher(horizontal_graph, pauli_graph)
    return next(graph_matcher.subgraph_isomorphisms_iter())


def _node_commutator(node1, node2, invol_type):
    if invol_type == "BDI":
        a, b, c, d = sorted(node1 + node2)
        if a == b:
            return (c, d)
        if b == c:
            return (a, d)
        return (a, b)
    elif invol_type == "DIII":
        i, j, t1, s1 = node1
        k, l, t2, s2 = node2
        new_t = "A" if t1 == t2 else "B"
        new_s = "+" if s1 == s2 else "-"
        a, b = {i, j, k, l}.difference({i, j} & {k, l})
        if a > b:
            a, b = b, a
        return (a, b, new_t, new_s)

    raise ValueError


def map_hor_com_hor(mapping, pauli_graph, invol_type=None):
    """Extend an initialized mapping from a horizontal subspace to horizontal Pauli
    words by computing all accessible first-order commutators."""
    inv_mapping = {val: key for key, val in mapping.items()}
    for p1, p2 in pauli_graph.edges():
        com_node = _node_commutator(inv_mapping[p1], inv_mapping[p2], invol_type)
        mapping[com_node] = p1._commutator(p2)[0]

    return mapping


def _yield_sorted_ids_no_collision(i, j, n):
    # Iterate from 0 to n-1 and return sorted versions of (i, k), (k, j)
    for k in range(i):
        yield (k, i), (k, j)
    for k in range(i + 1, j):
        yield (i, k), (k, j)
    for k in range(j + 1, n):
        yield (i, k), (j, k)


def all_pre_commutators(node, n, invol_type):
    if invol_type == "BDI":
        yield from _yield_sorted_ids_no_collision(*node, n)
    if invol_type == "DIII":
        i, j, t, s = node
        signs = [("+", "+"), ("-", "-")] if s == "+" else [("+", "-"), ("-", "+")]
        types = [("A", "A"), ("B", "B")] if t == "A" else [("A", "B")]
        for ids1, ids2 in _yield_sorted_ids_no_collision(i, j, n // 2):
            for s1, s2 in signs:
                for t1, t2 in types:
                    yield ids1 + (t1, s1), ids2 + (t2, s2)


def choose_generic_first_missing(missing, n, invol_type):
    if invol_type in {"BDI", "DIII"}:
        cand = missing[0]
        assert cand[0] == 0
        if invol_type == "DIII":
            assert cand[2] == "A"
            assert cand[3] == "+"
        return cand, 0


def anticommuting_nodes(node, n, invol_type):
    if invol_type == "BDI":
        for i in range(1, n):
            yield (0, i)
    if invol_type == "DIII":
        for i in range(1, n):
            yield (0, i, "A", "+")
            yield (0, i, "A", "-")
            yield (0, i, "B", "+")
            yield (0, i, "B", "-")


def map_coms(mapping, missing, missing_ops, n, invol_type):
    """Extend a partial mapping by filling in gaps that are commutators of elements
    that exist already in the mapping."""

    missing_idx = 0
    last_reset_length = -1
    while missing:
        if missing_idx == len(missing):
            # Reset position, starting a new recursion loop
            if len(missing) == last_reset_length:
                # Already reset with missing_idx pointing to the end. No extension seems possible
                return mapping, missing, missing_ops
            last_reset_length = len(missing)
            missing_idx = 0

        node = missing[missing_idx]
        for n1, n2 in all_pre_commutators(node, n, invol_type):
            if n1 in mapping and n2 in mapping:
                com_pw = mapping[n1]._commutator(mapping[n2])[0]
                mapping[node] = com_pw
                missing_ops.remove(com_pw)
                missing.pop(missing_idx)
                break
        else:
            missing_idx += 1

    return mapping, [], []


def map_choice(mapping, missing, missing_ops, n, invol_type):
    node, node_idx = choose_generic_first_missing(missing, n, invol_type)
    ac_nodes = anticommuting_nodes(node, n, invol_type)
    for op_idx, op in enumerate(missing_ops):
        if any(op.commutes_with(mapping[ac_node]) for ac_node in ac_nodes if ac_node in mapping):
            continue
        mapping[node] = op
        missing.pop(node_idx)
        missing_ops.pop(op_idx)
        return mapping, missing, missing_ops
    raise ValueError("No compatible choice could be made from the missing operators.")


def _so_signs_from_star(mapping, n, star_signs=None):
    """Check the Pauli basis and determine all signs from its (0, i) star."""
    n = positive_integer(n, "n")
    if n < 2:
        raise ValueError("An so(n) Pauli mapping requires n >= 2.")
    planes = set(combinations(range(n), 2))
    if not isinstance(mapping, Mapping) or set(mapping) != planes:
        raise ValueError("The mapping must contain every so(n) rotation plane exactly once.")
    words = list(mapping.values())
    if not all(isinstance(word, PauliWord) for word in words) or len(set(words)) != len(words):
        raise ValueError("The mapping must be a bijection onto distinct PauliWords.")
    if any(pauli not in ("I", "X", "Y", "Z") for word in words for pauli in word.values()):
        raise ValueError("PauliWord entries must contain only I, X, Y, Z.")

    if star_signs is None:
        signs = {(0, i): 1 for i in range(1, n)}
        signs[(0, n // 2)] = -1
    else:
        signs = {(0, i): star_signs[(0, i)] for i in range(1, n)}
    for i, j in combinations(range(1, n), 2):
        word, coefficient = mapping[(0, i)]._commutator(mapping[(0, j)])
        if coefficient not in (-2j, 2j) or word != mapping[(i, j)]:
            raise ValueError(f"The so(n) star commutator has the wrong Pauli word at plane {(i, j)}.")
        signs[(i, j)] = int((2j * signs[(0, i)] * signs[(0, j)] / coefficient).real)
    return signs


def _validate_so_mapping(mapping, signs, n):
    """Verify the Lie map i*P_ij -> 2*sign_ij*(E_ij-E_ji) and normalize signs.

    The star relation [P_0i, P_0j] = 2j*s_0i*s_0j/s_ij*P_ij
    makes the star Pauli words Clifford generators. Their pair products then
    satisfy every remaining so(n) commutator. Distinct Pauli words ensure
    injectivity, including for so(4); so(2) has no star pairs to check.
    Any valid choice of star signs is accepted.
    """
    if not isinstance(mapping, Mapping) or not isinstance(signs, Mapping) or set(signs) != set(mapping):
        raise ValueError("The signs must contain one sign for every mapped rotation plane.")
    signs = {plane: finite_real_scalar(sign, f"sign for {plane}") for plane, sign in signs.items()}
    if any(sign not in (-1.0, 1.0) for sign in signs.values()):
        raise ValueError("Each mapping sign must be +1 or -1.")
    expected = _so_signs_from_star(mapping, n, signs)
    if signs != expected:
        raise ValueError("The mapping signs do not satisfy the so(n) star commutators.")
    return signs


def make_signs(mapping, n, invol_type):
    """Construct consistent BDI signs directly, retaining the standard star gauge."""
    assert invol_type == "BDI"
    return _so_signs_from_star(mapping, n)


def map_simple_to_irrep(ops, horizontal_ops=None, n=None, invol_type=None, invol_kwargs=None):
    """Map a complete so(n) Pauli-word basis to signed BDI rotation planes.

    ``horizontal_ops`` supplies the horizontal generators or their initial mapping.
    Other involutions and automatic horizontal-generator selection are unsupported.
    """
    assert all(isinstance(op, PauliWord) for op in ops)
    n = positive_integer(n, "n")
    assert invol_type == "BDI"
    if invol_kwargs is None:
        invol_kwargs = {}

    if horizontal_ops is None:
        raise NotImplementedError("This is the simpler scenario, but it is not implemented yet.")
    if isinstance(horizontal_ops, dict):
        # If a dictionary is passed, assume that it already contains a mapping for the horizontal
        # operators, rather than just the operators.
        mapping = horizontal_ops
        pauli_graph = anticom_graph_pauli(mapping.values())
        assert all(
            isinstance(key, tuple) and len(key) == 2 and isinstance(op, PauliWord)
            for key, op in horizontal_ops.items()
        )
    else:
        assert all(isinstance(op, PauliWord) for op in horizontal_ops)

        horizontal_graph = anticom_graph_irrep(n, invol_type, invol_kwargs)
        pauli_graph = anticom_graph_pauli(horizontal_ops)

        mapping = map_horizontal_subgraph(pauli_graph, horizontal_graph)

    all_nodes = list(combinations(range(n), r=2))

    mapping = map_hor_com_hor(mapping, pauli_graph, invol_type=invol_type)
    missing = [node for node in all_nodes if node not in mapping]
    missing_ops = list(set(ops).difference(set(mapping.values())))

    mapping, missing, missing_ops = map_coms(
        mapping, missing, missing_ops, n, invol_type=invol_type
    )

    while missing:
        assert missing_ops
        mapping, missing, missing_ops = map_choice(mapping, missing, missing_ops, n, invol_type)
        mapping, missing, missing_ops = map_coms(
            mapping, missing, missing_ops, n, invol_type=invol_type
        )

    assert not missing_ops
    assert len(mapping) == len(all_nodes)

    return mapping, make_signs(mapping, n, invol_type)


def map_irrep_to_matrices(mapping, signs, n, invol_type):
    return {op: signs[node] * E(node, n, invol_type) for node, op in mapping.items()}


def irrep_dot(coeffs, generators, mapping, signs=None, n=None, invol_type=None):
    """Build ``sum_i coeffs[i] * generators[i]`` directly in the irrep.

    Two mapping conventions are accepted:

    - ``mapping = {node: (PauliWord, sign)}`` with ``signs=None``, as produced by
      ``make_so_2n_full_mapping_str``-style helpers, and
    - ``mapping = {node: PauliWord}`` together with ``signs = {node: sign}``, which is
      what :func:`map_simple_to_irrep` returns.

    The legacy positional call ``irrep_dot(coeffs, generators, mapping, n, invol_type)``
    still works. Coefficients must be finite and real, with exactly one per
    generator. An empty sum returns an ``(n, n)`` zero matrix.

    Raises:
        ValueError: If dimensions or coefficients are invalid, a generator is
            missing from the mapping, or a selected BDI node/sign is invalid.
    """
    if isinstance(signs, (int, np.integer)):
        # Legacy positional signature: signs slot held n, n slot held invol_type.
        signs, n, invol_type = None, signs, invol_type if invol_type is not None else n

    n = positive_integer(n, "n")
    generators = list(generators)
    coeffs = real_coefficients(coeffs, len(generators))
    if not isinstance(mapping, Mapping) or (signs is not None and not isinstance(signs, Mapping)):
        raise ValueError("mapping and signs must be dictionaries or other mappings.")
    inv_mapping = {}
    for node, entry in mapping.items():
        if signs is None:
            try:
                op, sign = entry
            except (TypeError, ValueError) as exc:
                raise ValueError("Each mapping entry must contain a (generator, sign) pair.") from exc
        else:
            op = entry
        if op not in generators:
            continue
        if signs is not None:
            if node not in signs:
                raise ValueError(f"Missing sign for mapped generator {op!r} at node {node!r}.")
            sign = signs[node]
        if op in inv_mapping:
            raise ValueError(f"Generator {op!r} is assigned to more than one mapping node.")
        if invol_type == "BDI":
            if (
                not isinstance(node, tuple) or len(node) != 2
                or any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                       for index in node)
                or not 0 <= node[0] < node[1] < n
            ):
                raise ValueError(f"Invalid BDI mapping node {node!r}; expected 0 <= i < j < {n}.")
            sign = finite_real_scalar(sign, f"sign for node {node!r}")
            if sign not in (-1.0, 1.0):
                raise ValueError(f"The BDI sign for node {node!r} must be +1 or -1.")
        inv_mapping[op] = (node, sign)

    out = np.zeros((n, n))
    for c, gen in zip(coeffs, generators):
        if gen not in inv_mapping:
            raise ValueError(f"Generator {gen!r} is missing from the irrep mapping.")
        node, sign = inv_mapping[gen]
        out = out + c * sign * E(node, n, invol_type)
    return out


def map_matrix_to_reducible(matrix, mapping, signs, invol_type):
    assert invol_type == "BDI"
    op = {}
    for i, j in zip(*np.where(matrix)):
        if i < j:
            op[mapping[(i, j)]] = matrix[i, j] / 2 / signs[(i, j)]

    return PauliSentence(op)


def E(node, n, invol_type):
    if invol_type not in {"BDI", "DIII", "AIII"}:
        raise NotImplementedError(
            f"Matrix generators for involution {invol_type!r} are not implemented; "
            "choose BDI, DIII or AIII."
        )
    if invol_type == "BDI":
        e = np.zeros((n, n))
        i, j = node
        e[i, j] = 2
        e[j, i] = -2
    if invol_type == "DIII":
        e = np.zeros((n, n))
        i, j, t, s = node
        sign = 1 if s == "+" else -1
        if t == "A":
            e[i, j] = 1
            e[j, i] = -1
            e[i + n // 2, j + n // 2] = sign
            e[j + n // 2, i + n // 2] = -sign
        if t == "B":
            e[i, j + n // 2] = 1
            e[j + n // 2, i] = -1
            e[j, i + n // 2] = sign
            e[i + n // 2, j] = -sign

    if invol_type == "AIII":
        e = np.zeros((n, n), dtype=complex)
        i, j, t = node
        if t == "X":
            e[i, j] = e[j, i] = 1j
        if t == "Y":
            e[i, j] = 1
            e[j, i] = -1
        if t == "Z":
            e[i, i] = 1j
            e[i + 1, i + 1] = -1j

    return e


def make_so_2n(n):
    """Create all Pauli words for the reducible so(2n) representation implemented
    by the transverse field XY model, i.e., generated by XX couplings, YY couplings, and
    single-qubit Z operators."""
    algebra = [
        PauliWord({w: P1, v: P2} | {i: "Z" for i in range(w + 1, v)})
        for w, v in combinations(range(n), r=2)
        for P1, P2 in product("XY", repeat=2)
    ]
    algebra += [PauliWord({w: "Z"}) for w in range(n)]
    return algebra


def make_so_2n_horizontal_mapping(n):
    """Create a default reducible-to-irreducible mapping for some horizontal operators
    in a BDI decomposition of so(2n). The mapped operators are XX couplings, YY couplings,
    and single-qubit Z operators."""
    mapping = {(i, i + n): PauliWord({i: "Z"}) for i in range(n)}
    mapping |= {(i, i + n + 1): PauliWord({i: "X", i + 1: "X"}) for i in range(n - 1)}
    mapping |= {(i, i + n - 1): PauliWord({i - 1: "Y", i: "Y"}) for i in range(1, n)}
    return mapping


def make_so_2n_full_mapping(n, xy_symmetric=False):
    """Return the TF-XY so(2n) Pauli mapping and its historical sign convention.

    Contains :func:`make_so_2n_horizontal_mapping`; XX, YY and Z terms are horizontal.
    """
    if xy_symmetric:
        return _so_2n_full_mapping_xy(n)

    upper_left, lower_right, upper_right, lower_left = {}, {}, {}, {}
    for i, j in combinations(range(n), 2):
        chain = {w: "Z" for w in range(i + 1, j)}
        upper_left[(i, j)] = PauliWord({i: "X", j: "Y"} | chain)
        lower_right[(n + i, n + j)] = PauliWord({i: "Y", j: "X"} | chain)
        upper_right[(i, n + j)] = PauliWord({i: "X", j: "X"} | chain)
        lower_left[(j, n + i)] = PauliWord({i: "Y", j: "Y"} | chain)
    mapping = upper_left | lower_right | upper_right | lower_left
    mapping |= {(i, n + i): PauliWord({i: "Z"}) for i in range(n)}

    signs = {(i, n + i): -1 for i in range(n)}
    signs |= {(i, j): 1 for i in range(n) for j in range(i + 1, 2 * n)}
    signs |= {(i, j): -1 for i in range(n, 2 * n) for j in range(i, 2 * n)}
    return mapping, signs


def make_so_2n_full_mapping_str(n, xy_symmetric=False):
    """Return TF-XY rotation planes as (compressed Pauli string, sign) entries.

    Numbers count leading/trailing identities and intervening Z factors.
    The Z-field sign is -1 in this string convention.
    """
    if xy_symmetric:
        raise ValueError

    upper_left, lower_right, upper_right, lower_left = {}, {}, {}, {}
    for i, j in combinations(range(n), 2):
        upper_left[(i, j)] = (f"{i}X{j - i - 1}Y{n-j-1}", 1)
        lower_right[(n + i, n + j)] = (f"{i}Y{j - i - 1}X{n-j-1}", -1)
        upper_right[(i, n + j)] = (f"{i}X{j - i - 1}X{n-j-1}", 1)
        lower_left[(j, n + i)] = (f"{i}Y{j - i - 1}Y{n-j-1}", 1)
    mapping = upper_left | lower_right | upper_right | lower_left
    mapping |= {(i, n + i): (f"{i}Z{n - i - 1}", -1) for i in range(n)}

    return mapping


def _make_tfXY_coeffs(n, coefficients):
    if coefficients == "random":
        alphas = np.random.normal(0.6, 1.0, size=n - 1)
        betas = np.random.normal(0.3, 1.2, size=n - 1)
        gammas = np.random.normal(0.0, 0.3, size=n)
    elif coefficients == "random TF":
        alphas = np.ones(n - 1)
        betas = np.ones(n - 1)
        gammas = np.random.normal(0.0, 0.3, size=n)
    elif coefficients == "uniform":
        alphas = np.ones(n - 1)
        betas = np.ones(n - 1)
        gammas = np.ones(n)

    norm = np.linalg.norm(np.concatenate([alphas, betas, gammas]))
    return alphas / norm, betas / norm, gammas / norm


def make_tfXY_hamiltonian_irrep(n, coefficients="random"):
    """Create the transverse-field XY model Hamiltonian on n qubits,
    represented in a free-fermionic picture as 2n x 2n matrix.
    This function uses the hardcoded mapping from the manuscript for this particular
    model.
    The Hamiltonian is normalized to trace norm 1.
    """
    alphas, betas, gammas = _make_tfXY_coeffs(n, coefficients)
    H_irrep = np.diag(-gammas, k=n) - np.diag(-gammas, k=-n)
    H_irrep += np.diag(alphas, k=n + 1) - np.diag(alphas, k=-n - 1)
    _betas = np.concatenate([[0], betas, [0]])
    H_irrep += np.diag(_betas, k=n - 1) - np.diag(_betas, k=-n + 1)
    return H_irrep


def make_tfXY_hamiltonian_qubits(n, coefficients="random"):
    """Create the transverse-field XY model Hamiltonian on n qubits,
    in its original representation on qubits.
    The Hamiltonian is normalized to trace norm 1.
    """
    alphas, betas, gammas = _make_tfXY_coeffs(n, coefficients)
    coeffs = np.concatenate([alphas, betas, gammas])
    couplings = [X(w) @ X(w + 1) for w in range(n - 1)] + [Y(w) @ Y(w + 1) for w in range(n - 1)]
    Zs = [Z(w) for w in range(n)]
    generators = couplings + Zs
    H = qml.dot(coeffs, generators)
    generators = [next(iter(op.pauli_rep)) for op in generators]
    return H, generators, coeffs
