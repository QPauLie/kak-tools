"""Map a Pauli-word representation of so(n) onto signed BDI rotation planes (paper App. F)."""

from collections.abc import Mapping
from itertools import combinations
import networkx as nx
import numpy as np

from pennylane.pauli import PauliWord, PauliSentence
from .pauli_dlas import anticom_graph_pauli
from ._validation import (
    finite_real_scalar, nonnegative_integer, positive_integer, real_coefficients, require,
    resolve_bdi_partition,
)


def _require_bdi(invol_type):
    """The Pauli-level mapping exists for the BDI involution only."""
    if invol_type != "BDI":
        raise NotImplementedError(f"Only BDI is implemented, not {invol_type!r}.")


def anticom_graph_irrep(n, invol_type=None, invol_kwargs=None):
    """Anticommutation graph of the horizontal rotation planes of BDI(p, q) in so(n).

    Two planes anticommute iff they share an index, so the horizontal planes ``(i, j)``
    with ``i < p <= j`` form the rook graph on a ``p x q`` board (paper Fig. 13).
    """
    _require_bdi(invol_type)
    p, _ = resolve_bdi_partition(n, invol_kwargs)
    edges_hor = [((i, j), (i, l)) for i in range(p) for j in range(p, n) for l in range(j + 1, n)]
    edges_hor += [((i, j), (k, j)) for i in range(p) for j in range(p, n) for k in range(i + 1, p)]
    horizontal_graph = nx.Graph()
    # Include the isolated generator of BDI(1, 1).
    horizontal_graph.add_nodes_from((i, j) for i in range(p) for j in range(p, n))
    horizontal_graph.add_edges_from(edges_hor)
    return horizontal_graph


class HorizontalEmbeddingError(ValueError):
    """The horizontal Pauli words do not fit the horizontal subspace of the involution."""


def _word_key(word):
    """Sort key for Pauli words, so the mapping never depends on the hash seed."""
    return sorted((type(wire).__name__, wire, pauli) for wire, pauli in word.items())


def _canonical(word):
    """The same Pauli word with its wires in sorted order, so ``word.wires`` is predictable."""
    try:
        return PauliWord(dict(sorted(word.items(), key=lambda item: (type(item[0]).__name__, item[0]))))
    except TypeError:  # wires of mixed, unorderable types: keep the given order
        return word


def _slot_key(slot):
    """Sort key for Majorana slots (cliques of words, or a word's private slot)."""
    if isinstance(slot, tuple):
        return (1, _word_key(slot[0]), slot[1])
    return (0, sorted(_word_key(word) for word in slot))


def _majorana_slots(pauli_graph):
    """Recover the Majorana slots that the horizontal Pauli words pair up (paper App. F.6).

    A horizontal word is, up to a phase, the product of a row Majorana and a column
    Majorana, and two words anticommute iff they share one. The words sharing a slot
    with ``word`` thus form a clique in the anticommutation graph, and since the slot
    graph is bipartite (hence triangle-free) the neighbourhood of ``word`` is the disjoint
    union of at most two such cliques with no edges between them. Returns the slot graph,
    whose edges carry the word joining their two slots.
    """
    slots = nx.Graph()
    for word in pauli_graph:
        neighbourhood = pauli_graph.subgraph(pauli_graph[word])
        # networkx iterates the induced node set, so order the cliques ourselves.
        cliques = sorted(
            (frozenset(clique) for clique in nx.connected_components(neighbourhood)),
            key=lambda clique: sorted(_word_key(w) for w in clique),
        )
        if len(cliques) > 2 or any(
            neighbourhood.subgraph(clique).number_of_edges() != len(clique) * (len(clique) - 1) // 2
            for clique in cliques
        ):
            raise HorizontalEmbeddingError(
                f"The horizontal Pauli word {word!r} anticommutes with words that cannot all "
                "share a rotation-plane index with it."
            )
        ends = [clique | {word} for clique in cliques]
        # A slot shared with no other word is private to this word.
        ends += [(word, k) for k in range(len(ends), 2)]
        slots.add_edge(*ends, word=word)
    return slots


def map_horizontal_subgraph(pauli_graph, p, q):
    """Place the horizontal words on rotation planes ``(i, j)`` with ``i < p <= j``.

    The horizontal planes anticommute iff they share an index, i.e. they form the line
    graph of the complete bipartite graph on ``p`` rows and ``q`` columns (paper App. F.6).
    The words embed iff their Majorana slot graph is bipartite and, orienting every
    connected component of it independently, its two sides fit into the rows and columns.
    """
    slots = _majorana_slots(pauli_graph)
    components = []
    for component in nx.connected_components(slots):
        try:
            colour = nx.bipartite.color(slots.subgraph(component))
        except nx.NetworkXError as exc:
            raise HorizontalEmbeddingError(
                "The horizontal Pauli words do not anticommute like rotation planes."
            ) from exc
        sides = sorted(
            (sorted((s for s in component if colour[s] == c), key=_slot_key) for c in (0, 1)),
            key=lambda side: _slot_key(side[0]),
        )
        components.append(tuple(sides))

    # Every slot becomes a row or a column, so the orientations must land the row total
    # in [total - q, p]; track which totals are reachable component by component.
    total = slots.number_of_nodes()
    reachable = [{0: None}]
    for first, second in components:
        step = {}
        for rows in reachable[-1]:
            step.setdefault(rows + len(first), (rows, False))
            step.setdefault(rows + len(second), (rows, True))
        reachable.append(step)
    fitting = [rows for rows in reachable[-1] if total - q <= rows <= p]
    if not fitting:
        raise HorizontalEmbeddingError(
            f"The horizontal Pauli words need more than {p} rows or {q} columns of BDI({p}, {q})."
        )
    flips = []
    rows = fitting[0]
    for step in reversed(reachable[1:]):
        rows, flipped = step[rows]
        flips.append(flipped)

    index = {}
    next_row, next_column = 0, p
    for (first, second), flipped in zip(components, reversed(flips), strict=True):
        rows, columns = (second, first) if flipped else (first, second)
        index |= dict(zip(rows, range(next_row, next_row + len(rows)), strict=True))
        index |= dict(zip(columns, range(next_column, next_column + len(columns)), strict=True))
        next_row += len(rows)
        next_column += len(columns)
    return {_plane(index[a], index[b]): word for a, b, word in slots.edges(data="word")}


def _plane(i, j):
    return (i, j) if i < j else (j, i)


def _complete_mapping(mapping, n):
    """Map the remaining so(n) planes through commutators of mapped planes (paper App. F.3).

    ``[P_ab, P_ak]`` is proportional to ``P_bk``, so every newly mapped plane is combined
    with the mapped planes sharing one of its indices. The proportionality signs are fixed
    afterwards by :func:`make_signs`. Iterating the growing list visits every mapped
    plane exactly once.
    """
    planes = list(mapping)
    for a, b in planes:
        for k in range(n):
            if k in (a, b):
                continue
            for shared, other in ((a, b), (b, a)):
                partner = mapping.get(_plane(shared, k))
                target = _plane(other, k)
                if partner is None or target in mapping:
                    continue
                word, coefficient = mapping[(a, b)]._commutator(partner)
                if coefficient == 0:
                    raise ValueError(
                        f"The Pauli words at planes {(a, b)} and {_plane(shared, k)} commute, "
                        f"so they do not represent so({n})."
                    )
                mapping[target] = _canonical(word)
                planes.append(target)


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
    _require_bdi(invol_type)
    return _so_signs_from_star(mapping, n)


def map_simple_to_irrep(ops, horizontal_ops=None, n=None, invol_type=None, invol_kwargs=None):
    """Map a complete so(n) Pauli-word basis to signed BDI rotation planes.

    ``horizontal_ops`` supplies the horizontal generators, which must generate the
    algebra, or a partial mapping ``{plane: word}`` of horizontal planes to complete.
    Other involutions and automatic horizontal-generator selection are unsupported.

    Raises:
        HorizontalEmbeddingError: If the horizontal words do not fit BDI(p, q).
        ValueError: If the horizontal words do not generate so(n), or ``ops`` is not
            the basis they generate.
    """
    require(all(isinstance(op, PauliWord) for op in ops), "ops must contain PauliWords.")
    n = positive_integer(n, "n")
    _require_bdi(invol_type)
    p, q = resolve_bdi_partition(n, invol_kwargs)

    if horizontal_ops is None:
        raise NotImplementedError("Automatic selection of horizontal generators is not implemented.")
    if isinstance(horizontal_ops, Mapping):
        mapping = dict(horizontal_ops)
        require(
            all(isinstance(op, PauliWord) for op in mapping.values())
            and set(mapping) <= {(i, j) for i in range(p) for j in range(p, n)},
            f"A horizontal mapping must map planes (i, j) with i < {p} <= j < {n} to PauliWords.",
        )
    else:
        horizontal_ops = list(horizontal_ops)
        require(all(isinstance(op, PauliWord) for op in horizontal_ops), "horizontal_ops must contain PauliWords.")
        mapping = map_horizontal_subgraph(anticom_graph_pauli(horizontal_ops), p, q)

    _complete_mapping(mapping, n)
    if len(mapping) < n * (n - 1) // 2:
        raise ValueError("The horizontal Pauli words do not generate so(n); supply a generating set.")
    if set(mapping.values()) != set(ops):
        raise ValueError(
            f"ops must be the {len(mapping)} Pauli words that the horizontal words generate."
        )
    return mapping, make_signs(mapping, n, invol_type)


def map_irrep_to_matrices(mapping, signs, n, invol_type):
    return {op: signs[node] * E(node, n, invol_type) for node, op in mapping.items()}


def irrep_dot(coeffs, generators, mapping, signs=None, *, n, invol_type="BDI"):
    """Build ``sum_i coeffs[i] * generators[i]`` directly in the irrep.

    Two mapping conventions are accepted:

    - ``mapping = {node: PauliWord}`` together with ``signs = {node: sign}``, which is
      what :func:`map_simple_to_irrep` returns, and
    - ``mapping = {node: (PauliWord, sign)}`` with ``signs=None``, as produced by
      ``make_so_2n_full_mapping_str``-style helpers.

    Coefficients must be finite and real, with exactly one per generator. An empty
    sum returns an ``(n, n)`` zero matrix.

    Raises:
        ValueError: If dimensions or coefficients are invalid, a generator is
            missing from the mapping, or a selected BDI node/sign is invalid.
    """
    _require_bdi(invol_type)
    n = positive_integer(n, "n")
    generators = list(generators)
    coeffs = real_coefficients(coeffs, len(generators))
    if not isinstance(mapping, Mapping) or (signs is not None and not isinstance(signs, Mapping)):
        raise ValueError("mapping and signs must be dictionaries or other mappings.")
    if signs is None:
        try:
            pairs = {node: (op, sign) for node, (op, sign) in mapping.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("Each mapping entry must contain a (generator, sign) pair.") from exc
        mapping = {node: op for node, (op, _) in pairs.items()}
        signs = {node: sign for node, (_, sign) in pairs.items()}

    wanted = set(generators)
    inv_mapping = {}
    for node, op in mapping.items():
        if op not in wanted:
            continue
        if node not in signs:
            raise ValueError(f"Missing sign for mapped generator {op!r} at node {node!r}.")
        if op in inv_mapping:
            raise ValueError(f"Generator {op!r} is assigned to more than one mapping node.")
        try:
            i, j = (nonnegative_integer(index, "plane index") for index in node)
        except (TypeError, ValueError):
            i = j = None
        if i is None or not i < j < n:
            raise ValueError(f"Invalid BDI mapping node {node!r}; expected 0 <= i < j < {n}.")
        sign = finite_real_scalar(signs[node], f"sign for node {node!r}")
        if sign not in (-1.0, 1.0):
            raise ValueError(f"The BDI sign for node {node!r} must be +1 or -1.")
        inv_mapping[op] = ((i, j), sign)

    out = np.zeros((n, n))
    for c, gen in zip(coeffs, generators, strict=True):
        if gen not in inv_mapping:
            raise ValueError(f"Generator {gen!r} is missing from the irrep mapping.")
        node, sign = inv_mapping[gen]
        out = out + c * sign * E(node, n, invol_type)
    return out


def map_matrix_to_reducible(matrix, mapping, signs, invol_type):
    _require_bdi(invol_type)
    op = {}
    for i, j in zip(*np.where(matrix), strict=True):
        if i < j:
            op[mapping[(i, j)]] = matrix[i, j] / 2 / signs[(i, j)]

    return PauliSentence(op)


def E(node, n, invol_type="BDI"):
    """Return the so(n) generator ``2 (E_ij - E_ji)`` of the rotation plane ``node = (i, j)``."""
    _require_bdi(invol_type)
    i, j = node
    e = np.zeros((n, n))
    e[i, j] = 2
    e[j, i] = -2
    return e
