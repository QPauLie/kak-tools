"""This file contains tools for handling DLAs made of Pauli words."""

import copy
import warnings
from itertools import combinations, product
import networkx as nx

import pennylane as qml

from ._validation import require


def anticom_graph_pauli(paulis):
    """Compute the anticommutation graph of a set of Pauli words.

    Args:
        paulis (List[qml.pauli.PauliWord]): The Pauli words.

    Returns
        networkx.Graph: The anticommutation graph, which is an undirected, unweighted
        graph.
    """
    require(all(isinstance(p, qml.pauli.PauliWord) for p in paulis), "paulis must contain PauliWords.")
    graph = nx.Graph()
    graph.add_nodes_from(paulis)
    graph.add_edges_from(
        ((p1, p2) for p1, p2 in combinations(paulis, r=2) if not p1.commutes_with(p2))
    )
    return graph


def split_pauli_algebra(dla, verbose=False):
    """Split a list of Pauli words that make up a DLA into multiple sublists
    that make up the connected components of the anticommutation graph of the DLA.
    Note that these components may not be simple by themselves, but can be a semisimple
    algebra with 2^{n_C} isomorphic simple components, c.f., Theorm 2 in
    `Aguilar et al. <https://arxiv.org/pdf/2408.00081>`__.

    Args:
        dla (List[qml.pauli.PauliWord]): List of Pauli words that make up the DLA.
        verbose (bool): Whether or not to print a status report.

    Returns:
        list[set[qml.pauli.PauliWord]]: A list of sets, with each set containing the Pauli
        words that make up a connected component of the anticommutation graph of ``dla``.

    """
    require(all(isinstance(op, qml.pauli.PauliWord) for op in dla), "dla must contain PauliWords.")
    # Create fully disconnected graph with Pauli words as nodes
    graph = anticom_graph_pauli(dla)
    # Get connected components of the graph. The components are given as collections of nodes
    comps = list(nx.connected_components(graph))
    num_comps = len(comps)
    if num_comps == 1:
        comps = comps[0]
    if verbose:
        dims = len(comps) if num_comps == 1 else [len(comp) for comp in comps]
        plural = "s" * (num_comps > 1)
        print(f"Found {num_comps} component{plural} with dimension{plural} {dims}.")

    return comps


def lie_closure_pauli_words(generators, verbose=False, max_iterations=10000, full_size=None):
    """Compute the Lie closure of a list of Pauli words.

    Args:
        generators (List[qml.pauli.PauliWord]): The generators of the algebra.
        verbose (bool): Whether to print status updates while closing the set.
        max_iterations (int): Maximum number of iterations, corresponding to max commutator order
        full_size (int): Size of the closed algebra. If provided and this number of generators is
            found, the iteration will be interrupted, to save cost. Note that this means that
            if the wrong size is provided, the closure might fail.

    Returns:
        List[qml.pauli.PauliWord]: The elements of the closed Pauli word Lie algebra.
    """

    require(all(isinstance(op, qml.pauli.PauliWord) for op in generators), "generators must contain PauliWords.")
    dla = copy.copy(generators)
    # Membership is tested once per commutator, so keep a set alongside the list: `in` on
    # a list of d Pauli words is O(d), which makes the closure O(d^3) rather than O(d^2).
    seen = set(dla)
    epoch = 0
    old_length = 0  # dummy value
    new_length = initial_length = len(dla)

    while (new_length > old_length) and (epoch < max_iterations):
        if verbose:
            print(f"epoch {epoch+1} of lie_closure, DLA size is {new_length}")

        for pw1, pw2 in product(dla[:initial_length], dla[old_length:new_length]):
            if pw1.commutes_with(pw2):
                continue
            com = pw1._matmul(pw2)[0]
            if com not in seen:
                seen.add(com)
                dla.append(com)
                if len(dla) == full_size:
                    # The caller told us how big the algebra is, so we are done.
                    if verbose > 0:
                        print(f"Reached the announced DLA size of {full_size}")
                    return dla

        # Updated number of linearly independent PauliSentences from previous and current step
        old_length = new_length
        new_length = len(dla)
        epoch += 1

        if epoch == max_iterations:
            warnings.warn(
                f"reached the maximum number of iterations {max_iterations}", UserWarning, stacklevel=2
            )
        if new_length == full_size:
            break

    if verbose > 0:
        print(f"After {epoch} epochs, reached a DLA size of {new_length}")

    return dla
