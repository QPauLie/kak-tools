# KAK tools
Tools for automatic, deterministic KAK decompositions of Lie algebras.

The package can be pip-installed, for example via `pip install -e .` while in the main directory.

This repository contains code that is part of our publication on KAK decompositions for compilation.
It is structured as follows:
- `kak_tools/`: Source code files containing all tools
- `notebooks/`: Notebooks/scripts containing examples, applications and visualizations
- `tests/`: Minimal tests for the tools in `kak_tools/`

## PauLie bridge

A KAK decomposition needs to know *which* algebra it is decomposing: the irrep size `n`
and a Cartan involution. On its own this package either takes that on faith — `n_so = 2 * n`
is hard-coded in `kak_tools/full_workflows.py` — or guesses it from the dimension via
`identify_algebra`, which is deliberately non-unique. A 21-dimensional simple DLA, for
instance, is consistent with both `so(7)` and `sp(3)`, and those need different involutions.

[PauLie](https://github.com/QPauLie/PauLie) settles the question. It classifies the dynamical
Lie algebra of a set of Pauli-string generators exactly and in polynomial time, returning a
decomposition such as `so(8)` or `u(1)+2*su(2)`. `kak_tools.paulie_bridge` wires the two
together, so the decomposition is configured from the classification instead of by hand:

```python
from kak_tools import kak_decomposition, classify_dla

n = 4
generators = (
    [f"{'I' * w}XX{'I' * (n - w - 2)}" for w in range(n - 1)]
    + [f"{'I' * w}YY{'I' * (n - w - 2)}" for w in range(n - 1)]
    + [f"{'I' * w}Z{'I' * (n - w - 1)}" for w in range(n)]
)

classify_dla(generators).algebra      # 'so(8)' -- one answer, not a candidate list

result = kak_decomposition(generators, coefficients, time=0.83)
result.irrep_size                     # 8, taken from the classification
result.pauli_rotations                # [(PauliWord, angle, 'k1' | 'k2' | 'a0' | 'a'), ...]
result.cartan_angles                  # the only t-dependent rotations
result.reconstruction_error           # ~1e-14; the rotations are checked against exp(t H)
```

`notebooks/paulie_bridge_example.py` runs the whole thing end to end, on several models.

### What the bridge provides

| Function | Purpose | Delegates to |
| --- | --- | --- |
| `classify_dla` | Converts kak-tools-flavoured generators to a PauLie collection and returns its classification, wrapped as a `DLAInfo` | `PauliStringCollection.get_class` |
| `DLAInfo.algebra` / `.dim` / `.components` | Name, dimension and summands | `get_algebra`, `get_dla_dim`, `get_subalgebras` |
| `DLAInfo.matrix_basis` / `.orthogonal_basis` | Matrix basis of the algebra, as named and as `so(m)` | `get_algebra_basis`, `get_so_basis` |
| `DLAInfo.is_algebra` | Isomorphism tests, including the low-rank coincidences | `Classification.is_algebra` |
| `DLAInfo.orthogonal_size` | The `m` for which the DLA is `so(m)` | `get_simple_dim` + `is_algebra` |
| `dla_pauli_basis` | Pauli-word basis of the DLA, with the dimension known in advance | `lie_closure_pauli_words(..., full_size=)` |
| `labelled_matrix_basis` | PauLie's `so(m)` basis matrices, labelled with the Pauli words they correspond to | `get_so_basis` + `map_simple_to_irrep` |
| `map_dla_to_irrep` | `map_simple_to_irrep`, with `n` supplied by the classification | `map_simple_to_irrep` |
| `kak_decomposition` | The full workflow: generators in, verified Pauli rotations out | `recursive_bdi` etc. |
| `pauli_string_to_word` / `pauli_word_to_string` | PauLie `PauliString` to/from PennyLane `PauliWord` | — |

Nothing about the algebra is recomputed here: `DLAInfo` holds PauLie's `Classification`
object and forwards to it, and `DLAComponent` is parsed straight out of PauLie's own
naming. The bridge only supplies the one thing PauLie has no reason to know, namely which
irrep size and involution `kak_tools` should be configured with — and even the `so(m)`
identification is PauLie's `is_algebra`, not a local table of isomorphisms.

The two bases are deliberately distinct. `matrix_basis` is `get_algebra_basis()` verbatim,
so it follows the classified presentation — a DLA named `2*so(3)` is based in `6x6`.
`orthogonal_basis` is the `so(m)` presentation that `kak_tools` decomposes in, so the same
algebra is based in `4x4`. For a DLA PauLie already names `so(m)` the two coincide.

Generators may be given as Pauli strings (`"XXII"`), PennyLane `PauliWord`s, PennyLane
operators (`qml.X(0) @ qml.X(1)`), PauLie `PauliString`s or a PauLie `PauliStringCollection`.

### Installation

```sh
pip install -e .        # pulls in paulie, which requires Python >= 3.12
```

### Scope and known limitations

- The Pauli-word pipeline covers `so(m)` with a `BDI` involution, which is what
  `map_simple_to_irrep` implements. The low-rank coincidences are recognised, so a DLA
  PauLie names `2*so(3)` is still decomposed as `so(4)`.
- **Odd `m` is not supported.** The top-level split is then `BDI(p, p+1)`, whose horizontal
  cosine-sine decomposition has a gauge freedom that `dense_cartan.bdi` does not fix, and
  the `k1 = k2.T` relation fails for most Hamiltonians. `kak_decomposition` raises a
  `NotImplementedError` explaining this rather than returning a wrong answer. Even `m` --
  what qubit models with a free-fermionic DLA give -- is fully supported.
- Algebras that are not `so(m)` are refused with a message naming what PauLie found. The
  matrix-level routines in `numerical_decompositions` cover the other classical types, and
  can be called directly.
- `kak_decomposition` validates its own output by default: it recomposes `exp(t H)` from the
  Pauli rotations and raises if the result does not match.

### Fixes made along the way

Wiring PauLie in exposed three bugs in the decomposition path, since PauLie hands the
pipeline algebras it was never run on by hand. All three are fixed here:

1. `dense_cartan.angles_to_reducible` assigned the cosine-sine angles of a block of width
   `w` to the Pauli words indexed `(i, i + w // 2)`. The rotation actually couples `i` with
   `i + (w - w // 2)`. The two agree for even `w`, so this only surfaced once `so(6)`,
   `so(10)`, `so(12)`, ... were being decomposed -- where it silently produced a wrong
   decomposition rather than an error.
2. `dense_cartan.bdi` raised a bare `ValueError` whenever scipy's `cossin` returned a
   decomposition with `k11[:, i] == -k21[i]`, which is a sign gauge that leaves
   `k1 @ a @ k2` invariant and can simply be repaired (flip the row and shift the angle by
   `pi`). It is now repaired. Whether it triggered was coefficient-dependent -- for `so(4)`
   it was tripping on roughly three quarters of random Hamiltonians.
3. `dense_cartan.group_matrix_to_reducible` read a rotation angle off `arcsin` of the
   off-diagonal entry, so a `2x2` block equal to `diag(-1, -1)` -- a rotation by exactly
   `pi`, with no off-diagonal entry to be found -- was silently dropped. Angles are now read
   with `arctan2` and leftover `-1` pairs are emitted explicitly. This is what a
   translation-invariant Hamiltonian with uniform coefficients runs into.

`lie_closure_pauli_words` also picked up two fixes while the bridge was leaning on it: it
referenced `warnings` without importing it, so hitting `max_iterations` raised `NameError`
instead of warning; and it tested `com not in dla` against a *list*, which is `O(dim)` per
commutator and makes the closure `O(dim^3)`. It now keeps a set alongside, and honours
`full_size` as soon as the algebra is complete rather than at the end of the sweep.

Smaller ones: `map_to_irrep.irrep_dot` did not accept the `(mapping, signs)` pair that
`map_simple_to_irrep` returns, so `full_workflows.complete_workflow_tfXY` raised a
`TypeError` on every call; the `coefficients` argument of the `full_workflows` entry points
was ignored in favour of a hard-coded `"random"`; and `tests/test_map_to_irrep.py` imported
`structure_constants_dense` from a PennyLane location that no longer exists.
