# KAK tools

Deterministic Cartan decompositions for quantum compilation. The package provides
matrix-level routines for the classical Cartan types and a Pauli-level BDI workflow
using [PauLie](https://github.com/QPauLie/PauLie) for algebra classification.

## Installation and tests

Requires Python >= 3.12 and PauLie >= 0.2.2:

```sh
pip install -e .
```

For the reproducible CPU test environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
```

`constraints-test.txt` records tested versions, including PennyLane 0.45.1 and
JAX/JAXLIB 0.7.1. `requirements.txt` also includes notebook/development tools.
Tests cover representative partitions and concrete numerical regressions.

## Native PauLie classification and KAK compilation

Use PauLie's factory and `Classification` directly. The KAK bridge supplies the
Pauli closure, signed matrix mapping and compilation specific to this package.

```python
from paulie import get_pauli_string
from kak_tools import kak_decomposition

generators = ["XX", "YY", "ZI", "IZ"]
classification = get_pauli_string(generators).get_class()
result = kak_decomposition(generators, [1.0, 0.7, -0.3, 0.5], time=0.83)

print(classification.get_algebra(), classification.get_dla_dim())
print(result.classification.get_orthogonal_size(), result.n_qubits)
result.pauli_rotations    # (PauliWord, coefficient, kind) triples
result.cartan_angles      # (PauliWord, rate) pairs, independent of time
result.reconstruct(4.2)   # reuse the compilation at another time
```

For an already complete generator list, omit `n` in `get_pauli_string`:
`get_pauli_string(list, n=...)` expands local translations, rather than just padding.
The example in `notebooks/paulie_bridge_example.py` uses PauLie's native
`get_all_k_local` to build an open chain and checks the physical evolution.

| API | Purpose |
| --- | --- |
| `Classification.get_algebra()`, `.get_dla_dim()`, `.get_subalgebras()` | Algebra name, dimension and summands |
| `Classification.is_simple()`, `.get_simple_component()` | Test simplicity and retrieve the unique simple summand |
| `Classification.get_orthogonal_size()` | Find an isomorphic so(m) presentation, or return `None` |
| `Classification.get_algebra_basis()` | Basis in PauLie's classified presentation |
| `paulie.common.algebra_basis.get_so_basis(m)` | Basis in the chosen so(m) presentation |
| `dla_pauli_basis` | Complete native Pauli closure, converted to PennyLane words |
| `map_dla_to_irrep` | Return `(mapping, signs, classification)` for signed rotation planes |
| `labelled_matrix_basis` | Label PauLie's orthogonal matrices with verified Pauli words |
| `pauli_string_to_word`, `pauli_word_to_string` | Convert between PauLie and PennyLane |

For `2*so(3)`, the classified basis uses 6×6 matrices, while `get_so_basis(4)` uses
4×4 matrices. Dimension alone is insufficient: so(7) and sp(3) both have dimension 21.
`get_simple_component()` raises PauLie's `ClassificationException` for nonsimple algebras.

### Migration

`classify_dla`, `DLAInfo` and `DLAComponent` are removed. Use the native factory above
for strings/native inputs; `as_pauli_collection(generators).get_class()` preserves mixed/PennyLane normalization.
`result.classification` replaces `result.info`; register width is `result.n_qubits`.
`dla_pauli_basis` and `map_dla_to_irrep` no longer accept `info`. Pass the returned
classification to `labelled_matrix_basis(mapping, signs, classification, n_qubits=...)`.
`full_workflows.paulie_workflow(..., t0=t)` is removed; use
`kak_decomposition(..., time=t).pauli_rotations`.

### Inputs and conventions

The bridge accepts strings, PauLie strings/collections, PennyLane `PauliWord`s and
real multiples of single Pauli operators, including iterators. Operator sums are
rejected: splitting a sum changes the independently generated algebra. Separate
`coefficients` multiply intrinsic weights before duplicate Hamiltonian terms combine.
Trailing identities and explicit Identity wires preserve register width. Native bit
inputs are normalized to consistent endianness. Coefficients and times must be finite
real scalars; numeric complex scalars with zero imaginary part are accepted.

The physical convention is **exp(+it ΣcP)**. A PennyLane `PauliRot` takes
`-2 * coefficient`, additionally multiplied by time for central `a0` rates.
`matrix_factors` stores `(matrix, start, end, kind)` group factors at the original
time; these differ from `recursive_bdi`'s arrays of CS angles.

### Algorithm and scope

- The Pauli workflow requires a compatible so(m) basis and horizontal generators
  for BDI(p,q). Even and odd m work. Supplying only `p` or `q` in `invol_kwargs`
  infers the complement; both sizes must be positive and sum to m. The default
  partition is balanced; alternatives are not searched automatically.
- Abstract isomorphism with so(m) does not guarantee a signed bijection between
  individual Pauli words and rotation planes. Independent su(2) factors on separate
  qubits may require compilation component by component.
- Zero separate coefficients retain the generator algebra. Intrinsically zero
  PennyLane operators have no Pauli support and are omitted.
- The bridge computes the full native closure before checking its classified
  dimension. `dla_pauli_basis(..., strict=False)` warns on a dimension mismatch.
- A complete, distinct Pauli basis and its star commutators certify the signed
  mapping. A signed SVD factors H=KAKᵀ; the right gate list is the exact inverse of
  the left, preserving the physical lift. Central rates remain unwrapped at all times.
- No rotations are discarded by default. Explicit `tol` drops small vertical
  rotations while retaining central rates. `validate=True` checks reconstruction;
  floating-point errors can accumulate at very large times.
- `dense_cartan.bdi` handles horizontal or general group matrices, returning block
  factors and CS angles. `numerical_decompositions.bdi_kak` returns full matrices.
  Other classical types are available through `numerical_decompositions`; CII
  supports unequal/empty partitions and repeated or endpoint angles. Its factors
  are checked in double precision; arbitrary noisy inputs and relative accuracy
  at arbitrarily small angles are not guaranteed.
