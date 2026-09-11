# KAK tools

Deterministic Cartan decompositions for quantum compilation. The package contains
matrix-level routines for the classical Cartan types and a Pauli-level BDI workflow
configured by [PauLie](https://github.com/QPauLie/PauLie).

## Installation and tests

Requires Python >= 3.12 and PauLie >= 0.2.2:

```sh
pip install -e .
```

Use a virtual environment for the reproducible CPU test dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
```

`constraints-test.txt` records the tested versions, including PennyLane 0.45.1 and
JAX/JAXLIB 0.7.1. `requirements.txt` additionally includes notebook/development tools.
The tests use representative partitions and fixed regressions rather than broad
Cartesian sweeps over dimensions, seeds and validation flags.

## PauLie bridge

PauLie classifies the generator algebra; the bridge chooses its orthogonal
presentation and constructs a verified Pauli-word mapping. Dimension alone would
be ambiguous: both so(7) and sp(3) have dimension 21.

```python
from kak_tools import classify_dla, kak_decomposition

# Two-qubit transverse-field XY model.
generators = ["XX", "YY", "ZI", "IZ"]
info = classify_dla(generators)
result = kak_decomposition(generators, [1.0, 0.7, -0.3, 0.5], time=0.83)

print(info.algebra, result.irrep_size, result.reconstruction_error)
result.pauli_rotations    # (PauliWord, coefficient, kind) triples
result.cartan_angles      # (PauliWord, rate) pairs, independent of time
result.reconstruct(4.2)   # reuse the compilation at another time
```

See `notebooks/paulie_bridge_example.py` for more models.

| API | Purpose |
| --- | --- |
| `classify_dla` | Wrap PauLie's classification in `DLAInfo` |
| `DLAInfo.algebra`, `.dim`, `.components` | Algebra name, dimension and summands |
| `DLAInfo.is_algebra`, `.orthogonal_size`, `.is_simple`, `.simple_component` | Delegate algebra properties to PauLie |
| `DLAInfo.matrix_basis`, `.orthogonal_basis` | Basis in the classified presentation or its so(m) presentation |
| `dla_pauli_basis` | Complete native Pauli closure, checked against the classified dimension |
| `map_dla_to_irrep` | Map the Pauli basis to signed rotation planes |
| `labelled_matrix_basis` | Label PauLie's orthogonal matrices with the verified Pauli words |
| `kak_decomposition` | Compile a horizontal Hamiltonian into reusable Pauli rotations |
| `pauli_string_to_word`, `pauli_word_to_string` | Convert between PauLie strings and PennyLane words |

`matrix_basis` follows the classified presentation, so 2*so(3) uses 6×6 matrices;
`orthogonal_basis` follows its so(4) presentation and uses 4×4 matrices.
`simple_component` propagates PauLie's `ClassificationException` for nonsimple algebras.

### Inputs and conventions

Generators can be strings, PauLie strings/collections, PennyLane `PauliWord`s or
real multiples of single Pauli operators, including iterators. Operator sums are
rejected because splitting them would change the independently generated algebra.
The separate `coefficients` argument has one entry per original term; it multiplies
intrinsic weights before duplicate Hamiltonian terms are combined.

Trailing identities and explicit Identity wires preserve register width. Native
bit inputs are normalized to consistent endianness. Coefficients and times must be
finite real scalars; numeric complex scalars with zero imaginary part are accepted.

The physical convention is **exp(+it ΣcP)**. A PennyLane `PauliRot` takes
`-2 * coefficient`, additionally multiplied by time for central `a0` rates.
`matrix_factors` stores `(matrix, start, end, kind)` group factors at the original
time; these differ from `recursive_bdi`'s arrays of CS angles.

### Algorithm and scope

- The Pauli workflow requires a compatible so(m) basis and horizontal generators
  for BDI(p,q). Even and odd m work. `invol_kwargs={"p": p}` or `{"q": q}` infers
  the complement; supplying both requires positive sizes with p+q=m. The default
  partition is balanced; alternatives are not searched automatically.
- Abstract isomorphism with so(m) does not guarantee a signed bijection between
  individual Pauli words and rotation planes. Independent su(2) factors on separate
  qubits may require compilation component by component.
- Zeros in the separate `coefficients` argument retain the generator algebra.
  Intrinsically zero PennyLane operators have no Pauli support and are omitted.
- The bridge computes the full native closure before checking its dimension.
  Supplied `DLAInfo` is checked against a fresh classification because PauLie's
  classification objects are mutable. Incompatible metadata raises, or warns and
  uses the actual algebra with `dla_pauli_basis(..., strict=False)`.
- A complete, distinct Pauli basis and its star commutators certify the signed
  so(m) mapping. A signed SVD then factors the horizontal Hamiltonian as H=KAKᵀ.
  The right gate list is the exact inverse of the left, preserving the physical
  lift. Central rates remain unwrapped for zero, negative and later times.
- No rotations are discarded by default. Explicit `tol` approximates by dropping
  small vertical rotations; central rates are retained. `validate=True` checks
  reconstruction at the supplied time. Floating-point errors can accumulate at
  very large times.
- `dense_cartan.bdi` handles horizontal or general group matrices, returning block
  factors and CS angles. `numerical_decompositions.bdi_kak` returns full matrices
  using the same general BDI kernel.
- Other classical types are available through `numerical_decompositions`.
  `cii_kak` uses coupled symplectic bases without angle clustering, including
  unequal/empty partitions and repeated or endpoint angles. Factor structure is
  checked in double precision; arbitrary noisy inputs and relative precision at
  arbitrarily small angles are not guaranteed.

The implementation separates input normalization (`_pauli_inputs`), shared
parameter checks (`_validation`), mapping (`map_to_irrep`), numerical kernels and
the coordinating bridge. `notebooks/` contains examples and applications.
