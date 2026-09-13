# KAK tools

This fork extends [the original kak-tools](https://github.com/dwierichs/kak-tools)
with a Pauli-level BDI compilation workflow using
[PauLie](https://github.com/QPauLie/PauLie) for algebra classification. It adds
Pauli/PennyLane input conversion, signed mappings to so(m) matrices and reusable
Pauli-rotation decompositions, including odd matrix sizes and explicit BDI partitions.

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

`constraints-test.txt` records tested versions, including PennyLane 0.45.1.
The optional extras are `test` (pytest), `jax` (the JAX/JAXLIB 0.7.1 pair
PennyLane 0.45.1 documents, for running compiled circuits in JAX) and
`notebooks` (matplotlib, tqdm, jupyter); `requirements.txt` installs all of them.
Tests cover representative partitions and concrete numerical regressions.

### Working against a PauLie checkout

An editable PauLie install records its version at install time. If the checkout
has since bumped its version, `pip install -e .` resolves `paulie>=0.2.2` from
PyPI and replaces the editable checkout with the wheel. Either re-install the
checkout first, so its metadata satisfies the pin locally,

```sh
pip install -e /path/to/PauLie
pip install -e .
```

or skip dependency resolution for this package altogether:

```sh
pip install -e . --no-deps
pip install -c constraints-test.txt pytest
```

Repeat the PauLie re-install after every PauLie version bump.

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
result.pauli_rotations    # PauliRotation(word, coefficient, kind) records
result.cartan_angles      # (PauliWord, rate) pairs, independent of time
result.partition          # the BDI(p, q) sizes that were used
result.reconstruct(4.2)   # reuse the compilation at another time
result.pennylane_ops(4.2) # qml.PauliRot gates in circuit order
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
| `dla_pauli_basis` | Complete native Pauli closure, converted to PennyLane words |
| `map_dla_to_irrep(generators, n_qubits=..., invol_kwargs=...)` | Return `(mapping, signs, classification)` for signed rotation planes |
| `labelled_matrix_basis(mapping, signs, classification, n_qubits=...)` | Verified signed so(m) generator matrices `2 * sign * (E_ij - E_ji)` per Pauli word |
| `kak_decomposition`, `KAKResult`, `PauliRotation` | Compile `exp(time * H)` into typed Pauli rotations |
| `reconstruct_from_pauli_rotations` | Recompose a matrix from ordered rotations |
| `pauli_string_to_word`, `pauli_word_to_string` | Convert between PauLie and PennyLane |

For `2*so(3)`, the classified basis uses 6×6 matrices, while the bridge works in the
4×4 so(4) presentation. Dimension alone is insufficient: so(7) and sp(3) both have
dimension 21. `get_simple_component()` raises PauLie's `ClassificationException` for
nonsimple algebras.

### Inputs and conventions

The bridge accepts strings, PauLie strings/collections, PennyLane `PauliWord`s and
real multiples of single Pauli operators, including iterators. Operator sums are
rejected: splitting a sum changes the independently generated algebra. Separate
`coefficients` multiply intrinsic weights before duplicate Hamiltonian terms combine.
Trailing identities and explicit Identity wires preserve register width. Native bit
inputs are normalized to consistent endianness. Coefficients and times must be finite
real scalars; numeric complex scalars with zero imaginary part are accepted.
For mixed or PennyLane inputs, `as_pauli_collection(generators).get_class()`
returns the native PauLie classification after input normalization.

The physical convention is **exp(+it ΣcP)**. `result.pauli_rotations` lists the
factors of the left-to-right matrix product U = R₁ R₂ … R_N with R_k = exp(+i c_k P_k),
which is what `reconstruct` multiplies out. A circuit applies the gates in **reversed**
list order, each as `qml.PauliRot(-2 * c_k, P_k)` with `c_k` additionally multiplied by
time for central `a0` rates; `result.pennylane_ops(time)` returns exactly that list.

### Algorithm and scope

- The Pauli workflow requires a compatible so(m) basis and horizontal generators
  for BDI(p,q). Even and odd m work. Supplying only `p` or `q` in `invol_kwargs`
  infers the complement; both sizes must be positive and sum to m. Without
  `invol_kwargs` the balanced partition is tried first and the unbalanced ones next,
  since some generator sets are horizontal only there; `result.partition` records
  the choice.
- Abstract isomorphism with so(m) does not guarantee a signed bijection between
  individual Pauli words and rotation planes. Independent su(2) factors on separate
  qubits may require compilation component by component.
- Every listed generator belongs to the generator family, whatever its weight;
  intrinsic PennyLane factors and separate coefficients only shape the Hamiltonian.
  Widen the register with `n_qubits` or an Identity factor on a term.
- The bridge computes the full native closure before checking its classified
  dimension. `dla_pauli_basis(..., strict=False)` warns on a dimension mismatch.
- A complete, distinct Pauli basis and its star commutators certify the signed
  mapping. A signed SVD factors H=KAKᵀ; the right gate list is the exact inverse of
  the left, preserving the physical lift. Central rates remain unwrapped at all times.
- No rotations are discarded by default. Explicit `tol` drops small vertical
  rotations while retaining central rates. `validate=True` checks the time-independent
  identity K₁AK₁ᵀ = H (reported as `reconstruction_error`) and the recomposed
  exp(time H), with tolerances scaled by |H| and |time H| respectively.
- Numerical corrections cover CII with unequal/empty partitions and repeated or
  endpoint angles, and DIII with degenerate eigenvalues and small rotations.
  Factors are checked in double precision; arbitrary noisy inputs and relative
  accuracy at arbitrarily small angles are not guaranteed.
