"""End-to-end example: PauLie classifies the DLA, kak_tools decomposes it.

Run with ``python notebooks/paulie_bridge_example.py`` after installing both packages::

    pip install -e .
    pip install paulie

The script walks through four things:

1. What ``kak_tools`` can say about a DLA on its own (a list of candidates).
2. What PauLie says (one answer).
3. The full KAK decomposition, configured from PauLie's answer, verified numerically.
4. The same pipeline on a model that is *not* the transverse-field XY chain, which the
   hard-coded workflow in ``kak_tools.full_workflows`` cannot handle.
"""

import numpy as np

from kak_tools import (
    classify_dla,
    dla_pauli_basis,
    identify_algebra,
    kak_decomposition,
)


def k_local(pattern, n_qubits):
    """All translations of ``pattern`` along an open chain of ``n_qubits`` qubits."""
    width = len(pattern)
    return [
        "".join(["I"] * w + list(pattern) + ["I"] * (n_qubits - w - width))
        for w in range(n_qubits - width + 1)
    ]


def model(patterns, n_qubits):
    """Generators of a translation-invariant model on an open chain."""
    generators = []
    for pattern in patterns:
        generators += k_local(pattern, n_qubits)
    return list(dict.fromkeys(generators))


def rule(title):
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------------------
# 1. Identifying the algebra without PauLie
# ---------------------------------------------------------------------------

rule("1. What kak_tools can tell on its own")

generators = model(["XX", "XZ"], n_qubits=4)
print(f"generators              {generators}")

dla = dla_pauli_basis(generators)
print(f"DLA dimension           {len(dla)}")

candidates = identify_algebra(list(dla))
print("identify_algebra says   " + " or ".join(
    f"{'' if mult == 1 else f'{mult} x '}{kind}({size})" for mult, kind, size in candidates
))
print("...which is as far as a dimension count can get you: so(7) and sp(3) are both")
print("21-dimensional, and they need different Cartan involutions.")


# ---------------------------------------------------------------------------
# 2. What PauLie says
# ---------------------------------------------------------------------------

rule("2. What PauLie says")

info = classify_dla(generators)
print(f"algebra                 {info.algebra}")
print(f"components              {[str(c) for c in info.components]}")
print(f"dimension               {info.dim}  (matches the closure: {len(dla)})")
print(f"simple                  {info.is_simple}")
print(f"so(m) presentation      m = {info.orthogonal_size}")
print("\nOne answer, no enumeration of the algebra required to get it.")


# ---------------------------------------------------------------------------
# 3. Full decomposition of the transverse-field XY model
# ---------------------------------------------------------------------------

rule("3. Fixed-depth decomposition of exp(t H), tfXY on 4 qubits")

n_qubits = 4
generators = model(["XX", "YY", "Z"], n_qubits)
rng = np.random.default_rng(20250805)
coefficients = rng.normal(0.0, 1.0, len(generators))
coefficients /= np.linalg.norm(coefficients)
time = 0.83

info = classify_dla(generators)
print(f"PauLie classification   {info.algebra}   (dim {info.dim})")

result = kak_decomposition(generators, coefficients, time=time)
print(f"irrep size              {result.irrep_size}")
print(f"involution              {result.involution}")
print(f"Pauli rotations         {len(result.pauli_rotations)}")
print(f"Cartan (a0) rotations   {len(result.cartan_angles)}   <- the only t-dependent ones")
print(f"reconstruction error    {result.reconstruction_error:.2e}")

print("\nThe central Cartan block, whose angles are linear in t:")
for word, angle in result.cartan_angles:
    print(f"    exp({angle * time:+.5f} * {word})")

vertical = [item for item in result.pauli_rotations if item[2] in ("k1", "k2")]
print(f"\nFirst few of the {len(vertical)} vertical (K) rotations, independent of t:")
for word, angle, kind in vertical[:4]:
    print(f"    [{kind}] exp({angle:+.5f} * {word})")

assert np.allclose(result.reconstruct(), result.unitary_irrep, atol=1e-8)
print("\nRecomposing the rotations reproduces exp(t H) exactly.")
print("The matrices used throughout are PauLie's own so(m) basis, labelled with the Pauli")
print("words that map_simple_to_irrep matched them to:")
print(f"    PauLie get_algebra_basis()  {info.matrix_basis.shape}")
print(f"    PauLie so(m) basis          {info.orthogonal_basis.shape}")
print(f"    labelled with Pauli words   {len(result.algebra_basis)} entries")

print("\nRe-running at a different time reuses the same circuit structure:")
for other_time in [0.1, 1.7, 4.2]:
    other = kak_decomposition(generators, coefficients, time=other_time)
    same_structure = [w for w, _, _ in other.pauli_rotations] == [
        w for w, _, _ in result.pauli_rotations
    ]
    print(
        f"    t = {other_time:4.1f}: same Pauli sequence = {same_structure}, "
        f"error {other.reconstruction_error:.1e}"
    )


# ---------------------------------------------------------------------------
# 4. Models the hard-coded workflow does not cover
# ---------------------------------------------------------------------------

rule("4. Other models, same code path")

for patterns, n_qubits in [
    (["XX", "Z"], 4),  # transverse-field Ising
    (["XX", "Z"], 6),
    (["XX", "YY", "Z"], 5),  # transverse-field XY, odd chain
    (["XY"], 6),  # so(6), reached through PauLie's 'so(6)' naming
    (["XY"], 4),  # PauLie says '2*so(3)'; that is so(4)
]:
    generators = model(patterns, n_qubits)
    info = classify_dla(generators)
    coefficients = np.linspace(0.2, 1.0, len(generators))
    try:
        result = kak_decomposition(generators, coefficients, time=0.6)
        status = (
            f"{len(result.pauli_rotations):3d} rotations, "
            f"error {result.reconstruction_error:.1e}"
        )
    except NotImplementedError as exc:
        status = f"not supported: {str(exc).split('.')[0]}"
    print(f"  {'+'.join(patterns):12s} on {n_qubits} qubits -> {info.algebra:10s} {status}")

print(
    "\nNone of this needed the algebra to be supplied by hand: PauLie's classification\n"
    "picks the irrep size and involution for every one of them."
)
