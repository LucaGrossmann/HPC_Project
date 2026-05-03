# Part 0 — Python Reference Implementation

A pure-NumPy implementation of t-SVDM and t-SVDMII, living in
[`HPC_Project/cuda/`](../../cuda/). Serves as the **correctness
oracle** for Parts 1–5 (serial C++, OpenMP, MPI, CuPy, Julia). Cross-
validated against [`mprod_package`](https://pypi.org/project/mprod-package/),
the canonical open-source reference implementation of
[Kilmer et al. 2021](https://doi.org/10.1073/pnas.2015851118).

## 1. Why this exists

Every implementation in Parts 1–5 is a performance rewrite of the same
math. We need one implementation that is *obviously correct* and trivial
to read, so we can point at it when a C++ or GPU port disagrees. That's
Part 0.

Key constraint: Part 0 code is **not** on the performance hot path. It
is optimized for readability first. Speed is measured elsewhere.

## 2. Module layout

All code lives in `HPC_Project/cuda/`. The `cuda/` name is a Part 4
artifact — plan.md §3 places the Python source there so the same
modules can later be imported with CuPy arrays (GPU) at Part 4 time,
no copy-paste.

```
cuda/
├── tsvdm_ops.py      # 5 building-block ops
├── tsvdm_core.py     # tsvdm, tsvdmii, reconstruct
├── tsvdm_utils.py    # errors, random orthogonal, validation
├── requirements.txt
├── README.md
└── tests/
    ├── conftest.py           # seeded (A, M) fixtures across 3 sizes
    ├── test_ops.py           # per-op unit tests (11 tests)
    ├── test_reconstruction.py # the north-star assertion
    ├── test_orthogonality.py # slicewise U^T U ≈ I
    ├── test_tsvdmii.py       # truncation behavior, Eckart-Young
    ├── test_edge_cases.py    # n=1, m=p square, identity M
    └── test_oracle.py        # cross-check vs mprod_package
```

Dependency order: `tsvdm_ops` is leaf; `tsvdm_core` and `tsvdm_utils`
depend on `tsvdm_ops`; tests depend on everything.

## 3. Tensor shape convention — the single most important choice

All Python tensors are shape **`(n, m, p)`**, C-order. Frontal slice
*i* is `A[i]`, a contiguous `(m, p)` block.

```
   paper / C++                    Python (this implementation)
   shape  (m, p, n)               shape  (n, m, p)
   ──────────────────             ──────────────────
   A[:, :, k] is slice k          A[k] is slice k
```

**Why**: `np.linalg.svd(A)` broadcasts over leading axes. Passing an
`(n, m, p)` tensor triggers a single batched SVD with no Python loop,
no transpose, no hidden copy. With shape `(m, p, n)` we'd need a
`transpose(2, 0, 1)` round-trip at every SVD call — slower and noisier
to read.

**Rejected alternative**: `(m, p, n)` for paper/C++ consistency. That
consistency belongs in the C++ code (Part 1+), where column-major
layout makes `A[:, :, k]` contiguous for LAPACK. Python uses C-order;
using the paper's shape would fight NumPy's batching.

**Risk**: shape mismatch at the Python ↔ C++ boundary. Mitigation: the
oracle test (`test_oracle.py`) explicitly transposes `(n, m, p) →
(m, p, n)` before calling mprod, and the C++ fixture loader (Part 1)
will apply the inverse transpose on load. Both code paths must
document the convention in their file headers.

## 4. Algorithms

### 4.1 t-SVDM (`tsvdm(A, M)`)

Input: `A` shape `(n, m, p)`, `M` shape `(n, n)` orthogonal.

```
 1.  Â = A ×_3 M                             # mode-3 transform
 2.  for i in 0..n-1:
         Û[i], Ŝ[i], V̂[i]^T = SVD(Â[i])      # batched via np.linalg.svd
 3.  U = Û ×_3 M^T,  S = Ŝ ×_3 M^T,  V = V̂ ×_3 M^T
```

Because `M` is orthogonal, `M^{-1} = M^T` — we never explicitly invert.

Output shapes (thin SVD, `r = min(m, p)`):

| Factor | Shape |
|---|---|
| `U` | `(n, m, r)` |
| `S` | `(n, r, r)`  (each slice diagonal) |
| `V` | `(n, p, r)` |

### 4.2 t-SVDMII (`tsvdmii(A, M, gamma)`)

Post-processes a t-SVDM with an energy threshold `γ ∈ (0, 1]`.

```
 1.  Â = A ×_3 M; slice-wise SVD → Û, ŝ, V̂  (stay in transform domain)
 2.  v = sort([ŝ_{i,j}^2 for all i, j], descending)
 3.  J = smallest index with cumsum(v)[J] > γ · ‖Â‖_F²     (strict)
 4.  τ = √(v_J)                                             ← see note
 5.  for each slice i:  keep columns j where ŝ_{i,j} ≥ τ
 6.  return ragged (Û_trunc, Ŝ_trunc, V̂_trunc) in transform domain
```

**Return format is ragged (Python lists of per-slice ndarrays).**
Different slices keep different numbers of components. Zero-padding
to a common rank would destroy the storage savings (see paper §6D).
Callers that need a spatial-domain tensor pass the ragged factors
through `reconstruct(...)`, which pads internally and applies `M^T`.

**Note on τ.** Paper Algorithm 3 line 6 literally prints `τ := v_J`
where `v_J` is a *squared* singular value. But line 8 of the same
algorithm ("keep if `ŝ_{i,j} ≥ τ`") compares against singular values,
not squared ones. The paper's surrounding text is consistent with
`τ = √(v_J)`, so that's what we implement. plan.md §1 flags this and
`mprod_package` uses the same choice. **Any future port of t-SVDMII
must preserve `τ = √(v_J)`** — don't copy the paper's line-6 text
blindly.

### 4.3 `reconstruct(U, S, V, M)`

Polymorphic:
- If `U, S, V` are ndarrays (full t-SVDM output): `Û = U ×_3 M`, etc.,
  then `Â = Û @ Ŝ @ V̂^T` (facewise matmul), then `A ≈ Â ×_3 M^T`.
- If `U, S, V` are lists (t-SVDMII output, already in transform
  domain): assemble `Â[i] = U[i] @ S[i] @ V[i]^T` slice-by-slice into
  a dense `Â`, then apply `M^T`.

The branch keeps the ragged/full distinction internal to this function;
callers of either algorithm use the same API.

## 5. Design decisions

For each decision, the alternative we rejected and why.

### 5.1 NumPy-only (not CuPy-dispatching)

**Chose**: direct `np.einsum`, `np.linalg.svd`, `np.eye`, ...
**Rejected**: `xp = cupy.get_array_module(A)` pattern from plan.md §5.2.
**Why**: Part 4 (CuPy/GPU) is two steps away. Writing `xp.` in front of
every NumPy call adds noise without paying for itself until then. The
refactor to add dispatch at Part 4 time is mechanical (~10 line-level
substitutions across two files); we'd rather push it off than read
awkward code for weeks.

**Handoff to Part 4**: replace `np` with `xp = cupy.get_array_module(A)`
at the top of `tsvdm_core.py` and `tsvdm_ops.py`. The only non-trivial
call is `np.linalg.svd` — CuPy's equivalent has the same signature.

### 5.2 Shape `(n, m, p)` in Python (see §3 above)

### 5.3 t-SVDMII returns ragged in transform domain

**Chose**: list of per-slice ndarrays, no `M^T` applied.
**Rejected**: zero-padded dense ndarray in spatial domain.
**Why**: paper §6D — padding destroys the compression. The transform-
domain factors are the actual stored quantities; the spatial
approximation is a derived view. `reconstruct(...)` is the single point
where the `M^T` and padding happen.

### 5.4 τ = √(v_J) not v_J (see §4.2 note above)

### 5.5 Cross-check against `mprod_package`, not a hand fixture

**Chose**: `test_oracle.py` runs `mprod.decompositions.svdm` on the
same seeded input and compares reconstructions.
**Rejected**: commit a `.npz` file with hand-computed expected outputs.
**Why**: a single fixture proves the code matches itself across runs —
that's weaker than proving it matches an independent implementation of
the same algorithm. `mprod_package` is BSD-3 and authored by the same
group that popularized the algorithm; it's as close to a reference as
exists.

**Caveat**: `mprod.dimensionality_reduction` (their t-SVDMII) breaks on
numpy 2.x due to a removed `np.sctypeDict` entry, so we only cross-
check t-SVDM. t-SVDMII is a well-defined post-process on top of t-SVDM
factors, validated internally (γ=1 ≡ full t-SVDM, monotonic error,
Eckart-Young bound).

### 5.6 `star_m_product` and `star_m_transpose` implemented but unused by t-SVDM/II

**Chose**: implement all 5 ops from plan.md §6.
**Rejected**: skip the two ops that don't appear on the t-SVDM/II hot
path.
**Why**: matches the public-library API surface (plan.md §5.2), costs
~15 lines total, and serves as a sanity check that our mode-3 product
composes correctly (tested via `test_star_m_transpose_shape_and_inverse`
and `test_star_m_product_reduces_to_facewise_when_M_is_identity`).

### 5.7 Sign-correct the diagonal of R in `random_orthogonal`

**Chose**: multiply Q columns by `sign(diag(R))` after QR.
**Rejected**: use plain `np.linalg.qr` output.
**Why**: without the sign correction, the distribution over `O(n)` is
not Haar (some regions are over-represented). This matters for
reproducibility: the test seeds should produce statistically clean
orthogonal matrices. One extra line; worth it.

## 6. Test strategy

Every test fixes a seed (`np.random.default_rng(0)`) and runs on three
sizes: `(4, 8, 8)`, `(8, 16, 16)`, `(16, 32, 24)`. Total suite: 49
tests, < 1 second on a laptop. Parametrization keeps test code short;
the parameter ids in pytest output (`n8_m16_p16` etc.) make failures
traceable.

| File | What it proves |
|---|---|
| `test_ops.py` | Each op satisfies its algebraic identity (e.g., mode-3 of identity is a no-op; mode-3 is reversible by M^T; facewise = per-slice matmul; star-M transpose squares to identity). |
| `test_reconstruction.py` | **The** assertion: `‖A − reconstruct(tsvdm(A,M))‖_F / ‖A‖_F < 1e-10` on three sizes. |
| `test_orthogonality.py` | `Û[i]` and `V̂[i]` slicewise orthogonal (transform-domain property). |
| `test_tsvdmii.py` | γ=1 ≡ full t-SVDM; error monotone in γ; kept ranks ≤ r; error ≤ tail energy (Eckart-Young). |
| `test_edge_cases.py` | n=1, m=p square, identity M, γ near zero. |
| `test_oracle.py` | Our reconstruction agrees with `mprod_package`'s to 1e-10 on a seeded `(6, 10, 8)` tensor. |

Comparison is always on **reconstructions**, never raw factors — SVD
factors have column-level sign ambiguity that masks correctness when
compared directly.

## 7. Known limitations

- **float64 real only.** Complex / float32 paths are not written. If
  we later discover a need, both fall out of NumPy for free, but the
  test suite would need new fixtures.
- **No randomized variant.** Plan.md reserves that for post-deadline
  (`randomized_tsvdm` in the public library).
- **No benchmarks.** Part 0 is correctness only. Timing lives in Parts
  1–5.
- **Ragged factor lists are Python lists, not ndarrays.** This means
  slicing across slices (e.g., "the 3rd-largest singular value of
  slice 5") requires `S_list[5][2, 2]`, not `S[5, 2, 2]`. That's fine
  for the compression use case but might bite if t-SVDMII factors need
  to be passed to highly-vectorized downstream code.
- **mprod t-SVDMII cross-check disabled** (their dimensionality-reduction
  submodule is broken on numpy 2.x). Fix upstream, or pin numpy <2, or
  accept the gap — we chose the last for now.

## 8. Handoff to Parts 1–5

Parts 1–5 should:

1. Load a **fixture `.npz` file** generated from this Python
   implementation (seeded input + expected reconstruction). Each part
   adds its own `tests/fixtures/` as needed; we'll produce the first
   fixture when Part 1 starts.
2. For their own correctness check: compute `‖A - reconstruct(tsvdm(A,M))‖_F / ‖A‖_F`
   and assert `< 1e-10`. This is the same assertion this suite uses.
3. For cross-implementation validation (Part 6 report): run the same
   seeded `(A, M)` through every implementation and assert pairwise
   agreement to `1e-8` on the reconstruction.

The C++ shape convention (`(m, p, n)`, column-major) differs from the
Python one (`(n, m, p)`, C-order). Loading a fixture requires one
transpose at the boundary. Document this clearly in every C++ file
header.

## 9. Reproducing

```bash
# Dependencies
conda run -n claude pip install -r HPC_Project/cuda/requirements.txt

# Full test suite (< 1 second)
conda run -n claude python -m pytest HPC_Project/cuda/tests -v

# Interactive smoke test
conda run -n claude python -c "
import numpy as np
from cuda import tsvdm, reconstruct, random_orthogonal, relative_error
rng = np.random.default_rng(0)
A = rng.standard_normal((4, 8, 8))
M = random_orthogonal(4, rng)
U, S, V = tsvdm(A, M)
print(f'rel error: {relative_error(A, reconstruct(U, S, V, M)):.2e}')
"
```

Expected: 49 passed; smoke test prints `rel error: ~1e-15`.
