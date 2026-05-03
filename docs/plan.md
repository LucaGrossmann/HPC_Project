# `tsvdm` — Project Plan

**Project:** CS 2050 final + open-source research library + short arxiv preprint.
**Algorithm:** t-SVDM and t-SVDMII from Kilmer, Horesh, Avron & Newman (2021, PNAS).
**Primary goals, in order:**
1. Full marks on the 6 course parts.
2. A polished public GitHub repository other researchers can use.
3. Language parity across Python / Julia / C++.
4. Short arxiv preprint summarizing the parallelism study.

**Strategic structure:** build everything in a single course repo (`HPC_Project/`). After May 7, extract the reusable Python + Julia pieces into a separate public library repo and publish to PyPI. No parallel development of two repos, no vendoring during the course, no private/public juggling. Post-deadline extraction is a short one-time job (~250 lines of Python + ~80 lines of Julia) and avoids any ambiguity about where code originated.

**Reference oracle:** [`mprod_package`](https://github.com/UriaMorP/mprod_package) on PyPI (BSD-3, authored by UriaMorP) is already the software implementation of the Kilmer et al. 2021 paper — it includes `tsvdm` and t-SVDMII on CPU Python. We use it as the validation oracle for every implementation we write, rather than writing our own Python reference from scratch. This changes the arxiv contribution framing (§7) and eliminates the "first open-source implementation" claim — the contribution is the **HPC/parallel study** (OpenMP, MPI, CUDA, Julia) plus the randomized variant, validated against `mprod_package`.

**North star for every decision:** simplicity and correctness. A smaller, well-tested, well-documented deliverable beats a sprawling one.

---

## 1. Algorithm recap (what we actually compute)

Given a third-order tensor $A \in \mathbb{R}^{m \times p \times n}$ and an orthogonal matrix $M \in \mathbb{R}^{n \times n}$:

**t-SVDM** (Algorithm 2 in the paper):
1. $\hat{A} = A \times_3 M$  *(apply $M$ along tube fibers)*
2. For $i = 1, \dots, n$: $(\hat{U}_{:,:,i},\ \hat{S}_{:,:,i},\ \hat{V}_{:,:,i}) = \mathrm{svd}(\hat{A}_{:,:,i})$  *(n independent slice SVDs)*
3. $U = \hat{U} \times_3 M^{-1}$, $S = \hat{S} \times_3 M^{-1}$, $V = \hat{V} \times_3 M^{-1}$

**t-SVDMII** (Algorithm 3): post-process a t-SVDM with energy threshold $\gamma \in (0, 1]$:
1. Build the vector $v$ of all squared singular values $(\hat{\sigma}^{(i)}_j)^2$ across all slices; sort descending.
2. Find the smallest index $J$ s.t. cumulative sum $w_J > \gamma \|\hat{A}\|_F^2$ (strict inequality, per paper Algorithm 3 line 5). Set $\tau = \sqrt{v_J}$ — i.e., the $J$-th largest singular value. (Note: the paper's Algorithm 3 line 6 prints `τ := v_J`, which conflicts with the surrounding text's `τ := √(v_J)`; the text is correct since line 8 compares singular values against $\tau$, not squared singular values. Document this in the `tsvdmii` docstring.)
3. For each face $i$, set $\rho_i$ = number of singular values $\hat{\sigma}^{(i)}_j \geq \tau$.
4. **Return the truncated factors in the transform domain, ragged over slices:** `(Û_{:,1:ρᵢ,i}, Ŝ_{1:ρᵢ,1:ρᵢ,i}, V̂_{:,1:ρᵢ,i})` for each $i$. Do **not** transform back — per paper §6D, moving back to the spatial domain requires padding each slice to a common rank and destroys the storage savings. A separate `reconstruct(U, S, V, M)` does the $\times_3 M^{-1}$ transform and produces the spatial-domain approximation $A_\rho$ when the caller needs it.

**Where the work lives:** step 2 of t-SVDM ($n$ slice SVDs) is the dominant cost and is embarrassingly parallel. Steps 1 and 3 of t-SVDM are a **single dense GEMM** each — mode-3 product $A \times_3 M$ is $M \cdot A_{(3)}$ where $A_{(3)}$ is $n \times mp$, so one $(n \times n) \times (n \times mp)$ matmul does the whole transform. (The batched-matmul pattern shows up in the facewise product $\hat{A}_{:,:,i} \cdot \hat{B}_{:,:,i}$, which isn't on the t-SVDM path.) Step 4 of t-SVDMII is a sort + prefix scan over $\leq n \min(m,p)$ values and is cheap.

**Why this is an ideal HPC target:** clean, data-parallel, maps to every paradigm (OpenMP parallel-for, MPI distribute-over-slices, CUDA batched SVD).

---

## 2. Core design decisions (locked in)

| Decision | Choice | Why |
|---|---|---|
| Algorithms implemented | t-SVDM + t-SVDMII only | Scope discipline |
| Choice of M | Arbitrary dense orthogonal (real) | No fast-transform special cases |
| Numeric type | `float64` | Robustness first |
| Complex support | No | Doubles test surface for ~0 gain |
| CPU SVD routine | `dgesdd` (NumPy default, LAPACK divide-and-conquer) | ~3× faster than `dgesvd`, robust |
| GPU SVD routine | `cupy.linalg.svd` (batched, uses cuSOLVER under the hood) | One line, correct, fast |
| Memory layout (Python) | Shape `(n, m, p)`, C-order (row-major) | Frontal slice is `A[i]`, contiguous; `np.linalg.svd(A)` / `cp.linalg.svd(A)` batch natively over the leading axis, no hidden copies. |
| Memory layout (C++) | Shape `(m, p, n)`, Fortran/column-major | Frontal slice `A[:,:,k]` is an $mp$-element contiguous block, directly passable to `dgesdd` without a copy. Document this layout choice in every source-file header. |
| Validation oracle | [`mprod_package`](https://pypi.org/project/mprod-package/) (pip install) | Already implements t-SVDM/II per the paper; BSD-3. Use its outputs to validate every implementation we write (reconstruction error to 1e−10 on seeded inputs). |
| Public-library name | `tsvdm` (if free on PyPI), else `tsvdm-hpc` / `ptsvdm` | `mprod-package` already occupies the reference-implementation niche; check PyPI availability before Phase G. |
| License | MIT | Max adoption; compatible with citing BSD-3 `mprod_package` in docs. |
| Python API | Pure Python (NumPy CPU / CuPy GPU, dispatches on array type) | No C++ bindings = no build pain for researchers. |
| Julia role | Implementation lives in `HPC_Project/additional/`; post-May-7 extracted to public repo. | Single location during the course; no vendoring during development. |
| C++ role | Standalone binaries for course Parts 1–3 (`serial/`, `openmp/`, `mpi/`). | Not called from Python; not part of the post-deadline Python library. |
| CUDA (Part 4) | CuPy script + Python source living in `HPC_Project/cuda/`. | Post-deadline this same Python source gets extracted (unchanged) to the public repo. |
| Repo strategy | **Single course repo during development; public library repo extracted post-May-7.** | No vendoring, no two-repo maintenance, no question about where code originated. |

---

## 3. Repository layout — one course repo, with post-deadline extraction

### 3.1 The course repo: `HPC_Project` (course submission)

```
HPC_Project/                                # ~/Desktop/Y1S2/HPC/HPC_Project/
├── README.md                               # Brief overview; after May 7, add a link to the extracted public library.
│
├── serial/                                 # ── COURSE PART 1 ── C++ standalone
│   ├── tsvdm_serial.cpp
│   ├── Makefile
│   ├── submit.slurm
│   └── results/
│
├── openmp/                                 # ── COURSE PART 2 ── C++ standalone
│   ├── tsvdm_openmp.cpp
│   ├── Makefile
│   ├── submit.slurm
│   └── results/
│
├── mpi/                                    # ── COURSE PART 3 ── C++ standalone
│   ├── tsvdm_mpi.cpp
│   ├── Makefile
│   ├── submit.slurm
│   └── results/
│
├── cuda/                                   # ── COURSE PART 4 ── CuPy
│   ├── tsvdm_core.py                       # Core t-SVDM / t-SVDMII (NumPy + CuPy via array-module dispatch)
│   ├── tsvdm_ops.py                        # mode-3 product, facewise, helpers
│   ├── tsvdm_utils.py                      # Validation, reconstruction, errors
│   ├── run_cupy.py                         # Driver: generate data, time, save CSV
│   ├── tests/                              # Pytest: reconstruction, oracle agreement vs mprod_package
│   ├── submit.slurm
│   └── results/
│
├── additional/                             # ── COURSE PART 5 ── Julia
│   ├── tsvdm.jl                            # Core Julia implementation
│   ├── run_julia.jl                        # Driver
│   ├── Project.toml
│   ├── submit.slurm
│   └── results/
│
├── report/                                 # ── COURSE PART 6 ──
│   ├── report.md                           # ~2500-word blog-style writeup
│   └── figures/                            # Final plots + profiler screenshots
│
└── docs/                                   # Course-level planning (this file lives here)
    └── plan.md
```

**Key invariants:**
- Folder structure is exactly what the spec requires (`README.md`, `report/`, `serial/`, `openmp/`, `mpi/`, `cuda/`, `additional/`), plus a `docs/` folder for planning notes.
- Python implementation lives natively in `cuda/`. It is written once there; no copies elsewhere during the course.
- Julia implementation lives natively in `additional/`. Same principle.
- `serial/`, `openmp/`, `mpi/` are C++ standalone — no Python dependency.
- `report/` pulls benchmark plots from `HPC_Project/*/results/*.csv`. The course submission is fully self-contained.
- During development, validate every implementation against `pip install mprod-package` on reference inputs (reconstruction error < 1e−10). No separate "private reference library" is needed.

### 3.2 Post-May-7 public library (separate repo, extracted after grading)

After the course submission is locked in, copy the contents of `HPC_Project/cuda/tsvdm_*.py` + `HPC_Project/additional/tsvdm.jl` into a fresh public repo (`tsvdm/` on github.com, published to PyPI). Add the library polish described in §5 (mkdocs site, CI, examples, randomized variant). Nothing from the course repo needs to be "pre-prepared" for this extraction — a clean Python package + Julia module with standard naming is enough to lift cleanly.

The extraction is one-time, post-deadline, and small (~250 lines of Python + ~80 lines of Julia).

---

## 4. Part-by-part implementation plan

### Phase 0 — Python reference + test harness (before any course part)

Build the Python implementation that will live in `HPC_Project/cuda/` first. Everything else is validated against it and against `mprod_package`.

**In `HPC_Project/cuda/`:**
- `tsvdm_ops.py` — the 5 building blocks from §6 (mode-3 product, facewise product, ⋆M product, ⋆M transpose, Frobenius norm).
- `tsvdm_core.py` — `tsvdm(A, M)` and `tsvdmii(A, M, gamma)`, using `cupy.get_array_module(A)` (or `array_api_compat.get_namespace`) for NumPy/CuPy dispatch.
- `tsvdm_utils.py` — validation, reconstruction, compression ratio, relative error.
- `tests/` — reconstruction < 1e−10, orthogonality, theoretical error bound, hand-computed fixture, **cross-check against `mprod_package` output on a seeded input**.

**Why first:** every subsequent implementation (C++ serial/OpenMP/MPI, Julia) needs a correctness oracle. This Python module is that oracle (cross-checked itself against `mprod_package`). It's also ~250 lines total.

### Part 1 — Serial C++ (10 pts)

**Goal:** baseline performance for comparisons.

**Implementation:**
1. Linear algebra backend: LAPACK directly via `dgesdd`, linked with OpenBLAS or MKL. Avoid Eigen's built-in SVD (slower for this regime).
2. Single file `tsvdm_serial.cpp`, one public function `tsvdm(double* A, int m, int p, int n, const double* M, double* U, double* S, double* V)`.
3. Data layout: frontal-slice-contiguous column-major. Document in file header.
4. Inner structure:
   - `apply_M_mode3`: materialize the mode-3 unfolding and call `dgemm` for a single large GEMM.
   - `slice_svd_loop`: for $i = 0 \dots n-1$ call `dgesdd`.
   - `apply_Minv_mode3`: same as first, with $M^T$ (orthogonal).
5. Timing: `std::chrono::steady_clock`, core function wall time only.
6. Validation: random $A$, random orthogonal $M$ (QR of Gaussian); reconstruct in C++ and compare to the `tsvdm` Python library. Require `‖A − Ã‖_F < 1e−10`.
7. Slurm script: single node, single thread, pins to one core, writes CSV row per run. **Wall time ≤ 10 min** (course staff will run it — see §9).

**Sweep (fits 10-min cap):** 3 sizes × **3 reps** = 9 runs.
- Small (64×64×64), medium (256×256×256), large (512×512×256).
- Report median + min wall time per (size, rep). CSV columns: `impl,size,rep,t_core_ms`.

### Part 2 — OpenMP (10 pts)

**Parallelization:**
- Over the n slice SVDs — embarrassingly parallel: `#pragma omp parallel for schedule(dynamic)`.
- Mode-3 transforms: split `A_(3)` (which is `n × mp`, column-major) into `T` contiguous column slabs — each slab is a contiguous range of `(j, k)` outer pairs — and have each OpenMP thread call `dgemm` on its own slab. One GEMM per thread, not per `(j, k)` pair. `M` is shared read-only; each slab of `A_(3)` and of the output is thread-private. Alternative: wrap just the transform in its own parallel region and set `OPENBLAS_NUM_THREADS=T` there, letting BLAS parallelize internally.
- **No nested parallelism.** BLAS threads internally. Set `OMP_NUM_THREADS=X` for the parallel region and `OPENBLAS_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`) to serialize inner BLAS calls. Document this in the Slurm script.

**Deliverables:**
- `openmp/tsvdm_openmp.cpp` — diff from serial is ~10 lines + `<omp.h>`.
- Slurm script wall time ≤ 10 min.

**Sweep (fits 10-min cap):**
- **Strong scaling:** threads ∈ {1, 4, 16} × medium size × 3 reps = 9 runs. Plot speedup vs. threads.
- **Strong scaling at large size:** threads ∈ {1, 16} × large × 3 reps = 6 runs. Establishes that scaling holds at the bandwidth-bound regime.
- **Weak scaling:** drop from in-script sweep — covered conceptually in the report by referencing the strong-scaling efficiency curve. Optional weak-scaling sweep can run locally outside the staff script if time permits.

### Part 3 — MPI (10 pts)

**Strategy: distribute along mode-3 (frontal slices).** Each rank owns a contiguous chunk of n slices.

**Communication pattern:**
1. **Initial distribution:** rank 0 generates $A$ with a seeded RNG and `MPI_Scatterv`s it. (For large benchmarks, each rank generates its own slices locally — cheaper.)
2. **Mode-3 forward transform $A \to \hat{A}$:** each rank owns its slices but the transform spans all slices. Use `MPI_Alltoallv` with derived datatypes to redistribute to a tube-contiguous layout, apply $M$ locally, `MPI_Alltoallv` back.
3. **Slice SVDs:** fully local. No communication.
4. **Inverse transform:** symmetric to step 2.
5. **t-SVDMII threshold τ:** `MPI_Allgatherv` to collect all $\hat{\sigma}^{(i)}_j$ (small payload), then each rank sorts independently and finds τ.

**Deliverables:**
- `mpi/tsvdm_mpi.cpp` — explicit comments on the `MPI_Alltoallv` transpose.
- Slurm script wall time ≤ 10 min.

**Sweep (fits 10-min cap, single Slurm allocation):**
- Request `--nodes=8 --ntasks-per-node=1 --time=00:10:00`. Step inside the script with `srun -N P -n P` (HW3 Q4 pattern).
- **Strong scaling:** P ∈ {1, 2, 4, 8} nodes × 1 rank/node × medium size × 3 reps = 12 runs.
- **Weak scaling:** P ∈ {1, 4} nodes × scaled size (problem grows with P) × 3 reps = 6 runs.
- Communication vs. compute breakdown logged per run; reported as table in the writeup.

(16 nodes is sufficient for a clean log-scale plot. The drop from "8 ranks/node" to "1 rank/node" trades ranks-per-node for a cleaner story; HW3 Q4 used the same shape and produced publishable scaling figures.)

**Why this pattern** (explanation to include in the report): slice-distribution lets the dominant compute (SVDs) run with zero communication. The two all-to-alls cost $O(mpn/P)$ per rank, while compute is $O(mpn \cdot r/P)$ where $r \sim \min(m,p)$. Surface-to-volume ratio is favorable as long as the per-slice work is nontrivial.

### Part 4 — CUDA via CuPy (10 pts)

Self-contained `cuda/` folder (the Python source lives here natively — no vendoring):
```
cuda/
├── tsvdm_core.py       # Core t-SVDM/II, NumPy/CuPy dispatch via get_array_module
├── tsvdm_ops.py        # mode-3 product, facewise, helpers
├── tsvdm_utils.py      # validation, reconstruction, errors
├── run_cupy.py         # Driver: generate data, time, save CSV
├── tests/
│   └── test_oracle.py  # Reconstruction + cross-check vs mprod_package
└── submit.slurm
```

**Critical pre-work for Part 4:** confirm that `cupy.linalg.svd` actually **batches** on the cluster's GPU for the slice sizes we care about (256×256 and up). Under the hood CuPy dispatches to `gesvdjBatched` (small matrices, typically ≤ 32×32) or falls back to serial-over-batch for larger matrices depending on the cuSOLVER version. If serial-over-batch is what we get for 256×256 slices, speed-up over CPU will be underwhelming. Fallback plan: loop SVDs across multiple CUDA streams (one per slice), or call raw `cusolverDnDgesvdaStridedBatched` via CuPy's low-level bindings. Run a 10-line microbenchmark on the cluster GPU before writing Part 4 to find out which regime you're in.

`run_cupy.py`, roughly:
```python
import cupy as cp
import time, csv, argparse
from tsvdm_core import tsvdmii  # local module in cuda/

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int); ap.add_argument("--p", type=int); ap.add_argument("--n", type=int)
ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()

rng = cp.random.default_rng(0)
A = rng.standard_normal((args.m, args.p, args.n), dtype=cp.float64)
M, _ = cp.linalg.qr(rng.standard_normal((args.n, args.n), dtype=cp.float64))

_ = tsvdmii(A, M, args.gamma); cp.cuda.Stream.null.synchronize()  # warmup

times = []
for _ in range(args.reps):
    cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    U, S, V = tsvdmii(A, M, args.gamma)
    cp.cuda.Stream.null.synchronize(); times.append(time.perf_counter() - t0)

# write CSV row ...
```

Because `tsvdm_core.py` uses `cupy.get_array_module(A)` internally, passing CuPy arrays makes the same code run on GPU. The same `tsvdm_core.py` powers Part 4 (CuPy on GPU) and is also called by the test harness on CPU (NumPy), so there's no duplicate code between "library" and "course driver" — they're the same file.

**Sweep (fits 10-min cap):** 3 sizes × 3 reps = 9 runs. GPU SVDs are fast — easily fits with headroom for warmup, host↔device timing, and the Nsight Systems profile run. Slurm wall time ≤ 10 min.

**Benchmarks to run:**
- Host↔device transfer time separately from compute.
- Vs. CPU (serial + OpenMP at 16 threads — the trimmed Part 2 peak, not 32).
- Nsight Systems trace, single representative run (see §10 Profiling protocol).

> **Performance framing note (lifted from `cluster-environment.md` §1.3):** L4 is an inference-class card with ~0.49 TFLOPS theoretical FP64. Expect modest (not order-of-magnitude) speedup over 16-thread Cascade Lake on `float64`. Frame the report around "speedup vs. serial" and "bandwidth utilization vs. peak", not absolute FLOPS.

### Part 5 — Julia (additional paradigm) (10 pts)

Implementation lives directly in `HPC_Project/additional/tsvdm.jl` — ~80 lines using `LinearAlgebra` + broadcasting. `run_julia.jl` includes `tsvdm.jl`, generates seeded data, times the core function, and writes CSV to `results/`. `Project.toml` pins Julia/package versions for reproducibility.

**Report angle:** Julia's single-source CPU/GPU story via `CUDA.jl` — swap `Array` for `CuArray` and the same code runs on GPU. Contrast with the C++/CUDA code-duplication pain.

**Optional stretch** (skip if it takes more than a short session): `using KernelAbstractions` to run on AMD/Intel/Apple GPU from the same source — earns the "hardware-agnostic" bullet you mentioned originally.

**Sweep (fits 10-min cap):** 3 sizes × 3 reps = 9 runs. Budget an extra ~30 s for first-run JIT warmup (do a throwaway call before the timed loop). Slurm wall time ≤ 10 min.

**Benchmarks:** time vs. Python/NumPy (CPU), time vs. CuPy (GPU), startup/JIT warmup note.

### Part 6 — Report (30 pts)

**~2500 words, blog-style:**
1. **Intro** (~300): motivation, what t-SVDM/II do, why this is an HPC target.
2. **Algorithm** (~400): just enough math. One equation for t-SVDM, one for t-SVDMII, one figure adapted from the paper with citation.
3. **Methods** (~700): one subsection per implementation — parallelization, decisions, Slurm config.
4. **Results** (~700): strong/weak scaling, implementation comparison, profiler highlights. At least one "unexpected observation" analyzed honestly.
5. **Conclusion** (~400): practitioner recommendations, limitations, future work. Mention the public `tsvdm` library.

**Figures** (all reproducible from the benchmark CSVs):
- Strong scaling: OpenMP + MPI.
- Weak scaling: OpenMP + MPI.
- All implementations comparison bar chart.
- One VTune screenshot + one Nsight Systems timeline.
- One algorithm schematic (adapted from paper Fig. 2).

### Part 7 — Professionalism (20 pts)

Levers: reproducibility, clean code, organized deliverables, compelling README. Comes almost free from §5 below.

---

## 5. Post-deadline public library (in detail)

**None of this happens during the course.** After May 7, extract the existing `HPC_Project/cuda/tsvdm_*.py` + `HPC_Project/additional/tsvdm.jl` into a fresh public repo and add the polish below. The sections here describe the end-state, not in-course work.

Positioning note: [`mprod_package`](https://github.com/UriaMorP/mprod_package) (BSD-3) is already the canonical Python reference implementation of the Kilmer et al. 2021 paper. Our public library therefore differentiates on: (a) CuPy GPU support out of the box via array-module dispatch, (b) randomized t-SVDM variant, (c) explicit HPC-study provenance (accompanying benchmarks + arxiv preprint across OpenMP/MPI/CUDA/Julia). We cite `mprod_package` in our docs as the reference CPU implementation we validated against.

### 5.1 README structure

1. One-line tagline + badges: CI status, PyPI version, license, Python versions, docs link, paper DOI (after arxiv).
2. 30-second install: `pip install tsvdm`.
3. 30-second example: ~10-line snippet compressing a random tensor + reconstructing.
4. Feature table.
5. Link to docs.
6. Link to paper.
7. Citation (BibTeX).
8. Contributing + license.

### 5.2 Python package surface

Public API — under 200 lines of actual code:
- `tsvdm(A, M)` → `(U, S, V)`
- `tsvdmii(A, M, gamma)` → `(U, S, V)`
- `reconstruct(U, S, V, M)` → `A_approx`
- `mode3_product(A, M)`, `facewise_product(A, B)`, `star_m_product(A, B, M)`, `star_m_transpose(A, M)`
- `compression_ratio(original, compressed)`, `relative_error(A, A_hat)`

**Array-type dispatch:** use `xp = cupy.get_array_module(A)` (or `array_api_compat.get_namespace`) so NumPy and CuPy arrays both work through one code path. Docstrings numpy-style, which feeds the API reference auto-generator.

### 5.3 Documentation

- `mkdocs-material`, theme readthedocs.
- Pages: Home, Install, Algorithm primer (MathJax), Tutorials (rendered notebooks), API reference (auto-generated via `mkdocstrings`).
- Hosted on GitHub Pages via `.github/workflows/docs.yml`.

### 5.4 Tutorial notebooks

Two, living in `tsvdm/docs/tutorials/` and `tsvdm/examples/`:
- **Hyperspectral compression** — Indian Pines (Purdue, public) or Salinas. Recreate Kilmer et al. Fig. 5 on public data.
- **Grayscale video compression** — Xiph.org "akiyo" clip (CC). Frame reconstructions at varying γ vs. plain matrix SVD.

These double as publication-quality figures for the arxiv paper.

### 5.5 Testing

- Framework: pytest + pytest-cov.
- Per operation: (1) against a hand-computed fixture, (2) property tests (shape preservation, U/V orthogonality, Eckart–Young bound), (3) edge cases ($n = 1$, $m = p$, non-square $M$).
- Coverage target: >80% (measured by `pytest --cov=tsvdm`). Not a release blocker.
- CI: `.github/workflows/ci.yml` runs pytest on Python 3.10–3.12 on Ubuntu. GPU tests marked `@pytest.mark.gpu` and skipped in CI (run locally on cluster).

### 5.6 Benchmarks

Library benchmarks live inside `tsvdm/benchmarks/` and exercise only the Python/Julia paths.

- `benchmarks/run.py` — `--impl {numpy,cupy,julia,randomized}`, `--size {small,medium,large}`, `--reps`. Writes CSV rows.
- `benchmarks/plot.py` — reads CSVs, produces strong/weak/implementation PNGs + PDFs.
- `benchmarks/README.md` — reproduction instructions.

The **course** benchmarks for `serial/openmp/mpi` live in `HPC_Project/` and have their own CSVs. `report/generate_figures.py` merges both sets for cross-implementation plots.

### 5.7 Packaging and releases (post-May-7)

- `pyproject.toml` with `build-backend = "hatchling.build"`. Base deps: `numpy`. Optional extras: `[gpu]` → `cupy-cuda12x`, `[examples]` → `jupyter, matplotlib, scikit-image`.
- **First public release (after grading):**
  1. Verify `tsvdm` free on PyPI and TestPyPI.
  2. `python -m build`.
  3. `twine upload --repository testpypi dist/*` → install-test on clean venv.
  4. `twine upload dist/*` → live.
  5. Flip the GitHub repo to public, push all history.
- **Versioning:** start at `0.1.0`. Pre-1.0 signals unstable API.
- **CHANGELOG.md:** simple markdown, one section per version.
- **Auto-release** via `.github/workflows/release.yml` (OIDC trusted publishing) can come later.

### 5.8 Reproducibility

- `pyproject.toml` with version ranges (for downstream installs).
- `environment.yml` committed for bit-exact benchmark reproduction.
- Seeded RNG everywhere, seed documented.
- Skip Docker/Nix unless requested.

---

## 6. Operations to implement (complete list)

### `src/tsvdm/ops.py`
1. `mode3_product(A, M)` — `A ×₃ M`; single `einsum`.
2. `facewise_product(A, B)` — pointwise frontal-slice matmul; single `einsum`.
3. `star_m_product(A, B, M)` — paper's ⋆M.
4. `star_m_transpose(A, M)` — paper's transpose under ⋆M.
5. `frobenius_norm(A)` — thin wrapper.

### `src/tsvdm/core.py`
6. `tsvdm(A, M) → (U, S, V)`.
7. `tsvdmii(A, M, gamma) → (U, S, V)`.
8. `reconstruct(U, S, V, M) → A_approx`.

### `src/tsvdm/randomized.py` (paper contribution)
9. `randomized_tsvdm(A, M, rank, oversample=10) → (U, S, V)` — Halko-style per slice.

### `src/tsvdm/utils.py`
10. `compression_ratio`, `relative_error`, `_validate_inputs`.

**Skipped on purpose:** identity tensor, tensor inverse under ⋆M, normalization helpers, broader ⋆M algebra. Not needed for t-SVDM/II.

---

## 7. arxiv paper strategy

**Target:** short (~8 page) preprint. Single-author.

**Contribution framing (revised after finding `mprod_package`):**
- **Not** "first open-source implementation" — [`mprod_package`](https://github.com/UriaMorP/mprod_package) on PyPI (BSD-3, authored by UriaMorP, built for the Kilmer et al. 2021 paper) already provides CPU-Python t-SVDM and t-SVDMII.
- Revised pitch: **"Parallel t-SVDM: a study of t-SVDM/II performance across shared-memory (OpenMP), distributed-memory (MPI), and GPU (CUDA/CuPy) backends, with a Julia cross-language reference and a randomized variant."** The novelty is the HPC study and the GPU path, not the algorithm itself.
- Explicitly cite `mprod_package` as the CPU reference and as the validation oracle. Collaborate-friendly framing: your library is complementary (adds GPU + HPC) rather than competing.

**Structure:**
1. Intro + related work + contributions.
2. Background — ⋆M product, t-SVDM, t-SVDMII.
3. Implementation — parallelization strategies.
4. Randomized variant.
5. Experiments — benchmark plots + randomized vs. deterministic.
6. Conclusion.

**Content reuse:** ~80% from the course report, expanded with the randomized SVD section + more formal algorithmic pseudocode + references (including Kilmer et al. 2021 and `mprod_package`). CITATION.cff + BibTeX block in README once posted.

---

## 8. Testing and correctness philosophy

- Every algorithmic claim has a test assertion.
- For each implementation (serial, OpenMP, MPI, CuPy, Julia): the final assertion is `‖reconstruct(tsvdm(A, M)) − A‖_F / ‖A‖_F < 1e−10` on a seeded random tensor.
- Cross-implementation: same `(A, M)` through every implementation recovers agreement to 1e−8. Tiny `.npz` fixture committed to `HPC_Project/cuda/tests/fixtures/` during the course; moves to `tsvdm/tests/fixtures/` post-extraction.
- **Correctness is not negotiable.** A faster implementation that disagrees with the Python library is wrong.

---

## 9. Benchmarking methodology

> **Hard cap (course-imposed):** every staff-runnable Slurm script in `serial/`, `openmp/`, `mpi/`, `cuda/`, `additional/` must complete in **≤ 10 min wall time**. The per-Part sweeps in §4 are designed against this cap. If richer plots are wanted, run a longer sweep locally outside the staff script and pre-populate `results/*.csv` before submission — the staff script still runs and produces a valid (smaller) subset of the same data.

**Problem sizes:**
- **Small:** 64 × 64 × 64 (~1 MB; fits in L3; overhead-dominated regime).
- **Medium:** 256 × 256 × 256 (~128 MB; realistic single-node).
- **Large:** 512 × 512 × 256 (~512 MB; memory-bandwidth regime).

(A larger ~2 GB tensor was originally on the books for MPI but does not fit a 10-min staff run on 8 nodes. Skipped.)

For each (implementation, size): **3 reps**, report median + min. Seeded RNG + seeded orthogonal $M$ (QR of Gaussian).

**Metrics:**
- Wall time of the core function (GPU: exclude host↔device transfer from core time, report transfer separately).
- Strong scaling: speedup vs. serial.
- Weak scaling: time at fixed work-per-rank (only where the per-Part sweep includes a weak run; otherwise discussed conceptually).
- CuPy: effective bandwidth vs. peak.
- MPI: communication vs. compute breakdown.

**Profiling:** see §10 (separate section).

---

## 10. Profiling protocol

Two profiles total. The course rubric requires *"at least one profiling tool (VTune for CPU, Nsight Systems or Nsight Compute for GPU) to identify bottlenecks or explain observed performance trends"* — we run one CPU and one GPU for safety. Anything fancier (Nsight Compute kernel-level, VTune memory-access / threading analyses) is **explicitly skipped** — overkill for the rubric.

| Profile | Tool | Target | Slurm script | Expected output |
|---|---|---|---|---|
| CPU | Intel VTune (`hotspots` collection) | OpenMP Part 2, 16 threads, medium size | `openmp/profile.slurm` | Hotspot list — `dgesdd_` and OpenBLAS GEMM dominate |
| GPU | Nsight Systems (`nsys profile`) | CuPy Part 4, medium size | `cuda/profile.slurm` | Timeline showing batched SVD kernel + host↔device transfers |

**CPU run command** (inside `openmp/profile.slurm`):
```bash
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

vtune -collect hotspots -result-dir results/vtune_omp16 -- ./tsvdm_openmp <args>
vtune -report summary -result-dir results/vtune_omp16 > results/vtune_summary.txt
```

**GPU run command** (inside `cuda/profile.slurm`):
```bash
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

nsys profile --output=results/nsys_cupy --force-overwrite=true \
    python run_cupy.py --m 256 --p 256 --n 256 --reps 1
```

**Deliverables per profile:**
1. **One screenshot** — VTune Hotspots view → `report/figures/profile_vtune.png`; Nsight Systems timeline view → `report/figures/profile_nsys.png`. Open the result file in the GUI, screenshot the most informative pane, save PNG.
2. **One paragraph** in `report.md` Methods section: tool used, exact command, top hotspot identified, what it explains about the observed scaling trend.

**Do not commit raw traces** (`vtune_omp16/` directory or `*.nsys-rep` files — large, opaque). Add to `serial/.gitignore` patterns or a top-level `.gitignore`. The screenshot is what the report needs; the raw trace is for local inspection.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MPI transpose harder than expected | Fall back to "replicate $A$, distribute only SVDs" for a working baseline; discuss tradeoff in report. |
| cuSOLVER batched SVD size limits (see Part 4 pre-work note) | Microbenchmark `cp.linalg.svd` on cluster GPU early. If serial-over-batch: loop SVDs across CUDA streams or use `cusolverDnDgesvdaStridedBatched` directly. |
| BLAS + OpenMP nested parallelism | Set `OPENBLAS_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`) in every Slurm script; document. |
| Tutorial dataset licensing | Indian Pines (public research), Xiph.org videos (CC). Avoid MATLAB built-ins. |
| Randomized SVD drag | Python-only, ~50 lines, no MPI/CUDA variants in v1. Strictly post-deadline. |
| arxiv novelty bar | `mprod_package` already covers the "first open-source impl" story. Novelty is the HPC study + GPU + randomized variant (see §7). |
| PyPI name `tsvdm` taken (post-deadline) | Already partly occupied conceptually by `mprod_package`. Check `pip install tsvdm` when ready. Fallbacks: `tsvdm-hpc`, `ptsvdm`, `tsvdm-gpu`. |
| Cluster MPI node availability + BLAS/CUDA versions | Confirm via §14 resolution steps before designing weak-scaling sweep. |

---

## 12. Suggested execution order

Python-first for correctness oracle, C++ course parts next for breadth, extract public library after May 7.

**Phase A — Python implementation in `HPC_Project/cuda/` (course-repo local):**
1. Create `HPC_Project/cuda/`. Write `tsvdm_ops.py` + `tsvdm_core.py` + `tsvdm_utils.py`. Iterate until reconstruction error < 1e−12 on a small random tensor.
2. `pip install mprod-package`. Write `tests/test_oracle.py` that runs our t-SVDM/II on a seeded input and checks reconstruction agrees with `mprod_package` output to 1e−10.
3. Write the rest of the test suite: per-op fixtures, orthogonality of U/V, Eckart–Young bound, edge cases.

**Phase B — Course C++ parts (independent of Python):**
4. Resolve cluster unknowns (see §14). Know your BLAS, MPI, CUDA versions before writing C++.
5. `HPC_Project/serial/` — LAPACK-linked C++. Validate vs. Python via saved `.npz` fixture (use `cnpy` or write a small binary dumper).
6. `HPC_Project/openmp/` — diff from serial, `OPENBLAS_NUM_THREADS=1`, thread sweep.
7. `HPC_Project/mpi/` — `MPI_Alltoallv` transpose. 1 rank smoke test, then scale.

**Phase C — Julia in `HPC_Project/additional/`:**
8. Write `tsvdm.jl` + `run_julia.jl` + `Project.toml`. Validate reconstruction vs. Python `HPC_Project/cuda/tsvdm_core.py`.

**Phase D — Part 4 CuPy driver:**
9. Run the cuSOLVER batching microbenchmark from §4 Part 4 on the cluster GPU. Pick the right path (batched vs stream-per-slice) accordingly.
10. Write `run_cupy.py` driver + Slurm script. The core implementation is already `HPC_Project/cuda/tsvdm_core.py` from Phase A, reused verbatim.

**Phase E — Benchmarks + report (course):**
11. Run all five implementations across sizes on the cluster. CSVs → `HPC_Project/*/results/`.
12. Profiling per §10 protocol: VTune on OpenMP (16 threads), Nsight Systems on CuPy. Screenshots → `HPC_Project/report/figures/`.
13. Write `HPC_Project/report/report.md`. Verify `HPC_Project/` structure matches the spec exactly (§14 checklist).
14. **Submit course (May 7).**

**Phase F — Extract public library (post-May-7):**
15. Create fresh public repo `github.com/<user>/tsvdm` (or fallback name if taken — see §11). MIT license.
16. Copy `HPC_Project/cuda/tsvdm_{core,ops,utils}.py` into `src/tsvdm/`. Add `__init__.py` re-exports. Write `pyproject.toml`.
17. Copy `HPC_Project/additional/tsvdm.jl` into `julia/`.
18. Add `randomized.py` + tests + benchmark.
19. Add `docs/` (mkdocs-material site, two tutorial notebooks), `benchmarks/run.py` + `plot.py`, `.github/workflows/{ci,docs}.yml`.

**Phase G — Public release:**
20. Verify PyPI name is free. `python -m build`, TestPyPI → PyPI. Tag `v0.1.0`.

**Phase H — Paper:**
21. Reformat report + expand with randomized SVD results → arxiv preprint. Cite `mprod_package` + Kilmer et al. 2021.
22. Add DOI badge + CITATION.cff to `tsvdm` README.

---

## 13. Open questions still worth confirming

- **PyPI name `tsvdm`** free? Check before Phase G (post-deadline). `mprod_package` is distinct — unlikely to conflict by name, but verify.
- **Cluster BLAS, MPI compiler/version, Julia presence** — ✅ resolved (2026-05-01). See `docs/cluster-environment.md`. Summary: OpenBLAS via Spack, GCC 11.5.0, Open MPI 4.1.7, no system Julia.
- **Cluster MPI node cap** — ✅ effectively closed. 16 nodes confirmed working in HW3 Q4 and is sufficient for a clean log-scale strong/weak scaling plot (1, 2, 4, 8 nodes covers four doublings); no need to push further.
- **GPU model** — ✅ resolved (2026-05-01). 4× NVIDIA L4 per node, 24 GB each, sm_89, driver 570.172.08, CUDA 12.8. See `cluster-environment.md` §1.3.
- **`nvcc --version`, CuPy version, `cp.linalg.svd` batching microbenchmark** — deferred to start of Phase D per current execution order. Tracked in §14.3.
- **Tutorial hyperspectral dataset** — Indian Pines (145×145×200) vs. Salinas (512×217×204). Resolve during Phase F. Former is smaller and standard; latter is more visually compelling.

None block starting Phase A. `mprod_package` existence and role as CPU reference is already resolved (§7, §11).

---

## 14. Cluster unknowns — how to resolve them

These checks need to happen **before Phase B** (C++ implementation) so the build toolchain and benchmark design are grounded in what the cluster actually has. Most are one-minute commands on the login node.

### 14.1 BLAS/LAPACK (for Part 1 serial, Part 2 OpenMP) — ✅ RESOLVED

**Answer:** **OpenBLAS**, installed via Spack. Threading control: `OPENBLAS_NUM_THREADS=1` (set `MKL_NUM_THREADS=1` defensively). Linker pattern requires explicit Spack path with rpath:

```makefile
BLAS := $(shell spack location -i openblas)
CXXFLAGS += -I$(BLAS)/include
LDFLAGS  += -L$(BLAS)/lib -Wl,-rpath,$(BLAS)/lib -lopenblas -lm
```

Established by reading `HPC_HW/homework-3-main/question-6/submit.slurm` (literally a batched-SVD with OpenBLAS + OpenMP — same pattern this project needs). See `docs/cluster-environment.md` §1.1.

### 14.2 MPI (for Part 3) — partially resolved

**Resolved (2026-05-01):**
- MPI implementation: **Open MPI 4.1.7**.
- Compiler wrapper: `mpicc`/`mpicxx` wrapping **GCC 11.5.0** (Red Hat 11.5.0-5).
- Confirmed working at 16 nodes × 1 task × 1 cpu (HW3 Q4) and 2 nodes × 1 task × 16 cpus (lecture-16/ex-3).

**Still open — node/rank caps for the weak-scaling sweep design:**
```bash
sinfo -o "%P %c %m %G %D"                                  # partitions, cores/node, GPUs/node, nodes
sacctmgr -n show assoc user=$USER format=qos,maxjobs,maxsubmit,maxwall
```

Run on the login node before Phase B step 7 (MPI design). If the cluster has fewer nodes available than HW3 Q4 used (16), redesign the sweep to fit.

### 14.3 CUDA / cuSOLVER (for Part 4) — partially resolved

**Resolved (2026-05-01):**
- Compute capability: **8.9 (Ada Lovelace)** — `-arch=sm_89` per `Lecture Notes/lecture-11.md`, `lecture-12.md`.
- GPU model: **4× NVIDIA L4 per GPU node**, 24 GB each (~72 W TDP) — `nvidia-smi`.
- Driver / driver-side CUDA: **570.172.08 / 12.8** — `nvidia-smi`.
- Inspection recipe: `Lecture Notes/lecture-11.md` `inspect-gpu.slurm` template (lscpu, lstopo, nvidia-smi, nvcc --version).
- **Implication for Part 4:** L4 is inference-class (~0.49 TFLOPS FP64). Frame results around speedup-vs-serial and bandwidth utilization, not absolute FLOPS. See `cluster-environment.md` §1.3 note.

**Deferred to start of Phase D (per current execution order):**
- `nvcc --version` (returned "command not found" in user's session — likely wrong Spack env active or `nvcc` provisioned in a sub-path).
- CuPy version + runtime version.
- `cp.linalg.svd` batching microbenchmark — drives Part 4 implementation choice (one-line batched call vs. stream-per-slice fallback vs. raw `cusolverDnDgesvdaStridedBatched`).

**How to resolve at Phase D start (interactive Slurm session on `gpu` partition):**
```bash
srun --partition=gpu --gres=gpu:1 --exclusive --time=00:30:00 --pty bash
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

# If nvcc still missing:
which nvcc || find $(spack location -e CS-2050-gpu)/.spack-env/view -name nvcc 2>/dev/null
nvcc --version
python -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.runtimeGetVersion())"
```

**The batching microbenchmark** (run once, 30 seconds):
```python
import cupy as cp, time
n, m = 64, 256
A = cp.random.standard_normal((n, m, m), dtype=cp.float64)
cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
U, S, V = cp.linalg.svd(A)                                 # batched call
cp.cuda.Stream.null.synchronize(); print(f"batched {n} SVDs of {m}×{m}: {time.perf_counter()-t0:.3f}s")

cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
for i in range(n):
    cp.linalg.svd(A[i])                                    # serial per-slice
cp.cuda.Stream.null.synchronize(); print(f"serial {n} SVDs of {m}×{m}: {time.perf_counter()-t0:.3f}s")
```

If batched time ≳ serial time / n, `cp.linalg.svd` is **not actually batching** for this size and you need the stream-per-slice fallback (or raw `cusolverDnDgesvdaStridedBatched` via CuPy's low-level bindings). If batched ≈ serial/something-good, we're fine.

**Resolve before:** Phase D step 9.

### 14.4 Julia (for Part 5) — ✅ RESOLVED

**Answer (2026-05-01):** Julia is **provisioned in the `CS-2050` Spack env**, not on the default `$PATH`. That's why a bare `which julia` failed — Spack must be activated first. Confirmed by `lecture-examples/lecture-18/example-3/submit.sh`, which calls `julia` directly after `spack env activate -p CS-2050`. **No `juliaup` install needed.**

**Bonus:** lecture-18 example-3 also documents the Julia + MPI recipe — bind `MPI.jl` to the cluster's Open MPI 4.1.7 via `MPIPreferences.use_system_binary(; extra_paths=[$(mpicc --showme:libdirs)])`. Useful if Part 5 wants to demo MPI from Julia.

**Remaining action:** confirm version inside the activated env (`julia --version`) and pin in `additional/Project.toml`.

### 14.5 Repo-structure sanity check (pre-submission)

Before pushing the final commit on May 7, verify against the spec (`docs/CS-2050_project.pdf` §3):
```bash
ls HPC_Project/                # Must contain: README.md, report/, serial/, openmp/, mpi/, cuda/, additional/
for d in serial openmp mpi cuda additional; do
  ls HPC_Project/$d/ | grep -E '\.(slurm|sh)$' || echo "MISSING Slurm script in $d/"
  ls HPC_Project/$d/results/ 2>/dev/null || echo "MISSING results/ in $d/"
done
```

Each of the 5 implementation folders must contain: source files, Slurm scripts that reproduce results, and results/plots.
