# `cuda/` — Python reference (Part 0) + CUDA via CuPy (Part 4)

Pure-NumPy implementation of t-SVDM (Part 0 reference / oracle), plus
the GPU driver `run_cupy.py` (Part 4) that reuses the same code path
through `cupy.get_array_module`.

## Layout

| File | Role |
|---|---|
| `tsvdm_ops.py` | Building-block ops (`mode3_product`, `facewise_product`, ⋆M product, ⋆M transpose). NumPy/CuPy-agnostic via `cupy.get_array_module`. |
| `tsvdm_core.py` | `tsvdm` (the algorithm) and `reconstruct` (rebuild `A_approx` from `U`, `S`, `V`, `M`). The reference oracle for every other backend. |
| `tsvdm_utils.py` | Helpers: `random_orthogonal`, `relative_error`, `compression_ratio`, `_validate_inputs`. |
| `run_cupy.py` | Part 4 GPU driver. `--gen M P N` to time a freshly seeded tensor on GPU; fixture path to validate against a committed `.bin`. |
| `gen_fixture.py` | Writes the `(A, M)` `.bin` fixtures used by `serial/`, `openmp/`, `mpi/`. Invoked by `serial/gen_fixtures.slurm`. |
| `compare_cpp_python.py` | Element-wise C++-vs-Python validator: runs a C++ binary with `--dump`, loads the result, compares against `tsvdm_core.tsvdm`. |
| `submit.slurm` | Part 4 GPU job. Self-creates `cuda/.venv/`, pip-installs CuPy 13.6.0 on first run, then runs the size sweep + fixture cross-check. |
| `requirements.txt` | Python deps for the CPU/Part 0 path: `numpy`, `pytest`, `mprod-package` (the oracle). |
| `__init__.py` | Re-exports `tsvdm`, `reconstruct`, `random_orthogonal`, `relative_error` so `from cuda import tsvdm` works. |
| `tests/` | pytest suite: reconstruction error, U/V orthogonality, edge cases, and `test_oracle.py` (cross-check vs. `mprod_package`). |

Tensor shape convention: `A` has shape `(n, m, p)`. Frontal
slice *i* is `A[i]`.

## Install (CPU / Part 0)

```bash
pip install -r requirements.txt
```

## Install (GPU / Part 4)

The course's `CS-2050-gpu` Spack environment ships **Numba CUDA** as
its primary Python-on-GPU tool, **not** CuPy. t-SVDM needs batched
dense SVD, which Numba CUDA does not expose as a primitive
(`numba.cuda` is a JIT for kernel code, not a linear-algebra library).
Per instructor guidance, we install CuPy into a project-local
virtualenv at `cuda/.venv/`. The venv is **self-created** by
`cuda/submit.slurm` on first run and **reused** on subsequent runs —
no manual setup required:

```bash
[cluster]$ cd cuda && sbatch submit.slurm
```

First run takes ~60 s extra (venv creation + CuPy install at the
pinned version `13.6.0`); subsequent runs no-op the install path.
The venv is gitignored. To rebuild from scratch:

```bash
[cluster]$ rm -rf cuda/.venv
[cluster]$ sbatch cuda/submit.slurm   # recreates venv on next run
```

This approach is reproducible: anyone cloning the repo gets the
same CuPy version with one `sbatch`. CuPy is acknowledged as a valid
GPU Python option in the course materials (`Lecture Notes/lecture-{10,21,23}.md`,
`final-exam-study-guide.md` §8.4, `Practice Exam` Q23) — it is not
foreign to the course toolchain, just not pre-installed.

## Run tests

```bash
python -m pytest cuda/tests -v
```

## Quick example

```python
import numpy as np
from cuda import tsvdm, reconstruct, random_orthogonal, relative_error

rng = np.random.default_rng(0)
A = rng.standard_normal((4, 8, 8))       # (n, m, p)
M = random_orthogonal(4, rng)            # (n, n)

U, S, V = tsvdm(A, M)
print(relative_error(A, reconstruct(U, S, V, M)))   # ~1e-15
```
