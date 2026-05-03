# Cluster Environment

Reference notes for the cluster the course uses, consolidated from
homework Slurm scripts (`HPC_HW/homework-{1,2,3}-main/`), lecture
examples (`lecture-examples/lecture-{04,15,16}/`), and direct queries
on the login node (2026-05-01).

This file exists so `Makefile`s, Slurm scripts, and the report can cite
exact toolchain details without re-discovering them every time.

---

## 1. Resolved

### 1.1 Toolchain (CPU)

| Item | Value | Source |
|---|---|---|
| OS / arch | Red Hat Linux on Intel **Cascade Lake** | CMake-generated paths in `lecture-examples/lecture-15/example-4/build/Makefile` (`linux-cascadelake-...`) |
| C/C++ compiler | **`g++` (GCC) 11.5.0** (Red Hat 11.5.0-5, 20240719) | `mpicc --version` on login node |
| Standard | `-std=c++17` | `HPC_HW/homework-3-main/question-6/submit.slurm` |
| Optimization | `-O2` or `-O3` | HW3 Q5 / Q6 |
| OpenMP | Built-in via `-fopenmp` | HW3 Q5 / Q6 |
| BLAS / LAPACK | **OpenBLAS**, installed via Spack | HW3 Q6 links `-lopenblas`, resolves path with `spack location -i openblas` |
| Threading control | `OPENBLAS_NUM_THREADS=1` (set `MKL_NUM_THREADS=1` defensively too) | HW3 Q6, lecture-15/example-4 |

**Important quirk:** OpenBLAS is **not** on the default linker path. Builds must explicitly add the Spack include/lib dirs and rpath:

```makefile
BLAS := $(shell spack location -i openblas)
CXXFLAGS += -I$(BLAS)/include
LDFLAGS  += -L$(BLAS)/lib -Wl,-rpath,$(BLAS)/lib -lopenblas -lm
```

Without `-Wl,-rpath`, the binary links fine on the login node but fails at runtime in a Slurm job (`error while loading shared libraries: libopenblas.so`).

### 1.2 MPI

| Item | Value | Source |
|---|---|---|
| Implementation | **Open MPI 4.1.7** | `mpirun --version` on login node |
| Compiler wrapper | `mpicxx` / `mpicc` (wraps `g++ 11.5.0`) | `mpicc --version` |
| Launch | `srun` (not `mpirun`) inside Slurm | All HW MPI scripts |
| Confirmed working at | 16 nodes × 1 task × 1 cpu (HW3 Q4); 2 nodes × 1 task × 16 cpus (lecture-16/ex-3) | HW Slurm scripts |
| Cluster MPI cap | **Not yet confirmed** — need `sinfo` + `sacctmgr` | — |

### 1.3 GPU / CUDA

| Item | Value | Source |
|---|---|---|
| Spack env | **`CS-2050-gpu`** (separate from CPU env `CS-2050`) | `HPC_HW/homework-3-main/question-5/submit-gpu.sh` |
| Slurm partition | `gpu` (with `--exclusive`) | HW3 Q5 |
| CUDA compiler | `nvcc -O3 -arch=sm_89` | `Lecture Notes/lecture-11.md`, `lecture-12.md` |
| GPU compute capability | **8.9 (Ada Lovelace)** | `-arch=sm_89` flag in lecture-11, lecture-12 |
| GPU model | **4× NVIDIA L4 per GPU node**, 24 GB each, ~72 W TDP | `nvidia-smi` on GPU node (2026-05-01) |
| Driver / driver-side CUDA | Driver **570.172.08**, CUDA **12.8** | `nvidia-smi` (2026-05-01) |
| Mixed C++/CUDA build | `find_package(CUDAToolkit) && enable_language(CUDA)` in CMake | `lecture-examples/lecture-20/example-2/md_simulator/CMakeLists.txt` |
| `nvcc --version` | **Deferred** — interactive GPU-node session returned "command not found"; suspect wrong Spack env active or `nvcc` provisioned in a sub-path. Resolve at start of Phase D. | — |
| CuPy version | **Deferred** to Phase D start | — |
| `cp.linalg.svd` batching at our slice sizes | **Deferred** to Phase D start (microbenchmark in §3.3) | — |

> **Performance note — L4 is an inference-class card.** Theoretical FP64 throughput on L4 is ~0.49 TFLOPS (vs. ~30 TFLOPS FP16). Our `float64` SVDs may be only modestly faster than the 16-thread Cascade Lake CPU baseline, not orders of magnitude. Plan the Part 4 results discussion around this — speedup vs. serial and bandwidth utilization vs. peak are the right framings; do not anchor on FP16-class numbers.

**GPU inspection script lifted from lecture-11:**
```bash
#!/bin/bash -l
#SBATCH --job-name=inspect-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

lscpu
lstopo --output-format svg -v --no-io > output.svg
nvidia-smi
gcc --version
nvcc --version
```

### 1.4 Julia — ✅ available via Spack env

| Item | Value | Source |
|---|---|---|
| Default `$PATH` | **Not** on system path (`which julia` fails on login node without Spack activated). | Login-node check (2026-05-01) |
| Inside CS-2050 Spack env | **Available** — `julia` is provisioned in the `CS-2050` Spack env. No `juliaup` install needed. | `lecture-examples/lecture-18/example-3/submit.sh` calls `julia` directly after `spack env activate -p CS-2050` |
| Threading | `julia -t N example.jl` for N threads (`Base.Threads.@threads`). | `lecture-examples/lecture-18/example-2/example.jl` |
| MPI integration | Use `MPI.jl` + `MPIPreferences.use_system_binary` to bind to the cluster's Open MPI 4.1.7. Recipe below. | `lecture-examples/lecture-18/example-3/submit.sh` |
| Action required | Confirm version inside the activated env: `julia --version`. Then pin in `additional/Project.toml`. | — |

**MPI-Julia recipe lifted from lecture-18 example-3** (one-time setup per environment):
```bash
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

export MPI_LIBDIRS=$(mpicc --showme:libdirs)
julia -e 'using Pkg; Pkg.add("MPI"); Pkg.add("MPIPreferences")'
julia -e 'using MPIPreferences; MPIPreferences.use_system_binary(; extra_paths=[ENV["MPI_LIBDIRS"]])'

srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=1 julia example.jl
```

### 1.5 Slurm

| Item | Value | Source |
|---|---|---|
| Partitions seen in course material | `general` (default CPU), `metal`, `gpu` | grep across HW + lectures |
| Per-node CPU counts seen | 16, 24, 36 (varies by node / partition) | HW Slurm scripts |
| Module system | **None** — Spack envs only, no `module load` anywhere | grep across HW + lectures |
| Spack activation | `. ~/161588/spack/share/spack/setup-env.sh && spack env activate -p CS-2050` | All HW Slurm scripts |
| OMP env conventions | `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, `MKL_NUM_THREADS=1` | `lecture-15/example-4/run_openmp.slurm` |
| Thread sweep convention | 1, 2, 4, 8, 16 (HW3 Q6); 1, 2, 4, 8, 18, 36 on a 36-core node (HW2 Q3) | HW Slurm scripts |

---

## 2. Reusable templates

### 2.1 CPU Makefile pattern (lifted from HW3 Q6)

```makefile
SPACK_OPENBLAS := $(shell spack location -i openblas)

CXX      := g++
CXXFLAGS := -O2 -std=c++17 -fopenmp -I$(SPACK_OPENBLAS)/include
LDFLAGS  := -L$(SPACK_OPENBLAS)/lib -Wl,-rpath,$(SPACK_OPENBLAS)/lib
LDLIBS   := -lopenblas -lm

target: source.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)
```

### 2.2 CPU Slurm header (OpenMP, single node, exclusive)

```bash
#!/usr/bin/env bash
#SBATCH --job-name=tsvdm_omp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=tsvdm_omp_%j.out

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

for t in 1 2 4 8 16; do
    OMP_NUM_THREADS=$t ./tsvdm_openmp <args>
done
```

### 2.3 MPI Slurm header (HW3 Q4 strong-scaling pattern)

```bash
#!/bin/bash
#SBATCH --job-name=tsvdm_mpi
#SBATCH --nodes=N
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=tsvdm_mpi_%j.out

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

for P in 1 2 4 8 16; do
    srun -N $P -n $P ./tsvdm_mpi <args>
done
```

### 2.4 GPU Slurm header (HW3 Q5 pattern)

```bash
#!/bin/bash
#SBATCH --job-name=tsvdm_cuda
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

python run_cupy.py --m 256 --p 256 --n 256
```

---

## 3. Open items — what to run next

### 3.1 Login node (~ 30 seconds)

```bash
sinfo -o "%P %c %m %G %D"
sacctmgr -n show assoc user=$USER format=qos,maxjobs,maxsubmit,maxwall
```

Output goes into §1.5 (real partition list, cores/node, GPUs/node) and §1.2 (MPI cap).

### 3.2 Confirm Julia version inside the Spack env (~ 30 seconds, login node)

Julia ships with the `CS-2050` Spack env (no `juliaup` install needed — confirmed via `lecture-examples/lecture-18/example-3/submit.sh`). Just verify:
```bash
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050
julia --version
```

Record the version in §1.4 above and pin it in `additional/Project.toml`.

### 3.3 GPU node — deferred to Phase D start

`nvidia-smi` was run on 2026-05-01 and recorded above (§1.3): 4× L4, driver 570.172.08, CUDA 12.8. `nvcc --version` returned "command not found" in that session — most likely the user activated `CS-2050` rather than `CS-2050-gpu`, or `nvcc` is provisioned in a sub-path of the env view. **Per user instruction, all remaining GPU recon (specific `nvcc --version`, CuPy version, batching microbenchmark) is deferred to the start of Phase D** (Part 4 implementation).

When Phase D begins, run on a `gpu`-partition node:
```bash
srun --partition=gpu --gres=gpu:1 --exclusive --time=00:30:00 --pty bash
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

# If nvcc is still not found, try:
which nvcc || find $(spack location -e CS-2050-gpu)/.spack-env/view -name nvcc 2>/dev/null
nvcc --version
python -c "import cupy; print('cupy', cupy.__version__, 'rt', cupy.cuda.runtime.runtimeGetVersion())"
```

Then run the **batching microbenchmark** (decides whether `cp.linalg.svd` truly batches at our slice sizes, or silently serializes — drives the Part 4 implementation choice):

```python
import cupy as cp, time

for m in (64, 128, 256, 512):
    n = 64
    A = cp.random.standard_normal((n, m, m), dtype=cp.float64)
    cp.linalg.svd(A); cp.cuda.Stream.null.synchronize()  # warmup

    cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    cp.linalg.svd(A)
    cp.cuda.Stream.null.synchronize()
    t_batched = time.perf_counter() - t0

    cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
    for i in range(n):
        cp.linalg.svd(A[i])
    cp.cuda.Stream.null.synchronize()
    t_serial = time.perf_counter() - t0

    print(f"m={m:4d}  batched={t_batched*1e3:8.2f}ms  "
          f"serial={t_serial*1e3:8.2f}ms  ratio={t_serial/t_batched:.2f}x")
```

**Decision rule:**
- `ratio` >> 1 → CuPy is genuinely batching → Part 4 is one line (`cp.linalg.svd(A)`).
- `ratio` ~ 1 → CuPy serializes internally → fall back to either CUDA streams (one stream per slice) or raw `cusolverDnDgesvdaStridedBatched` via CuPy low-level bindings.

Append the table of `(m, t_batched, t_serial, ratio)` into §1.3 of this file once collected.

---

## 4. Source-of-truth pointers

- HW3 Q6 (`HPC_HW/homework-3-main/question-6/`) — closest analog to the project's serial+OpenMP target. The `submit.slurm` is the canonical OpenBLAS+OpenMP build pattern.
- HW3 Q4 (`question-4/job.slurm`) — canonical MPI strong/weak scaling launcher.
- HW3 Q5 (`question-5/submit-gpu.sh`) — canonical GPU partition + `CS-2050-gpu` env activation.
- `lecture-examples/lecture-15/example-4/run_openmp.slurm` — canonical OMP env settings (`OMP_PLACES=cores`, `OMP_PROC_BIND=close`).
- `lecture-examples/lecture-16/example-3/run_mpi.slurm` — multi-node MPI on the `general` partition.
- `lecture-examples/lecture-17/example-1/CMakeLists.txt` — canonical CMake pattern for OpenBLAS (`set(BLA_VENDOR OpenBLAS); find_package(BLAS REQUIRED)`) + Spack-env include path.
- `lecture-examples/lecture-18/example-3/submit.sh` — canonical Julia + MPI recipe via `MPIPreferences.use_system_binary`.
- `lecture-examples/lecture-20/example-2/md_simulator/CMakeLists.txt` — canonical mixed C++ / CUDA / OpenMP / MPI CMake template.
- `Lecture Notes/lecture-11.md` — `inspect-gpu` Slurm template (lscpu, lstopo, nvidia-smi, nvcc --version) and `nvcc -O3 -arch=sm_89` compile pattern.
