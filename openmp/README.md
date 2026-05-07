# `openmp/` — Part 2: shared-memory OpenMP

Diff from the serial baseline is small: `#pragma omp parallel for
schedule(dynamic)` on the per-slice SVD loop, plus matching pragmas
on three helper loops (assemble `S_hat`, transpose `V^T`, facewise
reconstruction matmul). Mode-3 GEMMs stay single-threaded
(`OPENBLAS_NUM_THREADS=1`) to avoid nested parallelism.

## Layout

| File | Role |
|---|---|
| `tsvdm_openmp.cpp` | OpenMP-pragma'd version of the serial source: `parallel for schedule(dynamic)` on `sliceSvdAll`, `schedule(static)` on the three helper loops, mode-3 GEMMs left serial. |
| `Makefile` | g++ with `-O3 -march=native -fopenmp` plus `-g -fno-omit-frame-pointer` (so VTune can attribute samples), Spack OpenBLAS via `-Wl,-rpath`. |
| `submit.slurm` | Strong-scaling sweep over `T ∈ {1, 2, 4, 8, 16}` on a single fixture (default medium). One run per thread count, all on one allocation. |
| `weak.slurm` | Weak-scaling sweep: `T ∈ {1, 2, 4, 8, 16}` paired with `n = 32·T` from the `weak_n*.bin` fixtures so per-thread work stays constant. |
| `profile.slurm` | VTune Hotspots collection at `T=16` on the medium fixture; emits `vtune_summary_v*.txt`, `vtune_hotspots_v*.{txt,csv}`, and the raw `vtune_omp16_v*/` tree. |
| `results/` | Per-Slurm-job stdout, scaling plots (`omp_strong.pdf`, `omp_weak.pdf`), and VTune text reports + raw trees (gitignored). |

## Run

```bash
[cluster]$ cd openmp
[cluster]$ sbatch submit.slurm                              # default: medium fixture
[cluster]$ sbatch submit.slurm ../serial/fixtures/large.bin # large
[cluster]$ sbatch weak.slurm                                # weak scaling
[cluster]$ sbatch profile.slurm                             # VTune
```

VTune outputs are large and gitignored. View the GUI tree on a
machine with VTune installed; the text summaries are committed.
