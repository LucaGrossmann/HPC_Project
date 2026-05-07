# Incomplete sections + bugs to address

Generated 2026-05-07. Prioritized by what blocks the writeup PDF from
shipping vs. what just makes the report sharper.

---

## Blocking: writeup won't compile cleanly without these

### B1. ~~Missing image files referenced in the introduction~~ ✅ RESOLVED

`report/figures/{tensor.jpg, mode_3_prod.jpg, tSVDM_diagram.png}`
are now present and referenced from the writeup. The
\texttt{tSVDM\_diagram} caption is annotated "adapted from
\cite{kilmer2021tensor}".

### B2. ~~CuPy not yet running on the cluster~~ ✅ APPROACH CHANGED (per instructor)

The earlier `pip install --user` workaround failed because:
1. The `CS-2050-gpu` env's Python (3.12) lacks pip entirely
   (`No module named pip`).
2. Bare `pip` on PATH targeted a different Python (3.9), so files
   landed in `~/.local/lib/python3.9/...` where 3.12 never looks.

**New approach** (instructor-recommended): install CuPy into a
project-local venv at `cuda/.venv/`. `cuda/submit.slurm` is now
self-healing — creates the venv on first run, reuses thereafter:

```bash
[cluster]$ cd ~/HPC_Project && git pull
[cluster]$ cd cuda && sbatch submit.slurm
```

CuPy version is pinned to `13.6.0` for reproducibility. The venv
script handles the case where the parent Python lacks bundled pip
wheels (falls back to `get-pip.py` via curl). Open question: whether
compute nodes have outbound HTTPS for the `bootstrap.pypa.io` /
PyPI fetches. If they don't, the install will fail with a network
error — paste the error and we'll cache the `cupy-cuda12x` wheel
locally.

### B3. Profiler runs still pending

§3.5 currently has placeholder figure boxes for both VTune (CPU) and
Nsight Systems (GPU). To clear those:

- **VTune**: `cd openmp && sbatch profile.slurm`. Generates
  `openmp/results/vtune_omp16/` (raw) plus
  `vtune_summary.txt` and `vtune_hotspots.txt`. Open in VTune GUI
  (or export HTML and view in Safari, since macOS lost the native
  GUI). Screenshot the Hotspots view → `report/figures/profile_vtune.png`.
- **Nsight Systems**: `cd cuda && sbatch profile.slurm` (script
  exists per the runbook). Produces `cuda/results/nsys_cupy.nsys-rep`.
  Open locally in `nsys-ui` (free macOS download from NVIDIA).
  Screenshot the timeline → `report/figures/profile_nsys.png`.

Once those PNGs exist, replace the two `\fbox{...}` placeholder
figures with `\includegraphics` calls.

---

## Non-blocking: writeup claims to verify or correct

### C1. §2.4 CuPy — `cusolverDnXgesvdjBatched` claim is unverified

The methods text says CuPy "dispatches internally to cuSOLVER's
batched Jacobi kernel `cusolverDnXgesvdjBatched`; we confirm this in
Nsight Systems". Two issues:

1. The Jacobi batched routine has a per-matrix size limit (typically
   ≤ 32×32 in older cuSOLVER versions; the upper bound has shifted
   across versions but stays small). At our slice sizes
   (256×256, 512×512), CuPy almost certainly falls back to a
   non-batched path or to `gesvdaStridedBatched`.
2. The "we confirm this in Nsight Systems" sentence is a forward
   reference to a profile run that hasn't happened yet.

**Suggested rewrite** (defer the kernel-name claim until Nsight is
run):

> The dominant per-slice SVDs collapse into one batched call,
> `cp.linalg.svd(A_hat, full_matrices=False)`, which dispatches
> internally to cuSOLVER. Section~\ref{subsec:profiling} reports
> which specific cuSOLVER routine is selected at our slice sizes.

### C2. §1 — "All five are validated against `mprod_package`"

Currently the writeup claims all five backends are validated against
the canonical Python implementation. Only the Python reference itself
(`cuda/tsvdm_core.py`) is cross-validated against `mprod_package` via
`cuda/tests/test_oracle.py`. The C++/MPI/CuPy/Julia backends are
self-validated (relative reconstruction error) and the Julia driver's
"cross-check vs. fixture" line at the bottom of
`additional/results/tsvdm_julia-110245.out` does cross-check the
fixture, but they're not all run through `mprod_package` directly.

**Fix**: soften to "validated by reconstruction error and
cross-validated against the Python reference, which is itself
oracle-tested against `mprod_package`". One word per backend in
§3.1 already nearly says this; tighten the §1 sentence to match.

### C3. §2.3 MPI — emphasize the redundant work

The methods section already mentions that every rank runs the forward
mode-3 dgemm on the full tensor. This is the load-bearing detail
behind the poor weak scaling discussed in §3.3 — worth making the
connection explicit. One sentence in §2.3 along the lines of "this
redundancy is intentional (it lets the SVD step run with no further
communication) but it caps weak-scaling efficiency, as Figure~\ref{fig:weak} shows" would tie the methods and results together.

---

## Code-level bugs / smells

### D1. Serial `large.bin` timed out at 5 min cap

`serial/results/tsvdm_serial-110232.out` is just a SLURM cancellation
notice. We bumped the cap to 15 min for personal runs but the staff
default needs to fit in 10. Three options:

1. Skip serial-large in the staff sweep, run it locally for the
   writeup data (already done — we have it elsewhere via OpenMP T=1).
2. Shrink the "large" definition to 384³.
3. Reduce reps from 30 → 10 in the binary so 30s/rep × 30 reps becomes
   30s/rep × 10 reps.

Currently the writeup uses OpenMP T=1 numbers as the serial baseline
where serial itself isn't available. Acceptable, since OpenMP at
$T=1$ is the same code path with one thread.

### D2. `cuda_cpp/tsvdm_cuda.cu` — built but never run

- `cusolverDnDgesvd` is **deprecated** in CUDA 12+ and emits warnings
  at compile time. If they're noisy, add `-Wno-deprecated-declarations`
  to NVCCFLAGS.
- Multi-arch (`sm_86 + sm_89`) compile may fail if the cluster's
  `nvcc` is older than 12.0. Smoke-test with `nvcc --version` first;
  if it fails, drop to `-arch=sm_86` only (A10G) and retry on a
  different node if Slurm puts you on an L4.
- The `m ≥ p` constraint at line ~285 only matters for non-square
  slices; all benchmark sizes are square so it should never trip.
- Untimed reconstruction uses `cublasDgemmStridedBatched`, introduced
  in CUDA 8.0 — fine on any modern toolkit.
- Smoke test command: `cd cuda_cpp && make && ./tsvdm_cuda
  ../serial/fixtures/small.bin` — should print rel_err ≈ 1e-15.

### D3. `cuda/run_cupy.py` — when CuPy comes back, the `--gen` path skips fixture validation

The CuPy driver runs `--gen 64 64 64`, `--gen 256 256 256`, `--gen
512 512 256` (no fixtures), then a single fixture cross-check at the
end. That's three timing rows + one validation row. Verify the
output line format from the driver matches what `parse_results.py`
expects (it should — same template as the C++ binaries). If the
parser reports 0 rows on a successful CuPy run, the regex needs a
new branch for whatever token CuPy uses (probably `backend=cupy` or
no token at all, like Julia). Easy fix once we have a working .out
file to inspect.

---

## What's already populated in the writeup

Sections fully populated with data + figures:

- §3.1 Correctness — relative-error numbers, oracle pipeline
- §3.2 Strong scaling — both fixtures, OpenMP + MPI, parallel-efficiency analysis
- §3.3 Weak scaling — both backends with diagnostic discussion
- §3.4 Cross-implementation comparison — CPU + Julia bars (CuPy column placeholder in the figure)
- §4 Conclusion — practitioner takeaway, limitations, future work

Figures generated and referenced from the writeup:

- `report/figures/strong_scaling.pdf`
- `report/figures/weak_scaling.pdf`
- `report/figures/efficiency_bars.pdf`
- `report/figures/cross_impl_comparison.pdf`
- Per-impl plots: `serial/results/serial_size_scaling.pdf`,
  `openmp/results/{omp_strong,omp_weak}.pdf`,
  `mpi/results/{mpi_strong,mpi_weak}.pdf`,
  `additional/results/julia_size_scaling.pdf`
