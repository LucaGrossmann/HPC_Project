#!/usr/bin/env bash
# Part 5 — Julia. Single-node CPU job, BLAS pinned to 1 thread.

#SBATCH --job-name=tsvdm_julia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

# Activate the course Spack env (top-level CLAUDE.md "Cluster Setup").
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

# Pin BLAS to 1 thread — match the other CPU baselines.
export OPENBLAS_NUM_THREADS=1

# Print Julia version for reproducibility.
julia --version
echo "host=$(hostname)"
echo

# Size sweep (plan.md §4 Part 5).
echo "=== size sweep ==="
for size in "64 64 64" "256 256 256" "512 512 256"; do
    julia tsvdm.jl --gen $size --reps 3
done
echo

# Cross-check against the committed serial fixture.
echo "=== cross-check vs. serial fixture ==="
julia tsvdm.jl ../serial/fixtures/medium.bin --reps 1
