#!/usr/bin/env bash
# Part 2 — OpenMP. Strong-scaling sweep, single node, BLAS pinned to 1 thread.
#
# Strategy (plan.md §4 Part 2):
#   - OMP_NUM_THREADS sweep over {1, 2, 4, 8, 16}.
#   - OPENBLAS_NUM_THREADS=1 to avoid nested parallelism (BLAS x OpenMP).
#   - Default fixture = ../serial/fixtures/medium.bin (64x64x32). Pass another
#     fixture as the first arg to override.
#
# Wall time budget: ≤ 10 min per the project staff cap (plan.md §9).

#SBATCH --job-name=tsvdm_omp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

# Activate the course Spack env (top-level CLAUDE.md "Cluster Setup").
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

# Pin BLAS to 1 thread — the only parallelism we want is the outer OpenMP loop.
export OPENBLAS_NUM_THREADS=1

# Standard OpenMP placement — keep threads on cores, packed close.
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd "$(dirname "$0")"
mkdir -p results

FIXTURE="${1:-../serial/fixtures/medium.bin}"

echo "fixture=$FIXTURE"
echo "host=$(hostname)  cpus_per_task=$SLURM_CPUS_PER_TASK"
echo

for t in 1 2 4 8 16; do
    OMP_NUM_THREADS=$t ./tsvdm_openmp "$FIXTURE" --reps 3
done
