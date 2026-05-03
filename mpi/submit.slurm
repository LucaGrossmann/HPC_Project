#!/bin/bash
# Part 3 — MPI. Strong-scaling sweep across nodes, BLAS pinned to 1 thread.
#
# Strategy (plan.md §4 Part 3):
#   - Allocate 8 nodes, 1 task per node; sweep srun -N P -n P over P ∈ {1,2,4,8}.
#   - OPENBLAS_NUM_THREADS=1: BLAS stays single-threaded so MPI is the only
#     parallelism. (Hybrid MPI+OpenMP is out of scope for the simple version.)
#
# Wall time budget: ≤ 10 min per the project staff cap (plan.md §9).

#SBATCH --job-name=tsvdm_mpi
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

# Activate the course Spack env (top-level CLAUDE.md "Cluster Setup").
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

# Pin BLAS to 1 thread per rank.
export OPENBLAS_NUM_THREADS=1

cd "$(dirname "$0")"
mkdir -p results

FIXTURE="${1:-../serial/fixtures/medium.bin}"

echo "fixture=$FIXTURE  host=$(hostname)"
echo

for P in 1 2 4 8; do
    srun -N $P -n $P ./tsvdm_mpi "$FIXTURE" --reps 3
done
