#!/bin/bash
# Part 1 — serial baseline. One rank, one thread, BLAS pinned to 1 thread.
#
#SBATCH --job-name=tsvdm_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=results/%x-%j.out

# Activate the course Spack env (plan.md §"Cluster Setup" in the top-level CLAUDE.md).
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

# Keep BLAS single-threaded — the serial baseline must stay single-threaded.
export OPENBLAS_NUM_THREADS=1

cd "$(dirname "$0")"
mkdir -p results

./tsvdm_serial "${1:-fixtures/small.bin}"