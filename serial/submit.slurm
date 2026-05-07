#!/bin/bash
# Part 1 — serial baseline. One rank, one thread, BLAS pinned to 1 thread.
#
#SBATCH --job-name=tsvdm_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

export OPENBLAS_NUM_THREADS=1

make clean && make -s

srun ./tsvdm_serial "${1:-fixtures/small.bin}"
exit $?
