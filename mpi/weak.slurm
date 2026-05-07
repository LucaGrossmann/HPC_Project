#!/bin/bash
# MPI weak scaling: n_base × P slices at P ranks (one rank per node).
#
#SBATCH --job-name=tsvdm_mpi_weak
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

export OPENBLAS_NUM_THREADS=1

make clean && make -s

# Ranks P   →   fixture with n = 32 · P
declare -A FIX=( [1]=32 [2]=64 [4]=128 [8]=256 )

for P in 1 2 4 8; do
    srun -N $P -n $P ./tsvdm_mpi ../serial/fixtures/weak_n${FIX[$P]}.bin
done