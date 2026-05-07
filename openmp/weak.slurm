#!/usr/bin/env bash
# OpenMP weak scaling: n_base × T slices at T threads.
#
#SBATCH --job-name=tsvdm_omp_weak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

export OPENBLAS_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

make clean && make -s

# Threads T   →   fixture with n = 32 · T  (weak: per-thread work fixed)
declare -A FIX=( [1]=32 [2]=64 [4]=128 [8]=256 [16]=512 )

for t in 1 2 4 8 16; do
    OMP_NUM_THREADS=$t \
        ./tsvdm_openmp ../serial/fixtures/weak_n${FIX[$t]}.bin
done