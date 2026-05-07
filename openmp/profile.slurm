#!/usr/bin/env bash
# VTune hotspots collection — Part 2 OpenMP, 16 threads, medium fixture.
#
#SBATCH --job-name=tsvdm_omp_vtune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

# Pin BLAS to one thread so the only parallelism is the explicit OpenMP
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close

make clean && make -s

FIXTURE="${1:-../serial/fixtures/medium.bin}"
TAG="${2:-v1}"
mkdir -p results

# Hotspots collection. -knob sampling-interval=1 means 1ms (default 10ms);
# the medium fixture runs ~1s at 16 threads so default 10ms gives only ~100
# samples, too few. 1ms gives ~1000 samples.
vtune -collect hotspots \
      -knob sampling-mode=hw \
      -knob sampling-interval=1 \
      -result-dir results/vtune_omp16_$TAG \
      -- ./tsvdm_openmp "$FIXTURE"

# Text summary for the report (independent of GUI).
vtune -report summary -result-dir results/vtune_omp16_$TAG \
      > results/vtune_summary_$TAG.txt

# Top hotspots by CPU time, grouped by function.
vtune -report hotspots -result-dir results/vtune_omp16_$TAG \
      -group-by function -limit 20 \
      > results/vtune_hotspots_$TAG.txt

# Same hotspots in CSV form 
vtune -report hotspots -result-dir results/vtune_omp16_$TAG \
      -group-by function -limit 20 -format csv -csv-delimiter comma \
      > results/vtune_hotspots_$TAG.csv