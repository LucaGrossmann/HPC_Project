#!/bin/bash
# Generate every fixture the C++ implementations need (small/medium/large
# for size sweeps + weak_n{32..512} for weak scaling).
#
# Submit from the repo root:
#   sbatch serial/gen_fixtures.slurm
#
#SBATCH --job-name=gen_fixtures
#SBATCH --output=serial/results/gen_fixtures_%j.out
#SBATCH --error=serial/results/gen_fixtures_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p serial/fixtures serial/results

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS=1

echo "host=$(hostname)  python=$(python --version 2>&1)"
echo

echo "=== Fixed-size fixtures ==="
python cuda/gen_fixture.py \
    --rows 64  --cols 64  --slices 64  --seed 0 \
    --out serial/fixtures/small.bin

python cuda/gen_fixture.py \
    --rows 256 --cols 256 --slices 256 --seed 0 \
    --out serial/fixtures/medium.bin

python cuda/gen_fixture.py \
    --rows 512 --cols 512 --slices 256 --seed 0 \
    --out serial/fixtures/large.bin

echo
echo "=== Weak-scaling fixtures ==="
# (m, p) fixed at 256x256, n scales with parallelism (per-worker work fixed).
# n_base = 32 → covers OpenMP T ∈ {1,2,4,8,16} and MPI P ∈ {1,2,4,8}.
for n in 32 64 128 256 512; do
    python cuda/gen_fixture.py \
        --rows 256 --cols 256 --slices "$n" --seed 0 \
        --out "serial/fixtures/weak_n${n}.bin"
done

echo
echo "=== Done. Fixtures on disk: ==="
ls -lh serial/fixtures/
