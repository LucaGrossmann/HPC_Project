#!/bin/bash
# Part 4 — CUDA via CuPy. GPU partition, single node.
#
# Strategy (plan.md §4 Part 4):
#   - 3 sizes × 3 reps = 9 timed runs.
#   - Plus one fixture cross-check vs. the committed serial fixture.
#   - L4 is inference-class (~0.49 TFLOPS FP64); expect modest speedup over
#     16-thread Cascade Lake (cluster-environment.md §1.3).
#
# Wall time budget: ≤ 10 min per the project staff cap (plan.md §9).

#SBATCH --job-name=tsvdm_cuda
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

# Activate the GPU Spack env (note: distinct from the CPU CS-2050 env).
. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

cd "$(dirname "$0")"
mkdir -p results

# One-time recon: print versions so we can tell what we ran on.
nvidia-smi -L || true
python -c "import cupy; print('cupy', cupy.__version__, 'cuda_runtime', cupy.cuda.runtime.runtimeGetVersion())"
echo

# Strong-scaling sweep across problem sizes (plan.md §4 Part 4).
echo "=== size sweep (3 sizes × 3 reps) ==="
for size in "64 64 64" "256 256 256" "512 512 256"; do
    python run_cupy.py --gen $size --reps 3
done
echo

# Cross-check against the committed serial fixture (sanity, 1-rep).
echo "=== cross-check vs. serial fixture ==="
python run_cupy.py ../serial/fixtures/medium.bin --reps 1
