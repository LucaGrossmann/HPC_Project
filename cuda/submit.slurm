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

# CuPy is not provisioned in CS-2050-gpu (the course teaches Numba CUDA as
# its primary Python-on-GPU tool). t-SVDM needs batched SVD, which Numba
# doesn't expose, so we use CuPy — installed user-locally on first run.
# Idempotent: no-op if already installed.
#
# Note: the Spack env's Python is a venv with system-site-packages
# disabled, so pip install --user lands in ~/.local but the env's
# Python won't see it. Prepend the user-site dir to PYTHONPATH so the
# import resolves.
PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$HOME/.local/lib/python${PYVER}/site-packages:${PYTHONPATH:-}"

# Use `python -m pip` (not bare `pip`) so the install targets the active
# env's Python. Bare `pip` on this cluster resolves to a different Python
# (3.9), which drops packages into ~/.local/lib/python3.9/... where the
# env's Python 3.12 will never find them. Two complications:
#   1. The env's Python 3.12 ships without pip — bootstrap it via ensurepip.
#   2. Both steps are idempotent: if pip / cupy already exist, they no-op.
python -m pip --version >/dev/null 2>&1 || python -m ensurepip --user --quiet
python -c "import cupy" 2>/dev/null || python -m pip install --user --quiet cupy-cuda12x

# One-time recon: print versions so we can tell what we ran on.
nvidia-smi -L || true
python -c "import cupy; print('cupy', cupy.__version__, 'cuda_runtime', cupy.cuda.runtime.runtimeGetVersion())"
echo

# Strong-scaling sweep across problem sizes (plan.md §4 Part 4).
echo "=== size sweep ==="
for size in "64 64 64" "256 256 256" "512 512 256"; do
    python run_cupy.py --gen $size
done
echo

# Cross-check against the committed serial fixture.
echo "=== cross-check vs. serial fixture ==="
python run_cupy.py ../serial/fixtures/medium.bin
