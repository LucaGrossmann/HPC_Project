#!/bin/bash
# Part 4 — CUDA via CuPy. GPU partition, single node.
# Sweep: 3 sizes × 3 reps + one fixture cross-check.

#SBATCH --job-name=tsvdm_cuda
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050-gpu

# CuPy is not in CS-2050-gpu; install into a project-local venv (cuda/.venv/),
# self-created on first run, reused thereafter. See cuda/README.md.
VENV=".venv"
CUPY_VER="13.6.0"

if [ ! -d "$VENV" ]; then
    # Some Spack-built Pythons ship without pip wheels; fall back to get-pip.py.
    if ! python -m venv "$VENV" 2>/dev/null; then
        python -m venv "$VENV" --without-pip
        source "$VENV/bin/activate"
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | python
        deactivate
    fi
fi
source "$VENV/bin/activate"

if ! python -c "import cupy; assert cupy.__version__ == '$CUPY_VER'" 2>/dev/null; then
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet "cupy-cuda12x==$CUPY_VER"
fi

nvidia-smi -L || true
python -c "import cupy; print('cupy', cupy.__version__, 'cuda_runtime', cupy.cuda.runtime.runtimeGetVersion())"
echo

echo "=== size sweep ==="
for size in "64 64 64" "256 256 256" "512 512 256"; do
    python run_cupy.py --gen $size
done
echo

echo "=== cross-check vs. serial fixture ==="
python run_cupy.py ../serial/fixtures/medium.bin
