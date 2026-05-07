#!/bin/bash
# Part 4 — CUDA via CuPy. GPU partition, single node.
#
# Strategy (plan.md §4 Part 4):
#   - 3 sizes × 3 reps = 9 timed runs.
#   - Plus one fixture cross-check vs. the committed serial fixture.
#
# CuPy provisioning
# -----------------
# CuPy is not installed in the course-managed CS-2050-gpu Spack env
# (the env teaches Numba CUDA as its primary Python-on-GPU tool, per
# instructor guidance). We install it into a project-local virtualenv
# at cuda/.venv/, self-created on first run, reused thereafter. This
# is reproducible: any clone of the repo can run `sbatch submit.slurm`
# and end up with CuPy at the pinned version below.
#
# Wall time budget: ≤ 10 min per the project staff cap (plan.md §9).
# First run pays ~60s overhead (venv + CuPy install); subsequent runs no-op.

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

# --- Self-healing venv setup ----------------------------------------------
VENV=".venv"
CUPY_VER="13.6.0"

if [ ! -d "$VENV" ]; then
    echo "==> Creating venv at $(pwd)/$VENV"
    # The Spack env's Python may ship without bundled pip wheels, so the
    # default `python -m venv` (which calls ensurepip internally) can fail.
    # Try the default first; on failure, fall back to a no-pip venv +
    # bootstrap pip via get-pip.py.
    if ! python -m venv "$VENV" 2>/dev/null; then
        echo "    default venv creation failed; bootstrapping pip via get-pip.py"
        python -m venv "$VENV" --without-pip
        # shellcheck source=/dev/null
        source "$VENV/bin/activate"
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | python
        deactivate
    fi
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

# Idempotent CuPy install: only re-install if missing or wrong version.
if ! python -c "import cupy; assert cupy.__version__ == '$CUPY_VER'" 2>/dev/null; then
    echo "==> Installing cupy-cuda12x==$CUPY_VER into $VENV"
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet "cupy-cuda12x==$CUPY_VER"
fi

# --- Sanity print ---------------------------------------------------------
nvidia-smi -L || true
python -c "import cupy; print('cupy', cupy.__version__, 'cuda_runtime', cupy.cuda.runtime.runtimeGetVersion())"
echo

# --- Strong-scaling sweep -------------------------------------------------
echo "=== size sweep ==="
for size in "64 64 64" "256 256 256" "512 512 256"; do
    python run_cupy.py --gen $size
done
echo

# --- Cross-check against the committed serial fixture --------------------
echo "=== cross-check vs. serial fixture ==="
python run_cupy.py ../serial/fixtures/medium.bin
