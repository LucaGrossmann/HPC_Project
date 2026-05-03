#!/bin/bash
# Part 5 — Julia. Single-node CPU job. Julia is provisioned in the CS-2050
# Spack env (cluster-environment.md §1.4).

#SBATCH --job-name=tsvdm_julia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

. ~/161588/spack/share/spack/setup-env.sh
spack env activate -p CS-2050

cd "$(dirname "$0")"
mkdir -p results

# Print Julia version for reproducibility.
julia --version
echo

# 3 sizes × 3 reps per plan.md §4 Part 5 (10-min cap with JIT-warmup budget).
for size in "64 64 64" "256 256 256" "512 512 256"; do
    julia tsvdm.jl --gen $size --reps 3
done

# Cross-check vs. the committed serial fixture (sanity).
julia tsvdm.jl ../serial/fixtures/medium.bin --reps 1
