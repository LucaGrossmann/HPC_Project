"""Benchmark: C++ serial tsvdm vs Python Part 0 tsvdm.

Generates a fresh fixture for each (m, p, n), runs both implementations
for ``--reps`` repetitions (with one untimed warmup each), reports
median wall time + min + max, and prints a comparison table.

Only ``tsvdm`` is timed — not reconstruction or validation.

Usage
-----
    python tools/benchmark_cpp_vs_python.py --reps 10

    # custom size sweep
    python tools/benchmark_cpp_vs_python.py \\
        --sizes 32,32,16 128,128,32 256,256,64
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cuda.tsvdm_core import tsvdm
from cuda.tsvdm_utils import random_orthogonal

DEFAULT_SIZES = [
    (32, 32, 16),
    (64, 64, 32),
    (128, 128, 32),
    (256, 256, 64),
]

FIXTURE_OUT = re.compile(r"median_ms=([\d.]+) min_ms=([\d.]+) max_ms=([\d.]+)")


def time_python(m: int, p: int, n: int, reps: int, seed: int = 0):
    """Return (median_ms, min_ms, max_ms) for Python tsvdm."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m, p))
    M = random_orthogonal(n, rng)

    # Warmup.
    _ = tsvdm(A, M)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = tsvdm(A, M)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return median(times), min(times), max(times)


def time_cpp(binary: Path, gen_fixture: Path, m: int, p: int, n: int,
             reps: int, seed: int = 0):
    """Return (median_ms, min_ms, max_ms) for C++ tsvdm_serial."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        fixture_path = Path(tmp.name)
    try:
        subprocess.run(
            [sys.executable, str(gen_fixture),
             "--rows", str(m), "--cols", str(p), "--slices", str(n),
             "--seed", str(seed), "--out", str(fixture_path)],
            check=True, capture_output=True,
        )
        result = subprocess.run(
            [str(binary), str(fixture_path), "--reps", str(reps)],
            check=True, capture_output=True, text=True,
        )
    finally:
        fixture_path.unlink(missing_ok=True)

    m_out = FIXTURE_OUT.search(result.stdout)
    if not m_out:
        raise RuntimeError(f"could not parse C++ output: {result.stdout!r}")
    return (float(m_out.group(1)),
            float(m_out.group(2)),
            float(m_out.group(3)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--sizes", nargs="*", default=None,
                    help="List of m,p,n triples (comma-separated).")
    ap.add_argument("--binary", type=Path,
                    default=ROOT / "serial" / "tsvdm_serial")
    ap.add_argument("--gen-fixture", type=Path,
                    default=ROOT / "cuda" / "gen_fixture.py")
    args = ap.parse_args()

    if args.sizes:
        sizes = [tuple(int(x) for x in s.split(",")) for s in args.sizes]
    else:
        sizes = DEFAULT_SIZES

    if not args.binary.exists():
        print(f"error: C++ binary not found at {args.binary}; run `make` first",
              file=sys.stderr)
        return 2

    print(f"{'size':>14}  {'Python med':>12}  {'C++ med':>10}  "
          f"{'speedup':>8}  {'Py min':>8}  {'C++ min':>8}")
    print("-" * 78)

    for (m, p, n) in sizes:
        py_med, py_min, py_max = time_python(m, p, n, args.reps)
        cpp_med, cpp_min, cpp_max = time_cpp(
            args.binary, args.gen_fixture, m, p, n, args.reps
        )
        speedup = py_med / cpp_med if cpp_med > 0 else float("inf")
        size_str = f"{m:3d}x{p:3d}x{n:3d}"
        print(f"{size_str:>14}  "
              f"{py_med:10.2f}ms  "
              f"{cpp_med:8.2f}ms  "
              f"{speedup:7.2f}x  "
              f"{py_min:6.2f}ms  "
              f"{cpp_min:6.2f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
