"""Element-wise cross-check: C++ serial vs Python reference.

Runs ``serial/tsvdm_serial`` on a fixture, dumping its reconstruction
``Aapprox`` to a binary file. Then loads the same fixture in Python,
runs ``cuda.tsvdm_core.tsvdm + reconstruct``, and compares the two
reconstructions element-wise.

Usage
-----
    python tools/compare_cpp_python.py serial/fixtures/small.bin

Exits non-zero if max |A_cpp - A_py| / max |A| exceeds 1e-10.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cuda.tsvdm_core import tsvdm, reconstruct


def load_fixture(path: Path):
    """Load a .bin fixture into Python's (n, m, p) layout."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"TSVD":
            raise ValueError(f"bad magic in {path}: {magic!r}")
        m, p, n = struct.unpack("<iii", f.read(12))
        # A on disk: n slices, each (m, p) column-major.
        A_bytes = f.read(m * p * n * 8)
        M_bytes = f.read(n * n * 8)

    # Read per-slice column-major (m, p) then stack along axis 0.
    slices = [
        np.frombuffer(A_bytes, dtype=np.float64, count=m * p,
                      offset=k * m * p * 8).reshape((p, m)).T
        for k in range(n)
    ]
    A = np.stack(slices, axis=0)                            # (n, m, p)
    M = np.frombuffer(M_bytes, dtype=np.float64).reshape((n, n)).T.copy()
    return A, M, (m, p, n)


def load_cpp_dump(path: Path, m: int, p: int, n: int) -> np.ndarray:
    """Load the C++ reconstruction dump (column-major (m, p, n)) into
    Python's (n, m, p) layout."""
    raw = np.fromfile(path, dtype=np.float64, count=m * p * n)
    slices = [raw[k * m * p:(k + 1) * m * p].reshape((p, m)).T
              for k in range(n)]
    return np.stack(slices, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--binary", type=Path,
                    default=ROOT / "serial" / "tsvdm_serial",
                    help="path to the compiled C++ binary")
    ap.add_argument("--atol", type=float, default=1e-10,
                    help="absolute tolerance for element-wise agreement")
    args = ap.parse_args()

    if not args.binary.exists():
        print(f"error: C++ binary not found at {args.binary}; run `make` first",
              file=sys.stderr)
        return 2

    A_py, M_py, (m, p, n) = load_fixture(args.fixture)
    print(f"fixture: m={m} p={p} n={n}")

    # 1. Run C++ with a dump path so it writes Aapprox to disk.
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        dump_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [str(args.binary), str(args.fixture), "--dump", str(dump_path)],
            capture_output=True, text=True, check=True,
        )
        print(f"cpp stdout: {result.stdout.strip()}")
        A_cpp = load_cpp_dump(dump_path, m, p, n)
    finally:
        dump_path.unlink(missing_ok=True)

    # 2. Python reference.
    U, S, V = tsvdm(A_py, M_py)
    A_py_approx = reconstruct(U, S, V, M_py)

    # 3. Element-wise comparison.
    diff = A_cpp - A_py_approx
    max_abs_diff = float(np.max(np.abs(diff)))
    max_abs_A    = float(np.max(np.abs(A_py)))
    rel_diff     = float(np.linalg.norm(diff) / np.linalg.norm(A_py))

    cpp_recon_err = float(np.linalg.norm(A_cpp - A_py) / np.linalg.norm(A_py))
    py_recon_err  = float(np.linalg.norm(A_py_approx - A_py) / np.linalg.norm(A_py))

    print()
    print(f"  rel_err (cpp vs A)       = {cpp_recon_err:.3e}")
    print(f"  rel_err (python vs A)    = {py_recon_err:.3e}")
    print(f"  rel_diff (cpp vs python) = {rel_diff:.3e}")
    print(f"  max |cpp - python|       = {max_abs_diff:.3e}")
    print(f"  max |A|                  = {max_abs_A:.3e}")

    if rel_diff > args.atol:
        print(f"\nFAIL: rel_diff {rel_diff:.3e} > atol {args.atol:.0e}")
        return 1
    print(f"\nOK: C++ and Python agree to {args.atol:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
