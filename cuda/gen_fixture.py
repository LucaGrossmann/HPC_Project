"""Generate a seeded (A, M) binary fixture for the C++ implementations.

On-disk format
--------------
Little-endian, ``sizeof(double) == 8``, ``sizeof(int32) == 4``.

    offset    size              field
    --------  ----------------  ----------------------------------------
    0         4 bytes           magic = "TSVD" (ASCII)
    4         4 bytes           int32 m
    8         4 bytes           int32 p
    12        4 bytes           int32 n
    16        m*p*n * 8         A: n slices, each (m, p) column-major
    16+mpn*8  n*n * 8           M: (n, n) column-major

Layout conversion
-----------------
Part 0 (Python) uses shape ``(n, m, p)`` C-order. Part 1+ (C++) uses shape
``(m, p, n)`` column-major. We do the conversion here: for each slice k,
write ``A_python[k]`` in Fortran-order (column-major). That matches the
C++ side reading ``m*p`` contiguous doubles per slice.

Usage
-----
    python cuda/gen_fixture.py \\
        --rows 8 --cols 8 --slices 4 --seed 0 \\
        --out serial/fixtures/small.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda.tsvdm_core import tsvdm, reconstruct
from cuda.tsvdm_utils import random_orthogonal, relative_error

MAGIC = b"TSVD"


def write_fixture(path: Path, m: int, p: int, n: int, seed: int) -> float:
    """Write a seeded fixture and return the Python reference recon error."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m, p))          # Python layout (n, m, p)
    M = random_orthogonal(n, rng)               # (n, n)

    # Sanity: run our own decomposition so the sidecar records what C++
    # should match.
    U, S, V = tsvdm(A, M)
    rel_err = relative_error(A, reconstruct(U, S, V, M))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iii", m, p, n))
        # A: one slice at a time, each written in column-major (Fortran order).
        for k in range(n):
            f.write(np.asfortranarray(A[k]).tobytes())
        # M: column-major.
        f.write(np.asfortranarray(M).tobytes())

    return rel_err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, required=True, help="m (rows per slice)")
    ap.add_argument("--cols", type=int, required=True, help="p (cols per slice)")
    ap.add_argument("--slices", type=int, required=True, help="n (number of slices)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rel_err = write_fixture(args.out, args.rows, args.cols, args.slices, args.seed)
    print(
        f"wrote {args.out} ({args.out.stat().st_size} bytes) "
        f"m={args.rows} p={args.cols} n={args.slices} seed={args.seed} "
        f"python_rel_err={rel_err:.2e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
