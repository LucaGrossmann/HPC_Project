#!/usr/bin/env python3
"""Part 4 — CuPy GPU t-SVDM driver.

Usage:
    ./run_cupy.py <fixture.bin>               # validate against fixture
    ./run_cupy.py --gen 256 256 256           # benchmark a fresh tensor

Layout convention: A is shape (n, m, p) C-order, M is (n, n). This
matches the existing NumPy reference in tsvdm_core.py — *not* the C++
column-major layout. Fixtures are loaded slice-by-slice and stacked.
"""

import argparse
import struct
import time

import numpy as np
import cupy as cp


def sync() -> None:
    cp.cuda.Stream.null.synchronize()


# --- Algorithm (the "CUDA" part — every call below uses cp.*) ----------------

def mode3(X, A):
    """Mode-3 product: apply the (n, n) matrix X along A's leading axis.

    A has shape (n, m, p) C-order. Reshaping to (n, m*p) is a stride-only
    view of the same bytes, so the whole mode-3 product is one cuBLAS
    dgemm: X @ A_flat, reshaped back. Same trick as the C++ baseline.
    """
    n = A.shape[0]
    return (X @ A.reshape(n, -1)).reshape(A.shape)


def tsvdm_cupy(A, M):
    """t-SVDM. A: (n, m, p), M: (n, n). Returns U (n, m, r), S (n, r, r), V (n, p, r)."""
    n, m, p = A.shape
    r = min(m, p)

    A_hat = mode3(M, A)

    # Batched SVD over the leading (n) axis. CuPy dispatches to cuSOLVER.
    U_hat, s_hat, Vt_hat = cp.linalg.svd(A_hat, full_matrices=False)

    # Pack singular values into a dense f-diagonal tensor (n, r, r).
    S_hat = cp.zeros((n, r, r), dtype=A.dtype)
    di = cp.arange(r)
    S_hat[:, di, di] = s_hat

    V_hat = Vt_hat.transpose(0, 2, 1)                     # (n, r, p) -> (n, p, r)

    # Inverse mode-3 (apply M^T, since M is orthogonal).
    Mt = M.T
    return mode3(Mt, U_hat), mode3(Mt, S_hat), mode3(Mt, V_hat)


def reconstruct_cupy(U, S, V, M):
    """Reverse t-SVDM: A_approx = M^T x_3 ((M x_3 U) (M x_3 S) (M x_3 V)^T)."""
    U_hat = mode3(M, U)
    S_hat = mode3(M, S)
    V_hat = mode3(M, V)
    A_hat = cp.matmul(cp.matmul(U_hat, S_hat), V_hat.transpose(0, 2, 1))
    return mode3(M.T, A_hat)


# --- Loading / generation ----------------------------------------------------

def load_fixture(path: str):
    """Read a .bin fixture (TSVD magic + (m, p, n) header + A + M)."""
    with open(path, "rb") as f:
        if f.read(4) != b"TSVD":
            raise ValueError(f"bad fixture magic in {path}")
        m, p, n = struct.unpack("<iii", f.read(12))
        A_bytes = f.read(m * p * n * 8)
        M_bytes = f.read(n * n * 8)

    # On disk: per-slice column-major (m, p). Load as (n, m, p) C-order.
    slices = [
        np.frombuffer(A_bytes, dtype=np.float64, count=m * p,
                      offset=k * m * p * 8).reshape((p, m)).T
        for k in range(n)
    ]
    A_cpu = np.stack(slices, axis=0)                      # (n, m, p)
    M_cpu = np.frombuffer(M_bytes, dtype=np.float64).reshape((n, n)).T.copy()
    return cp.asarray(A_cpu), cp.asarray(M_cpu), (m, p, n)


def generate(m: int, p: int, n: int, seed: int = 0):
    """Seed-generate A and an orthogonal M with shape (n, n) directly on the GPU."""
    rng = cp.random.default_rng(seed)
    A = rng.standard_normal((n, m, p), dtype=cp.float64)
    G = rng.standard_normal((n, n), dtype=cp.float64)
    Q, _ = cp.linalg.qr(G)
    return A, Q, (m, p, n)


# --- Driver ------------------------------------------------------------------

REPS = 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fixture", nargs="?",
                    help=".bin fixture path (mutually exclusive with --gen)")
    ap.add_argument("--gen", nargs=3, type=int, metavar=("M", "P", "N"),
                    help="seed-generate (m, p, n) tensor instead of loading")
    ap.add_argument("--dump", type=str,
                    help="write reconstruction to this path (column-major (m, p, n))")
    args = ap.parse_args()

    if args.fixture and args.gen:
        ap.error("use --fixture OR --gen, not both")
    if not args.fixture and not args.gen:
        ap.error("provide a fixture path or --gen M P N")

    if args.fixture:
        A, M, (m, p, n) = load_fixture(args.fixture)
    else:
        A, M, (m, p, n) = generate(*args.gen)

    # Warmup — pays for any lazy CUDA / kernel-cache init. Not timed.
    tsvdm_cupy(A, M)
    sync()

    times = []
    for _ in range(REPS):
        sync()
        t0 = time.perf_counter()
        U, S, V = tsvdm_cupy(A, M)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()

    A_approx = reconstruct_cupy(U, S, V, M)
    sync()
    rel_err = float(cp.linalg.norm(A_approx - A) / cp.linalg.norm(A))

    median_ms = times[len(times) // 2]
    print(f"m={m} p={p} n={n} reps={REPS} "
          f"rel_err={rel_err:.3e} median_ms={median_ms:.3f} "
          f"min_ms={times[0]:.3f} max_ms={times[-1]:.3f}")

    if args.dump:
        # Match the C++ dump format: column-major (m, p, n), one slice at a time.
        A_cpu = cp.asnumpy(A_approx)
        with open(args.dump, "wb") as f:
            for k in range(n):
                # tobytes(order='F') writes column-major; default 'C' would
                # give row-major regardless of the array's memory layout.
                f.write(A_cpu[k].tobytes(order="F"))

    return 0 if rel_err < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
