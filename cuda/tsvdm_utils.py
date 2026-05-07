"""Utilities: validation, random orthogonal matrices, error metrics."""

from __future__ import annotations

import numpy as np


def random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random real orthogonal ``(n, n)`` matrix from QR of a Gaussian.

    The Haar distribution on O(n) is obtained by sign-correcting the
    diagonal of R so that R_ii > 0. We apply that correction so the
    output is a proper Haar sample (not just any orthogonal matrix).
    """
    G = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(G)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return Q * d  # broadcast across columns


def relative_error(A: np.ndarray, A_approx: np.ndarray) -> float:
    """Relative Frobenius error ``‖A − A_approx‖_F / ‖A‖_F``."""
    return float(np.linalg.norm(A - A_approx) / np.linalg.norm(A))


def compression_ratio(A: np.ndarray, stored_values: int) -> float:
    """Ratio of stored floats in the approximation to A's total size.

    ``stored_values`` is the total number of floats kept across U, S, V.
    """
    return float(stored_values) / float(A.size)


def _validate_inputs(A: np.ndarray, M: np.ndarray) -> None:
    """Check shape and layout constraints used throughout the package."""
    if A.ndim != 3:
        raise ValueError(f"A must be 3-D (shape (n, m, p)); got ndim={A.ndim}")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2-D; got shape {M.shape}")
    n = A.shape[0]
    if M.shape[0] != n:
        raise ValueError(
            f"M side length {M.shape[0]} must equal n={n} (A.shape[0])"
        )
