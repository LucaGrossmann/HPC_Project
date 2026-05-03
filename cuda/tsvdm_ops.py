"""Building-block tensor operations for t-SVDM.

Shape convention
----------------
All tensors have shape ``(n, m, p)`` in C-order. The leading axis ``n``
indexes frontal slices; ``A[i]`` is the i-th slice as a contiguous
``(m, p)`` matrix. This lets ``np.linalg.svd(A)`` batch natively over
the leading axis. Note this differs from the paper and C++ code, which
use ``(m, p, n)``.

Matrix ``M`` has shape ``(n, n)`` and is real orthogonal.
"""

from __future__ import annotations

import numpy as np


def mode3_product(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Mode-3 product ``A ×_3 M``.

    Applies ``M`` along the tube axis. Every tube fiber
    ``A[:, i, j]`` (length ``n``) is replaced by ``M @ A[:, i, j]``.

    Parameters
    ----------
    A : ndarray, shape (n, m, p)
    M : ndarray, shape (n, n)

    Returns
    -------
    ndarray, shape (n, m, p)
    """
    return np.einsum("ij,jkl->ikl", M, A)


def facewise_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Facewise (frontal-slice) matmul.

    Output slice ``C[i] = A[i] @ B[i]``. This is not the ⋆_M product —
    it's the component of ⋆_M in the transform domain.

    Parameters
    ----------
    A : ndarray, shape (n, m, k)
    B : ndarray, shape (n, k, p)

    Returns
    -------
    ndarray, shape (n, m, p)
    """
    return A @ B


def star_m_product(A: np.ndarray, B: np.ndarray, M: np.ndarray) -> np.ndarray:
    """The ⋆_M tensor-tensor product.

    Defined as ``A ⋆_M B = (Â ▽ B̂) ×_3 M^T`` where ``▽`` is the
    facewise product and ``Â = A ×_3 M``. Not used on the t-SVDM hot
    path; kept for API parity with the public library surface
    (plan.md §5.2).

    Parameters
    ----------
    A : ndarray, shape (n, m, k)
    B : ndarray, shape (n, k, p)
    M : ndarray, shape (n, n), orthogonal

    Returns
    -------
    ndarray, shape (n, m, p)
    """
    A_hat = mode3_product(A, M)
    B_hat = mode3_product(B, M)
    C_hat = facewise_product(A_hat, B_hat)
    return mode3_product(C_hat, M.T)


def star_m_transpose(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transpose under the ⋆_M algebra.

    Given ``A`` of shape ``(n, m, p)``, returns a tensor of shape
    ``(n, p, m)`` such that ``(A ⋆_M B)^T = B^T ⋆_M A^T``. Computed
    by facewise transpose in the transform domain, then inverse mode-3.

    Parameters
    ----------
    A : ndarray, shape (n, m, p)
    M : ndarray, shape (n, n), orthogonal

    Returns
    -------
    ndarray, shape (n, p, m)
    """
    A_hat = mode3_product(A, M)
    A_hat_T = A_hat.transpose(0, 2, 1)
    return mode3_product(A_hat_T, M.T)


def frobenius_norm(A: np.ndarray) -> float:
    """Frobenius norm of a tensor (flattened 2-norm)."""
    return float(np.linalg.norm(A))
