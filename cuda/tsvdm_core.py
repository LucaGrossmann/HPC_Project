"""Core algorithms: ``tsvdm``, ``reconstruct``.

References
----------
Kilmer, Horesh, Avron, Newman (2021, PNAS). Algorithm 2.

Tensor shape convention is ``(n, m, p)`` — see ``tsvdm_ops.py`` docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .tsvdm_ops import mode3_product
from .tsvdm_utils import _validate_inputs


def tsvdm(
    A: np.ndarray, M: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """t-SVDM decomposition (Kilmer et al. 2021, Algorithm 2).

    Thin factorization: ``A ≈ U ⋆_M S ⋆_M V^T`` with ``r = min(m, p)``.

    Parameters
    ----------
    A : ndarray, shape (n, m, p)
    M : ndarray, shape (n, n), orthogonal

    Returns
    -------
    U : ndarray, shape (n, m, r)
    S : ndarray, shape (n, r, r)   (f-diagonal; each slice is diagonal)
    V : ndarray, shape (n, p, r)
    """
    _validate_inputs(A, M)

    A_hat = mode3_product(A, M)
    U_hat, s_hat, Vt_hat = np.linalg.svd(A_hat, full_matrices=False)

    # s_hat has shape (n, r); build the f-diagonal tensor (n, r, r)
    n, r = s_hat.shape
    S_hat = np.zeros((n, r, r), dtype=A.dtype)
    diag_idx = np.arange(r)
    S_hat[:, diag_idx, diag_idx] = s_hat

    # Inverse mode-3 transform: M is orthogonal, so M^{-1} = M^T.
    Mt = M.T
    U = mode3_product(U_hat, Mt)
    S = mode3_product(S_hat, Mt)
    V = mode3_product(Vt_hat.transpose(0, 2, 1), Mt)
    return U, S, V


def reconstruct(
    U: np.ndarray, S: np.ndarray, V: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Reconstruct ``A_approx`` from t-SVDM factors.

    ``A = U ⋆_M S ⋆_M V^T``. In the transform domain this is just a facewise
    matmul: ``Â = Û @ Ŝ @ V̂^T``, then ``A = Â ×_3 M^T``.

    Parameters
    ----------
    U, S, V : ndarrays of shape (n, m, r), (n, r, r), (n, p, r).
    M       : ndarray, shape (n, n), orthogonal.

    Returns
    -------
    A_approx : ndarray, shape (n, m, p)
    """
    U_hat = mode3_product(U, M)
    S_hat = mode3_product(S, M)
    V_hat = mode3_product(V, M)
    A_hat = U_hat @ S_hat @ V_hat.transpose(0, 2, 1)
    return mode3_product(A_hat, M.T)
