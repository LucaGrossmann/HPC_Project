"""Core algorithms: ``tsvdm``, ``tsvdmii``, ``reconstruct``.

References
----------
Kilmer, Horesh, Avron, Newman (2021, PNAS). Algorithms 2 and 3.

Tensor shape convention is ``(n, m, p)`` — see ``tsvdm_ops.py`` docstring.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from .tsvdm_ops import mode3_product
from .tsvdm_utils import _validate_inputs

Factor = Union[np.ndarray, List[np.ndarray]]


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


def tsvdmii(
    A: np.ndarray, M: np.ndarray, gamma: float
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Truncated t-SVDM (Kilmer et al. 2021, Algorithm 3).

    Keeps the leading singular components (in the transform domain)
    whose squared values cover an energy fraction of ``gamma``.

    Factors are returned **in the transform domain**, ragged across
    slices — each slice ``i`` may have a different kept rank ``ρ_i``.
    This is intentional: per paper §6D, projecting back to the spatial
    domain requires zero-padding every slice to ``max ρ_i``, which
    destroys the storage savings. Call ``reconstruct(U, S, V, M)`` to
    get the spatial-domain approximation when needed.

    Parameters
    ----------
    A     : ndarray, shape (n, m, p)
    M     : ndarray, shape (n, n), orthogonal
    gamma : float in (0, 1]; fraction of energy to retain.

    Returns
    -------
    U_list : list of n ndarrays, each shape (m, ρ_i)
    S_list : list of n ndarrays, each shape (ρ_i, ρ_i)  (diagonal)
    V_list : list of n ndarrays, each shape (p, ρ_i)

    """
    _validate_inputs(A, M)
    if not (0 < gamma <= 1):
        raise ValueError(f"gamma must be in (0, 1]; got {gamma}")

    A_hat = mode3_product(A, M)
    U_hat, s_hat, Vt_hat = np.linalg.svd(A_hat, full_matrices=False)
    # s_hat shape: (n, r) where r = min(m, p). Each row: singular values of slice i.

    squared = (s_hat ** 2).ravel()
    order = np.argsort(-squared)  # descending
    v_sorted = squared[order]

    total_energy = float(np.sum(squared))
    target = gamma * total_energy

    # Smallest J such that cumsum[J] > target (strict, per plan.md §1).
    cumsum = np.cumsum(v_sorted)
    J_candidates = np.nonzero(cumsum > target)[0]
    if J_candidates.size == 0:
        # gamma == 1 and floating-point rounding: take all.
        J = len(v_sorted) - 1
    else:
        J = int(J_candidates[0])

    tau = float(np.sqrt(v_sorted[J]))

    U_list: List[np.ndarray] = []
    S_list: List[np.ndarray] = []
    V_list: List[np.ndarray] = []
    for i in range(A.shape[0]):
        kept = int(np.sum(s_hat[i] >= tau))
        U_list.append(U_hat[i, :, :kept].copy())
        S_list.append(np.diag(s_hat[i, :kept]))
        V_list.append(Vt_hat[i, :kept, :].T.copy())

    return U_list, S_list, V_list


def reconstruct(
    U: Factor, S: Factor, V: Factor, M: np.ndarray
) -> np.ndarray:
    """Reconstruct ``A_approx`` from t-SVDM or t-SVDMII factors.

    Accepts either full ndarrays (from ``tsvdm``, shape ``(n, m, r)``
    etc.) or ragged lists of per-slice ndarrays (from ``tsvdmii``).

    Parameters
    ----------
    U, S, V : ndarrays of shape (n, m, r), (n, r, r), (n, p, r),
              **or** Python lists of n per-slice matrices.
    M       : ndarray, shape (n, n), orthogonal.

    Returns
    -------
    A_approx : ndarray, shape (n, m, p)
    """
    if isinstance(U, list):
        return _reconstruct_ragged(U, S, V, M)
    return _reconstruct_full(U, S, V, M)


def _reconstruct_full(
    U: np.ndarray, S: np.ndarray, V: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Reconstruction for the non-truncated (full ndarray) case.

    A = U ⋆_M S ⋆_M V^T. In the transform domain this is just a facewise
    matmul: Â = Û @ Ŝ @ V̂^T, then A = Â ×_3 M^T.
    """
    U_hat = mode3_product(U, M)
    S_hat = mode3_product(S, M)
    V_hat = mode3_product(V, M)
    A_hat = U_hat @ S_hat @ V_hat.transpose(0, 2, 1)
    return mode3_product(A_hat, M.T)


def _reconstruct_ragged(
    U_list: List[np.ndarray],
    S_list: List[np.ndarray],
    V_list: List[np.ndarray],
    M: np.ndarray,
) -> np.ndarray:
    """Reconstruction for t-SVDMII ragged, transform-domain factors.

    Factors are already in the transform domain with per-slice rank
    ``ρ_i``. We form ``Â[i] = U_list[i] @ S_list[i] @ V_list[i]^T``
    slice-by-slice into a dense ``Â`` (shape ``(n, m, p)``), then apply
    the inverse mode-3 transform.
    """
    n = len(U_list)
    m = U_list[0].shape[0]
    p = V_list[0].shape[0]
    A_hat = np.zeros((n, m, p), dtype=U_list[0].dtype)
    for i in range(n):
        A_hat[i] = U_list[i] @ S_list[i] @ V_list[i].T
    return mode3_product(A_hat, M.T)
