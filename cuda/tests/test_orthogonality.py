"""Slicewise orthogonality of the transform-domain U and V.

In the transform domain, Û[i] and V̂[i] are the 2D SVD factors of
Â[i], so they satisfy Û[i]^T Û[i] = I and V̂[i]^T V̂[i] = I."""

from __future__ import annotations

import numpy as np

from cuda.tsvdm_ops import mode3_product
from cuda.tsvdm_core import tsvdm


def test_U_slicewise_orthogonal(A, M):
    U, _, _ = tsvdm(A, M)
    U_hat = mode3_product(U, M)
    n, _, r = U_hat.shape
    for i in range(n):
        UtU = U_hat[i].T @ U_hat[i]
        assert np.allclose(UtU, np.eye(r), atol=1e-10)


def test_V_slicewise_orthogonal(A, M):
    _, _, V = tsvdm(A, M)
    V_hat = mode3_product(V, M)
    n, _, r = V_hat.shape
    for i in range(n):
        VtV = V_hat[i].T @ V_hat[i]
        assert np.allclose(VtV, np.eye(r), atol=1e-10)
