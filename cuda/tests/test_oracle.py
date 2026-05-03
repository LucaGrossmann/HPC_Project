"""Cross-validate our tsvdm against ``mprod_package`` (the canonical
CPU reference implementation of Kilmer et al. 2021).

mprod uses tensor layout ``(m, p, n)`` while we use ``(n, m, p)``. We
transpose at the boundary. Because SVD factors are unique only up to
sign flips on each column, we compare the **reconstruction** of A
(which is sign-invariant), not the raw factors.
"""

from __future__ import annotations

import numpy as np
import pytest

from cuda.tsvdm_core import tsvdm, reconstruct
from cuda.tsvdm_utils import random_orthogonal, relative_error

mprod = pytest.importorskip("mprod")
mprod_decompositions = pytest.importorskip("mprod.decompositions")


def test_tsvdm_matches_mprod_reconstruction():
    rng = np.random.default_rng(42)
    n, m, p = 6, 10, 8

    # Same A and M for both implementations.
    A_ours = rng.standard_normal((n, m, p))                        # (n, m, p)
    A_mprod = A_ours.transpose(1, 2, 0).copy()                     # (m, p, n)
    M = random_orthogonal(n, rng)

    # Ours.
    U, S, V = tsvdm(A_ours, M)
    A_approx_ours = reconstruct(U, S, V, M)

    # mprod: wrap M as tube-fiber callables; x_m3(M) applies M along axis 2.
    fun_m = mprod.x_m3(M)
    inv_m = mprod.x_m3(M.T)
    u_ref, s_ref, v_ref = mprod_decompositions.svdm(A_mprod, fun_m, inv_m)

    # mprod's s is (k, n); rebuild the f-diagonal (m, p, n) then reconstruct.
    k = s_ref.shape[0]
    s_dense = np.zeros((k, k, n))
    diag = np.arange(k)
    s_dense[diag, diag, :] = s_ref
    # Reconstruct via mprod's own m-product: u ⋆ s ⋆ v^T.
    A_approx_ref = mprod.m_prod(
        mprod.m_prod(u_ref, s_dense, fun_m, inv_m),
        mprod.tensor_mtranspose(v_ref, fun_m, inv_m),
        fun_m, inv_m,
    )
    # Transpose mprod's reconstruction back to our (n, m, p) layout.
    A_approx_ref = A_approx_ref.transpose(2, 0, 1)

    # Both should recover A (self-consistency) to ~1e-12.
    assert relative_error(A_ours, A_approx_ours) < 1e-10
    assert relative_error(A_ours, A_approx_ref) < 1e-10
    # And the two reconstructions should agree with each other.
    assert relative_error(A_approx_ours, A_approx_ref) < 1e-10
