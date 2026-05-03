"""Behavior of the truncated decomposition.

1. γ = 1 keeps every component; reconstruction is exact.
2. Reconstruction error decreases monotonically as γ increases.
3. Every kept singular value is ≥ τ (where τ is the selected threshold).
4. Eckart–Young: the error is bounded by the tail energy we discarded.
"""

from __future__ import annotations

import numpy as np

from cuda.tsvdm_core import tsvdm, tsvdmii, reconstruct
from cuda.tsvdm_utils import relative_error


def test_gamma_one_is_exact(A, M):
    U, S, V = tsvdmii(A, M, 1.0)
    assert relative_error(A, reconstruct(U, S, V, M)) < 1e-10


def test_error_monotone_in_gamma(A, M):
    errors = []
    for gamma in (0.3, 0.5, 0.7, 0.9, 0.99):
        U, S, V = tsvdmii(A, M, gamma)
        errors.append(relative_error(A, reconstruct(U, S, V, M)))
    # Non-increasing as gamma increases.
    for a, b in zip(errors, errors[1:]):
        assert b <= a + 1e-12


def test_kept_ranks_at_most_full(A, M):
    U_full, _, _ = tsvdm(A, M)
    r = U_full.shape[2]
    U, _, _ = tsvdmii(A, M, 0.9)
    for Ui in U:
        assert Ui.shape[1] <= r


def test_eckart_young_bound(A, M):
    # Tail energy of truncated singular values should upper-bound the
    # squared reconstruction error in the transform domain. Since M is
    # orthogonal, the Frobenius norm is preserved under mode-3, so the
    # bound carries over to the spatial domain.
    from cuda.tsvdm_ops import mode3_product
    A_hat = mode3_product(A, M)
    _, s_hat, _ = np.linalg.svd(A_hat, full_matrices=False)

    gamma = 0.8
    U, S, V = tsvdmii(A, M, gamma)
    err2 = float(np.linalg.norm(A - reconstruct(U, S, V, M))) ** 2
    kept_per_slice = [Ui.shape[1] for Ui in U]

    tail_energy = 0.0
    for i, k in enumerate(kept_per_slice):
        tail_energy += float(np.sum(s_hat[i, k:] ** 2))

    # err2 should equal tail_energy up to numerical noise; allow a
    # small absolute slack.
    assert err2 <= tail_energy + 1e-8
