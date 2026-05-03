"""Unit tests for the 5 building-block operations in tsvdm_ops."""

from __future__ import annotations

import numpy as np
import pytest

from cuda.tsvdm_ops import (
    mode3_product,
    facewise_product,
    star_m_product,
    star_m_transpose,
    frobenius_norm,
)
from cuda.tsvdm_utils import random_orthogonal


def test_mode3_product_shape(A, M):
    assert mode3_product(A, M).shape == A.shape


def test_mode3_product_identity(A):
    n = A.shape[0]
    I = np.eye(n)
    assert np.allclose(mode3_product(A, I), A)


def test_mode3_product_inverse(A, M):
    # Orthogonal M: applying M then M.T restores A.
    B = mode3_product(A, M)
    C = mode3_product(B, M.T)
    assert np.allclose(C, A, atol=1e-12)


def test_mode3_product_matches_explicit_loop(A, M):
    n = A.shape[0]
    expected = np.zeros_like(A)
    for k in range(n):
        for l in range(n):
            expected[k] += M[k, l] * A[l]
    assert np.allclose(mode3_product(A, M), expected)


def test_facewise_product_matches_per_slice(rng):
    n, m, k, p = 5, 4, 3, 6
    A = rng.standard_normal((n, m, k))
    B = rng.standard_normal((n, k, p))
    C = facewise_product(A, B)
    assert C.shape == (n, m, p)
    for i in range(n):
        assert np.allclose(C[i], A[i] @ B[i])


def test_star_m_product_reduces_to_facewise_when_M_is_identity(rng):
    n, m, k, p = 4, 5, 3, 6
    A = rng.standard_normal((n, m, k))
    B = rng.standard_normal((n, k, p))
    I = np.eye(n)
    assert np.allclose(star_m_product(A, B, I), facewise_product(A, B))


def test_star_m_transpose_shape_and_inverse(A, M):
    AT = star_m_transpose(A, M)
    assert AT.shape == (A.shape[0], A.shape[2], A.shape[1])
    # Transposing twice is identity (up to float precision).
    ATT = star_m_transpose(AT, M)
    assert np.allclose(ATT, A, atol=1e-12)


def test_frobenius_norm_matches_numpy(A):
    assert frobenius_norm(A) == pytest.approx(float(np.linalg.norm(A)))


def test_frobenius_norm_preserved_by_mode3(A, M):
    # Orthogonal mode-3 transform preserves the Frobenius norm.
    assert frobenius_norm(mode3_product(A, M)) == pytest.approx(frobenius_norm(A))
