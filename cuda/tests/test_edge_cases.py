"""Edge cases: n=1, square m=p, identity M."""

from __future__ import annotations

import numpy as np

from cuda.tsvdm_core import tsvdm, reconstruct
from cuda.tsvdm_utils import random_orthogonal, relative_error


def test_single_slice():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((1, 6, 5))
    M = random_orthogonal(1, rng)   # just ±1
    U, S, V = tsvdm(A, M)
    assert relative_error(A, reconstruct(U, S, V, M)) < 1e-12


def test_square_slices():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((4, 7, 7))
    M = random_orthogonal(4, rng)
    U, S, V = tsvdm(A, M)
    assert relative_error(A, reconstruct(U, S, V, M)) < 1e-12


def test_identity_M():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 6, 8))
    M = np.eye(5)
    # With M = I, each slice's SVD is independent — verify reconstruction.
    U, S, V = tsvdm(A, M)
    assert relative_error(A, reconstruct(U, S, V, M)) < 1e-12
