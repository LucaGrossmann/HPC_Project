"""Reconstruction is the north-star correctness assertion.

If ``‖A − reconstruct(tsvdm(A, M), M)‖_F / ‖A‖_F < 1e-10`` then the
decomposition is correct. Every other test sharpens this in some
axis."""

from __future__ import annotations

from cuda.tsvdm_core import tsvdm, reconstruct
from cuda.tsvdm_utils import relative_error


def test_tsvdm_reconstruction(A, M):
    U, S, V = tsvdm(A, M)
    A_approx = reconstruct(U, S, V, M)
    assert relative_error(A, A_approx) < 1e-10
