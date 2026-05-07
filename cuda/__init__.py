"""Part 0 — Python reference implementation of t-SVDM.

This package is the correctness oracle for all other implementations
(Parts 1-5: serial C++, OpenMP, MPI, CuPy, Julia).
"""

from .tsvdm_core import tsvdm, reconstruct
from .tsvdm_ops import (
    mode3_product,
    facewise_product,
    star_m_product,
    star_m_transpose,
    frobenius_norm,
)
from .tsvdm_utils import relative_error, compression_ratio, random_orthogonal

__all__ = [
    "tsvdm",
    "reconstruct",
    "mode3_product",
    "facewise_product",
    "star_m_product",
    "star_m_transpose",
    "frobenius_norm",
    "relative_error",
    "compression_ratio",
    "random_orthogonal",
]
