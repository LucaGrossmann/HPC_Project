"""Shared pytest fixtures: seeded (A, M) at several sizes."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cuda.tsvdm_utils import random_orthogonal


# (n, m, p) sizes covered across the suite. Small on purpose — tests
# should take seconds, not minutes. Large-scale validation lives in
# Parts 1-5 benchmarks, not here.
SIZES = [
    (4, 8, 8),
    (8, 16, 16),
    (16, 32, 24),
]


@pytest.fixture(params=SIZES, ids=lambda s: f"n{s[0]}_m{s[1]}_p{s[2]}")
def size(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def A(size, rng):
    n, m, p = size
    return rng.standard_normal((n, m, p))


@pytest.fixture
def M(size, rng):
    n = size[0]
    return random_orthogonal(n, rng)
