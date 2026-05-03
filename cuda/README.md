# `cuda/` — Python reference implementation (Part 0)

Pure-NumPy implementation of t-SVDM and t-SVDMII. 

## Layout

| File | Role |
|---|---|
| `tsvdm_ops.py` | Building-block ops (mode-3 product, facewise, ⋆M, etc.) |
| `tsvdm_core.py` | `tsvdm`, `tsvdmii`, `reconstruct` |
| `tsvdm_utils.py` | `random_orthogonal`, `relative_error`, `compression_ratio` |
| `tests/` | pytest suite (reconstruction, orthogonality, oracle cross-check) |

Tensor shape convention: `A` has shape `(n, m, p)`. Frontal
slice *i* is `A[i]`.

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest cuda/tests -v
```

## Quick example

```python
import numpy as np
from cuda import tsvdm, reconstruct, random_orthogonal, relative_error

rng = np.random.default_rng(0)
A = rng.standard_normal((4, 8, 8))       # (n, m, p)
M = random_orthogonal(4, rng)            # (n, n)

U, S, V = tsvdm(A, M)
print(relative_error(A, reconstruct(U, S, V, M)))   # ~1e-15
```
